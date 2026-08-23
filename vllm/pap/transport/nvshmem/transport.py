# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention--Projection transport over same-host NVSHMEM P2P."""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecTransportClosed,
    PAPTensorTransport,
)
from vllm.pap.protocol.offload_exec import (
    dtype_from_name,
    dtype_name,
    layer_index_and_template,
    layer_name_from_template,
)
from vllm.pap.transport.nvshmem.runtime import PAPNVSHMEMError
from vllm.pap.transport.nvshmem.world import (
    get_pap_nvshmem_world,
)

_DEFAULT_BUFFER_BYTES = 16 * 1024 * 1024
_CONTROL_MAGIC = b"PNSH"
_CONTROL_HEADER = struct.Struct("<4sBBHIHdIHHH")
_CONTROL_ROW = struct.Struct("<HI")
_METADATA_VERSION = 3

_READY_QKV = 0
_READY_OUTPUT = 1
_RELEASE_QKV = 2


@dataclass
class _QKVStepPlan:
    descriptor: PAPOffloadExecBatchDescriptor
    dtype: torch.dtype
    qkv_width: int
    layer_count: int
    layer_template: tuple[str, str]
    qkv_tensor: torch.Tensor | None = None
    step_context: Any = None


@cache
def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _encode_step_plan(
    descriptor: PAPOffloadExecBatchDescriptor,
    *,
    dtype: torch.dtype,
    qkv_width: int,
    layer_count: int,
) -> bytes:
    template = descriptor.metadata_template
    if template is None:
        request_ids = tuple(item.request_id for item in descriptor.items)
        steps = tuple(int(item.step) for item in descriptor.items)
        scales = tuple(float(item.scale) for item in descriptor.items)
    else:
        request_ids = tuple(str(value) for value in template["r"])
        steps = tuple(int(value) for value in template["s"])
        scales = tuple(float(value) for value in template["a"])
    if not request_ids or not (len(request_ids) == len(steps) == len(scales)):
        raise PAPNVSHMEMError("PAP NVSHMEM step rows are malformed")
    scale = scales[0]
    if any(value != scale for value in scales):
        raise PAPNVSHMEMError("PAP NVSHMEM step has mixed attention scales")
    if qkv_width <= 0 or qkv_width >= 1 << 32:
        raise PAPNVSHMEMError("PAP NVSHMEM QKV width is out of range")
    if layer_count <= 0 or layer_count >= 1 << 16:
        raise PAPNVSHMEMError("PAP NVSHMEM layer count is out of range")
    if len(request_ids) >= 1 << 16:
        raise PAPNVSHMEMError("PAP NVSHMEM row count is out of range")

    layer_name = descriptor.layer_name.encode("utf-8")
    encoded_dtype_name = dtype_name(dtype).encode("ascii")
    batch_suffix = (
        descriptor.batch_id_suffix
        or ",".join(
            f"{request_id}@{step}" for request_id, step in zip(request_ids, steps)
        )
    ).encode("utf-8")
    static_fields = (layer_name, encoded_dtype_name, batch_suffix)
    if any(len(value) >= 1 << 16 for value in static_fields):
        raise PAPNVSHMEMError("PAP NVSHMEM step string is too large")

    encoded_rows: list[tuple[bytes, int]] = []
    rows_bytes = 0
    for request_id, step in zip(request_ids, steps):
        encoded_id = request_id.encode("utf-8")
        if len(encoded_id) >= 1 << 16 or step < 0 or step >= 1 << 32:
            raise PAPNVSHMEMError("PAP NVSHMEM step row is out of range")
        encoded_rows.append((encoded_id, step))
        rows_bytes += _CONTROL_ROW.size + len(encoded_id)
    record_bytes = (
        _CONTROL_HEADER.size + sum(len(value) for value in static_fields) + rows_bytes
    )
    record = bytearray(record_bytes)
    _CONTROL_HEADER.pack_into(
        record,
        0,
        _CONTROL_MAGIC,
        _METADATA_VERSION,
        0,
        layer_count,
        qkv_width,
        len(request_ids),
        scale,
        record_bytes,
        len(layer_name),
        len(encoded_dtype_name),
        len(batch_suffix),
    )
    cursor = _CONTROL_HEADER.size
    for value in static_fields:
        record[cursor : cursor + len(value)] = value
        cursor += len(value)
    for encoded_id, step in encoded_rows:
        _CONTROL_ROW.pack_into(record, cursor, len(encoded_id), step)
        cursor += _CONTROL_ROW.size
        record[cursor : cursor + len(encoded_id)] = encoded_id
        cursor += len(encoded_id)
    return bytes(record)


def _decode_step_plan(
    control: torch.Tensor,
    *,
    capacity: int,
) -> tuple[PAPOffloadExecBatchDescriptor, torch.dtype, int, int]:
    host_view = memoryview(control.numpy())
    if capacity < _CONTROL_HEADER.size:
        raise PAPNVSHMEMError("PAP NVSHMEM control record is truncated")
    (
        magic,
        version,
        flags,
        layer_count,
        qkv_width,
        row_count,
        scale,
        record_bytes,
        layer_name_bytes,
        dtype_name_bytes,
        batch_suffix_bytes,
    ) = _CONTROL_HEADER.unpack_from(host_view, 0)
    if magic != _CONTROL_MAGIC or version != _METADATA_VERSION or flags != 0:
        raise PAPNVSHMEMError("PAP NVSHMEM control header is incompatible")
    if record_bytes < _CONTROL_HEADER.size or record_bytes > capacity:
        raise PAPNVSHMEMError("PAP NVSHMEM control record length is invalid")

    cursor = _CONTROL_HEADER.size

    def read_string(size: int, encoding: str) -> str:
        nonlocal cursor
        end = cursor + size
        if end > record_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM control string is truncated")
        value = bytes(host_view[cursor:end]).decode(encoding)
        cursor = end
        return value

    layer_name = read_string(layer_name_bytes, "utf-8")
    dtype_name = read_string(dtype_name_bytes, "ascii")
    batch_suffix = read_string(batch_suffix_bytes, "utf-8")
    request_ids: list[str] = []
    steps: list[int] = []
    for _ in range(row_count):
        if cursor + _CONTROL_ROW.size > record_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM control row is truncated")
        request_id_bytes, step = _CONTROL_ROW.unpack_from(host_view, cursor)
        cursor += _CONTROL_ROW.size
        request_ids.append(read_string(request_id_bytes, "utf-8"))
        steps.append(step)
    if cursor != record_bytes or not request_ids:
        raise PAPNVSHMEMError("PAP NVSHMEM control record has trailing data")
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=(),
        batch_id_suffix=batch_suffix,
        metadata_template={
            "r": tuple(request_ids),
            "s": tuple(steps),
            "a": (float(scale),) * len(request_ids),
        },
    )
    return descriptor, dtype_from_name(dtype_name), qkv_width, layer_count


class PAPNVSHMEMTransport:
    """One peer view over a process-global NVSHMEM world."""

    transport = PAPTensorTransport.NVSHMEM

    def __init__(
        self,
        *,
        actor_id: str,
        device: torch.device,
        buffer_bytes: int | None = None,
    ) -> None:
        self.actor_id = str(actor_id)
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise PAPNVSHMEMError("PAP NVSHMEM requires an indexed CUDA device")
        configured_buffer_bytes = os.environ.get("PAP_NVSHMEM_BUFFER_BYTES")
        self.buffer_bytes = int(
            buffer_bytes
            if buffer_bytes is not None
            else configured_buffer_bytes or _DEFAULT_BUFFER_BYTES
        )
        if self.buffer_bytes <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM buffer size must be positive")
        self.world = get_pap_nvshmem_world(
            device_index=self.device.index,
            buffer_bytes=self.buffer_bytes,
        )
        self.peer_rank: int | None = None
        self._world_ready = False
        self._stopped = threading.Event()
        self._closed = False
        self._send_lock = threading.Lock()
        self._step_prepare_handler: Any = None
        self._qkv_send_generation = 1
        self._qkv_recv_generation = 1
        self._last_qkv_sent = 0
        self._qkv_plan: _QKVStepPlan | None = None
        self._payload_views: dict[
            tuple[int, tuple[int, int], torch.dtype, int], torch.Tensor
        ] = {}
        with torch.accelerator.device_index(self.device.index):
            self._qkv_stream = torch.Stream(device=self.device)
            control_bytes = self.world.config.control_bytes
            self._control_host = torch.empty(
                control_bytes,
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._control_send = torch.empty(
                control_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        self._graph_layer_count = 0

    @property
    def local_agent_metadata(self) -> bytes:
        payload = {
            "version": _METADATA_VERSION,
            "transport": "nvshmem_graph",
            "hostname": socket.gethostname(),
            "rank": self.world.rank,
            "world_size": self.world.world_size,
            "buffer_bytes": self.buffer_bytes,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def bind_peer(self, peer_agent_metadata: bytes) -> None:
        if self._closed:
            raise PAPNVSHMEMError("PAP NVSHMEM transport is closed")
        try:
            payload = json.loads(peer_agent_metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PAPNVSHMEMError("invalid PAP NVSHMEM peer metadata") from exc
        if payload.get("version") != _METADATA_VERSION:
            raise PAPNVSHMEMError("PAP NVSHMEM metadata version mismatch")
        if payload.get("transport") != "nvshmem_graph":
            raise PAPNVSHMEMError("PAP NVSHMEM peer uses another transport")
        if str(payload.get("hostname")) != socket.gethostname():
            raise PAPNVSHMEMError("PAP NVSHMEM Graph requires the same host")
        if int(payload.get("world_size", -1)) != self.world.world_size:
            raise PAPNVSHMEMError("PAP NVSHMEM world size mismatch")
        if int(payload.get("buffer_bytes", -1)) != self.buffer_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM buffer size mismatch")
        peer_rank = int(payload.get("rank", -1))
        if peer_rank < 0 or peer_rank >= self.world.world_size:
            raise PAPNVSHMEMError("PAP NVSHMEM peer rank is invalid")
        if peer_rank == self.world.rank:
            raise PAPNVSHMEMError("PAP NVSHMEM cannot bind its own PE")
        if self.peer_rank is not None and self.peer_rank != peer_rank:
            raise PAPNVSHMEMError("PAP NVSHMEM transport changed peer rank")
        self.peer_rank = peer_rank
        self._ensure_world_ready()

    def send_step_prepare(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        dtype: torch.dtype,
        qkv_width: int = 0,
        layer_count: int = 0,
    ) -> None:
        if qkv_width <= 0 or layer_count <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM requires a whole-step Graph plan")
        layer_info = layer_index_and_template(descriptor.layer_name)
        if layer_info is None or layer_info[0] != 0:
            raise PAPNVSHMEMError("PAP NVSHMEM step plan must start at layer zero")
        self._send_control(
            _encode_step_plan(
                descriptor,
                dtype=dtype,
                qkv_width=qkv_width,
                layer_count=layer_count,
            )
        )

    def recv_graph_step_plan(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, torch.Tensor, Any]:
        """Receive one CPU step plan without consuming per-layer payloads."""
        self._require_open()
        self._ensure_world_ready()
        if self._qkv_plan is None:
            self._receive_step_plan()
        plan = self._qkv_plan
        if plan is None or plan.qkv_tensor is None:
            raise PAPNVSHMEMError("PAP NVSHMEM graph step plan is invalid")
        descriptor = PAPOffloadExecBatchDescriptor(
            layer_name=layer_name_from_template(plan.layer_template, 0),
            items=(),
            batch_id_suffix=plan.descriptor.batch_id_suffix,
            metadata_template=plan.descriptor.metadata_template,
        )
        self._qkv_plan = None
        return descriptor, plan.qkv_tensor, plan.step_context

    def graph_begin_step(
        self,
        *,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Capture the per-peer device epoch increment once per replay."""
        self._require_graph_ready(layer_count)
        self.world.runtime.graph_advance_epoch(
            epoch=self._graph_epoch_tensor(),
            stream=stream,
        )

    def graph_wait_qkv(
        self,
        *,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Wait for Projection to publish one QKV tensor."""
        self._graph_wait(
            kind=_READY_QKV,
            source_rank=self._require_peer_rank(),
            layer_index=layer_index,
            layer_count=layer_count,
            generation_delta=0,
            stream=stream,
        )

    def graph_send_output(
        self,
        tensor: torch.Tensor,
        *,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Put one Attention output and publish its generation."""
        self._graph_put(
            tensor,
            ready_kind=_READY_OUTPUT,
            layer_index=layer_index,
            layer_count=layer_count,
            stream=stream,
        )

    def graph_qkv_view(
        self,
        *,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the stable QKV view written by the bound Projection PE."""
        return self._graph_payload_view(
            shape=shape,
            dtype=dtype,
        )

    def graph_output_view(
        self,
        *,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the stable output view written by the bound Attention PE."""
        return self._graph_payload_view(
            shape=shape,
            dtype=dtype,
        )

    def graph_dispatch_routed_qkv(
        self,
        tensor: torch.Tensor,
        *,
        packed: torch.Tensor,
        route_indices: torch.Tensor,
        route_counts: torch.Tensor,
        peer_ranks: torch.Tensor,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Dispatch variable row counts to every active PA in one kernel."""
        self._require_graph_ready(layer_count)
        self.world.runtime.graph_dispatch_qkv(
            data=self._data(),
            data_slot_bytes=self.buffer_bytes,
            source=tensor,
            packed=packed,
            route_indices=route_indices,
            route_counts=route_counts,
            peer_ranks=peer_ranks,
            signals=self._graph_signals(),
            epochs=self._graph_epochs(),
            layer_count=layer_count,
            layer_index=layer_index,
            stream=stream,
        )

    def graph_gather_routed_output(
        self,
        output: torch.Tensor,
        *,
        route_indices: torch.Tensor,
        route_counts: torch.Tensor,
        peer_ranks: torch.Tensor,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Wait and scatter every active PA output in one barrier kernel."""
        self._require_graph_ready(layer_count)
        self.world.runtime.graph_gather_output(
            data=self._data(),
            data_slot_bytes=self.buffer_bytes,
            output=output,
            route_indices=route_indices,
            route_counts=route_counts,
            peer_ranks=peer_ranks,
            signals=self._graph_signals(),
            epochs=self._graph_epochs(),
            layer_count=layer_count,
            layer_index=layer_index,
            stream=stream,
        )

    def set_step_prepare_handler(self, handler: Any) -> None:
        self._step_prepare_handler = handler

    def step_prepare_stream(self) -> torch.Stream:
        return self._qkv_stream

    def stop_receiving(self) -> None:
        self._stopped.set()
        if not self._world_ready or self.peer_rank is None:
            return
        signal_i32 = self._signals().int32_tensor
        if signal_i32 is None:
            return
        signal_index = self.world.signal_offset(
            _READY_QKV,
            self.peer_rank,
        ) // struct.calcsize("i")
        with torch.accelerator.device_index(self.device.index):
            signal_i32[signal_index].fill_((1 << 31) - 1)

    def close(self) -> None:
        self._closed = True
        self._stopped.set()

    def _send_control(self, record: bytes) -> None:
        self._require_open()
        self._ensure_world_ready()
        peer_rank = self._require_peer_rank()
        record_bytes = len(record)
        if record_bytes > self.world.config.control_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM step metadata is too large")
        source = torch.frombuffer(bytearray(record), dtype=torch.uint8)
        with self._send_lock:
            generation = self._qkv_send_generation
            stream = self._qkv_stream
            current = torch.accelerator.current_stream(self.device)
            stream.wait_stream(current)
            if self._last_qkv_sent:
                self.world.runtime.wait_signal_on_stream(
                    signal=self._signals(),
                    signal_offset=self.world.signal_offset(
                        _RELEASE_QKV,
                        peer_rank,
                    ),
                    generation=self._last_qkv_sent,
                    stream=stream,
                )
            self._control_host[:record_bytes].copy_(source)
            with torch.accelerator.stream(stream):
                self._control_send[:record_bytes].copy_(
                    self._control_host[:record_bytes],
                    non_blocking=True,
                )
            self.world.runtime.put_signal_on_stream(
                destination=self._control(),
                destination_offset=self.world.control_slot_offset(self.world.rank),
                source=self._control_send,
                num_bytes=record_bytes,
                signal=self._signals(),
                signal_offset=self.world.signal_offset(_READY_QKV, self.world.rank),
                generation=generation,
                peer=peer_rank,
                stream=stream,
            )
            self._qkv_send_generation += 1
            self._last_qkv_sent = generation

    def _receive_step_plan(self) -> None:
        generation = self._qkv_recv_generation
        peer_rank = self._require_peer_rank()
        stream = self._qkv_stream
        self.world.runtime.wait_signal_on_stream(
            signal=self._signals(),
            signal_offset=self.world.signal_offset(_READY_QKV, peer_rank),
            generation=generation,
            stream=stream,
        )
        control = self._control()
        offset = self.world.control_slot_offset(peer_rank)
        with torch.accelerator.stream(stream):
            self._control_host.copy_(
                control.tensor.narrow(0, offset, self.world.config.control_bytes),
                non_blocking=True,
            )
        stream.synchronize()
        self._require_open()
        descriptor, dtype, qkv_width, layer_count = _decode_step_plan(
            self._control_host,
            capacity=self.world.config.control_bytes,
        )
        layer_info = layer_index_and_template(descriptor.layer_name)
        if layer_info is None:
            raise PAPNVSHMEMError("PAP NVSHMEM step layer name is invalid")
        plan = _QKVStepPlan(
            descriptor=descriptor,
            dtype=dtype,
            qkv_width=qkv_width,
            layer_count=layer_count,
            layer_template=layer_info[1],
        )
        if plan.qkv_width <= 0 or plan.layer_count <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM step plan shape is invalid")
        qkv_num_bytes = (
            descriptor.item_count * plan.qkv_width * _element_size(plan.dtype)
        )
        plan.qkv_tensor = self._payload_view(
            source_rank=peer_rank,
            shape=(descriptor.item_count, plan.qkv_width),
            dtype=plan.dtype,
            num_bytes=qkv_num_bytes,
        )
        self._qkv_recv_generation += 1
        self._release(
            release_kind=_RELEASE_QKV,
            generation=generation,
            stream=stream,
        )
        if self._step_prepare_handler is not None:
            with torch.accelerator.stream(stream):
                plan.step_context = self._step_prepare_handler(descriptor, plan.dtype)
        self._qkv_plan = plan

    def _release(
        self,
        *,
        release_kind: int,
        generation: int,
        stream: torch.Stream,
    ) -> None:
        self.world.runtime.signal_on_stream(
            signal=self._signals(),
            signal_offset=self.world.signal_offset(release_kind, self.world.rank),
            generation=generation,
            peer=self._require_peer_rank(),
            stream=stream,
        )

    def _payload_view(
        self,
        *,
        source_rank: int,
        shape: tuple[int, int],
        dtype: torch.dtype,
        num_bytes: int,
    ) -> torch.Tensor:
        if num_bytes > self.buffer_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM received payload exceeds its slot")
        cache_key = (source_rank, shape, dtype, num_bytes)
        cached = self._payload_views.get(cache_key)
        if cached is not None:
            return cached
        data = self._data().tensor
        offset = self.world.data_slot_offset(source_rank)
        tensor = data.narrow(0, offset, num_bytes).view(dtype).reshape(shape)
        self._payload_views[cache_key] = tensor
        return tensor

    def _graph_payload_view(
        self,
        *,
        shape: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self._ensure_world_ready()
        num_bytes = shape[0] * shape[1] * _element_size(dtype)
        return self._payload_view(
            source_rank=self._require_peer_rank(),
            shape=shape,
            dtype=dtype,
            num_bytes=num_bytes,
        )

    def _graph_wait(
        self,
        *,
        kind: int,
        source_rank: int,
        layer_index: int,
        layer_count: int,
        generation_delta: int,
        stream: torch.Stream,
    ) -> None:
        self._require_graph_ready(layer_count)
        self.world.runtime.graph_wait_signal(
            signal=self._graph_signals(),
            signal_offset=self.world.signal_offset(kind, source_rank),
            epoch=self._graph_epoch_tensor(),
            layer_count=layer_count,
            layer_index=layer_index,
            generation_delta=generation_delta,
            stream=stream,
        )

    def _graph_put(
        self,
        tensor: torch.Tensor,
        *,
        ready_kind: int,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        self._require_graph_ready(layer_count)
        source = tensor.detach()
        num_bytes = source.numel() * source.element_size()
        if num_bytes > self.buffer_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM graph payload exceeds its slot")
        self.world.runtime.graph_put_signal(
            destination=self._data(),
            destination_offset=self.world.data_slot_offset(self.world.rank),
            source=source,
            num_bytes=num_bytes,
            signal=self._graph_signals(),
            signal_offset=self.world.signal_offset(
                ready_kind,
                self.world.rank,
            ),
            epoch=self._graph_epoch_tensor(),
            layer_count=layer_count,
            layer_index=layer_index,
            peer=self._require_peer_rank(),
            stream=stream,
        )

    def _require_graph_ready(self, layer_count: int) -> None:
        self._require_open()
        self._ensure_world_ready()
        if layer_count <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM graph layer count is invalid")
        if self._graph_layer_count not in (0, layer_count):
            raise PAPNVSHMEMError("PAP NVSHMEM graph layer count changed")
        self._graph_layer_count = layer_count

    def _data(self):
        if self.world.data is None:
            raise PAPNVSHMEMError("PAP NVSHMEM data allocation is unavailable")
        return self.world.data

    def _control(self):
        if self.world.control is None:
            raise PAPNVSHMEMError("PAP NVSHMEM control allocation is unavailable")
        return self.world.control

    def _signals(self):
        if self.world.signals is None:
            raise PAPNVSHMEMError("PAP NVSHMEM signal allocation is unavailable")
        return self.world.signals

    def _graph_signals(self):
        if self.world.graph_signals is None:
            raise PAPNVSHMEMError("PAP NVSHMEM graph signals are unavailable")
        return self.world.graph_signals

    def _graph_epoch_tensor(self) -> torch.Tensor:
        epochs = self._graph_epochs()
        peer_rank = self._require_peer_rank()
        return epochs.narrow(0, peer_rank, 1)

    def _graph_epochs(self) -> torch.Tensor:
        epochs = self.world.graph_epochs
        if epochs is None:
            raise PAPNVSHMEMError("PAP NVSHMEM graph epochs are unavailable")
        return epochs

    def _require_peer_rank(self) -> int:
        if self.peer_rank is None:
            raise PAPNVSHMEMError("PAP NVSHMEM peer is not bound")
        return self.peer_rank

    def _ensure_world_ready(self) -> None:
        if self._world_ready:
            return
        self.world.wait_ready()
        self._world_ready = True

    def _require_open(self) -> None:
        if self._closed or self._stopped.is_set():
            raise PAPOffloadExecTransportClosed("PAP NVSHMEM transport is closed")


def build_nvshmem_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
) -> PAPNVSHMEMTransport:
    """Build the NVSHMEM transport selected by PAP configuration."""
    return PAPNVSHMEMTransport(
        actor_id=actor_id,
        device=torch.device("cuda", local_rank),
        buffer_bytes=buffer_bytes,
    )
