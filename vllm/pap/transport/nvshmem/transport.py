# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention--Projection transport over same-host NVSHMEM P2P."""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
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
_GRAPH_ABORT = 3


@dataclass
class _QKVStepPlan:
    descriptor: PAPOffloadExecBatchDescriptor
    dtype: torch.dtype
    qkv_width: int
    layer_count: int
    layer_template: tuple[str, str]
    qkv_tensor: torch.Tensor | None = None
    step_context: Any = None


@dataclass
class _ProjectionPATrace:
    output_path: Path
    ring_steps: int
    sample_steps: int
    layer_count: int
    world_size: int
    root_rank: int
    start_ns: torch.Tensor
    end_ns: torch.Tensor
    step_ids: torch.Tensor
    route_counts: torch.Tensor
    peer_epochs: torch.Tensor
    dispatch_done_ns: torch.Tensor
    gather_done_ns: torch.Tensor
    step_counter: torch.Tensor
    current_step: torch.Tensor
    host_start_ns: torch.Tensor
    host_end_ns: torch.Tensor
    host_step_ids: torch.Tensor
    host_route_counts: torch.Tensor
    host_peer_epochs: torch.Tensor
    host_dispatch_done_ns: torch.Tensor
    host_gather_done_ns: torch.Tensor
    host_completion: torch.Tensor
    host_start_pointer: int
    host_end_pointer: int
    host_step_ids_pointer: int
    host_route_counts_pointer: int
    host_peer_epochs_pointer: int
    host_dispatch_done_pointer: int
    host_gather_done_pointer: int
    host_completion_pointer: int
    export_interval_seconds: float


@dataclass
class _AttentionKernelTrace:
    output_path: Path
    ring_steps: int
    sample_steps: int
    layer_count: int
    world_rank: int
    replay_start_ns: torch.Tensor
    step_start_ns: torch.Tensor
    start_ns: torch.Tensor
    end_ns: torch.Tensor
    step_ids: torch.Tensor
    host_replay_start_ns: torch.Tensor
    host_step_start_ns: torch.Tensor
    host_start_ns: torch.Tensor
    host_end_ns: torch.Tensor
    host_step_ids: torch.Tensor
    host_completion: torch.Tensor
    host_replay_start_pointer: int
    host_step_start_pointer: int
    host_start_pointer: int
    host_end_pointer: int
    host_step_ids_pointer: int
    host_completion_pointer: int
    export_interval_seconds: float
    step_metadata: OrderedDict[int, _AttentionStepMetadata]


@dataclass(frozen=True)
class _AttentionStepMetadata:
    local_epoch: int
    request_ids: tuple[str, ...]
    seq_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    request_block_counts: tuple[int, ...]
    request_leased_block_counts: tuple[int, ...]
    logical_context_tokens: int
    unique_context_tokens: int
    block_reference_count: int
    unique_block_count: int
    unique_leased_block_count: int
    common_prefix_blocks: int
    common_prefix_tokens: int
    common_prefix_savings_tokens: int
    attention_backend: str
    attention_reused_kv_tokens: int
    control_wait_ns: int
    control_decode_ns: int
    context_prepare_ns: int
    graph_lookup_ns: int
    graph_replay_submit_ns: int


_PROJECTION_PA_TRACE: _ProjectionPATrace | None = None
_ATTENTION_KERNEL_TRACE: _AttentionKernelTrace | None = None
_PROJECTION_PA_TRACE_LOCK = threading.Lock()


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
        self._lifecycle_lock = threading.Lock()
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
            self._qkv_stream = torch.cuda.Stream(device=self.device)
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

    def prepare_projection_pa_trace(self, layer_count: int) -> None:
        """Allocate stable Projection-side trace buffers before Graph capture."""
        global _PROJECTION_PA_TRACE

        output = os.environ.get("PAP_PROJECTION_PA_TRACE_OUTPUT")
        if not output or self.world.rank != self.world.config.root_rank:
            return
        ring_steps = int(os.environ.get("PAP_PROJECTION_PA_TRACE_RING_STEPS", 2048))
        sample_steps = int(os.environ.get("PAP_PROJECTION_PA_TRACE_SAMPLES", 512))
        if ring_steps <= 0 or sample_steps <= 0 or sample_steps > ring_steps:
            raise PAPNVSHMEMError("PAP Projection PA trace step counts are invalid")
        with _PROJECTION_PA_TRACE_LOCK:
            trace = _PROJECTION_PA_TRACE
            if trace is not None:
                if (
                    trace.layer_count != layer_count
                    or trace.world_size != self.world.world_size
                    or trace.output_path != Path(output)
                ):
                    raise PAPNVSHMEMError(
                        "PAP Projection PA trace configuration changed after capture"
                    )
                return
            with torch.accelerator.device_index(self.device.index):
                shape = (ring_steps, layer_count, self.world.world_size)
                host_start_ns = torch.zeros(shape, dtype=torch.uint64, pin_memory=True)
                host_end_ns = torch.zeros(shape, dtype=torch.uint64, pin_memory=True)
                host_step_ids = torch.zeros(
                    (ring_steps, self.world.world_size),
                    dtype=torch.uint64,
                    pin_memory=True,
                )
                host_route_counts = torch.zeros(
                    (ring_steps, self.world.world_size),
                    dtype=torch.int32,
                    pin_memory=True,
                )
                host_peer_epochs = torch.zeros(
                    (ring_steps, self.world.world_size),
                    dtype=torch.uint64,
                    pin_memory=True,
                )
                scalar_shape = (ring_steps, layer_count)
                host_dispatch_done_ns = torch.zeros(
                    scalar_shape, dtype=torch.uint64, pin_memory=True
                )
                host_gather_done_ns = torch.zeros(
                    scalar_shape, dtype=torch.uint64, pin_memory=True
                )
                host_completion = torch.zeros(
                    ring_steps, dtype=torch.uint64, pin_memory=True
                )
                trace = _ProjectionPATrace(
                    output_path=Path(output),
                    ring_steps=ring_steps,
                    sample_steps=sample_steps,
                    layer_count=layer_count,
                    world_size=self.world.world_size,
                    root_rank=self.world.config.root_rank,
                    start_ns=torch.zeros(shape, dtype=torch.uint64, device=self.device),
                    end_ns=torch.zeros(shape, dtype=torch.uint64, device=self.device),
                    step_ids=torch.zeros(
                        (ring_steps, self.world.world_size),
                        dtype=torch.uint64,
                        device=self.device,
                    ),
                    route_counts=torch.zeros(
                        (ring_steps, self.world.world_size),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    peer_epochs=torch.zeros(
                        (ring_steps, self.world.world_size),
                        dtype=torch.uint64,
                        device=self.device,
                    ),
                    dispatch_done_ns=torch.zeros(
                        scalar_shape, dtype=torch.uint64, device=self.device
                    ),
                    gather_done_ns=torch.zeros(
                        scalar_shape, dtype=torch.uint64, device=self.device
                    ),
                    step_counter=torch.zeros(1, dtype=torch.uint64, device=self.device),
                    current_step=torch.zeros(1, dtype=torch.uint64, device=self.device),
                    host_start_ns=host_start_ns,
                    host_end_ns=host_end_ns,
                    host_step_ids=host_step_ids,
                    host_route_counts=host_route_counts,
                    host_peer_epochs=host_peer_epochs,
                    host_dispatch_done_ns=host_dispatch_done_ns,
                    host_gather_done_ns=host_gather_done_ns,
                    host_completion=host_completion,
                    host_start_pointer=self.world.runtime.host_device_pointer(
                        host_start_ns
                    ),
                    host_end_pointer=self.world.runtime.host_device_pointer(
                        host_end_ns
                    ),
                    host_step_ids_pointer=self.world.runtime.host_device_pointer(
                        host_step_ids
                    ),
                    host_route_counts_pointer=self.world.runtime.host_device_pointer(
                        host_route_counts
                    ),
                    host_peer_epochs_pointer=self.world.runtime.host_device_pointer(
                        host_peer_epochs
                    ),
                    host_dispatch_done_pointer=self.world.runtime.host_device_pointer(
                        host_dispatch_done_ns
                    ),
                    host_gather_done_pointer=self.world.runtime.host_device_pointer(
                        host_gather_done_ns
                    ),
                    host_completion_pointer=self.world.runtime.host_device_pointer(
                        host_completion
                    ),
                    export_interval_seconds=float(
                        os.environ.get("PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS", 5)
                    ),
                )
                if trace.export_interval_seconds <= 0:
                    raise PAPNVSHMEMError(
                        "PAP Projection PA trace flush interval must be positive"
                    )
                trace.output_path.parent.mkdir(parents=True, exist_ok=True)
                _PROJECTION_PA_TRACE = trace
                threading.Thread(
                    target=self._projection_pa_trace_export_loop,
                    args=(trace,),
                    name="pap-projection-pa-trace-export",
                    daemon=True,
                ).start()

    def _projection_pa_trace_export_loop(self, trace: _ProjectionPATrace) -> None:
        while not self._stopped.wait(trace.export_interval_seconds):
            try:
                self.export_projection_pa_trace()
            except Exception as exc:
                error_path = trace.output_path.with_suffix(".error.txt")
                error_path.write_text(f"{type(exc).__name__}: {exc}\n")
                return

    def export_projection_pa_trace(self) -> Path | None:
        """Export the latest complete Projection-side PA layer samples."""
        trace = _PROJECTION_PA_TRACE
        if trace is None:
            return None
        if self.world.rank != trace.root_rank:
            return None
        with _PROJECTION_PA_TRACE_LOCK:
            return self._export_projection_pa_trace_locked(trace)

    def _export_projection_pa_trace_locked(self, trace: _ProjectionPATrace) -> Path:
        completion_before = trace.host_completion.clone().to(torch.int64)
        start_ns = trace.host_start_ns.clone().to(torch.int64)
        end_ns = trace.host_end_ns.clone().to(torch.int64)
        step_ids = trace.host_step_ids.clone().to(torch.int64)
        route_counts = trace.host_route_counts.clone()
        peer_epochs = trace.host_peer_epochs.clone().to(torch.int64)
        dispatch_done_ns = trace.host_dispatch_done_ns.clone().to(torch.int64)
        gather_done_ns = trace.host_gather_done_ns.clone().to(torch.int64)
        completion_after = trace.host_completion.clone().to(torch.int64)
        peer_ranks = tuple(
            rank for rank in range(trace.world_size) if rank != trace.root_rank
        )
        peer_index = torch.tensor(peer_ranks, dtype=torch.int64)
        start_ns = start_ns.index_select(2, peer_index)
        end_ns = end_ns.index_select(2, peer_index)
        step_ids = step_ids.index_select(1, peer_index)
        route_counts = route_counts.index_select(1, peer_index)
        peer_epochs = peer_epochs.index_select(1, peer_index)
        latency_ns = end_ns - start_ns
        valid = (
            completion_before.eq(completion_after)
            & completion_after.ge(2)
            & completion_after.remainder(2).eq(0)
            & step_ids[:, 0].mul(2).add(2).eq(completion_after)
            & step_ids.eq(step_ids[:, :1]).all(dim=1)
            & route_counts.gt(0).all(dim=1)
            & start_ns.gt(0).all(dim=(1, 2))
            & end_ns.ge(start_ns).all(dim=(1, 2))
            & dispatch_done_ns.gt(0).all(dim=1)
            & gather_done_ns.gt(0).all(dim=1)
        )
        slots = valid.nonzero(as_tuple=False).flatten()
        if slots.numel():
            order = torch.argsort(step_ids[slots, 0])
            ordered_slots = slots[order]
            consecutive = step_ids[ordered_slots[1:], 0].eq(
                step_ids[ordered_slots[:-1], 0] + 1
            )
            pair_indices = consecutive.nonzero(as_tuple=False).flatten()
            keep_steps = trace.ring_steps - 1
            base_slots = ordered_slots[pair_indices][-keep_steps:]
            next_slots = ordered_slots[pair_indices + 1][-keep_steps:]
        else:
            base_slots = slots
            next_slots = slots
        sampled_start = start_ns[base_slots]
        sampled_end = end_ns[base_slots]
        sampled_latency = latency_ns[base_slots]
        sampled_step_ids = step_ids[base_slots, 0]
        sampled_counts = route_counts[base_slots]
        sampled_peer_epochs = peer_epochs[base_slots]
        sampled_gather_done = gather_done_ns[base_slots]
        sampled_next_dispatch = dispatch_done_ns[next_slots, 0]
        projection_latency_ns = torch.empty_like(sampled_gather_done)
        projection_latency_ns[:, :-1] = (
            dispatch_done_ns[base_slots, 1:] - sampled_gather_done[:, :-1]
        )
        projection_latency_ns[:, -1] = (
            sampled_next_dispatch - sampled_gather_done[:, -1]
        )
        payload = {
            "start_ns": sampled_start,
            "end_ns": sampled_end,
            "latency_ns": sampled_latency,
            "step_id": sampled_step_ids,
            "route_counts": sampled_counts,
            "peer_epoch": sampled_peer_epochs,
            "projection_gather_done_ns": sampled_gather_done,
            "projection_next_dispatch_done_ns": torch.cat(
                (dispatch_done_ns[base_slots, 1:], sampled_next_dispatch[:, None]),
                dim=1,
            ),
            "projection_latency_ns": projection_latency_ns,
            "peer_ranks": torch.tensor(peer_ranks, dtype=torch.int32),
            "metadata": {
                "clock": "projection_gpu_globaltimer_ns",
                "start": "dispatch block start before QKV pack and NVSHMEM put",
                "end": "output-ready NVSHMEM signal observed before scatter copy",
                "shape": list(sampled_latency.shape),
                "projection_shape": list(projection_latency_ns.shape),
                "requested_samples": trace.sample_steps,
                "ring_steps": trace.ring_steps,
            },
        }
        output_path = trace.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
        latency_us = sampled_latency.to(torch.float64) / 1000.0
        projection_latency_us = projection_latency_ns.to(torch.float64) / 1000.0
        summary = {
            "output": str(output_path),
            "shape": list(sampled_latency.shape),
            "complete_samples": int(base_slots.numel()),
            "requested_samples": trace.sample_steps,
            "peer_ranks": list(peer_ranks),
            "first_step_id": (
                int(sampled_step_ids[0]) if sampled_step_ids.numel() else None
            ),
            "last_step_id": (
                int(sampled_step_ids[-1]) if sampled_step_ids.numel() else None
            ),
            "latency_us": {
                "mean": float(latency_us.mean()) if latency_us.numel() else None,
                "p50": (
                    float(torch.quantile(latency_us.flatten(), 0.5))
                    if latency_us.numel()
                    else None
                ),
                "p99": (
                    float(torch.quantile(latency_us.flatten(), 0.99))
                    if latency_us.numel()
                    else None
                ),
                "max": float(latency_us.max()) if latency_us.numel() else None,
            },
            "projection_latency_us": {
                "mean": (
                    float(projection_latency_us.mean())
                    if projection_latency_us.numel()
                    else None
                ),
                "p50": (
                    float(torch.quantile(projection_latency_us.flatten(), 0.5))
                    if projection_latency_us.numel()
                    else None
                ),
                "p99": (
                    float(torch.quantile(projection_latency_us.flatten(), 0.99))
                    if projection_latency_us.numel()
                    else None
                ),
                "max": (
                    float(projection_latency_us.max())
                    if projection_latency_us.numel()
                    else None
                ),
            },
        }
        summary_path = output_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return output_path

    def prepare_attention_kernel_trace(self, layer_count: int) -> None:
        """Allocate one PA's Attention-kernel trace before Graph capture."""
        global _ATTENTION_KERNEL_TRACE

        output = os.environ.get("PAP_PROJECTION_PA_TRACE_OUTPUT")
        if not output or self.world.rank == self.world.config.root_rank:
            return
        ring_steps = int(os.environ.get("PAP_PROJECTION_PA_TRACE_RING_STEPS", 2048))
        sample_steps = int(os.environ.get("PAP_PROJECTION_PA_TRACE_SAMPLES", 512))
        if ring_steps <= 0 or sample_steps <= 0 or sample_steps > ring_steps:
            raise PAPNVSHMEMError("PAP Attention trace step counts are invalid")
        peer_ranks = tuple(
            rank
            for rank in range(self.world.world_size)
            if rank != self.world.config.root_rank
        )
        pa_index = peer_ranks.index(self.world.rank)
        output_path = Path(output).with_name(f"attention_pa_{pa_index}_kernel_trace.pt")
        with _PROJECTION_PA_TRACE_LOCK:
            trace = _ATTENTION_KERNEL_TRACE
            if trace is not None:
                if trace.layer_count != layer_count:
                    raise PAPNVSHMEMError(
                        "PAP Attention trace layer count changed after capture"
                    )
                return
            shape = (ring_steps, layer_count)
            host_replay_start_ns = torch.zeros(
                ring_steps, dtype=torch.uint64, pin_memory=True
            )
            host_step_start_ns = torch.zeros(
                ring_steps, dtype=torch.uint64, pin_memory=True
            )
            host_start_ns = torch.zeros(shape, dtype=torch.uint64, pin_memory=True)
            host_end_ns = torch.zeros(shape, dtype=torch.uint64, pin_memory=True)
            host_step_ids = torch.zeros(ring_steps, dtype=torch.uint64, pin_memory=True)
            host_completion = torch.zeros(
                ring_steps, dtype=torch.uint64, pin_memory=True
            )
            with torch.accelerator.device_index(self.device.index):
                trace = _AttentionKernelTrace(
                    output_path=output_path,
                    ring_steps=ring_steps,
                    sample_steps=sample_steps,
                    layer_count=layer_count,
                    world_rank=self.world.rank,
                    replay_start_ns=torch.zeros(
                        ring_steps, dtype=torch.uint64, device=self.device
                    ),
                    step_start_ns=torch.zeros(
                        ring_steps, dtype=torch.uint64, device=self.device
                    ),
                    start_ns=torch.zeros(shape, dtype=torch.uint64, device=self.device),
                    end_ns=torch.zeros(shape, dtype=torch.uint64, device=self.device),
                    step_ids=torch.zeros(
                        ring_steps, dtype=torch.uint64, device=self.device
                    ),
                    host_replay_start_ns=host_replay_start_ns,
                    host_step_start_ns=host_step_start_ns,
                    host_start_ns=host_start_ns,
                    host_end_ns=host_end_ns,
                    host_step_ids=host_step_ids,
                    host_completion=host_completion,
                    host_replay_start_pointer=self.world.runtime.host_device_pointer(
                        host_replay_start_ns
                    ),
                    host_step_start_pointer=self.world.runtime.host_device_pointer(
                        host_step_start_ns
                    ),
                    host_start_pointer=self.world.runtime.host_device_pointer(
                        host_start_ns
                    ),
                    host_end_pointer=self.world.runtime.host_device_pointer(
                        host_end_ns
                    ),
                    host_step_ids_pointer=self.world.runtime.host_device_pointer(
                        host_step_ids
                    ),
                    host_completion_pointer=self.world.runtime.host_device_pointer(
                        host_completion
                    ),
                    export_interval_seconds=float(
                        os.environ.get("PAP_PROJECTION_PA_TRACE_FLUSH_SECONDS", 5)
                    ),
                    step_metadata=OrderedDict(),
                )
            if trace.export_interval_seconds <= 0:
                raise PAPNVSHMEMError(
                    "PAP Attention trace flush interval must be positive"
                )
            trace.output_path.parent.mkdir(parents=True, exist_ok=True)
            _ATTENTION_KERNEL_TRACE = trace
            threading.Thread(
                target=self._attention_kernel_trace_export_loop,
                args=(trace,),
                name=f"pap-attention-kernel-trace-export-pa{pa_index}",
                daemon=True,
            ).start()

    def _attention_kernel_trace_export_loop(self, trace: _AttentionKernelTrace) -> None:
        while not self._stopped.wait(trace.export_interval_seconds):
            try:
                self.export_attention_kernel_trace()
            except Exception as exc:
                error_path = trace.output_path.with_suffix(".error.txt")
                error_path.write_text(f"{type(exc).__name__}: {exc}\n")
                return

    def record_attention_step_trace_metadata(self, context: Any) -> None:
        """Record the exact request contexts consumed by one Graph replay."""
        trace = _ATTENTION_KERNEL_TRACE
        if trace is None or self.world.rank != trace.world_rank:
            return
        local_epoch = self._qkv_recv_generation - 2
        if local_epoch < 0:
            raise PAPNVSHMEMError("PAP Attention trace local epoch is invalid")
        graph_epoch: int | None = None
        for step_id, completion in zip(
            trace.host_step_ids.tolist(),
            trace.host_completion.tolist(),
        ):
            step_id = int(step_id)
            if int(completion) == step_id * 2 + 2:
                graph_epoch = max(graph_epoch or step_id, step_id)
        if graph_epoch is None:
            raise PAPNVSHMEMError("PAP Attention trace Graph epoch is unavailable")
        seq_lens = tuple(int(value) for value in context.result_seq_lens)
        layer_name = min(context.expected_layers)
        states = context.layer_states[layer_name]
        if len(states) != len(seq_lens):
            raise PAPNVSHMEMError("PAP Attention trace metadata row mismatch")

        request_block_counts: list[int] = []
        request_leased_block_counts: list[int] = []
        referenced_block_rows: list[tuple[int, ...]] = []
        unique_tokens_by_block: dict[int, int] = {}
        leased_blocks: set[int] = set()
        for state, seq_len in zip(states, seq_lens):
            remaining = seq_len
            referenced_blocks = 0
            for raw_block_id in state.block_ids:
                block_id = int(raw_block_id)
                leased_blocks.add(block_id)
                if remaining <= 0:
                    continue
                tokens = min(int(state.block_size), remaining)
                unique_tokens_by_block[block_id] = max(
                    unique_tokens_by_block.get(block_id, 0),
                    tokens,
                )
                referenced_blocks += 1
                remaining -= tokens
            if remaining:
                raise PAPNVSHMEMError(
                    "PAP Attention trace sequence exceeds its leased blocks"
                )
            request_block_counts.append(referenced_blocks)
            request_leased_block_counts.append(len(state.block_ids))
            referenced_block_rows.append(
                tuple(int(value) for value in state.block_ids[:referenced_blocks])
            )

        logical_context_tokens = sum(seq_lens)
        unique_context_tokens = sum(unique_tokens_by_block.values())
        if unique_context_tokens > logical_context_tokens:
            raise PAPNVSHMEMError("PAP Attention trace unique context is invalid")
        common_prefix_blocks = 0
        for block_ids in zip(*referenced_block_rows):
            if len(set(block_ids)) != 1:
                break
            common_prefix_blocks += 1
        block_size = int(states[0].block_size)
        common_prefix_tokens = common_prefix_blocks * block_size
        metadata = _AttentionStepMetadata(
            local_epoch=local_epoch,
            request_ids=tuple(str(value) for value in context.session_request_ids),
            seq_lens=seq_lens,
            prefix_lens=tuple(int(state.prefix_len) for state in states),
            request_block_counts=tuple(request_block_counts),
            request_leased_block_counts=tuple(request_leased_block_counts),
            logical_context_tokens=logical_context_tokens,
            unique_context_tokens=unique_context_tokens,
            block_reference_count=sum(request_block_counts),
            unique_block_count=len(unique_tokens_by_block),
            unique_leased_block_count=len(leased_blocks),
            common_prefix_blocks=common_prefix_blocks,
            common_prefix_tokens=common_prefix_tokens,
            common_prefix_savings_tokens=(common_prefix_tokens * (len(states) - 1)),
            attention_backend=(
                context.attention_kernel_plan.backend_name
                if context.attention_kernel_plan is not None
                else "triton"
            ),
            attention_reused_kv_tokens=(
                int(context.attention_kernel_plan.reused_kv_tokens)
                if context.attention_kernel_plan is not None
                else 0
            ),
            control_wait_ns=int(getattr(context, "_pap_trace_control_wait_ns", 0)),
            control_decode_ns=int(getattr(context, "_pap_trace_control_decode_ns", 0)),
            context_prepare_ns=int(
                getattr(context, "_pap_trace_context_prepare_ns", 0)
            ),
            graph_lookup_ns=int(getattr(context, "_pap_trace_graph_lookup_ns", 0)),
            graph_replay_submit_ns=int(
                getattr(context, "_pap_trace_graph_replay_submit_ns", 0)
            ),
        )
        with _PROJECTION_PA_TRACE_LOCK:
            existing = trace.step_metadata.get(graph_epoch)
            if existing is not None and existing != metadata:
                raise PAPNVSHMEMError(
                    "PAP Attention trace metadata changed for one Graph epoch"
                )
            trace.step_metadata[graph_epoch] = metadata
            while len(trace.step_metadata) > trace.ring_steps:
                trace.step_metadata.popitem(last=False)

    def export_attention_kernel_trace(self) -> Path | None:
        """Export one PA's latest complete Attention-kernel samples."""
        trace = _ATTENTION_KERNEL_TRACE
        if trace is None or self.world.rank != trace.world_rank:
            return None
        with _PROJECTION_PA_TRACE_LOCK:
            completion_before = trace.host_completion.clone().to(torch.int64)
            replay_start_ns = trace.host_replay_start_ns.clone().to(torch.int64)
            step_start_ns = trace.host_step_start_ns.clone().to(torch.int64)
            start_ns = trace.host_start_ns.clone().to(torch.int64)
            end_ns = trace.host_end_ns.clone().to(torch.int64)
            step_ids = trace.host_step_ids.clone().to(torch.int64)
            completion_after = trace.host_completion.clone().to(torch.int64)
            valid = (
                completion_before.eq(completion_after)
                & completion_after.ge(2)
                & completion_after.remainder(2).eq(0)
                & step_ids.mul(2).add(2).eq(completion_after)
                & replay_start_ns.gt(0)
                & step_start_ns.gt(0)
                & start_ns.gt(0).all(dim=1)
                & end_ns.ge(start_ns).all(dim=1)
            )
            slots = valid.nonzero(as_tuple=False).flatten()
            if slots.numel():
                has_metadata = torch.tensor(
                    [
                        int(step_ids[slot]) in trace.step_metadata
                        for slot in slots.tolist()
                    ],
                    dtype=torch.bool,
                )
                slots = slots[has_metadata]
            if slots.numel():
                order = torch.argsort(step_ids[slots])
                keep_steps = trace.ring_steps
                slots = slots[order][-keep_steps:]
            sampled_start = start_ns[slots]
            sampled_end = end_ns[slots]
            sampled_replay_start = replay_start_ns[slots]
            sampled_step_start = step_start_ns[slots]
            latency_ns = sampled_end - sampled_start
            sampled_step_ids = step_ids[slots]
            metadata_rows = [
                trace.step_metadata[int(epoch)] for epoch in sampled_step_ids.tolist()
            ]
            max_requests = max(
                (len(row.seq_lens) for row in metadata_rows),
                default=0,
            )
            seq_lens = torch.zeros(
                (len(metadata_rows), max_requests), dtype=torch.int32
            )
            prefix_lens = torch.zeros_like(seq_lens)
            request_block_counts = torch.zeros_like(seq_lens)
            request_leased_block_counts = torch.zeros_like(seq_lens)
            for index, row in enumerate(metadata_rows):
                request_count = len(row.seq_lens)
                seq_lens[index, :request_count] = torch.tensor(
                    row.seq_lens, dtype=torch.int32
                )
                prefix_lens[index, :request_count] = torch.tensor(
                    row.prefix_lens, dtype=torch.int32
                )
                request_block_counts[index, :request_count] = torch.tensor(
                    row.request_block_counts, dtype=torch.int32
                )
                request_leased_block_counts[index, :request_count] = torch.tensor(
                    row.request_leased_block_counts, dtype=torch.int32
                )
            payload = {
                "replay_start_ns": sampled_replay_start,
                "graph_start_ns": sampled_step_start,
                "start_ns": sampled_start,
                "end_ns": sampled_end,
                "latency_ns": latency_ns,
                "graph_epoch": sampled_step_ids,
                "local_epoch": torch.tensor(
                    [row.local_epoch for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "world_rank": trace.world_rank,
                "request_ids": [list(row.request_ids) for row in metadata_rows],
                "request_count": torch.tensor(
                    [len(row.seq_lens) for row in metadata_rows], dtype=torch.int32
                ),
                "seq_lens": seq_lens,
                "prefix_lens": prefix_lens,
                "request_block_counts": request_block_counts,
                "request_leased_block_counts": request_leased_block_counts,
                "logical_context_tokens": torch.tensor(
                    [row.logical_context_tokens for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "unique_context_tokens": torch.tensor(
                    [row.unique_context_tokens for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "block_reference_count": torch.tensor(
                    [row.block_reference_count for row in metadata_rows],
                    dtype=torch.int32,
                ),
                "unique_block_count": torch.tensor(
                    [row.unique_block_count for row in metadata_rows],
                    dtype=torch.int32,
                ),
                "unique_leased_block_count": torch.tensor(
                    [row.unique_leased_block_count for row in metadata_rows],
                    dtype=torch.int32,
                ),
                "common_prefix_blocks": torch.tensor(
                    [row.common_prefix_blocks for row in metadata_rows],
                    dtype=torch.int32,
                ),
                "common_prefix_tokens": torch.tensor(
                    [row.common_prefix_tokens for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "common_prefix_savings_tokens": torch.tensor(
                    [row.common_prefix_savings_tokens for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "attention_backend": [row.attention_backend for row in metadata_rows],
                "attention_reused_kv_tokens": torch.tensor(
                    [row.attention_reused_kv_tokens for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "control_wait_ns": torch.tensor(
                    [row.control_wait_ns for row in metadata_rows], dtype=torch.int64
                ),
                "control_decode_ns": torch.tensor(
                    [row.control_decode_ns for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "context_prepare_ns": torch.tensor(
                    [row.context_prepare_ns for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "graph_lookup_ns": torch.tensor(
                    [row.graph_lookup_ns for row in metadata_rows], dtype=torch.int64
                ),
                "graph_replay_submit_ns": torch.tensor(
                    [row.graph_replay_submit_ns for row in metadata_rows],
                    dtype=torch.int64,
                ),
                "metadata": {
                    "clock": "attention_gpu_globaltimer_ns",
                    "replay_start": "GPU marker enqueued immediately before replay",
                    "graph_start": "after graph epoch advance and before layer-0 wait",
                    "start": (
                        "after reshape_and_cache and before paged decode attention"
                    ),
                    "end": "after all paged decode attention kernels",
                    "shape": list(latency_ns.shape),
                    "ring_steps": trace.ring_steps,
                    "seq_lens_shape": list(seq_lens.shape),
                    "context_tokens": "exact seq_lens consumed by the kernel",
                    "unique_context_tokens": (
                        "per-block maximum referenced tokens after block-id dedup"
                    ),
                },
            }
            temporary_path = trace.output_path.with_suffix(
                trace.output_path.suffix + ".tmp"
            )
            torch.save(payload, temporary_path)
            temporary_path.replace(trace.output_path)
            latency_us = latency_ns.to(torch.float64) / 1000.0
            logical_context_tokens = payload["logical_context_tokens"].to(torch.float64)
            unique_context_tokens = payload["unique_context_tokens"].to(torch.float64)
            summary = {
                "output": str(trace.output_path),
                "shape": list(latency_ns.shape),
                "world_rank": trace.world_rank,
                "first_local_epoch": (
                    int(sampled_step_ids[0]) if sampled_step_ids.numel() else None
                ),
                "last_local_epoch": (
                    int(sampled_step_ids[-1]) if sampled_step_ids.numel() else None
                ),
                "latency_us": {
                    "mean": float(latency_us.mean()) if latency_us.numel() else None,
                    "p50": (
                        float(torch.quantile(latency_us.flatten(), 0.5))
                        if latency_us.numel()
                        else None
                    ),
                    "p99": (
                        float(torch.quantile(latency_us.flatten(), 0.99))
                        if latency_us.numel()
                        else None
                    ),
                    "max": (float(latency_us.max()) if latency_us.numel() else None),
                },
                "logical_context_tokens": {
                    "mean": (
                        float(logical_context_tokens.mean())
                        if logical_context_tokens.numel()
                        else None
                    ),
                    "min": (
                        int(logical_context_tokens.min())
                        if logical_context_tokens.numel()
                        else None
                    ),
                    "max": (
                        int(logical_context_tokens.max())
                        if logical_context_tokens.numel()
                        else None
                    ),
                },
                "logical_over_unique_context": (
                    float((logical_context_tokens / unique_context_tokens).mean())
                    if unique_context_tokens.numel()
                    else None
                ),
            }
            trace.output_path.with_suffix(".json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            return trace.output_path

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

    def probe_device_graph_launch(
        self,
        *,
        graph_handle: int,
        stream: torch.Stream,
    ) -> None:
        """Validate a captured Attention graph for GPU-side launch."""
        self.world.runtime.probe_device_graph_launch(
            graph_handle=graph_handle,
            stream=stream,
        )

    def create_device_graph_launch(
        self,
        *,
        graph_handle: int,
        stream: torch.Stream,
    ) -> int:
        """Create a GPU-launchable Attention graph executable."""
        return self.world.runtime.create_device_graph_launch(
            graph_handle=graph_handle,
            stream=stream,
        )

    def create_resident_graph_dispatcher(
        self,
        *,
        stream: torch.Stream,
        window_size: int,
    ) -> int:
        """Create the hardware-wait queue for dynamic Attention graphs."""
        return self.world.runtime.create_resident_graph_dispatcher(
            stream=stream,
            window_size=window_size,
        )

    def run_resident_device_graph(
        self,
        *,
        dispatcher_handle: int,
        executable_handle: int,
        generation: int,
    ) -> None:
        """Publish and run one graph through the hardware-wait dispatcher."""
        self.world.runtime.run_resident_device_graph(
            dispatcher_handle=dispatcher_handle,
            executable_handle=executable_handle,
            generation=generation,
        )

    def destroy_resident_graph_dispatcher(self, dispatcher_handle: int) -> None:
        """Stop the hardware-wait dispatcher."""
        self.world.runtime.destroy_resident_graph_dispatcher(dispatcher_handle)

    def launch_device_graph(
        self,
        *,
        executable_handle: int,
        stream: torch.Stream,
    ) -> None:
        """Launch an Attention graph through a GPU launcher kernel."""
        self.world.runtime.launch_device_graph(
            executable_handle=executable_handle,
            stream=stream,
        )

    def destroy_device_graph_launch(self, executable_handle: int) -> None:
        """Release a GPU-launchable Attention graph executable."""
        self.world.runtime.destroy_device_graph_launch(executable_handle)

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

    def graph_attention_kernel_trace_start(
        self,
        *,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Record the start of one PA's paged-decode Attention kernels."""
        self._graph_attention_kernel_trace_marker(
            layer_index=layer_index,
            layer_count=layer_count,
            is_start=True,
            stream=stream,
        )

    def graph_attention_kernel_trace_end(
        self,
        *,
        layer_index: int,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Record the end of one PA's paged-decode Attention kernels."""
        self._graph_attention_kernel_trace_marker(
            layer_index=layer_index,
            layer_count=layer_count,
            is_start=False,
            stream=stream,
        )

    def _graph_attention_kernel_trace_marker(
        self,
        *,
        layer_index: int,
        layer_count: int,
        is_start: bool,
        stream: torch.Stream,
    ) -> None:
        trace = _ATTENTION_KERNEL_TRACE
        if trace is None:
            return
        self._require_graph_ready(layer_count)
        self.world.runtime.trace_attention_marker(
            epoch=self._graph_epoch_tensor(),
            replay_start_ns=trace.replay_start_ns,
            step_start_ns=trace.step_start_ns,
            start_ns=trace.start_ns,
            end_ns=trace.end_ns,
            step_ids=trace.step_ids,
            host_replay_start_pointer=trace.host_replay_start_pointer,
            host_step_start_pointer=trace.host_step_start_pointer,
            host_start_pointer=trace.host_start_pointer,
            host_end_pointer=trace.host_end_pointer,
            host_step_ids_pointer=trace.host_step_ids_pointer,
            host_completion_pointer=trace.host_completion_pointer,
            trace_steps=trace.ring_steps,
            trace_layers=trace.layer_count,
            layer_index=layer_index,
            marker_kind=1 if is_start else 2,
            stream=stream,
        )

    def graph_attention_step_trace_start(
        self,
        *,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Record when one Attention whole-step Graph begins on the GPU."""
        trace = _ATTENTION_KERNEL_TRACE
        if trace is None:
            return
        self._require_graph_ready(layer_count)
        self.world.runtime.trace_attention_marker(
            epoch=self._graph_epoch_tensor(),
            replay_start_ns=trace.replay_start_ns,
            step_start_ns=trace.step_start_ns,
            start_ns=trace.start_ns,
            end_ns=trace.end_ns,
            step_ids=trace.step_ids,
            host_replay_start_pointer=trace.host_replay_start_pointer,
            host_step_start_pointer=trace.host_step_start_pointer,
            host_start_pointer=trace.host_start_pointer,
            host_end_pointer=trace.host_end_pointer,
            host_step_ids_pointer=trace.host_step_ids_pointer,
            host_completion_pointer=trace.host_completion_pointer,
            trace_steps=trace.ring_steps,
            trace_layers=trace.layer_count,
            layer_index=0,
            marker_kind=0,
            stream=stream,
        )

    def graph_attention_replay_trace_start(
        self,
        *,
        layer_count: int,
        stream: torch.Stream,
    ) -> None:
        """Record the GPU queue boundary immediately before Graph replay."""
        trace = _ATTENTION_KERNEL_TRACE
        if trace is None:
            return
        self._require_graph_ready(layer_count)
        self.world.runtime.trace_attention_marker(
            epoch=self._graph_epoch_tensor(),
            replay_start_ns=trace.replay_start_ns,
            step_start_ns=trace.step_start_ns,
            start_ns=trace.start_ns,
            end_ns=trace.end_ns,
            step_ids=trace.step_ids,
            host_replay_start_pointer=trace.host_replay_start_pointer,
            host_step_start_pointer=trace.host_step_start_pointer,
            host_start_pointer=trace.host_start_pointer,
            host_end_pointer=trace.host_end_pointer,
            host_step_ids_pointer=trace.host_step_ids_pointer,
            host_completion_pointer=trace.host_completion_pointer,
            trace_steps=trace.ring_steps,
            trace_layers=trace.layer_count,
            layer_index=0,
            marker_kind=3,
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
        trace = _PROJECTION_PA_TRACE
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
            trace_start_ns=trace.start_ns if trace is not None else None,
            trace_step_ids=trace.step_ids if trace is not None else None,
            trace_route_counts=trace.route_counts if trace is not None else None,
            trace_peer_epochs=trace.peer_epochs if trace is not None else None,
            trace_step_counter=trace.step_counter if trace is not None else None,
            trace_current_step=trace.current_step if trace is not None else None,
            trace_host_completion_pointer=(
                trace.host_completion_pointer if trace is not None else 0
            ),
            trace_steps=trace.ring_steps if trace is not None else 0,
            trace_layers=trace.layer_count if trace is not None else 0,
            stream=stream,
        )
        if trace is not None:
            self.world.runtime.trace_projection_dispatch_done(
                current_step=trace.current_step,
                dispatch_done_ns=trace.dispatch_done_ns,
                trace_steps=trace.ring_steps,
                trace_layers=trace.layer_count,
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
        trace = _PROJECTION_PA_TRACE
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
            trace_end_ns=trace.end_ns if trace is not None else None,
            trace_current_step=trace.current_step if trace is not None else None,
            trace_steps=trace.ring_steps if trace is not None else 0,
            trace_layers=trace.layer_count if trace is not None else 0,
            stream=stream,
        )
        if trace is not None:
            self.world.runtime.trace_projection_gather_done(
                current_step=trace.current_step,
                start_ns=trace.start_ns,
                end_ns=trace.end_ns,
                step_ids=trace.step_ids,
                route_counts=trace.route_counts,
                peer_epochs=trace.peer_epochs,
                dispatch_done_ns=trace.dispatch_done_ns,
                gather_done_ns=trace.gather_done_ns,
                host_start_pointer=trace.host_start_pointer,
                host_end_pointer=trace.host_end_pointer,
                host_step_ids_pointer=trace.host_step_ids_pointer,
                host_route_counts_pointer=trace.host_route_counts_pointer,
                host_peer_epochs_pointer=trace.host_peer_epochs_pointer,
                host_dispatch_done_pointer=trace.host_dispatch_done_pointer,
                host_gather_done_pointer=trace.host_gather_done_pointer,
                host_completion_pointer=trace.host_completion_pointer,
                trace_steps=trace.ring_steps,
                trace_layers=trace.layer_count,
                layer_index=layer_index,
                stream=stream,
            )

    def set_step_prepare_handler(self, handler: Any) -> None:
        self._step_prepare_handler = handler

    def step_prepare_stream(self) -> torch.Stream:
        return self._qkv_stream

    def stop_receiving(self) -> None:
        with self._lifecycle_lock:
            self._stopped.set()
            if not self._world_ready or self.peer_rank is None:
                return
            signal_i32 = self._signals().int32_tensor
            graph_signal_i32 = self._graph_signals().int32_tensor
            if signal_i32 is None or graph_signal_i32 is None:
                return
            control_signal_index = self.world.signal_offset(
                _READY_QKV,
                self.peer_rank,
            ) // struct.calcsize("i")
            graph_signal_index = self.world.signal_offset(
                _READY_QKV,
                self.peer_rank,
            ) // struct.calcsize("i")
            abort_signal_index = self.world.signal_offset(
                _GRAPH_ABORT,
                self.world.rank,
            ) // struct.calcsize("i")
            with torch.accelerator.device_index(self.device.index):
                stream = torch.cuda.current_stream(self.device)
                graph_signal_i32[abort_signal_index].fill_(1)
                graph_signal_i32[graph_signal_index].fill_((1 << 31) - 1)
                signal_i32[control_signal_index].fill_((1 << 31) - 1)
                stream.synchronize()

    def commit_received_step(self, callback: Any) -> bool:
        """Commit a completed Graph step unless shutdown won the race."""
        with self._lifecycle_lock:
            if self._closed or self._stopped.is_set():
                return False
            callback()
            return True

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
            current = torch.cuda.current_stream(self.device)
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
            with torch.cuda.stream(stream):
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
        trace_host = _ATTENTION_KERNEL_TRACE is not None
        control_wait_started_ns = time.perf_counter_ns() if trace_host else 0
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
        with torch.cuda.stream(stream):
            self._control_host.copy_(
                control.tensor.narrow(0, offset, self.world.config.control_bytes),
                non_blocking=True,
            )
        stream.synchronize()
        control_wait_ns = (
            time.perf_counter_ns() - control_wait_started_ns if trace_host else 0
        )
        self._require_open()
        control_decode_started_ns = time.perf_counter_ns() if trace_host else 0
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
        control_decode_ns = (
            time.perf_counter_ns() - control_decode_started_ns if trace_host else 0
        )
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
            context_prepare_started_ns = time.perf_counter_ns() if trace_host else 0
            with torch.cuda.stream(stream):
                plan.step_context = self._step_prepare_handler(descriptor, plan.dtype)
            if trace_host:
                plan.step_context._pap_trace_control_wait_ns = control_wait_ns
                plan.step_context._pap_trace_control_decode_ns = control_decode_ns
                plan.step_context._pap_trace_context_prepare_ns = (
                    time.perf_counter_ns() - context_prepare_started_ns
                )
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
            abort_signal_offset=self.world.signal_offset(
                _GRAPH_ABORT,
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
