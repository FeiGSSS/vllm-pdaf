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
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from vllm.pap.config import PAPStepTraceConfig, read_env_int
from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecTransportClosed,
    PAPTensorTransport,
)
from vllm.pap.protocol.offload_exec import (
    layer_index_and_template,
    layer_name_from_template,
)
from vllm.pap.transport.nvshmem.protocol import (
    METADATA_VERSION,
    decode_step_plan,
    encode_step_plan,
)
from vllm.pap.transport.nvshmem.runtime import PAPNVSHMEMError
from vllm.pap.transport.nvshmem.tracing import PAPNVSHMEMTraceRecorder
from vllm.pap.transport.nvshmem.world import (
    get_pap_nvshmem_world,
)

_DEFAULT_BUFFER_BYTES = 16 * 1024 * 1024

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


@cache
def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


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
        self.buffer_bytes = int(
            buffer_bytes
            if buffer_bytes is not None
            else read_env_int(
                os.environ, "PAP_NVSHMEM_BUFFER_BYTES", _DEFAULT_BUFFER_BYTES, minimum=1
            )
        )
        if self.buffer_bytes <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM buffer size must be positive")
        trace_config = PAPStepTraceConfig.from_env()
        self.world = get_pap_nvshmem_world(
            device_index=self.device.index,
            buffer_bytes=self.buffer_bytes,
        )
        self.peer_rank: int | None = None
        self._world_ready = False
        self._stopped = threading.Event()
        self.tracing = PAPNVSHMEMTraceRecorder(
            world=self.world,
            device=self.device,
            config=trace_config,
            stopped=self._stopped,
        )
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._step_prepare_handler: Any = None
        self._step_preflight_handler: Any = None
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

    @property
    def local_agent_metadata(self) -> bytes:
        payload = {
            "version": METADATA_VERSION,
            "transport": "nvshmem_graph",
            "hostname": socket.gethostname(),
            "rank": self.world.rank,
            "world_size": self.world.world_size,
            "buffer_bytes": self.buffer_bytes,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def record_attention_step_trace_metadata(self, context: Any) -> None:
        self.tracing.record_attention_step_trace_metadata(
            context, local_epoch=self._qkv_recv_generation - 2
        )

    def bind_peer(self, peer_agent_metadata: bytes) -> None:
        if self._closed:
            raise PAPNVSHMEMError("PAP NVSHMEM transport is closed")
        try:
            payload = json.loads(peer_agent_metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PAPNVSHMEMError("invalid PAP NVSHMEM peer metadata") from exc
        if payload.get("version") != METADATA_VERSION:
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
            encode_step_plan(
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
        trace = self.tracing.attention
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
        trace = self.tracing.attention
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
        trace = self.tracing.attention
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
        trace = self.tracing.projection
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
        trace = self.tracing.projection
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

    def set_step_preflight_handler(self, handler: Any) -> None:
        """Install the capacity fence run before a control slot is released."""
        self._step_preflight_handler = handler

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
        trace_host = self.tracing.attention is not None
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
        descriptor, dtype, qkv_width, layer_count = decode_step_plan(
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
        if self._step_preflight_handler is not None:
            self._step_preflight_handler(descriptor)
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
