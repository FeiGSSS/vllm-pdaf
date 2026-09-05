# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace-buffer ownership and CPU export for PAP's NVSHMEM data path."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.pap.config import PAPStepTraceConfig
from vllm.pap.transport.nvshmem.runtime import PAPNVSHMEMError
from vllm.pap.transport.nvshmem.world import PAPNVSHMEMWorld


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


class PAPNVSHMEMTraceRecorder:
    """Keep host-side trace allocation and export outside the transport."""

    def __init__(
        self,
        *,
        world: PAPNVSHMEMWorld,
        device: torch.device,
        config: PAPStepTraceConfig,
        stopped: threading.Event,
    ) -> None:
        self.world = world
        self.device = device
        self._trace_config = config
        self._stopped = stopped

    @property
    def projection(self) -> _ProjectionPATrace | None:
        return _PROJECTION_PA_TRACE

    @property
    def attention(self) -> _AttentionKernelTrace | None:
        return _ATTENTION_KERNEL_TRACE

    def prepare_projection_pa_trace(self, layer_count: int) -> None:
        """Allocate stable Projection-side trace buffers before Graph capture."""
        global _PROJECTION_PA_TRACE

        config = self._trace_config
        output = config.output
        if not output or self.world.rank != self.world.config.root_rank:
            return
        ring_steps, sample_steps = config.ring_steps, config.sample_steps
        with _PROJECTION_PA_TRACE_LOCK:
            trace = _PROJECTION_PA_TRACE
            if trace is not None:
                if (
                    trace.layer_count != layer_count
                    or trace.world_size != self.world.world_size
                    or trace.output_path != Path(output)
                    or trace.ring_steps != ring_steps
                    or trace.sample_steps != sample_steps
                    or trace.export_interval_seconds != config.export_interval_seconds
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
                    export_interval_seconds=config.export_interval_seconds,
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

        config = self._trace_config
        output = config.output
        if not output or self.world.rank == self.world.config.root_rank:
            return
        ring_steps, sample_steps = config.ring_steps, config.sample_steps
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
                if (
                    trace.layer_count != layer_count
                    or trace.world_rank != self.world.rank
                    or trace.output_path != output_path
                    or trace.ring_steps != ring_steps
                    or trace.sample_steps != sample_steps
                    or trace.export_interval_seconds != config.export_interval_seconds
                ):
                    raise PAPNVSHMEMError(
                        "PAP Attention trace configuration changed after capture"
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
                    export_interval_seconds=config.export_interval_seconds,
                    step_metadata=OrderedDict(),
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

    def record_attention_step_trace_metadata(
        self, context: Any, *, local_epoch: int
    ) -> None:
        """Record the exact request contexts consumed by one Graph replay."""
        trace = _ATTENTION_KERNEL_TRACE
        if trace is None or self.world.rank != trace.world_rank:
            return
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
