# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace-only Projection reporting for PAP Attention offload."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    pap_offload_exec_trace_id,
)

logger = init_logger(__name__)


@dataclass(slots=True)
class PAPProjectionTraceState:
    """Completed trace state for one Projection layer execution."""

    offload_exec_batches: Sequence[tuple[Any, ...]]
    step_groups: Sequence[Any]
    projection_timeline: dict[str, Any] | None
    pre_attn_compute_ms: float | None
    send_ms: float
    trigger_ms: float
    yield_ms: float
    recv_ms: float
    total_ms: float
    pre_attn_start_ns: int
    pre_attn_done_ns: int
    send_done_ns: int
    yield_start_ns: int
    yield_end_ns: int
    recv_done_ns: int
    contiguous_route_groups: int
    direct_qkv_groups: int
    packed_qkv_groups: int
    direct_output_rows: int
    scattered_output_rows: int
    output_origin: torch.cuda.Event | None
    output_ready_events: list[torch.cuda.Event]
    fanout_prepare_us: list[float]
    fanout_submit_us: list[float]
    fanout_submit_start_ns: list[int]
    local_batched_fanout_plan: Any
    batched_fanout_submit_us: float


def record_projection_trace(state: PAPProjectionTraceState) -> None:
    """Publish one completed Projection trace without touching the hot path."""
    offload_exec_batches = state.offload_exec_batches
    step_groups = state.step_groups
    projection_timeline = state.projection_timeline
    pre_attn_compute_ms = state.pre_attn_compute_ms
    trace_send_ms = state.send_ms
    trace_trigger_ms = state.trigger_ms
    trace_yield_ms = state.yield_ms
    trace_recv_ms = state.recv_ms
    trace_total_ms = state.total_ms
    pre_attn_start_ns = state.pre_attn_start_ns
    pre_attn_done_ns = state.pre_attn_done_ns
    trace_send_done_ns = state.send_done_ns
    trace_yield_start_ns = state.yield_start_ns
    trace_yield_end_ns = state.yield_end_ns
    trace_recv_done_ns = state.recv_done_ns
    trace_contiguous_route_groups = state.contiguous_route_groups
    trace_direct_qkv_groups = state.direct_qkv_groups
    trace_packed_qkv_groups = state.packed_qkv_groups
    trace_direct_output_rows = state.direct_output_rows
    trace_scattered_output_rows = state.scattered_output_rows
    trace_output_origin = state.output_origin
    trace_output_ready_events = state.output_ready_events
    trace_fanout_prepare_us = state.fanout_prepare_us
    trace_fanout_submit_us = state.fanout_submit_us
    trace_fanout_submit_start_ns = state.fanout_submit_start_ns
    local_batched_fanout_plan = state.local_batched_fanout_plan
    batched_fanout_submit_us = state.batched_fanout_submit_us

    if not offload_exec_batches:
        return

    def route_kv_tokens(
        descriptor: PAPOffloadExecBatchDescriptor,
    ) -> str:
        template = descriptor.metadata_template
        if template is None:
            return "0"
        total = 0
        for step in template.get("s", ()):
            total += int(step)
        return str(total)

    trace_batch_keys = "|".join(
        pap_offload_exec_trace_id(item[2].output_tensor_id)
        for item in offload_exec_batches
    )
    trace_route_rows = "|".join(
        str(item[2].item_count) for item in offload_exec_batches
    )
    trace_route_kv_tokens = "|".join(
        route_kv_tokens(item[2]) for item in offload_exec_batches
    )
    calls = sum(item[2].item_count for item in offload_exec_batches)
    if projection_timeline is not None:
        projection_timeline.update(
            {
                "layer": offload_exec_batches[0][2].layer_name,
                "batches": len(offload_exec_batches),
                "calls": calls,
                "pre_attn_compute_ms": pre_attn_compute_ms,
                "send_ms": trace_send_ms,
                "trigger_ms": trace_trigger_ms,
                "yield_ms": trace_yield_ms,
                "recv_ms": trace_recv_ms,
                "remote_total_ms": trace_total_ms,
                "batch_keys": trace_batch_keys,
                "route_rows": trace_route_rows,
                "route_kv_tokens": trace_route_kv_tokens,
                "pre_attn_start_ns": pre_attn_start_ns,
                "pre_attn_done_ns": pre_attn_done_ns,
                "send_done_ns": trace_send_done_ns,
                "yield_start_ns": trace_yield_start_ns,
                "yield_end_ns": trace_yield_end_ns,
                "recv_done_ns": trace_recv_done_ns,
                "route_groups": len(step_groups),
                "contiguous_route_groups": trace_contiguous_route_groups,
                "direct_qkv_groups": trace_direct_qkv_groups,
                "packed_qkv_groups": trace_packed_qkv_groups,
                "direct_output_rows": trace_direct_output_rows,
                "scattered_output_rows": trace_scattered_output_rows,
            }
        )
    logger.info(
        "PAP OFFLOAD_EXEC projection trace layer=%s batches=%d "
        "calls=%d send_ms=%.3f trigger_ms=%.3f "
        "yield_ms=%.3f recv_ms=%.3f total_ms=%.3f batch_keys=%s "
        "route_rows=%s route_kv_tokens=%s "
        "send_done_ns=%d yield_start_ns=%d yield_end_ns=%d "
        "recv_done_ns=%d route_groups=%d "
        "contiguous_route_groups=%d direct_qkv_groups=%d "
        "packed_qkv_groups=%d direct_output_rows=%d "
        "scattered_output_rows=%d",
        offload_exec_batches[0][2].layer_name,
        len(offload_exec_batches),
        calls,
        trace_send_ms,
        trace_trigger_ms,
        trace_yield_ms,
        trace_recv_ms,
        trace_total_ms,
        trace_batch_keys,
        trace_route_rows,
        trace_route_kv_tokens,
        trace_send_done_ns,
        trace_yield_start_ns,
        trace_yield_end_ns,
        trace_recv_done_ns,
        len(step_groups),
        trace_contiguous_route_groups,
        trace_direct_qkv_groups,
        trace_packed_qkv_groups,
        trace_direct_output_rows,
        trace_scattered_output_rows,
    )
    if trace_output_origin is not None and len(trace_output_ready_events) > 1:
        for ready_event in trace_output_ready_events:
            ready_event.synchronize()
        ready_times_ms = [
            trace_output_origin.elapsed_time(ready_event)
            for ready_event in trace_output_ready_events
        ]
        first_ready_ms = min(ready_times_ms)
        last_ready_ms = max(ready_times_ms)
        spread_ms = last_ready_ms - first_ready_ms
        spread_pct = spread_ms / first_ready_ms * 100.0 if first_ready_ms > 0 else 0.0
        logger.info(
            "PAP OFFLOAD_EXEC projection fan-in trace layer=%s "
            "peers=%d first_ready_ms=%.3f last_ready_ms=%.3f "
            "spread_ms=%.3f spread_over_fastest_pct=%.3f",
            offload_exec_batches[0][2].layer_name,
            len(ready_times_ms),
            first_ready_ms,
            last_ready_ms,
            spread_ms,
            spread_pct,
        )
    if trace_fanout_submit_us:
        first_submit_ns = trace_fanout_submit_start_ns[0]
        logger.info(
            "PAP OFFLOAD_EXEC projection fan-out trace layer=%s "
            "peers=%d qkv_host_to_first_submit_us=%.3f "
            "total_host_send_us=%.3f prepare_us=%s submit_us=%s "
            "submit_start_offsets_us=%s",
            offload_exec_batches[0][2].layer_name,
            len(trace_fanout_submit_us),
            max(
                0.0,
                (first_submit_ns - pre_attn_done_ns) / 1_000.0,
            ),
            (trace_send_done_ns - pre_attn_done_ns) / 1_000.0,
            "|".join(f"{value:.3f}" for value in trace_fanout_prepare_us),
            "|".join(f"{value:.3f}" for value in trace_fanout_submit_us),
            "|".join(
                f"{(value - first_submit_ns) / 1_000.0:.3f}"
                for value in trace_fanout_submit_start_ns
            ),
        )
    if local_batched_fanout_plan is not None:
        logger.info(
            "PAP OFFLOAD_EXEC batched fan-out trace layer=%s peers=%d submit_us=%.3f",
            offload_exec_batches[0][2].layer_name,
            len(offload_exec_batches),
            batched_fanout_submit_us,
        )
