# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare decode-step Attention state for the whole-step Graph."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.pap.attention.triton_backend import (
    PAPAttentionStepTensorCache,
    PAPPagedDecodeWorkspaceCache,
    build_paged_decode_workspace,
)
from vllm.pap.deferred_cuda_trace import (
    record_deferred_host_duration,
)
from vllm.pap.kv.decode_state import _DEFERRED_CUDA_TRACE_ENABLED
from vllm.pap.kv.metadata import (
    PAPPagedBlockTableBuffer,
    _coerce_block_id,
    build_unified_paged_flash_step_metadata,
)
from vllm.pap.kv.models import PAPAttentionStepContext
from vllm.pap.kv.registry import PAPAttentionRegistry

if TYPE_CHECKING:
    from vllm.pap.attention.backend import PAPAttentionSelector


def _offload_exec_batch_rows(
    descriptor: Any,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    tuple[float, ...],
]:
    template = getattr(descriptor, "metadata_template", None)
    if template is not None:
        request_ids = tuple(str(request_id) for request_id in template["r"])
        steps = tuple(int(step) for step in template["s"])
        scales = tuple(float(scale) for scale in template["a"])
    else:
        items = tuple(descriptor.items)
        request_ids = tuple(str(item.request_id) for item in items)
        steps = tuple(int(item.step) for item in items)
        scales = tuple(float(item.scale) for item in items)
    if not (len(request_ids) == len(steps) == len(scales)):
        raise RuntimeError("PAP OFFLOAD_EXEC batch descriptor length mismatch")
    return request_ids, steps, scales


def prepare_offload_exec_step(
    *,
    registry: PAPAttentionRegistry,
    descriptor: Any,
    dtype: torch.dtype,
    workspace_cache: PAPPagedDecodeWorkspaceCache | None = None,
    step_tensor_cache: PAPAttentionStepTensorCache | None = None,
    block_table_buffer: PAPPagedBlockTableBuffer | None = None,
    attention_kernel_selector: PAPAttentionSelector | None = None,
) -> PAPAttentionStepContext:
    """Prepare QKV-independent Attention state before layer-0 QKV arrives."""
    prepare_started = time.perf_counter()
    request_ids, steps, scales = _offload_exec_batch_rows(descriptor)
    (
        default_q_size,
        default_kv_size,
        num_heads,
        num_kv_heads,
        head_dim,
    ) = registry.offload_exec_shape_defaults
    lookup_started = time.perf_counter()
    context = registry.get_or_create_attention_step_context(
        request_ids=request_ids,
        decode_seq_lens=steps,
        scales=scales,
        layer_name=str(descriptor.layer_name),
        default_q_size=default_q_size,
        default_kv_size=default_kv_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if _DEFERRED_CUDA_TRACE_ENABLED:
        record_deferred_host_duration(
            "attention_step_context_lookup_wall_ms",
            (time.perf_counter() - lookup_started) * 1000.0,
        )
    lock_started = time.perf_counter()
    with context.lock:
        if _DEFERRED_CUDA_TRACE_ENABLED:
            record_deferred_host_duration(
                "attention_step_context_lock_wait_wall_ms",
                (time.perf_counter() - lock_started) * 1000.0,
            )
        if context.prepare_event is not None:
            return context
        states = context.layer_states[str(descriptor.layer_name)]
        if context.graph_slot_tensor is None:
            slot_started = time.perf_counter()
            # Keep Graph topology row-stable; -1 masks rows without a KV append.
            graph_slots = [-1] * len(context.request_ids)
            for index in context.active_indices:
                state = states[index]
                position = context.prior_seq_lens[index]
                block_size = int(state.block_size)
                logical_block = int(position) // block_size
                if logical_block >= len(state.block_ids):
                    raise RuntimeError(
                        "PAP step prepare slot exceeds sealed block table"
                    )
                physical_block = _coerce_block_id(state.block_ids[logical_block])
                graph_slots[index] = (
                    physical_block * block_size + int(position) % block_size
                )
            graph_slot_values = tuple(graph_slots)
            context.graph_slot_tensor = (
                step_tensor_cache.copy(
                    kind="graph_slots",
                    values=graph_slot_values,
                    dtype=torch.int64,
                    device=registry.storage_device,
                )
                if step_tensor_cache is not None
                else torch.tensor(
                    graph_slot_values,
                    dtype=torch.int64,
                    device=registry.storage_device,
                )
            )
            registry.record_attention_step_slot_plan_build()
            if _DEFERRED_CUDA_TRACE_ENABLED:
                record_deferred_host_duration(
                    "attention_step_slot_plan_wall_ms",
                    (time.perf_counter() - slot_started) * 1000.0,
                )
        if context.metadata is None:
            metadata_started = time.perf_counter()
            seq_lens_tensor = (
                step_tensor_cache.copy(
                    kind="seq_lens",
                    values=context.result_seq_lens,
                    dtype=torch.int32,
                    device=registry.storage_device,
                )
                if step_tensor_cache is not None
                else None
            )
            context.metadata = build_unified_paged_flash_step_metadata(
                states=states,
                seq_lens=context.result_seq_lens,
                device=registry.storage_device,
                seq_lens_tensor=seq_lens_tensor,
                block_table_buffer=block_table_buffer,
            )
            registry.record_attention_step_metadata_build()
            if _DEFERRED_CUDA_TRACE_ENABLED:
                record_deferred_host_duration(
                    "attention_step_metadata_wall_ms",
                    (time.perf_counter() - metadata_started) * 1000.0,
                )
        if context.paged_decode_workspace is None:
            workspace_started = time.perf_counter()
            query_template = torch.empty(
                (len(request_ids), context.num_heads, context.head_dim),
                dtype=dtype,
                device=registry.storage_device,
            )
            context.paged_decode_workspace = (
                workspace_cache.get(query_template)
                if workspace_cache is not None
                else build_paged_decode_workspace(query_template)
            )
            if _DEFERRED_CUDA_TRACE_ENABLED:
                record_deferred_host_duration(
                    "attention_step_workspace_wall_ms",
                    (time.perf_counter() - workspace_started) * 1000.0,
                )
        if not context.attention_kernel_plan_prepared:
            context.attention_kernel_plan = (
                attention_kernel_selector.plan(
                    step_signature=context.cache_key,
                    request_ids=context.request_ids,
                    topology_ids=context.topology_ids,
                    states=states,
                    seq_lens=context.result_seq_lens,
                    num_heads=context.num_heads,
                    num_kv_heads=context.num_kv_heads,
                    head_dim=context.head_dim,
                    scale=context.scale,
                    dtype=dtype,
                    device=registry.storage_device,
                )
                if attention_kernel_selector is not None
                else None
            )
            context.attention_kernel_plan_prepared = True
        if registry.storage_device.type == "cuda":
            event_started = time.perf_counter()
            context.prepare_event = torch.cuda.Event()
            context.prepare_event.record(
                torch.cuda.current_stream(registry.storage_device)
            )
            if _DEFERRED_CUDA_TRACE_ENABLED:
                record_deferred_host_duration(
                    "attention_step_event_wall_ms",
                    (time.perf_counter() - event_started) * 1000.0,
                )
    if _DEFERRED_CUDA_TRACE_ENABLED:
        record_deferred_host_duration(
            "attention_step_prepare_wall_ms",
            (time.perf_counter() - prepare_started) * 1000.0,
        )
    return context
