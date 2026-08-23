# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention compute path."""

from __future__ import annotations

import time
from typing import Any

import torch

from vllm.pap.attention.kernels import (
    PAPAttentionStepTensorCache,
    PAPPagedDecodeWorkspace,
    PAPPagedDecodeWorkspaceCache,
    build_paged_decode_workspace,
    run_paged_decode_attention,
)
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    end_deferred_cuda_span,
    record_deferred_host_duration,
)
from vllm.pap.kv.decode_state import _DEFERRED_CUDA_TRACE_ENABLED
from vllm.pap.kv.metadata import (
    PAPPagedBlockTableBuffer,
    PAPPagedFlashMetadata,
    _coerce_block_id,
    build_unified_paged_flash_step_metadata,
)
from vllm.pap.kv.models import PAPAttentionStepContext, PAPUnifiedPagedKVState
from vllm.pap.kv.observability import (
    log_kv_locality_profile as _log_kv_locality_profile,
)
from vllm.pap.kv.registry import PAPAttentionRegistry


def _compute_unified_paged_attention_batch(
    *,
    query_batch: torch.Tensor,
    states: list[PAPUnifiedPagedKVState],
    scale: float,
    layer_name: str,
    metadata: PAPPagedFlashMetadata,
    workspace: PAPPagedDecodeWorkspace,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor | None:
    """Run Prefill-owned paged decode Attention on one layer."""
    if not states:
        return None
    if not query_batch.is_cuda:
        return None
    base_kv = states[0].kv_cache
    if any(
        state.kv_cache.device != base_kv.device
        or state.kv_cache.shape != base_kv.shape
        or state.kv_cache.dtype != base_kv.dtype
        for state in states
    ):
        return None
    if base_kv.device != query_batch.device:
        return None

    if metadata.max_seq_len <= 0:
        return None
    key_cache, value_cache = base_kv.unbind(1)
    _log_kv_locality_profile(
        mode="unified",
        layer_name=layer_name,
        states=states,
        kv_cache=base_kv,
        key_cache=key_cache,
        value_cache=value_cache,
        layout=states[0].layout,
    )
    paged_start = time.perf_counter() if trace_stats is not None else 0.0
    use_deferred_flash_trace = (
        _DEFERRED_CUDA_TRACE_ENABLED
        and query_batch.is_cuda
        and torch.cuda.is_available()
    )
    if use_deferred_flash_trace:
        deferred_attention_trace = begin_deferred_cuda_span(
            "paged_fa_gpu_ms",
            torch.cuda.current_stream(query_batch.device),
        )
    start_event = end_event = None
    if (
        trace_stats is not None
        and not use_deferred_flash_trace
        and query_batch.is_cuda
        and torch.cuda.is_available()
    ):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(query_batch.device)
        start_event.record(stream)
    if use_deferred_flash_trace:
        try:
            output = run_paged_decode_attention(
                query=query_batch,
                key_cache=key_cache,
                value_cache=value_cache,
                metadata=metadata,
                workspace=workspace,
                scale=float(scale),
                block_size=int(states[0].block_size),
            )
        finally:
            end_deferred_cuda_span(deferred_attention_trace)
    else:
        output = run_paged_decode_attention(
            query=query_batch,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            workspace=workspace,
            scale=float(scale),
            block_size=int(states[0].block_size),
        )
    if end_event is not None:
        end_event.record(torch.cuda.current_stream(query_batch.device))
        end_event.synchronize()
    if trace_stats is not None:
        paged_done_ns = time.perf_counter_ns()
        paged_wall_ms = (time.perf_counter() - paged_start) * 1000.0
        if use_deferred_flash_trace:
            paged_kernel_ms = 0.0
        elif start_event is not None and end_event is not None:
            paged_kernel_ms = start_event.elapsed_time(end_event)
        else:
            paged_kernel_ms = paged_wall_ms
        trace_stats["paged_flash_ms"] = (
            trace_stats.get("paged_flash_ms", 0.0) + paged_wall_ms
        )
        trace_stats["paged_flash_kernel_ms"] = (
            trace_stats.get("paged_flash_kernel_ms", 0.0) + paged_kernel_ms
        )
        trace_stats["paged_flash_done_ns"] = float(paged_done_ns)
    return output


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
            if context.active_indices:
                slot_values = tuple(
                    graph_slots[index] for index in context.active_indices
                )
                context.slot_tensor = (
                    step_tensor_cache.copy(
                        kind="slots",
                        values=slot_values,
                        dtype=torch.int64,
                        device=registry.storage_device,
                    )
                    if step_tensor_cache is not None
                    else torch.tensor(
                        slot_values,
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
        if (
            context.active_index_tensor is None
            and context.active_indices
            and len(context.active_indices) != len(context.request_ids)
        ):
            context.active_index_tensor = (
                step_tensor_cache.copy(
                    kind="active_indices",
                    values=context.active_indices,
                    dtype=torch.int64,
                    device=registry.storage_device,
                )
                if step_tensor_cache is not None
                else torch.tensor(
                    context.active_indices,
                    dtype=torch.int64,
                    device=registry.storage_device,
                )
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


def compute_offload_exec_batch_output(
    *,
    registry: PAPAttentionRegistry,
    descriptor: Any,
    qkv_batch: torch.Tensor,
    step_context: PAPAttentionStepContext | None = None,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute one OFFLOAD_EXEC output batch via paged decode Attention."""

    if step_context is None:
        request_ids, steps, scales = _offload_exec_batch_rows(descriptor)
        (
            default_q_size,
            default_kv_size,
            num_heads_default,
            num_kv_heads_default,
            head_dim_default,
        ) = registry.offload_exec_shape_defaults
        step_context = registry.get_or_create_attention_step_context(
            request_ids=request_ids,
            decode_seq_lens=steps,
            scales=scales,
            layer_name=str(descriptor.layer_name),
            default_q_size=default_q_size,
            default_kv_size=default_kv_size,
            num_heads=num_heads_default,
            num_kv_heads=num_kv_heads_default,
            head_dim=head_dim_default,
        )
    else:
        request_ids = step_context.request_ids
    if int(qkv_batch.shape[0]) != len(request_ids):
        raise RuntimeError(
            "PAP OFFLOAD_EXEC batch QKV row count does not match descriptor"
        )
    if trace_stats is not None:
        trace_stats["pre_compute_start_ns"] = float(time.perf_counter_ns())

    shape_lookup_start = time.perf_counter() if trace_stats is not None else 0.0
    if trace_stats is not None:
        trace_stats["shape_lookup_ms"] += (
            time.perf_counter() - shape_lookup_start
        ) * 1000.0

    _q_size = step_context.q_size
    kv_size = step_context.kv_size
    num_heads = step_context.num_heads
    num_kv_heads = step_context.num_kv_heads
    head_dim = step_context.head_dim
    batch_size = len(request_ids)

    qkv_split_start = time.perf_counter() if trace_stats is not None else 0.0
    query_flat, key_flat, value_flat = qkv_batch.split(
        [_q_size, kv_size, kv_size],
        dim=-1,
    )
    query_batch = query_flat.view(batch_size, num_heads, head_dim)
    key_batch = key_flat.view(batch_size, num_kv_heads, head_dim)
    value_batch = value_flat.view(batch_size, num_kv_heads, head_dim)
    if trace_stats is not None:
        trace_stats["qkv_split_ms"] += (time.perf_counter() - qkv_split_start) * 1000.0

    query_move_start = time.perf_counter() if trace_stats is not None else 0.0
    if torch.cuda.is_available():
        query_batch = query_batch.to(registry.storage_device, non_blocking=True)
    if trace_stats is not None:
        trace_stats["query_move_ms"] += (
            time.perf_counter() - query_move_start
        ) * 1000.0

    layer_name = str(descriptor.layer_name)
    with step_context.lock:
        if (
            step_context.prepare_event is not None
            and not step_context.prepare_event_waited
        ):
            compute_stream = torch.cuda.current_stream(query_batch.device)
            prepare_wait_trace = begin_deferred_cuda_span(
                "attention_step_prepare_wait_gpu_ms",
                compute_stream,
            )
            try:
                compute_stream.wait_event(step_context.prepare_event)
            finally:
                end_deferred_cuda_span(prepare_wait_trace)
            step_context.prepare_event_waited = True
        unified_states = step_context.layer_states.get(layer_name)
        if unified_states is None:
            raise RuntimeError(
                f"PAP Attention step received unexpected layer {layer_name}"
            )
        layer_completed = layer_name in step_context.completed_layers
        expected_seq_lens = (
            step_context.result_seq_lens
            if layer_completed
            else step_context.prior_seq_lens
        )
        if tuple(int(state.seq_len) for state in unified_states) != expected_seq_lens:
            raise RuntimeError(
                "PAP Attention step observed a layer sequence-length drift "
                f"layer={layer_name}"
            )

        append_start = time.perf_counter() if trace_stats is not None else 0.0
        written = registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=step_context.session_request_ids,
            layer_name=layer_name,
            key_batch=key_batch,
            value_batch=value_batch,
            decode_seq_lens=step_context.decode_seq_lens,
            step_context=step_context,
            trace_stats=trace_stats,
        )
        expected_writes = 0 if layer_completed else len(step_context.active_indices)
        if written != expected_writes:
            raise RuntimeError(
                "PAP Attention step KV append row count mismatch "
                f"expected={expected_writes} written={written}"
            )
        if trace_stats is not None:
            trace_stats["append_kv_ms"] += (time.perf_counter() - append_start) * 1000.0

        metadata_build_ms = 0.0
        if step_context.metadata is None:
            metadata_start = time.perf_counter()
            step_context.metadata = build_unified_paged_flash_step_metadata(
                states=unified_states,
                seq_lens=step_context.result_seq_lens,
                device=query_batch.device,
            )
            metadata_build_ms = (time.perf_counter() - metadata_start) * 1000.0
            registry.record_attention_step_metadata_build()
        if trace_stats is not None:
            trace_stats["paged_metadata_ms"] += metadata_build_ms
            trace_stats["metadata_build_ms"] += metadata_build_ms
            trace_stats["pre_compute_done_ns"] = float(time.perf_counter_ns())

        if step_context.paged_decode_workspace is None:
            step_context.paged_decode_workspace = build_paged_decode_workspace(
                query_batch
            )

        unified_output = _compute_unified_paged_attention_batch(
            query_batch=query_batch,
            states=list(unified_states),
            scale=step_context.scale,
            layer_name=layer_name,
            metadata=step_context.metadata,
            workspace=step_context.paged_decode_workspace,
            trace_stats=trace_stats,
        )
        if unified_output is None:
            raise RuntimeError("PAP unified paged decode Attention failed")
        reshape_start = time.perf_counter() if trace_stats is not None else 0.0
        if unified_output.ndim == 3:
            unified_output = unified_output.reshape(batch_size, num_heads * head_dim)
        if trace_stats is not None:
            reshape_ms = (time.perf_counter() - reshape_start) * 1000.0
            trace_stats["reshape_ms"] = trace_stats.get("reshape_ms", 0.0) + reshape_ms
            trace_stats["attention_output_reshape_ms"] = (
                trace_stats.get("attention_output_reshape_ms", 0.0) + reshape_ms
            )
            trace_stats["post_compute_done_ns"] = float(time.perf_counter_ns())
        registry.complete_attention_step_layer(
            context=step_context,
            layer_name=layer_name,
        )
        return unified_output


def _finalize_offload_exec_compute_trace(
    trace_stats: dict[str, float] | None,
    compute_ms: float,
) -> None:
    if trace_stats is None:
        return
    explained_ms = sum(
        float(trace_stats.get(field, 0.0))
        for field in (
            "shape_lookup_ms",
            "qkv_split_ms",
            "query_move_ms",
            "append_kv_ms",
            "metadata_build_ms",
            "paged_flash_ms",
            "attention_output_reshape_ms",
        )
    )
    if float(trace_stats.get("compute_unaccounted_ms", 0.0)) <= 0.0:
        trace_stats["compute_unaccounted_ms"] = max(0.0, compute_ms - explained_ms)
