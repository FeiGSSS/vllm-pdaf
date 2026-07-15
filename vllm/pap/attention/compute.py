# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention compute path."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    end_deferred_cuda_span,
)
from vllm.pap.kv.metadata import (
    PAPPagedFlashMetadata,
    build_unified_paged_flash_metadata,
)
from vllm.pap.kv.decode_state import _DEFERRED_CUDA_TRACE_ENABLED
from vllm.pap.kv.models import PAPAttentionSession, PAPUnifiedPagedKVState
from vllm.pap.kv.observability import (
    log_kv_locality_profile as _log_kv_locality_profile,
)
from vllm.pap.kv.registry import (
    PAPAttentionRegistry,
    _DECODE_COMMIT_PATH,
    _prefill_control_endpoint,
)
from vllm.pap.protocol import pap_offload_exec_trace_id

logger = logging.getLogger("pap_attention")


def _offload_exec_attention_shapes(
    *,
    session: PAPAttentionSession,
) -> tuple[int, int, int, int, int]:
    q_size = session.q_size or int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0"))
    kv_size = session.kv_size or int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0"))
    num_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0"))
    num_kv_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0"))
    head_dim = int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0"))
    if q_size <= 0 or kv_size <= 0:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC requires q_size and kv_size in attention "
            "registration or PAP_OFFLOAD_EXEC_Q_SIZE/PAP_OFFLOAD_EXEC_KV_SIZE"
        )
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC requires PAP_OFFLOAD_EXEC_NUM_HEADS, "
            "PAP_OFFLOAD_EXEC_NUM_KV_HEADS, and PAP_OFFLOAD_EXEC_HEAD_DIM"
        )
    return q_size, kv_size, num_heads, num_kv_heads, head_dim


def _offload_exec_session(
    *,
    registry: PAPAttentionRegistry,
    request_id: str,
) -> tuple[str, PAPAttentionSession]:
    session_request_id = registry.resolve_session_request_id(request_id)
    if session_request_id is None:
        raise KeyError(request_id)
    session = registry.get_session(session_request_id)
    if session is None:
        raise KeyError(request_id)
    return session_request_id, session


def _run_paged_flash_varlen(
    *,
    flash_attn_varlen_func: Any,
    fa_version: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: PAPPagedFlashMetadata,
    scale: float,
    causal: bool,
    return_softmax_lse: bool,
) -> Any:
    output = torch.empty_like(query)
    return flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.seq_lens,
        max_seqlen_q=1,
        max_seqlen_k=metadata.max_seq_len,
        softmax_scale=float(scale),
        causal=causal,
        block_table=metadata.block_table,
        softcap=0.0,
        return_softmax_lse=return_softmax_lse,
        fa_version=fa_version,
    )


def _compute_unified_paged_flash_batch(
    *,
    query_batch: torch.Tensor,
    states: list[PAPUnifiedPagedKVState],
    scale: float,
    layer_name: str,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor | None:
    """Single-source Prefill-owned paged FA compute (Stage 4 unified path)."""
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

    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            get_flash_attn_version,
            is_flash_attn_varlen_func_available,
        )
    except Exception:
        return None
    if not is_flash_attn_varlen_func_available():
        return None

    metadata_start = time.perf_counter() if trace_stats is not None else 0.0
    metadata = build_unified_paged_flash_metadata(
        states=states, device=query_batch.device
    )
    if trace_stats is not None:
        metadata_done_ns = time.perf_counter_ns()
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
        trace_stats["paged_metadata_ms"] = (
            trace_stats.get("paged_metadata_ms", 0.0) + metadata_ms
        )
        trace_stats["metadata_build_ms"] = (
            trace_stats.get("metadata_build_ms", 0.0) + metadata_ms
        )
        trace_stats["pre_compute_done_ns"] = float(metadata_done_ns)
    if metadata.max_seq_len <= 0:
        return None

    fa_version = get_flash_attn_version(head_size=int(query_batch.shape[-1]))
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
        deferred_flash_trace = begin_deferred_cuda_span(
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
            result = _run_paged_flash_varlen(
                flash_attn_varlen_func=flash_attn_varlen_func,
                fa_version=fa_version,
                query=query_batch,
                key_cache=key_cache,
                value_cache=value_cache,
                metadata=metadata,
                scale=float(scale),
                causal=True,
                return_softmax_lse=False,
            )
        finally:
            end_deferred_cuda_span(deferred_flash_trace)
    else:
        result = _run_paged_flash_varlen(
            flash_attn_varlen_func=flash_attn_varlen_func,
            fa_version=fa_version,
            query=query_batch,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            scale=float(scale),
            causal=True,
            return_softmax_lse=False,
        )
    if end_event is not None:
        end_event.record(torch.cuda.current_stream(query_batch.device))
        end_event.synchronize()
    output = result[0] if isinstance(result, tuple) else result
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


def compute_offload_exec_batch_output(
    *,
    registry: PAPAttentionRegistry,
    descriptor: Any,
    qkv_batch: torch.Tensor,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute one OFFLOAD_EXEC attention output batch via paged FlashAttention."""

    request_ids, steps, scales = _offload_exec_batch_rows(descriptor)
    if int(qkv_batch.shape[0]) != len(request_ids):
        raise RuntimeError(
            "PAP OFFLOAD_EXEC batch QKV row count does not match descriptor"
        )
    if trace_stats is not None:
        trace_stats["pre_compute_start_ns"] = float(time.perf_counter_ns())

    shape_lookup_start = time.perf_counter() if trace_stats is not None else 0.0
    num_heads_default = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0"))
    num_kv_heads_default = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0"))
    head_dim_default = int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0"))
    session_entries = registry.offload_exec_batch_session_entries(
        request_ids,
        default_q_size=int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0")),
        default_kv_size=int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0")),
        num_heads=num_heads_default,
        num_kv_heads=num_kv_heads_default,
        head_dim=head_dim_default,
    )

    common_shape: tuple[int, int, int, int, int] | None = None
    common_scale: float | None = None
    for scale, session_entry in zip(scales, session_entries):
        shape = (
            session_entry.q_size,
            session_entry.kv_size,
            session_entry.num_heads,
            session_entry.num_kv_heads,
            session_entry.head_dim,
        )
        if common_shape is None:
            common_shape = shape
            common_scale = float(scale)
        elif shape != common_shape or float(scale) != common_scale:
            raise RuntimeError("PAP OFFLOAD_EXEC batch has mixed shapes or scales")
    if trace_stats is not None:
        trace_stats["shape_lookup_ms"] += (
            time.perf_counter() - shape_lookup_start
        ) * 1000.0

    assert common_shape is not None
    _q_size, kv_size, num_heads, num_kv_heads, head_dim = common_shape
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

    append_start = time.perf_counter() if trace_stats is not None else 0.0
    decode_seq_lens = list(steps)
    session_request_ids = tuple(
        session_entry.session_request_id for session_entry in session_entries
    )

    unified_states = registry.get_unified_paged_states(
        session_request_ids=session_request_ids,
        layer_name=descriptor.layer_name,
    )
    if unified_states is not None:
        commit_new_seq_lens: list[int | None] = [
            int(decode_len) if int(decode_len) > int(state.seq_len) else None
            for decode_len, state in zip(decode_seq_lens, unified_states)
        ]
        written = registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=session_request_ids,
            layer_name=descriptor.layer_name,
            key_batch=key_batch,
            value_batch=value_batch,
            decode_seq_lens=decode_seq_lens,
            trace_stats=trace_stats,
        )
        if any(seq_len is not None for seq_len in commit_new_seq_lens) and written <= 0:
            raise RuntimeError("PAP unified KV append wrote no rows")
        if trace_stats is not None:
            trace_stats["append_kv_ms"] += (time.perf_counter() - append_start) * 1000.0
        unified_output = _compute_unified_paged_flash_batch(
            query_batch=query_batch,
            states=unified_states,
            scale=common_scale,
            layer_name=descriptor.layer_name,
            trace_stats=trace_stats,
        )
        if unified_output is None:
            raise RuntimeError("PAP unified paged FlashAttention failed")
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
        for index, request_id in enumerate(request_ids):
            new_seq_len = commit_new_seq_lens[index]
            if new_seq_len is None:
                continue
            endpoint = _prefill_control_endpoint(
                session_entries[index].prefill_endpoint,
                _DECODE_COMMIT_PATH,
            )
            registry.record_decode_kv_ready(
                request_id=request_id,
                new_seq_len=new_seq_len,
                endpoint=endpoint,
            )
        return unified_output

    raise RuntimeError(
        "PAP Prefill-owned KV state missing for layer="
        f"{descriptor.layer_name}; sealed manifest handoff did not complete"
    )


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


def _combine_offload_exec_outputs(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def run_offload_exec_batch_once(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
    remote_address: str,
    descriptor: Any,
) -> None:
    """Receive one batched QKV tensor and send one batched attention output."""

    trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
    trace_recv_start_ns = 0
    trace_recv_done_ns = 0
    trace_compute_done_ns = 0
    trace_send_start_ns = 0
    trace_send_done_ns = 0
    trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
    if trace_offload_exec:
        trace_recv_start_ns = time.perf_counter_ns()
    qkv_batch = transport.recv_qkv_batch(
        descriptor,
        remote_address=remote_address,
    )
    trace_recv_ms = (
        (time.perf_counter() - trace_recv_start) * 1000.0 if trace_offload_exec else 0.0
    )
    if trace_offload_exec:
        trace_recv_done_ns = time.perf_counter_ns()
    if int(qkv_batch.shape[0]) != descriptor.item_count:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC batch QKV row count does not match descriptor"
        )
    trace_compute_start = time.perf_counter() if trace_offload_exec else 0.0
    trace_compute_stats = (
        {
            "append_kv_ms": 0.0,
            "pack_ms": 0.0,
            "sdpa_ms": 0.0,
            "reshape_ms": 0.0,
            "paged_metadata_ms": 0.0,
            "paged_flash_ms": 0.0,
            "metadata_build_ms": 0.0,
            "paged_flash_kernel_ms": 0.0,
            "attention_output_reshape_ms": 0.0,
            "compute_unaccounted_ms": 0.0,
            "fallback_ms": 0.0,
            "shape_lookup_ms": 0.0,
            "qkv_split_ms": 0.0,
            "query_move_ms": 0.0,
            "query_cat_ms": 0.0,
            "append_lock_wait_ms": 0.0,
            "append_prepare_ms": 0.0,
            "append_record_ms": 0.0,
            "append_tensor_ms": 0.0,
            "append_copy_ms": 0.0,
            "append_state_ms": 0.0,
            "pre_compute_start_ns": 0.0,
            "pre_compute_done_ns": 0.0,
            "paged_flash_done_ns": 0.0,
            "post_compute_done_ns": 0.0,
        }
        if trace_offload_exec
        else None
    )
    output_batch = compute_offload_exec_batch_output(
        registry=registry,
        descriptor=descriptor,
        qkv_batch=qkv_batch,
        trace_stats=trace_compute_stats,
    )
    trace_compute_ms = (
        (time.perf_counter() - trace_compute_start) * 1000.0
        if trace_offload_exec
        else 0.0
    )
    _finalize_offload_exec_compute_trace(trace_compute_stats, trace_compute_ms)
    if trace_offload_exec:
        trace_compute_done_ns = time.perf_counter_ns()
    trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
    if trace_offload_exec:
        trace_send_start_ns = time.perf_counter_ns()
    transport.send_output_batch(
        descriptor,
        output_batch,
        remote_address=remote_address,
    )
    if trace_offload_exec:
        trace_send_done_ns = time.perf_counter_ns()
        trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
        trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
        logger.info(
            "PAP OFFLOAD_EXEC attention batch trace layer=%s calls=%d "
            "recv_qkv_ms=%.3f compute_ms=%.3f send_output_ms=%.3f "
            "total_ms=%.3f append_kv_ms=%.3f pack_ms=%.3f "
            "sdpa_ms=%.3f reshape_ms=%.3f paged_metadata_ms=%.3f "
            "paged_flash_ms=%.3f fallback_ms=%.3f shape_lookup_ms=%.3f "
            "qkv_split_ms=%.3f query_move_ms=%.3f query_cat_ms=%.3f "
            "append_lock_wait_ms=%.3f append_prepare_ms=%.3f "
            "append_record_ms=%.3f append_tensor_ms=%.3f "
            "append_copy_ms=%.3f append_state_ms=%.3f "
            "metadata_build_ms=%.3f paged_flash_kernel_ms=%.3f "
            "attention_output_reshape_ms=%.3f compute_unaccounted_ms=%.3f "
            "qkv_shape=%s output_shape=%s batch_key=%s "
            "recv_done_ns=%d compute_done_ns=%d send_done_ns=%d "
            "recv_start_ns=%d pre_compute_start_ns=%d "
            "pre_compute_done_ns=%d paged_flash_done_ns=%d reshape_done_ns=%d "
            "send_start_ns=%d",
            descriptor.layer_name,
            descriptor.item_count,
            trace_recv_ms,
            trace_compute_ms,
            trace_send_ms,
            trace_total_ms,
            trace_compute_stats["append_kv_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["pack_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["sdpa_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["reshape_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["paged_metadata_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["paged_flash_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["fallback_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["shape_lookup_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["qkv_split_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["query_move_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["query_cat_ms"] if trace_compute_stats else 0.0,
            (
                trace_compute_stats["append_lock_wait_ms"]
                if trace_compute_stats
                else 0.0
            ),
            (trace_compute_stats["append_prepare_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_record_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_tensor_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_copy_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_state_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["metadata_build_ms"] if trace_compute_stats else 0.0),
            (
                trace_compute_stats["paged_flash_kernel_ms"]
                if trace_compute_stats
                else 0.0
            ),
            (
                trace_compute_stats["attention_output_reshape_ms"]
                if trace_compute_stats
                else 0.0
            ),
            (
                trace_compute_stats["compute_unaccounted_ms"]
                if trace_compute_stats
                else 0.0
            ),
            tuple(qkv_batch.shape),
            tuple(output_batch.shape),
            pap_offload_exec_trace_id(descriptor.output_tensor_id),
            trace_recv_done_ns,
            trace_compute_done_ns,
            trace_send_done_ns,
            trace_recv_start_ns,
            (
                int(trace_compute_stats.get("pre_compute_start_ns", 0.0))
                if trace_compute_stats
                else 0
            ),
            (
                int(trace_compute_stats.get("pre_compute_done_ns", 0.0))
                if trace_compute_stats
                else 0
            ),
            (
                int(trace_compute_stats.get("paged_flash_done_ns", 0.0))
                if trace_compute_stats
                else 0
            ),
            (
                int(trace_compute_stats.get("post_compute_done_ns", 0.0))
                if trace_compute_stats
                else 0
            ),
            trace_send_start_ns,
        )
