# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention mailbox and dispatcher runtime."""

from __future__ import annotations

import logging
import os
import time
from queue import Queue
from threading import Thread
from typing import Any

import torch

from vllm.pap.attention.compute import (
    _finalize_offload_exec_compute_trace,
    _offload_exec_batch_rows,
    compute_offload_exec_batch_output,
)
from vllm.pap.attention.dispatcher import (
    PAPAttentionDispatcher,
    PAPAttentionWorkItem,
)
from vllm.pap.kv.observability import pap_env_flag as _pap_env_flag
from vllm.pap.kv.state import PAPAttentionRegistry
from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    pap_offload_exec_trace_id,
)

logger = logging.getLogger("pap_attention")


def _recv_next_qkv_batch_message_or_tensor(
    transport: Any,
) -> tuple[Any, Any | None, torch.Tensor]:
    recv_message_fn = getattr(transport, "recv_next_qkv_batch_message", None)
    if callable(recv_message_fn):
        descriptor, qkv_message = recv_message_fn()
        return descriptor, qkv_message, qkv_message.tensor
    descriptor, qkv_batch = transport.recv_next_qkv_batch()
    return descriptor, None, qkv_batch


def _qkv_message_recv_trace(
    qkv_message: Any | None,
    recv_qkv_ms: float,
) -> dict[str, float]:
    trace = getattr(qkv_message, "recv_trace", None) or {}

    def trace_float(name: str) -> float:
        value = trace.get(name, 0.0)
        return float(value or 0.0)

    wait_ms = trace_float("wait_ms")
    read_ms = trace_float("read_total_ms")
    return {
        "wait_ms": wait_ms,
        "read_ms": read_ms,
        "materialize_ms": trace_float("materialize_ms"),
        "transfer_ms": trace_float("transfer_ms"),
        "wait_other_ms": max(0.0, wait_ms - read_ms),
        "unaccounted_ms": max(0.0, recv_qkv_ms - wait_ms),
    }


class _QKVBatchMessagePrefetcher:
    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self._requests: Queue[object] = Queue()
        self._results: Queue[tuple[bool, Any]] = Queue(maxsize=1)
        self._stop = object()
        self._thread = Thread(
            target=self._run,
            name="pap-attention-mailbox-prefetch",
            daemon=True,
        )
        self._thread.start()

    def prefetch(self) -> None:
        self._requests.put(None)

    def result(self) -> tuple[Any, Any | None, torch.Tensor]:
        ok, payload = self._results.get()
        if ok:
            return payload
        raise payload

    def close(self) -> None:
        self._requests.put(self._stop)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is self._stop:
                return
            try:
                payload = _recv_next_qkv_batch_message_or_tensor(self._transport)
            except BaseException as exc:
                self._results.put((False, exc))
            else:
                self._results.put((True, payload))


def _record_offload_exec_ready_event(qkv_batch: torch.Tensor) -> Any | None:
    if not qkv_batch.is_cuda:
        return None
    with torch.cuda.device(qkv_batch.device):
        ready_event = torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream(qkv_batch.device))
    return ready_event


def _wait_offload_exec_ready_event(item: PAPAttentionWorkItem) -> None:
    if item.ready_event is None:
        return
    with torch.cuda.device(item.qkv_batch.device):
        torch.cuda.current_stream(item.qkv_batch.device).wait_event(item.ready_event)


def _offload_exec_work_item_compatibility_key(
    item: PAPAttentionWorkItem,
) -> tuple[Any, ...]:
    """Return the same-layer ABI key used for ready-item combination."""

    descriptor = item.descriptor
    _request_ids, _steps, scales = _offload_exec_batch_rows(descriptor)
    common_scale: float | tuple[float, ...]
    if scales and all(scale == scales[0] for scale in scales):
        common_scale = scales[0]
    else:
        common_scale = scales
    qkv_batch = item.qkv_batch
    return (
        str(descriptor.layer_name),
        str(qkv_batch.device),
        str(qkv_batch.dtype),
        tuple(int(dim) for dim in qkv_batch.shape[1:]),
        common_scale,
    )


def _combine_offload_exec_work_items(
    items: tuple[PAPAttentionWorkItem, ...],
) -> tuple[PAPOffloadExecBatchDescriptor, torch.Tensor, tuple[int, ...]]:
    if not items:
        raise ValueError("PAP Attention combine requires at least one item")
    first_key = _offload_exec_work_item_compatibility_key(items[0])
    layer_name = str(items[0].descriptor.layer_name)
    request_ids: list[str] = []
    steps: list[int] = []
    scales: list[float] = []
    row_counts: list[int] = []
    qkv_batches: list[torch.Tensor] = []
    for item in items:
        if _offload_exec_work_item_compatibility_key(item) != first_key:
            raise RuntimeError(
                "PAP Attention attempted to combine incompatible work items"
            )
        descriptor = item.descriptor
        rows = _offload_exec_batch_rows(descriptor)
        request_ids.extend(rows[0])
        steps.extend(rows[1])
        scales.extend(rows[2])
        row_count = int(descriptor.item_count)
        if int(item.qkv_batch.shape[0]) != row_count:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC mailbox batch QKV row count does not match descriptor"
            )
        row_counts.append(row_count)
        qkv_batches.append(item.qkv_batch)
    combined_descriptor = PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=(),
        batch_id_suffix="combined-" + "-".join(str(item.peer_id) for item in items),
        metadata_template={
            "r": tuple(request_ids),
            "s": tuple(steps),
            "a": tuple(scales),
        },
    )
    combined_qkv = (
        qkv_batches[0] if len(qkv_batches) == 1 else torch.cat(qkv_batches, dim=0)
    )
    return combined_descriptor, combined_qkv, tuple(row_counts)


def run_offload_exec_mailbox_receiver_loop(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
    dispatcher: PAPAttentionDispatcher,
    peer_id: str,
) -> None:
    """Receive peer batches and transfer their ownership to a dispatcher."""

    trace_offload_exec = _pap_env_flag("PAP_OFFLOAD_EXEC_TRACE", False)
    while True:
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_recv_start_ns = time.perf_counter_ns() if trace_offload_exec else 0
        descriptor, qkv_message, qkv_batch = _recv_next_qkv_batch_message_or_tensor(
            transport
        )
        arrival_ns = time.perf_counter_ns()
        trace_recv_ms = (
            (time.perf_counter() - trace_recv_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        item = PAPAttentionWorkItem(
            descriptor=descriptor,
            qkv_batch=qkv_batch,
            transport=transport,
            peer_id=str(peer_id),
            arrival_ns=arrival_ns,
            input_message=qkv_message,
            ready_event=_record_offload_exec_ready_event(qkv_batch),
            trace_context={
                "enabled": trace_offload_exec,
                "total_start": trace_total_start,
                "recv_start_ns": trace_recv_start_ns,
                "recv_done_ns": arrival_ns,
                "recv_ms": trace_recv_ms,
                "recv_stats": (
                    _qkv_message_recv_trace(qkv_message, trace_recv_ms)
                    if trace_offload_exec
                    else None
                ),
            },
        )
        try:
            if int(qkv_batch.shape[0]) != descriptor.item_count:
                raise RuntimeError(
                    "PAP OFFLOAD_EXEC mailbox batch QKV row count does not "
                    "match descriptor"
                )
            registry.record_offload_exec_peer_batch(
                peer_id=str(peer_id),
                rows=descriptor.item_count,
            )
            dispatcher.enqueue(item)
            item.wait_completed()
        except BaseException:
            item.release_input()
            raise


def _new_offload_exec_compute_trace_stats() -> dict[str, float]:
    return {
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


def _execute_offload_exec_work_items(
    *,
    registry: PAPAttentionRegistry,
    items: tuple[PAPAttentionWorkItem, ...],
) -> None:
    """Combine one ready compatibility group and scatter its output."""

    if len(items) == 1:
        _execute_offload_exec_work_item(
            registry=registry,
            item=items[0],
        )
        return
    for item in items:
        _wait_offload_exec_ready_event(item)
    descriptor, qkv_batch, row_counts = _combine_offload_exec_work_items(items)
    trace_offload_exec = any(
        bool(item.trace_context.get("enabled", False)) for item in items
    )
    trace_compute_stats = (
        _new_offload_exec_compute_trace_stats() if trace_offload_exec else None
    )
    registry.record_offload_exec_compute(
        layer_name=descriptor.layer_name,
        rows=descriptor.item_count,
        source_batches=len(items),
    )
    trace_compute_start = time.perf_counter() if trace_offload_exec else 0.0
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
    if int(output_batch.shape[0]) != sum(row_counts):
        raise RuntimeError(
            "PAP combined Attention output row count does not match inputs"
        )
    trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
    row_start = 0
    for item, row_count in zip(items, row_counts):
        item.transport.send_output_batch(
            item.descriptor,
            output_batch.narrow(0, row_start, row_count),
            remote_address="",
        )
        row_start += row_count
    if not trace_offload_exec:
        return
    trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
    queue_wait_ms = max(
        (float(item.queue_wait_ns) / 1_000_000.0 for item in items),
        default=0.0,
    )
    logger.info(
        "PAP OFFLOAD_EXEC attention combined batch trace layer=%s calls=%d "
        "source_batches=%d compute_ms=%.3f send_output_ms=%.3f "
        "queue_wait_ms=%.3f qkv_shape=%s output_shape=%s batch_key=%s "
        "peers=%s",
        descriptor.layer_name,
        descriptor.item_count,
        len(items),
        trace_compute_ms,
        trace_send_ms,
        queue_wait_ms,
        tuple(qkv_batch.shape),
        tuple(output_batch.shape),
        pap_offload_exec_trace_id(descriptor.output_tensor_id),
        ",".join(item.peer_id for item in items),
    )


def _execute_offload_exec_work_item(
    *,
    registry: PAPAttentionRegistry,
    item: PAPAttentionWorkItem,
) -> None:
    """Compute and send one dispatcher-owned peer batch."""

    descriptor = item.descriptor
    qkv_batch = item.qkv_batch
    if int(qkv_batch.shape[0]) != descriptor.item_count:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC mailbox batch QKV row count does not match descriptor"
        )
    _wait_offload_exec_ready_event(item)
    trace_context = item.trace_context
    trace_offload_exec = bool(trace_context.get("enabled", False))
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
    registry.record_offload_exec_compute(
        layer_name=descriptor.layer_name,
        rows=descriptor.item_count,
        source_batches=1,
    )
    trace_compute_start = time.perf_counter() if trace_offload_exec else 0.0
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
    trace_compute_done_ns = time.perf_counter_ns() if trace_offload_exec else 0
    trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
    trace_send_start_ns = time.perf_counter_ns() if trace_offload_exec else 0
    item.transport.send_output_batch(
        descriptor,
        output_batch,
        remote_address="",
    )
    if not trace_offload_exec:
        return
    trace_send_done_ns = time.perf_counter_ns()
    trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
    trace_total_start = float(trace_context.get("total_start", trace_compute_start))
    trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
    trace_recv_stats = trace_context.get("recv_stats") or {}
    compute_stats = trace_compute_stats or {}

    def metric(stats: dict[str, Any], name: str) -> float:
        return float(stats.get(name, 0.0) or 0.0)

    fields = {
        "layer": descriptor.layer_name,
        "calls": descriptor.item_count,
        "recv_ms": float(trace_context.get("recv_ms", 0.0)),
        "compute_ms": trace_compute_ms,
        "send_ms": trace_send_ms,
        "total_ms": trace_total_ms,
        "recv_wait_ms": metric(trace_recv_stats, "wait_ms"),
        "recv_read_ms": metric(trace_recv_stats, "read_ms"),
        "recv_materialize_ms": metric(trace_recv_stats, "materialize_ms"),
        "recv_transfer_ms": metric(trace_recv_stats, "transfer_ms"),
        "recv_wait_other_ms": metric(trace_recv_stats, "wait_other_ms"),
        "recv_unaccounted_ms": metric(trace_recv_stats, "unaccounted_ms"),
        "append_kv_ms": metric(compute_stats, "append_kv_ms"),
        "pack_ms": metric(compute_stats, "pack_ms"),
        "sdpa_ms": metric(compute_stats, "sdpa_ms"),
        "reshape_ms": metric(compute_stats, "reshape_ms"),
        "paged_metadata_ms": metric(compute_stats, "paged_metadata_ms"),
        "paged_flash_ms": metric(compute_stats, "paged_flash_ms"),
        "fallback_ms": metric(compute_stats, "fallback_ms"),
        "shape_lookup_ms": metric(compute_stats, "shape_lookup_ms"),
        "qkv_split_ms": metric(compute_stats, "qkv_split_ms"),
        "query_move_ms": metric(compute_stats, "query_move_ms"),
        "query_cat_ms": metric(compute_stats, "query_cat_ms"),
        "append_lock_wait_ms": metric(compute_stats, "append_lock_wait_ms"),
        "append_prepare_ms": metric(compute_stats, "append_prepare_ms"),
        "append_record_ms": metric(compute_stats, "append_record_ms"),
        "append_tensor_ms": metric(compute_stats, "append_tensor_ms"),
        "append_copy_ms": metric(compute_stats, "append_copy_ms"),
        "append_state_ms": metric(compute_stats, "append_state_ms"),
        "metadata_build_ms": metric(compute_stats, "metadata_build_ms"),
        "paged_flash_kernel_ms": metric(
            compute_stats,
            "paged_flash_kernel_ms",
        ),
        "attention_output_reshape_ms": metric(
            compute_stats,
            "attention_output_reshape_ms",
        ),
        "compute_unaccounted_ms": metric(
            compute_stats,
            "compute_unaccounted_ms",
        ),
        "qkv_shape": tuple(qkv_batch.shape),
        "output_shape": tuple(output_batch.shape),
        "batch_key": pap_offload_exec_trace_id(descriptor.output_tensor_id),
        "recv_done_ns": int(trace_context.get("recv_done_ns", 0)),
        "compute_done_ns": trace_compute_done_ns,
        "send_done_ns": trace_send_done_ns,
        "recv_start_ns": int(trace_context.get("recv_start_ns", 0)),
        "pre_compute_start_ns": int(compute_stats.get("pre_compute_start_ns", 0.0)),
        "pre_compute_done_ns": int(compute_stats.get("pre_compute_done_ns", 0.0)),
        "paged_flash_done_ns": int(compute_stats.get("paged_flash_done_ns", 0.0)),
        "reshape_done_ns": int(compute_stats.get("post_compute_done_ns", 0.0)),
        "send_start_ns": trace_send_start_ns,
        "queue_wait_ms": item.queue_wait_ns / 1_000_000.0,
        "peer": item.peer_id,
        "arrival_ns": item.arrival_ns,
    }
    logger.info(
        "PAP OFFLOAD_EXEC attention mailbox batch trace layer=%(layer)s "
        "calls=%(calls)d recv_qkv_ms=%(recv_ms).3f "
        "compute_ms=%(compute_ms).3f send_output_ms=%(send_ms).3f "
        "total_ms=%(total_ms).3f recv_wait_ms=%(recv_wait_ms).3f "
        "recv_read_ms=%(recv_read_ms).3f "
        "recv_materialize_ms=%(recv_materialize_ms).3f "
        "recv_transfer_ms=%(recv_transfer_ms).3f "
        "recv_wait_other_ms=%(recv_wait_other_ms).3f "
        "recv_unaccounted_ms=%(recv_unaccounted_ms).3f "
        "append_kv_ms=%(append_kv_ms).3f pack_ms=%(pack_ms).3f "
        "sdpa_ms=%(sdpa_ms).3f reshape_ms=%(reshape_ms).3f "
        "paged_metadata_ms=%(paged_metadata_ms).3f "
        "paged_flash_ms=%(paged_flash_ms).3f fallback_ms=%(fallback_ms).3f "
        "shape_lookup_ms=%(shape_lookup_ms).3f "
        "qkv_split_ms=%(qkv_split_ms).3f query_move_ms=%(query_move_ms).3f "
        "query_cat_ms=%(query_cat_ms).3f "
        "append_lock_wait_ms=%(append_lock_wait_ms).3f "
        "append_prepare_ms=%(append_prepare_ms).3f "
        "append_record_ms=%(append_record_ms).3f "
        "append_tensor_ms=%(append_tensor_ms).3f "
        "append_copy_ms=%(append_copy_ms).3f "
        "append_state_ms=%(append_state_ms).3f "
        "metadata_build_ms=%(metadata_build_ms).3f "
        "paged_flash_kernel_ms=%(paged_flash_kernel_ms).3f "
        "attention_output_reshape_ms=%(attention_output_reshape_ms).3f "
        "compute_unaccounted_ms=%(compute_unaccounted_ms).3f "
        "qkv_shape=%(qkv_shape)s output_shape=%(output_shape)s "
        "batch_key=%(batch_key)s recv_done_ns=%(recv_done_ns)d "
        "compute_done_ns=%(compute_done_ns)d send_done_ns=%(send_done_ns)d "
        "recv_start_ns=%(recv_start_ns)d "
        "pre_compute_start_ns=%(pre_compute_start_ns)d "
        "pre_compute_done_ns=%(pre_compute_done_ns)d "
        "paged_flash_done_ns=%(paged_flash_done_ns)d "
        "reshape_done_ns=%(reshape_done_ns)d send_start_ns=%(send_start_ns)d "
        "queue_wait_ms=%(queue_wait_ms).3f peer=%(peer)s "
        "arrival_ns=%(arrival_ns)d",
        fields,
    )


def run_offload_exec_mailbox_loop(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
    peer_id: str | None = None,
) -> None:
    """Consume mailbox QKV messages and publish mailbox attention outputs."""

    trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    prefetch_enabled = _pap_env_flag(
        "PAP_ATTENTION_MAILBOX_PREFETCH", False
    ) and callable(getattr(transport, "recv_next_qkv_batch_message", None))
    peer_id = peer_id or str(getattr(transport, "actor_id", type(transport).__name__))
    prefetcher = _QKVBatchMessagePrefetcher(transport) if prefetch_enabled else None
    if prefetcher is not None:
        prefetcher.prefetch()
    while True:
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_recv_start_ns = 0
        trace_recv_done_ns = 0
        trace_compute_done_ns = 0
        trace_send_start_ns = 0
        trace_send_done_ns = 0
        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
        if trace_offload_exec:
            trace_recv_start_ns = time.perf_counter_ns()
        qkv_message = None
        if prefetcher is not None:
            descriptor, qkv_message, qkv_batch = prefetcher.result()
            prefetcher.prefetch()
        else:
            descriptor, qkv_message, qkv_batch = _recv_next_qkv_batch_message_or_tensor(
                transport
            )
        arrival_ns = time.perf_counter_ns()
        trace_recv_ms = (
            (time.perf_counter() - trace_recv_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        trace_recv_stats = (
            _qkv_message_recv_trace(qkv_message, trace_recv_ms)
            if trace_offload_exec
            else None
        )
        if trace_offload_exec:
            trace_recv_done_ns = time.perf_counter_ns()
        try:
            if int(qkv_batch.shape[0]) != descriptor.item_count:
                raise RuntimeError(
                    "PAP OFFLOAD_EXEC mailbox batch QKV row count does not "
                    "match descriptor"
                )
            registry.record_offload_exec_peer_batch(
                peer_id=peer_id,
                rows=descriptor.item_count,
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
                    "compute_done_ns": 0.0,
                    "post_compute_done_ns": 0.0,
                }
                if trace_offload_exec
                else None
            )
            registry.record_offload_exec_compute(
                layer_name=descriptor.layer_name,
                rows=descriptor.item_count,
                source_batches=1,
            )
            output_batch = compute_offload_exec_batch_output(
                registry=registry,
                descriptor=descriptor,
                qkv_batch=qkv_batch,
                trace_stats=trace_compute_stats,
            )
        finally:
            if qkv_message is not None:
                qkv_message.release()
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
            remote_address="",
        )
        if trace_offload_exec:
            trace_send_done_ns = time.perf_counter_ns()
            trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
            trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
            logger.info(
                "PAP OFFLOAD_EXEC attention mailbox batch trace layer=%s "
                "calls=%d recv_qkv_ms=%.3f compute_ms=%.3f "
                "send_output_ms=%.3f total_ms=%.3f "
                "recv_wait_ms=%.3f recv_read_ms=%.3f "
                "recv_materialize_ms=%.3f recv_transfer_ms=%.3f "
                "recv_wait_other_ms=%.3f recv_unaccounted_ms=%.3f "
                "append_kv_ms=%.3f "
                "pack_ms=%.3f sdpa_ms=%.3f reshape_ms=%.3f "
                "paged_metadata_ms=%.3f paged_flash_ms=%.3f fallback_ms=%.3f "
                "shape_lookup_ms=%.3f qkv_split_ms=%.3f query_move_ms=%.3f "
                "query_cat_ms=%.3f append_lock_wait_ms=%.3f "
                "append_prepare_ms=%.3f append_record_ms=%.3f "
                "append_tensor_ms=%.3f append_copy_ms=%.3f "
                "append_state_ms=%.3f metadata_build_ms=%.3f "
                "paged_flash_kernel_ms=%.3f attention_output_reshape_ms=%.3f "
                "compute_unaccounted_ms=%.3f qkv_shape=%s output_shape=%s "
                "batch_key=%s "
                "recv_done_ns=%d compute_done_ns=%d send_done_ns=%d "
                "recv_start_ns=%d pre_compute_start_ns=%d "
                "pre_compute_done_ns=%d paged_flash_done_ns=%d reshape_done_ns=%d "
                "send_start_ns=%d peer=%s arrival_ns=%d",
                descriptor.layer_name,
                descriptor.item_count,
                trace_recv_ms,
                trace_compute_ms,
                trace_send_ms,
                trace_total_ms,
                trace_recv_stats["wait_ms"] if trace_recv_stats else 0.0,
                trace_recv_stats["read_ms"] if trace_recv_stats else 0.0,
                trace_recv_stats["materialize_ms"] if trace_recv_stats else 0.0,
                trace_recv_stats["transfer_ms"] if trace_recv_stats else 0.0,
                trace_recv_stats["wait_other_ms"] if trace_recv_stats else 0.0,
                trace_recv_stats["unaccounted_ms"] if trace_recv_stats else 0.0,
                trace_compute_stats["append_kv_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["pack_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["sdpa_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["reshape_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["paged_metadata_ms"]
                if trace_compute_stats
                else 0.0,
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
                (
                    trace_compute_stats["append_prepare_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (
                    trace_compute_stats["append_record_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (
                    trace_compute_stats["append_tensor_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (trace_compute_stats["append_copy_ms"] if trace_compute_stats else 0.0),
                (
                    trace_compute_stats["append_state_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (
                    trace_compute_stats["metadata_build_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
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
                peer_id,
                arrival_ns,
            )
