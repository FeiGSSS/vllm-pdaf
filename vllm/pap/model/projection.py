# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side PAP Attention adapter."""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.config import (
    PAPOffloadExecTransport as PAPOffloadExecTransportKind,
)
from vllm.pap.config import parse_offload_exec_transport
from vllm.pap.mode import is_pap_request_id, pap_request_ids_are_routable
from vllm.pap.model.context import (
    PAPModelForwardBatch,
    pap_endpoint_for_tp_rank,
    pap_tensor_parallel_rank,
)
from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    pap_offload_exec_trace_id,
)

logger = init_logger(__name__)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_PAP_STEP_GROUPS_KEY = "_pap_qwen3_offload_exec_step_groups"


def _pap_env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_ENV_VALUES


def _pap_direct_qkv_send_enabled() -> bool:
    return (
        os.environ.get("PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND", "1").lower()
        in _TRUE_ENV_VALUES
    )


def _pap_offload_exec_session_request_id(
    request_id: str,
    prefill_kv_handle: Any,
) -> str:
    return str(prefill_kv_handle or request_id)


def _pap_pack_qkv_group_items(
    group_items: list[tuple[int, Any, tuple[torch.Tensor, ...]]],
) -> torch.Tensor:
    if len(group_items) == 1:
        return torch.cat(group_items[0][2], dim=-1)
    return torch.cat(
        [torch.cat(item[2], dim=-1) for item in group_items],
        dim=0,
    )


def _pap_req_indices_are_contiguous(req_indices: tuple[int, ...]) -> bool:
    if not req_indices:
        return False
    start = int(req_indices[0])
    return start >= 0 and req_indices == tuple(range(start, start + len(req_indices)))


def _pap_direct_qkv_batch_for_indices(
    qkv_batch: torch.Tensor | None,
    req_indices: tuple[int, ...],
) -> torch.Tensor | None:
    if qkv_batch is None or not req_indices:
        return None
    if qkv_batch.ndim != 2 or not qkv_batch.is_contiguous():
        return None
    if not _pap_req_indices_are_contiguous(req_indices):
        return None
    start = int(req_indices[0])
    stop = start + len(req_indices)
    if stop > int(qkv_batch.shape[0]):
        return None
    direct = qkv_batch[start:stop]
    return direct if direct.is_contiguous() else None


def _pap_route_index_tensor(
    additional_kwargs: dict[str, Any],
    req_indices: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    route_cache = additional_kwargs.setdefault(
        "_pap_qwen3_route_index_tensors",
        {},
    )
    cache_key = (str(torch.device(device)), req_indices)
    cached = route_cache.get(cache_key)
    if cached is not None:
        return cached
    index_tensor = torch.tensor(
        req_indices,
        dtype=torch.long,
        device=device,
    )
    route_cache[cache_key] = index_tensor
    return index_tensor


def _pap_qkv_batch_for_indices(
    qkv_batch: torch.Tensor | None,
    req_indices: tuple[int, ...],
    *,
    index_tensor: torch.Tensor | None,
) -> tuple[torch.Tensor | None, bool]:
    direct = _pap_direct_qkv_batch_for_indices(qkv_batch, req_indices)
    if direct is not None:
        return direct, True
    if (
        qkv_batch is None
        or qkv_batch.ndim != 2
        or not qkv_batch.is_contiguous()
        or not req_indices
        or index_tensor is None
    ):
        return None, False
    return torch.index_select(qkv_batch, 0, index_tensor), False


def _pap_scatter_attention_output_group(
    output: torch.Tensor,
    remote_output: torch.Tensor,
    *,
    req_indices: tuple[int, ...],
    index_tensor: torch.Tensor | None,
) -> None:
    if not req_indices:
        raise RuntimeError("PAP remote attention output has no route rows")
    remote_output = remote_output.to(
        device=output.device,
        dtype=output.dtype,
        non_blocking=True,
    )
    target_shape = (len(req_indices), *output.shape[1:])
    target_numel = math.prod(target_shape)
    if int(remote_output.numel()) != int(target_numel):
        raise RuntimeError(
            "PAP remote attention output shape mismatch: "
            f"got {tuple(remote_output.shape)}, expected {target_shape}"
        )
    remote_output = remote_output.reshape(target_shape)
    if _pap_req_indices_are_contiguous(req_indices):
        start = int(req_indices[0])
        output[start : start + len(req_indices)].copy_(remote_output)
        return
    if index_tensor is None:
        index_tensor = torch.tensor(
            req_indices,
            dtype=torch.long,
            device=output.device,
        )
    output.index_copy_(0, index_tensor, remote_output)


@dataclass(frozen=True)
class _PAPOffloadExecStepGroup:
    attention_endpoint: str
    offload_exec_zmq_endpoint: str
    req_indices: tuple[int, ...]
    batch_id_suffix: str
    metadata_template: dict[str, Any]


def _pap_offload_exec_step_groups(
    additional_kwargs: dict[str, Any],
    *,
    num_reqs: int,
    scaling: float,
) -> tuple[_PAPOffloadExecStepGroup, ...]:
    cached = additional_kwargs.get(_PAP_STEP_GROUPS_KEY)
    if cached is not None:
        return tuple(cached)

    request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
    route_groups = tuple(
        additional_kwargs.get("pap_offload_exec_route_groups") or ()
    )
    if not route_groups:
        raise RuntimeError("PAP attention missing OFFLOAD_EXEC route groups")

    attention_kv_installed = set(
        additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
    )
    prefix_len_by_request = (
        additional_kwargs.get("pap_prefill_prefix_len_by_request") or {}
    )
    prefill_kv_handle_by_request = (
        additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
    )
    step_groups: list[_PAPOffloadExecStepGroup] = []
    routed_req_indices: set[int] = set()
    for route_group in route_groups:
        attention_endpoint = pap_endpoint_for_tp_rank(
            route_group.get("attention_endpoint")
        )
        offload_exec_zmq_endpoint = pap_endpoint_for_tp_rank(
            route_group.get("offload_exec_zmq_endpoint")
        )
        if not attention_endpoint:
            raise RuntimeError(
                "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
            )
        if not offload_exec_zmq_endpoint:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC mailbox path missing pap_offload_exec_zmq_endpoint"
            )

        req_indices = tuple(
            int(req_index) for req_index in route_group.get("req_indices", ())
        )
        group_request_ids = tuple(
            str(request_id) for request_id in route_group.get("request_ids", ())
        )
        group_steps = tuple(int(step) for step in route_group.get("steps", ()))
        if not (len(req_indices) == len(group_request_ids) == len(group_steps)):
            raise RuntimeError("PAP OFFLOAD_EXEC route group is malformed")

        session_request_ids: list[str] = []
        for group_offset, req_index in enumerate(req_indices):
            if req_index < 0 or req_index >= num_reqs:
                raise RuntimeError("PAP OFFLOAD_EXEC route index out of range")
            request_id = group_request_ids[group_offset]
            if request_id != str(request_ids[req_index]):
                raise RuntimeError("PAP OFFLOAD_EXEC route request mismatch")
            if not is_pap_request_id(request_id):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            routed_req_indices.add(req_index)
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed:
                if not prefill_kv_handle:
                    raise RuntimeError("PAP missing local prefill KV handle")
                raise RuntimeError("PAP attention KV is not installed")
            session_request_ids.append(
                _pap_offload_exec_session_request_id(
                    request_id,
                    prefill_kv_handle,
                )
            )

        batch_id_suffix = ",".join(
            f"{request_id}@{step}"
            for request_id, step in zip(session_request_ids, group_steps)
        )
        step_groups.append(
            _PAPOffloadExecStepGroup(
                attention_endpoint=str(attention_endpoint),
                offload_exec_zmq_endpoint=str(offload_exec_zmq_endpoint),
                req_indices=req_indices,
                batch_id_suffix=batch_id_suffix,
                metadata_template={
                    "r": tuple(session_request_ids),
                    "s": group_steps,
                    "a": (float(scaling),) * len(group_steps),
                },
            )
        )

    if len(routed_req_indices) != num_reqs:
        raise RuntimeError("PAP OFFLOAD_EXEC route groups do not cover batch")

    result = tuple(step_groups)
    additional_kwargs[_PAP_STEP_GROUPS_KEY] = result
    return result


def _pap_offload_exec_transport_kind() -> PAPOffloadExecTransportKind:
    return parse_offload_exec_transport(
        os.environ.get("PAP_OFFLOAD_EXEC_TRANSPORT", "nixl_mailbox")
    )


@cache
def _pap_cached_offload_exec_transport(attention_endpoint: str):
    from vllm.pap.transport.factory import build_offload_exec_transport

    local_rank = pap_tensor_parallel_rank()
    actor_base = os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection")
    endpoint_hash = hashlib.sha1(attention_endpoint.encode("utf-8")).hexdigest()[:12]
    actor_id = f"{actor_base}-r{local_rank}-{endpoint_hash}"
    return build_offload_exec_transport(
        transport=_pap_offload_exec_transport_kind(),
        actor_id=actor_id,
        local_rank=local_rank,
    )


def _pap_offload_exec_transport_for_attention_endpoint(
    attention_endpoint: str | None,
    offload_exec_zmq_endpoint: str | None = None,
):
    del offload_exec_zmq_endpoint
    return _pap_cached_offload_exec_transport(str(attention_endpoint or ""))


def _pap_bind_offload_exec_mailbox_peer(
    transport: Any,
    attention_endpoint: str | None,
) -> None:
    if not attention_endpoint:
        raise RuntimeError(
            "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
        )
    if getattr(transport, "_pap_mailbox_bound", False):
        return
    from vllm.pap.attention.client import bind_offload_exec_mailbox

    peer_metadata = bind_offload_exec_mailbox(
        attention_endpoint=attention_endpoint,
        local_agent_metadata=transport.local_agent_metadata,
        source_id=(
            f"{os.environ.get('PAP_NIXL_MAILBOX_ACTOR_ID', 'projection')}"
            f"-r{pap_tensor_parallel_rank()}"
        ),
    )
    transport.bind_peer(peer_metadata)
    transport._pap_mailbox_bound = True
    transport._pap_mailbox_bound_attention_endpoint = attention_endpoint


@dataclass(slots=True)
class PAPProjectionAttentionAdapter:
    """Execute the Projection side of PAP Attention for one model layer."""

    layer_name: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scaling: float
    last_projection_timeline: dict[str, Any] | None = field(
        default=None,
        init=False,
    )

    def begin_step(self) -> None:
        """Reset per-forward Projection diagnostics."""
        self.last_projection_timeline = None

    def direct_qkv_send_enabled(self) -> bool:
        """Whether the current runtime accepts the packed QKV buffer."""
        return _pap_direct_qkv_send_enabled()

    def record_projection_timeline(self, timeline: dict[str, Any]) -> None:
        """Retain one layer timeline for outer model diagnostics."""
        self.last_projection_timeline = dict(timeline)

    def should_execute(self) -> bool:
        """Return whether the current forward is a valid PAP decode batch."""

        def reject(reason: str) -> bool:
            if _pap_env_enabled("PAP_DEBUG_DECISION"):
                logger.info(
                    "PAP attention disabled for %s: %s",
                    self.layer_name,
                    reason,
                )
            return False

        batch = PAPModelForwardBatch.current(self.layer_name)
        if batch is None:
            return reject("missing forward context")
        if not batch.enabled:
            return reject("pap_enabled is false")
        attn_metadata = batch.attention_metadata
        if attn_metadata is None:
            return reject("missing attn metadata")
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
            return reject(
                f"max_query_len={getattr(attn_metadata, 'max_query_len', None)}"
            )
        if batch.num_reqs <= 0:
            return reject(f"num_reqs={batch.num_reqs}")
        if len(batch.request_ids) < batch.num_reqs:
            return reject(
                "request_ids too short "
                f"len={len(batch.request_ids)} num_reqs={batch.num_reqs}"
            )
        if len(batch.num_scheduled_tokens) < batch.num_reqs:
            return reject(
                "num_scheduled_tokens too short "
                f"len={len(batch.num_scheduled_tokens)} num_reqs={batch.num_reqs}"
            )
        if not pap_request_ids_are_routable(batch.request_ids, batch.num_reqs):
            return reject(
                "non-PAP request id in scheduled batch "
                f"request_ids={batch.request_ids[: batch.num_reqs][:4]}"
            )
        if any(
            num_tokens != 1
            for num_tokens in batch.num_scheduled_tokens[: batch.num_reqs]
        ):
            return reject(
                "non-decode num_scheduled_tokens="
                f"{batch.num_scheduled_tokens[: batch.num_reqs]}"
            )
        installed = set(
            batch.additional_kwargs.get(
                "pap_attention_kv_installed_by_request"
            )
            or ()
        )
        active_request_ids = batch.request_ids[: batch.num_reqs]
        if not all(request_id in installed for request_id in active_request_ids):
            return reject(
                "attention KV not ready "
                f"request_ids={active_request_ids} installed={tuple(installed)[:4]}"
            )
        return True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pre_attn_compute_ms: float = 0.0,
        pre_attn_start_ns: int = 0,
        pre_attn_done_ns: int = 0,
        projection_timeline: dict[str, Any] | None = None,
        direct_qkv_send_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Offload one decode Attention layer and return its output."""
        batch = PAPModelForwardBatch.current(self.layer_name)
        if batch is None:
            raise RuntimeError("PAP attention requires forward context")
        attn_metadata = batch.attention_metadata
        if attn_metadata is None:
            raise RuntimeError(f"PAP attention missing metadata for {self.layer_name}")
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
            raise RuntimeError("PAP attention currently supports decode-only batches")

        request_ids = batch.request_ids
        num_reqs = batch.num_reqs
        num_actual_tokens = int(
            batch.additional_kwargs.get("pap_num_actual_tokens") or q.shape[0]
        )
        num_scheduled_tokens = batch.num_scheduled_tokens
        if num_reqs <= 0 or len(request_ids) < num_reqs:
            raise RuntimeError("PAP attention missing request ids")
        if num_actual_tokens < num_reqs:
            raise RuntimeError("PAP attention expected one actual token per request")
        if num_scheduled_tokens and any(
            num_tokens != 1 for num_tokens in num_scheduled_tokens[:num_reqs]
        ):
            raise RuntimeError("PAP attention currently supports one token per request")

        query = q.view(-1, self.num_heads, self.head_dim)
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view(-1, self.num_kv_heads, self.head_dim)
        step_groups = _pap_offload_exec_step_groups(
            batch.additional_kwargs,
            num_reqs=num_reqs,
            scaling=float(self.scaling),
        )
        all_requests_offloaded = (
            sum(len(group.req_indices) for group in step_groups) == num_reqs
        )
        direct_mailbox_output_enabled = _pap_env_enabled(
            "PAP_DIRECT_MAILBOX_OUTPUT"
        )
        output: torch.Tensor | None = None
        release_messages: list[Any] = []

        def get_copy_output_buffer() -> torch.Tensor:
            nonlocal output
            if output is None:
                output = (
                    torch.empty_like(query)
                    if all_requests_offloaded
                    else torch.zeros_like(query)
                )
            return output

        direct_qkv_send_enabled = _pap_direct_qkv_send_enabled()
        trace_offload_exec = _pap_env_enabled("PAP_OFFLOAD_EXEC_TRACE")
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_send_done_ns = 0
        trace_yield_start_ns = 0
        trace_yield_end_ns = 0
        trace_recv_done_ns = 0
        trace_recv_ms = 0.0
        trace_total_ms = 0.0
        trace_batch_keys = ""
        trace_contiguous_route_groups = 0
        trace_direct_qkv_groups = 0
        trace_packed_qkv_groups = 0
        trace_direct_output_rows = 0
        trace_scattered_output_rows = 0
        trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
        offload_exec_batches: list[
            tuple[
                str | None,
                str,
                PAPOffloadExecBatchDescriptor,
                tuple[int, ...],
                Any,
                torch.Tensor | None,
            ]
        ] = []
        for step_group in step_groups:
            attention_endpoint = step_group.attention_endpoint
            offload_exec_zmq_endpoint = step_group.offload_exec_zmq_endpoint
            req_indices = step_group.req_indices
            route_is_contiguous = _pap_req_indices_are_contiguous(req_indices)
            if trace_offload_exec and route_is_contiguous:
                trace_contiguous_route_groups += 1
            route_index_tensor = (
                None
                if route_is_contiguous
                else _pap_route_index_tensor(
                    batch.additional_kwargs,
                    req_indices,
                    device=query.device,
                )
            )
            transport = _pap_offload_exec_transport_for_attention_endpoint(
                attention_endpoint,
                offload_exec_zmq_endpoint,
            )
            batch_descriptor = PAPOffloadExecBatchDescriptor(
                layer_name=self.layer_name,
                items=(),
                batch_id_suffix=step_group.batch_id_suffix,
                metadata_template=step_group.metadata_template,
            )
            _pap_bind_offload_exec_mailbox_peer(transport, attention_endpoint)
            send_qkv_batch_direct = getattr(transport, "send_qkv_batch_direct", None)
            qkv_width = (
                self.num_heads * self.head_dim
                + 2 * self.num_kv_heads * self.head_dim
            )
            direct_qkv_batch: torch.Tensor | None = None
            direct_layout = False
            if direct_qkv_send_enabled and callable(send_qkv_batch_direct):
                direct_qkv_batch, direct_layout = _pap_qkv_batch_for_indices(
                    direct_qkv_send_buffer,
                    req_indices,
                    index_tensor=route_index_tensor,
                )
            if direct_qkv_batch is not None:
                if trace_offload_exec:
                    if direct_layout:
                        trace_direct_qkv_groups += 1
                    else:
                        trace_packed_qkv_groups += 1
                if int(direct_qkv_batch.shape[-1]) != qkv_width:
                    raise RuntimeError("PAP direct QKV batch width mismatch")
                send_qkv_batch_direct(
                    batch_descriptor,
                    direct_qkv_batch,
                    remote_address=offload_exec_zmq_endpoint,
                )
            else:
                if trace_offload_exec:
                    trace_packed_qkv_groups += 1
                group_items = [
                    (
                        req_index,
                        None,
                        (
                            query[req_index : req_index + 1].reshape(1, -1),
                            key[req_index : req_index + 1].reshape(1, -1),
                            value[req_index : req_index + 1].reshape(1, -1),
                        ),
                    )
                    for req_index in req_indices
                ]
                qkv_batch = _pap_pack_qkv_group_items(group_items)
                transport.send_qkv_batch(
                    batch_descriptor,
                    qkv_batch,
                    remote_address=offload_exec_zmq_endpoint,
                )
            offload_exec_batches.append(
                (
                    attention_endpoint,
                    offload_exec_zmq_endpoint,
                    batch_descriptor,
                    req_indices,
                    transport,
                    route_index_tensor,
                )
            )
        trace_send_ms = (
            (time.perf_counter() - trace_send_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        if trace_offload_exec:
            trace_send_done_ns = time.perf_counter_ns()

        trace_trigger_start = time.perf_counter() if trace_offload_exec else 0.0
        prepared_output_messages: list[Any | None] = []
        output_width = self.num_heads * self.head_dim
        for (
            _attention_endpoint,
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            transport,
            _route_index_tensor,
        ) in offload_exec_batches:
            output_message = transport.prepare_output_batch_message(
                batch_descriptor,
                shape=(len(req_indices), output_width),
                dtype=query.dtype,
                remote_address=offload_exec_zmq_endpoint,
            )
            prepared_output_messages.append(output_message)
        trace_trigger_ms = (
            (time.perf_counter() - trace_trigger_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        trace_yield_start = time.perf_counter() if trace_offload_exec else 0.0
        if trace_offload_exec:
            trace_yield_start_ns = time.perf_counter_ns()
        trace_yield_ms = 0.0
        if trace_offload_exec:
            trace_yield_end_ns = time.perf_counter_ns()
            trace_yield_ms = (time.perf_counter() - trace_yield_start) * 1000.0

        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0

        def record_projection_trace() -> None:
            nonlocal trace_batch_keys
            if not trace_offload_exec or not offload_exec_batches:
                return
            trace_batch_keys = "|".join(
                pap_offload_exec_trace_id(item[2].output_tensor_id)
                for item in offload_exec_batches
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

        for batch_index, (
            _attention_endpoint,
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            transport,
            route_index_tensor,
        ) in enumerate(offload_exec_batches):
            output_message = prepared_output_messages[batch_index]
            if output_message is not None:
                output_batch = output_message.tensor
            else:
                output_message = transport.recv_output_batch_message(
                    batch_descriptor,
                    remote_address=offload_exec_zmq_endpoint,
                )
                output_batch = output_message.tensor
            try:
                if int(output_batch.shape[0]) != batch_descriptor.item_count:
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC output batch row count mismatch"
                    )
                can_use_direct_output = (
                    direct_mailbox_output_enabled
                    and len(offload_exec_batches) == 1
                    and len(req_indices) == num_reqs
                    and req_indices == tuple(range(num_reqs))
                    and output_batch.device == query.device
                    and output_batch.dtype == query.dtype
                    and int(output_batch.numel())
                    == int(q.shape[0]) * self.num_heads * self.head_dim
                )
                if can_use_direct_output:
                    if trace_offload_exec:
                        trace_direct_output_rows += len(req_indices)
                    direct_output = output_batch.view(
                        q.shape[0], self.num_heads * self.head_dim
                    )
                    if output_message is not None:
                        release_messages.append(output_message)
                        output_message = None
                    if trace_offload_exec:
                        trace_recv_done_ns = time.perf_counter_ns()
                        trace_recv_ms = (
                            time.perf_counter() - trace_recv_start
                        ) * 1000.0
                        trace_total_ms = (
                            time.perf_counter() - trace_total_start
                        ) * 1000.0
                        record_projection_trace()
                    return direct_output, release_messages
                if trace_offload_exec:
                    trace_scattered_output_rows += len(req_indices)
                _pap_scatter_attention_output_group(
                    get_copy_output_buffer(),
                    output_batch,
                    req_indices=req_indices,
                    index_tensor=route_index_tensor,
                )
            finally:
                if output_message is not None:
                    output_message.release()
        if trace_offload_exec and offload_exec_batches:
            trace_recv_done_ns = time.perf_counter_ns()
            trace_recv_ms = (time.perf_counter() - trace_recv_start) * 1000.0
            trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
            record_projection_trace()
        final_output = get_copy_output_buffer()
        return final_output.view(q.shape[0], self.num_heads * self.head_dim), []
