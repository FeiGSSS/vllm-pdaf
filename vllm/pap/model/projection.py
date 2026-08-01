# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side PAP Attention adapter."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    end_deferred_cuda_span,
)
from vllm.pap.mode import pap_request_ids_are_routable
from vllm.pap.model.context import PAPModelForwardBatch
from vllm.pap.model.projection_io import (
    _pap_pack_qkv_group_items,
    _pap_qkv_batch_for_indices,
    _pap_req_indices_are_contiguous,
    _pap_route_index_tensor,
    _pap_scatter_attention_output_group,
)
from vllm.pap.model.projection_routing import (
    _pap_offload_exec_step_groups,
    _PAPOffloadExecStepGroup,
)
from vllm.pap.model.projection_trace import (
    PAPProjectionTraceState,
)
from vllm.pap.model.projection_trace import (
    record_projection_trace as record_projection_trace_snapshot,
)
from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    PAPStepPlannedOffloadExecTransport,
)
from vllm.pap.transport.projection import (
    _pap_bind_offload_exec_mailbox_peer,
    _pap_cached_step_planned_transport,
    _pap_offload_exec_transport_for_attention_endpoint,
)

logger = init_logger(__name__)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_PAP_LOCAL_BATCHED_FANOUT_PLAN_KEY = "_pap_qwen3_local_qkv_batched_fanout_plan"


def _pap_env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_ENV_VALUES


def _pap_direct_qkv_send_enabled() -> bool:
    return (
        os.environ.get("PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND", "1").lower()
        in _TRUE_ENV_VALUES
    )


@dataclass(slots=True)
class PAPProjectionAttentionAdapter:
    """Execute the Projection side of PAP Attention for one model layer."""

    layer_name: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scaling: float
    num_hidden_layers: int = 0
    _direct_qkv_send: bool = field(init=False, repr=False)
    _direct_mailbox_output: bool = field(init=False, repr=False)
    _trace_offload_exec: bool = field(init=False, repr=False)
    _debug_decision: bool = field(init=False, repr=False)
    _prepared_batch: PAPModelForwardBatch | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_projection_timeline: dict[str, Any] | None = field(
        default=None,
        init=False,
    )

    def __post_init__(self) -> None:
        """Resolve process-lifetime runtime switches once per model layer."""
        self._direct_qkv_send = _pap_direct_qkv_send_enabled()
        self._direct_mailbox_output = _pap_env_enabled("PAP_DIRECT_MAILBOX_OUTPUT")
        self._trace_offload_exec = _pap_env_enabled("PAP_OFFLOAD_EXEC_TRACE")
        self._debug_decision = _pap_env_enabled("PAP_DEBUG_DECISION")

    def begin_step(self) -> None:
        """Reset per-forward Projection diagnostics."""
        self.last_projection_timeline = None
        self._prepared_batch = None

    def direct_qkv_send_enabled(self) -> bool:
        """Whether the current runtime accepts the packed QKV buffer."""
        return self._direct_qkv_send

    def prepare_step(self, dtype: torch.dtype) -> None:
        """Publish layer-independent step state before layer-0 QKV compute."""
        if ".layers.0." not in self.layer_name:
            return
        batch = self._prepared_batch
        if batch is None:
            return
        step_groups = _pap_offload_exec_step_groups(
            batch.additional_kwargs,
            num_reqs=batch.num_reqs,
            scaling=float(self.scaling),
        )
        prepared_groups: list[
            tuple[
                _PAPOffloadExecStepGroup,
                Any,
                PAPStepPlannedOffloadExecTransport | None,
            ]
        ] = []
        for step_group in step_groups:
            transport = _pap_offload_exec_transport_for_attention_endpoint(
                step_group.attention_endpoint,
                step_group.offload_exec_zmq_endpoint,
            )
            _pap_bind_offload_exec_mailbox_peer(
                transport,
                step_group.attention_endpoint,
            )
            prepared_groups.append(
                (
                    step_group,
                    transport,
                    _pap_cached_step_planned_transport(step_group.attention_endpoint),
                )
            )

        qkv_width = (
            self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
        )
        from vllm.pap.transport.local.batched_fanout import (
            local_qkv_batched_fanout_available,
        )

        batched_fanout = (
            self._direct_qkv_send
            and self.num_hidden_layers > 0
            and local_qkv_batched_fanout_available()
            and os.environ.get(
                "PAP_LOCAL_BATCHED_FANOUT",
                "1",
            ).lower()
            in _TRUE_ENV_VALUES
            and all(
                _pap_req_indices_are_contiguous(step_group.req_indices)
                and step_planned_transport is not None
                for step_group, _transport, step_planned_transport in prepared_groups
            )
        )
        for step_group, _transport, step_planned_transport in prepared_groups:
            if step_planned_transport is None:
                continue
            step_planned_transport.send_step_prepare(
                PAPOffloadExecBatchDescriptor(
                    layer_name=self.layer_name,
                    items=(),
                    batch_id_suffix=step_group.batch_id_suffix,
                    metadata_template=step_group.metadata_template,
                ),
                dtype=dtype,
                remote_address=step_group.offload_exec_zmq_endpoint,
                descriptorless_qkv=batched_fanout,
                qkv_width=qkv_width if batched_fanout else 0,
                layer_count=(self.num_hidden_layers if batched_fanout else 0),
            )
        if batched_fanout:
            from vllm.pap.transport.local.batched_fanout import (
                build_local_qkv_batched_fanout_plan,
            )

            fanout_entries = [
                (
                    transport,
                    int(step_group.req_indices[0]),
                    len(step_group.req_indices),
                )
                for step_group, transport, _step_planned_transport in prepared_groups
            ]
            batch.additional_kwargs[_PAP_LOCAL_BATCHED_FANOUT_PLAN_KEY] = (
                build_local_qkv_batched_fanout_plan(
                    fanout_entries,
                    device=prepared_groups[0][1].device,
                    dtype=dtype,
                    qkv_width=qkv_width,
                    num_layers=self.num_hidden_layers,
                    num_source_rows=batch.num_reqs,
                )
            )

    def record_projection_timeline(self, timeline: dict[str, Any]) -> None:
        """Retain one layer timeline for outer model diagnostics."""
        self.last_projection_timeline = dict(timeline)

    def should_execute(self) -> bool:
        """Return whether the current forward is a valid PAP decode batch."""
        self._prepared_batch = None

        def reject(reason: str) -> bool:
            if self._debug_decision:
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
            batch.additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        active_request_ids = batch.request_ids[: batch.num_reqs]
        if not all(request_id in installed for request_id in active_request_ids):
            return reject(
                "attention KV not ready "
                f"request_ids={active_request_ids} installed={tuple(installed)[:4]}"
            )
        self._prepared_batch = batch
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
        reuse_query_output_buffer: bool = False,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Offload one decode Attention layer and return its output."""
        batch = self._prepared_batch or PAPModelForwardBatch.current(self.layer_name)
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
        direct_mailbox_output_enabled = self._direct_mailbox_output
        output: torch.Tensor | None = None
        release_messages: list[Any] = []

        def get_copy_output_buffer() -> torch.Tensor:
            nonlocal output
            if output is None:
                if all_requests_offloaded and reuse_query_output_buffer:
                    output = query
                elif all_requests_offloaded:
                    output = torch.empty_like(query)
                else:
                    output = torch.zeros_like(query)
            return output

        direct_qkv_send_enabled = self._direct_qkv_send
        trace_offload_exec = self._trace_offload_exec
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_send_done_ns = 0
        trace_yield_start_ns = 0
        trace_yield_end_ns = 0
        trace_recv_done_ns = 0
        trace_recv_ms = 0.0
        trace_total_ms = 0.0
        trace_contiguous_route_groups = 0
        trace_direct_qkv_groups = 0
        trace_packed_qkv_groups = 0
        trace_direct_output_rows = 0
        trace_scattered_output_rows = 0
        trace_fanout_prepare_us: list[float] = []
        trace_fanout_submit_us: list[float] = []
        trace_fanout_submit_start_ns: list[int] = []
        trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
        current_stream = torch.cuda.current_stream(query.device)
        trace_output_origin: torch.cuda.Event | None = None
        if trace_offload_exec:
            trace_output_origin = torch.cuda.Event(enable_timing=True)
            trace_output_origin.record(current_stream)
        remote_stage_trace = begin_deferred_cuda_span(
            "projection_remote_stage_gpu_ms",
            current_stream,
        )
        local_batched_fanout_plan = batch.additional_kwargs.get(
            _PAP_LOCAL_BATCHED_FANOUT_PLAN_KEY
        )
        batched_fanout_submit_us = 0.0
        if local_batched_fanout_plan is not None:
            if direct_qkv_send_buffer is None:
                raise RuntimeError(
                    "PAP local batched fan-out requires the direct QKV buffer"
                )
            batched_fanout_start_ns = (
                time.perf_counter_ns() if trace_offload_exec else 0
            )
            local_batched_fanout_plan.launch(direct_qkv_send_buffer)
            if trace_offload_exec:
                batched_fanout_submit_us = (
                    time.perf_counter_ns() - batched_fanout_start_ns
                ) / 1_000.0
        offload_exec_batches: list[
            tuple[
                str | None,
                str,
                PAPOffloadExecBatchDescriptor,
                tuple[int, ...],
                Any,
                PAPStepPlannedOffloadExecTransport | None,
                torch.Tensor | None,
            ]
        ] = []
        for step_group in step_groups:
            trace_group_prepare_start_ns = (
                time.perf_counter_ns() if trace_offload_exec else 0
            )
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
            step_planned_transport = _pap_cached_step_planned_transport(
                attention_endpoint
            )
            batch_descriptor = PAPOffloadExecBatchDescriptor(
                layer_name=self.layer_name,
                items=(),
                batch_id_suffix=step_group.batch_id_suffix,
                metadata_template=step_group.metadata_template,
            )
            _pap_bind_offload_exec_mailbox_peer(transport, attention_endpoint)
            use_fanout_stream = (
                len(step_groups) > 1 and step_planned_transport is not None
            )
            qkv_width = (
                self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
            )
            direct_qkv_batch: torch.Tensor | None = None
            direct_layout = False
            if local_batched_fanout_plan is None and direct_qkv_send_enabled:
                direct_qkv_batch, direct_layout = _pap_qkv_batch_for_indices(
                    direct_qkv_send_buffer,
                    req_indices,
                    index_tensor=route_index_tensor,
                )
            trace_group_submit_start_ns = (
                time.perf_counter_ns() if trace_offload_exec else 0
            )
            if local_batched_fanout_plan is not None:
                if trace_offload_exec:
                    trace_direct_qkv_groups += 1
            elif direct_qkv_batch is not None:
                if trace_offload_exec:
                    if direct_layout:
                        trace_direct_qkv_groups += 1
                    else:
                        trace_packed_qkv_groups += 1
                if int(direct_qkv_batch.shape[-1]) != qkv_width:
                    raise RuntimeError("PAP direct QKV batch width mismatch")
                if use_fanout_stream:
                    assert step_planned_transport is not None
                    step_planned_transport.send_qkv_batch_fanout(
                        batch_descriptor,
                        direct_qkv_batch,
                        remote_address=offload_exec_zmq_endpoint,
                    )
                else:
                    transport.send_qkv_batch_direct(
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
                if use_fanout_stream:
                    assert step_planned_transport is not None
                    step_planned_transport.send_qkv_batch_fanout(
                        batch_descriptor,
                        qkv_batch,
                        remote_address=offload_exec_zmq_endpoint,
                    )
                else:
                    transport.send_qkv_batch(
                        batch_descriptor,
                        qkv_batch,
                        remote_address=offload_exec_zmq_endpoint,
                    )
            if trace_offload_exec and local_batched_fanout_plan is None:
                trace_group_submit_done_ns = time.perf_counter_ns()
                trace_fanout_prepare_us.append(
                    (trace_group_submit_start_ns - trace_group_prepare_start_ns)
                    / 1_000.0
                )
                trace_fanout_submit_us.append(
                    (trace_group_submit_done_ns - trace_group_submit_start_ns) / 1_000.0
                )
                trace_fanout_submit_start_ns.append(trace_group_submit_start_ns)
            offload_exec_batches.append(
                (
                    attention_endpoint,
                    offload_exec_zmq_endpoint,
                    batch_descriptor,
                    req_indices,
                    transport,
                    step_planned_transport,
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
        prepared_output_streams: list[torch.cuda.Stream | None] = []
        prepared_output_traces: list[Any | None] = []
        trace_output_ready_events: list[torch.cuda.Event] = []
        output_width = self.num_heads * self.head_dim
        parallel_output_receives = len(offload_exec_batches) > 1
        output_inputs_ready: torch.cuda.Event | None = None
        if parallel_output_receives:
            get_copy_output_buffer()
            output_inputs_ready = torch.cuda.Event(
                enable_timing=trace_offload_exec,
            )
            output_inputs_ready.record(torch.cuda.current_stream(query.device))
        for (
            _attention_endpoint,
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            transport,
            step_planned_transport,
            _route_index_tensor,
        ) in offload_exec_batches:
            receive_stream = (
                step_planned_transport.output_receive_stream()
                if parallel_output_receives and step_planned_transport is not None
                else None
            )
            if receive_stream is None:
                output_message = transport.prepare_output_batch_message(
                    batch_descriptor,
                    shape=(len(req_indices), output_width),
                    dtype=query.dtype,
                    remote_address=offload_exec_zmq_endpoint,
                )
            else:
                with torch.cuda.stream(receive_stream):
                    assert output_inputs_ready is not None
                    receive_stream.wait_event(output_inputs_ready)
                    output_path_trace = begin_deferred_cuda_span(
                        f"pa_output_path_{len(prepared_output_messages)}_gpu_ms",
                        receive_stream,
                    )
                    output_message = transport.prepare_output_batch_message(
                        batch_descriptor,
                        shape=(len(req_indices), output_width),
                        dtype=query.dtype,
                        remote_address=offload_exec_zmq_endpoint,
                    )
                    if trace_offload_exec:
                        ready_event = torch.cuda.Event(enable_timing=True)
                        ready_event.record(receive_stream)
                        trace_output_ready_events.append(ready_event)
            if receive_stream is None:
                output_path_trace = None
            prepared_output_messages.append(output_message)
            prepared_output_streams.append(receive_stream)
            prepared_output_traces.append(output_path_trace)
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
            record_projection_trace_snapshot(
                PAPProjectionTraceState(
                    offload_exec_batches=offload_exec_batches,
                    step_groups=step_groups,
                    projection_timeline=projection_timeline,
                    pre_attn_compute_ms=pre_attn_compute_ms,
                    send_ms=trace_send_ms,
                    trigger_ms=trace_trigger_ms,
                    yield_ms=trace_yield_ms,
                    recv_ms=trace_recv_ms,
                    total_ms=trace_total_ms,
                    pre_attn_start_ns=pre_attn_start_ns,
                    pre_attn_done_ns=pre_attn_done_ns,
                    send_done_ns=trace_send_done_ns,
                    yield_start_ns=trace_yield_start_ns,
                    yield_end_ns=trace_yield_end_ns,
                    recv_done_ns=trace_recv_done_ns,
                    contiguous_route_groups=trace_contiguous_route_groups,
                    direct_qkv_groups=trace_direct_qkv_groups,
                    packed_qkv_groups=trace_packed_qkv_groups,
                    direct_output_rows=trace_direct_output_rows,
                    scattered_output_rows=trace_scattered_output_rows,
                    output_origin=trace_output_origin,
                    output_ready_events=trace_output_ready_events,
                    fanout_prepare_us=trace_fanout_prepare_us,
                    fanout_submit_us=trace_fanout_submit_us,
                    fanout_submit_start_ns=trace_fanout_submit_start_ns,
                    local_batched_fanout_plan=local_batched_fanout_plan,
                    batched_fanout_submit_us=batched_fanout_submit_us,
                )
            )

        output_scatter_events: list[torch.cuda.Event] = []
        for batch_index, (
            _attention_endpoint,
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            transport,
            _step_planned_transport,
            route_index_tensor,
        ) in enumerate(offload_exec_batches):
            output_message = prepared_output_messages[batch_index]
            output_receive_stream = prepared_output_streams[batch_index]
            output_path_trace = prepared_output_traces[batch_index]
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
                    end_deferred_cuda_span(remote_stage_trace)
                    return direct_output, release_messages
                if trace_offload_exec:
                    trace_scattered_output_rows += len(req_indices)
                if output_receive_stream is None:
                    _pap_scatter_attention_output_group(
                        get_copy_output_buffer(),
                        output_batch,
                        req_indices=req_indices,
                        index_tensor=route_index_tensor,
                    )
                else:
                    scatter_output = get_copy_output_buffer()
                    with torch.cuda.stream(output_receive_stream):
                        scatter_output.record_stream(output_receive_stream)
                        output_batch.record_stream(output_receive_stream)
                        if route_index_tensor is not None:
                            route_index_tensor.record_stream(output_receive_stream)
                        scatter_trace = begin_deferred_cuda_span(
                            "output_scatter_gpu_ms",
                            output_receive_stream,
                        )
                        try:
                            _pap_scatter_attention_output_group(
                                scatter_output,
                                output_batch,
                                req_indices=req_indices,
                                index_tensor=route_index_tensor,
                            )
                        finally:
                            end_deferred_cuda_span(scatter_trace)
                        output_message.release()
                        output_message = None
                        end_deferred_cuda_span(output_path_trace)
                        scatter_done = torch.cuda.Event()
                        scatter_done.record(output_receive_stream)
                        output_scatter_events.append(scatter_done)
            finally:
                if output_message is not None:
                    end_deferred_cuda_span(output_path_trace)
                if output_message is not None:
                    output_message.release()
        join_trace = (
            begin_deferred_cuda_span(
                "projection_join_wait_gpu_ms",
                current_stream,
            )
            if output_scatter_events
            else None
        )
        try:
            for scatter_done in output_scatter_events:
                current_stream.wait_event(scatter_done)
        finally:
            end_deferred_cuda_span(join_trace)
            end_deferred_cuda_span(remote_stage_trace)
        if trace_offload_exec and offload_exec_batches:
            trace_recv_done_ns = time.perf_counter_ns()
            trace_recv_ms = (time.perf_counter() - trace_recv_start) * 1000.0
            trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
            record_projection_trace()
        final_output = get_copy_output_buffer()
        return final_output.view(q.shape[0], self.num_heads * self.head_dim), []
