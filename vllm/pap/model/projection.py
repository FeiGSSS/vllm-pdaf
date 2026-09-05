# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side PAP Attention adapter for NVSHMEM whole-step Graphs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.pap.config import read_env_bool
from vllm.pap.mode import pap_request_ids_are_routable
from vllm.pap.model.context import PAPModelForwardBatch
from vllm.pap.model.step_graph import (
    PAPProjectionStepGraphContext,
    current_projection_step_graph_context,
    register_projection_step_graph_adapter,
)


@dataclass(slots=True)
class PAPProjectionAttentionExecution:
    """Execute one normalized Q/K/V Attention call through PAP Projection."""

    adapter: PAPProjectionAttentionAdapter

    def __call__(
        self,
        _attention: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        _kv_cache: torch.Tensor,
        _attn_metadata: Any,
        **_kwargs: Any,
    ) -> None:
        if not self.adapter.should_execute():
            output.zero_()
            return
        remote_output = self.adapter.execute(query, key, value)
        output.copy_(remote_output.reshape_as(output))


def bind_projection_attention_execution(
    static_forward_context: dict[str, Any],
) -> int:
    """Bind PAP Projection to generic vLLM Attention layers."""
    attention_layers = tuple(
        layer
        for layer in static_forward_context.values()
        if isinstance(layer, Attention)
    )
    layer_count = len(attention_layers)
    for attention in attention_layers:
        attention.set_execution_override(
            create_projection_attention_execution(attention, layer_count)
        )
    return layer_count


def create_projection_attention_execution(
    attention: Attention,
    layer_count: int,
) -> PAPProjectionAttentionExecution:
    """Create and register one Projection execution implementation."""
    scale = getattr(attention.impl, "scale", None)
    if scale is None:
        raise RuntimeError(
            f"PAP Attention backend has no scale: {attention.layer_name}"
        )
    adapter = PAPProjectionAttentionAdapter(
        layer_name=attention.layer_name,
        num_heads=attention.num_heads,
        num_kv_heads=attention.num_kv_heads,
        head_dim=attention.head_size,
        scaling=float(scale),
        num_hidden_layers=layer_count,
    )
    register_projection_step_graph_adapter(attention.layer_name, adapter)
    return PAPProjectionAttentionExecution(adapter)


logger = init_logger(__name__)


@dataclass(slots=True)
class PAPProjectionAttentionAdapter:
    """Execute one Projection layer inside the whole-step CUDA Graph."""

    layer_name: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scaling: float
    num_hidden_layers: int = 0
    _debug_decision: bool = field(init=False, repr=False)
    _qkv_width: int = field(init=False, repr=False)
    _output_width: int = field(init=False, repr=False)
    _prepared_batch: PAPModelForwardBatch | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._debug_decision = read_env_bool(os.environ, "PAP_DEBUG_DECISION")
        self._qkv_width = (
            self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
        )
        self._output_width = self.num_heads * self.head_dim

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
            return reject("request_ids shorter than num_reqs")
        if len(batch.num_scheduled_tokens) < batch.num_reqs:
            return reject("num_scheduled_tokens shorter than num_reqs")
        if not pap_request_ids_are_routable(batch.request_ids, batch.num_reqs):
            return reject("scheduled batch contains a non-PAP request")
        if any(tokens != 1 for tokens in batch.num_scheduled_tokens[: batch.num_reqs]):
            return reject("PAP Attention only supports decode tokens")
        installed = set(
            batch.additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        active_request_ids = batch.request_ids[: batch.num_reqs]
        if not all(request_id in installed for request_id in active_request_ids):
            return reject("Attention KV is not ready")
        self._prepared_batch = batch
        return True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Append graph-safe NVSHMEM dispatch/gather operations for one layer."""
        batch = self._prepared_batch or PAPModelForwardBatch.current(self.layer_name)
        if batch is None:
            raise RuntimeError("PAP attention requires forward context")
        context = current_projection_step_graph_context()
        if context is None:
            raise RuntimeError(
                "PAP Projection Attention requires the whole-step CUDA Graph path"
            )
        query = q.view(-1, self.num_heads, self.head_dim)
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view(-1, self.num_kv_heads, self.head_dim)
        return self._execute_step_graph(
            query=query,
            key=key,
            value=value,
            context=context,
        )

    def _execute_step_graph(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        context: PAPProjectionStepGraphContext,
    ) -> torch.Tensor:
        output = torch.empty(
            (query.shape[0], self._output_width),
            dtype=query.dtype,
            device=query.device,
        )
        stream = torch.cuda.current_stream(query.device)
        layer_index = context.layer_index(self.layer_name)
        qkv_batch = torch.cat(
            (
                query.reshape(query.shape[0], -1),
                key.reshape(key.shape[0], -1),
                value.reshape(value.shape[0], -1),
            ),
            dim=-1,
        )
        routed = context.routed
        routed.controller.graph_dispatch_routed_qkv(
            qkv_batch,
            packed=routed.packed_qkv,
            route_indices=routed.indices,
            route_counts=routed.counts,
            peer_ranks=routed.peer_ranks,
            layer_index=layer_index,
            layer_count=context.layer_count,
            stream=stream,
        )
        routed.controller.graph_gather_routed_output(
            output,
            route_indices=routed.indices,
            route_counts=routed.counts,
            peer_ranks=routed.peer_ranks,
            layer_index=layer_index,
            layer_count=context.layer_count,
            stream=stream,
        )
        return output


__all__ = [
    "PAPProjectionAttentionAdapter",
    "PAPProjectionAttentionExecution",
    "bind_projection_attention_execution",
    "create_projection_attention_execution",
]
