# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent PAP Attention execution binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.pap.model.projection import PAPProjectionAttentionAdapter
from vllm.pap.model.step_graph import register_projection_step_graph_adapter


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
        self.adapter.begin_step()
        if not self.adapter.should_execute():
            output.zero_()
            return
        remote_output, release_messages = self.adapter.execute(query, key, value)
        try:
            output.copy_(remote_output.reshape_as(output))
        finally:
            for message in release_messages:
                message.release()


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


__all__ = [
    "PAPProjectionAttentionExecution",
    "bind_projection_attention_execution",
    "create_projection_attention_execution",
]
