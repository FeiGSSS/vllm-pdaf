# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified PAP decode Attention dispatch."""

from __future__ import annotations

import torch

from vllm.pap.attention.kernels import (
    PAPPagedDecodeWorkspace,
    run_triton_paged_decode_attention,
)
from vllm.pap.attention.planning import PAPAttentionPlan
from vllm.pap.kv.metadata import PAPPagedFlashMetadata


def run_pap_decode_attention(
    *,
    attention_plan: PAPAttentionPlan | None,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: PAPPagedFlashMetadata,
    workspace: PAPPagedDecodeWorkspace,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Run the selected optimized backend or the Triton fallback."""
    if attention_plan is not None:
        return attention_plan.run_attention(query, key_cache, value_cache)
    return run_triton_paged_decode_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        metadata=metadata,
        workspace=workspace,
        scale=scale,
        block_size=block_size,
    )


__all__ = ["run_pap_decode_attention"]
