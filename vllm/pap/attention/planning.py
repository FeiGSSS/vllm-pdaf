# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interfaces for PAP decode Attention planning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch


class PAPAttentionPlan(Protocol):
    """One graph-stable optimized Attention execution plan."""

    backend_name: str
    reused_kv_tokens: int

    @property
    def graph_key(self) -> tuple[Any, ...]: ...

    @property
    def bound_tensors(self) -> tuple[torch.Tensor, ...]: ...

    def run_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor: ...


class PAPAttentionSelector(Protocol):
    """Select and prepare the Attention backend for one decode step."""

    def plan(
        self,
        *,
        step_signature: tuple[Any, ...],
        request_ids: Sequence[str],
        topology_ids: Sequence[int],
        states: Sequence[Any],
        seq_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PAPAttentionPlan | None: ...


__all__ = ["PAPAttentionPlan", "PAPAttentionSelector"]
