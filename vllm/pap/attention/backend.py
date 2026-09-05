# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend contracts, PAT/Triton selection, and unified execution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import torch

from vllm.pap.attention.pat_backend import PAPPATPlan, PAPPATPlanner
from vllm.pap.attention.triton_backend import (
    PAPPagedDecodeWorkspace,
    run_triton_paged_decode_attention,
)
from vllm.pap.kv.metadata import PAPPagedFlashMetadata


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


class PAPPATOrTritonSelector:
    """Reuse the previous decision and select PAT for any physical KV reuse."""

    def __init__(self, pat_planner: PAPPATPlanner | None = None) -> None:
        self.pat = pat_planner or PAPPATPlanner()
        self._has_previous_step = False
        self._last_structure_signature: tuple[Any, ...] | None = None
        self._last_metadata_signature: tuple[Any, ...] | None = None
        self._last_reused_kv_tokens = 0
        self._last_plan: PAPPATPlan | None = None
        self.metadata_reuses = 0
        self.incremental_metadata_reuses = 0
        self.prefix_rechecks = 0
        self.pat_rebuilds = 0
        self.triton_selections = 0

    @classmethod
    def create_if_available(
        cls,
    ) -> tuple[PAPPATOrTritonSelector | None, str | None]:
        unavailable_reason = PAPPATPlanner.unavailable_reason()
        if unavailable_reason is not None:
            return None, unavailable_reason
        return cls(), None

    def stats(self) -> dict[str, int]:
        return {
            "attention_kernel_metadata_reuses": self.metadata_reuses,
            "attention_kernel_incremental_metadata_reuses": (
                self.incremental_metadata_reuses
            ),
            "attention_kernel_prefix_rechecks": self.prefix_rechecks,
            "attention_kernel_pat_rebuilds": self.pat_rebuilds,
            "attention_kernel_triton_selections": self.triton_selections,
        }

    @staticmethod
    def _storage_id(state: Any) -> int:
        return int(state.kv_cache.untyped_storage().data_ptr())

    @classmethod
    def _structure_signature(
        cls,
        *,
        request_ids: Sequence[str],
        topology_ids: Sequence[int],
        states: Sequence[Any],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Any, ...]:
        return (
            tuple(map(str, request_ids)),
            tuple(map(int, topology_ids)),
            tuple(cls._storage_id(state) for state in states),
            tuple(int(state.block_size) for state in states),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            dtype,
            device,
        )

    @classmethod
    def _reused_kv_tokens(
        cls,
        states: Sequence[Any],
        seq_lens: Sequence[int],
    ) -> int:
        logical_blocks = 0
        unique_blocks_by_storage: dict[int, set[int]] = {}
        block_size = int(states[0].block_size) if states else 0
        for state, raw_seq_len in zip(states, seq_lens, strict=True):
            seq_len = int(raw_seq_len)
            if int(state.block_size) != block_size:
                raise ValueError("PAP Attention selector received mixed block sizes")
            used_blocks = math.ceil(seq_len / block_size)
            storage_id = cls._storage_id(state)
            logical_blocks += used_blocks
            unique_blocks_by_storage.setdefault(storage_id, set()).update(
                state.block_ids[:used_blocks]
            )
        unique_blocks = sum(map(len, unique_blocks_by_storage.values()))
        return (logical_blocks - unique_blocks) * block_size

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
    ) -> PAPPATPlan | None:
        batch_size = len(states)
        if not (len(request_ids) == len(topology_ids) == len(seq_lens) == batch_size):
            raise ValueError("PAP Attention selector batch metadata length mismatch")
        structure_signature = self._structure_signature(
            request_ids=request_ids,
            topology_ids=topology_ids,
            states=states,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        metadata_signature = (
            step_signature,
            structure_signature,
            tuple(map(int, seq_lens)),
        )
        if (
            self._has_previous_step
            and metadata_signature == self._last_metadata_signature
        ):
            self.metadata_reuses += 1
            return self._last_plan
        if (
            self._has_previous_step
            and structure_signature == self._last_structure_signature
        ):
            if self._last_plan is None:
                self.incremental_metadata_reuses += 1
                self._last_metadata_signature = metadata_signature
                return None
            if self._last_plan.update_decode_state(states, seq_lens):
                self.incremental_metadata_reuses += 1
                self._last_plan.reused_kv_tokens = self._last_reused_kv_tokens
                self._last_metadata_signature = metadata_signature
                return self._last_plan

        self.prefix_rechecks += 1
        reused_kv_tokens = self._reused_kv_tokens(states, seq_lens)

        plan = None
        if reused_kv_tokens > 0:
            plan = self.pat.plan(
                states=states,
                seq_lens=seq_lens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scale=scale,
                reused_kv_tokens=reused_kv_tokens,
                dtype=dtype,
                device=device,
            )
        if plan is not None:
            self.pat_rebuilds += 1
        else:
            self.triton_selections += 1

        self._has_previous_step = True
        self._last_structure_signature = structure_signature
        self._last_metadata_signature = metadata_signature
        self._last_reused_kv_tokens = reused_kv_tokens
        self._last_plan = plan
        return plan


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


__all__ = [
    "PAPAttentionPlan",
    "PAPAttentionSelector",
    "PAPPATOrTritonSelector",
    "run_pap_decode_attention",
]
