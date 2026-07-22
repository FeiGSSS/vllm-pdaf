# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP-owned paged decode-attention kernel integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.pap.kv.metadata import PAPPagedFlashMetadata

PAP_TRITON_DECODE_NUM_SPLITS = 4


@dataclass(frozen=True)
class PAPPagedDecodeWorkspace:
    """Step-owned scratch reused by every Attention layer."""

    partial: torch.Tensor
    lse: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    batch_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    num_splits: int = PAP_TRITON_DECODE_NUM_SPLITS

    def validate(self, query: torch.Tensor) -> None:
        signature = (
            int(query.shape[0]),
            int(query.shape[1]),
            int(query.shape[2]),
            query.dtype,
            query.device,
        )
        expected = (
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.dtype,
            self.device,
        )
        if signature != expected:
            raise RuntimeError(
                "PAP paged decode workspace does not match the query shape"
            )


def build_paged_decode_workspace(
    query: torch.Tensor,
) -> PAPPagedDecodeWorkspace:
    """Allocate the fixed split-4 scratch once for a decode step."""

    if query.ndim != 3:
        raise ValueError("PAP paged decode query must be rank 3")
    batch_size, num_heads, head_dim = map(int, query.shape)
    return PAPPagedDecodeWorkspace(
        partial=torch.empty(
            (
                batch_size,
                num_heads,
                PAP_TRITON_DECODE_NUM_SPLITS,
                head_dim + 1,
            ),
            dtype=torch.float32,
            device=query.device,
        ),
        lse=torch.empty(
            (batch_size, num_heads),
            dtype=torch.float32,
            device=query.device,
        ),
        k_scale=torch.ones((), dtype=torch.float32, device=query.device),
        v_scale=torch.ones((), dtype=torch.float32, device=query.device),
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=query.dtype,
        device=query.device,
    )


def run_paged_decode_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: PAPPagedFlashMetadata,
    workspace: PAPPagedDecodeWorkspace,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Run the current Triton paged-decode kernel without a layer fallback."""

    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd,
    )

    workspace.validate(query)
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError("PAP paged decode KV cache must be rank 4")
    if key_cache.shape != value_cache.shape:
        raise RuntimeError("PAP paged decode K/V cache shapes differ")
    if int(key_cache.shape[1]) != int(block_size):
        raise RuntimeError("PAP paged decode block size does not match KV cache")
    if int(query.shape[1]) % int(key_cache.shape[-2]) != 0:
        raise RuntimeError("PAP paged decode GQA head counts are incompatible")

    output = torch.empty_like(query)
    decode_attention_fwd(
        query,
        key_cache,
        value_cache,
        output,
        workspace.lse,
        metadata.block_table,
        metadata.seq_lens,
        workspace.partial,
        workspace.num_splits,
        float(scale),
        page_size=int(block_size),
        k_scale=workspace.k_scale,
        v_scale=workspace.v_scale,
    )
    return output


__all__ = [
    "PAPPagedDecodeWorkspace",
    "build_paged_decode_workspace",
    "run_paged_decode_attention",
]
