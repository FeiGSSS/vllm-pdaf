# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache layout adapters at the PAP/vLLM boundary."""

from __future__ import annotations

import torch


def split_paged_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FlashAttention K/V views for supported vLLM cache layouts."""
    if kv_cache.ndim == 4 and kv_cache.shape[-1] == 2 * head_dim:
        return kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
        key_cache, value_cache = kv_cache.unbind(1)
        return key_cache, value_cache
    raise ValueError(f"unsupported PAP KV cache shape: {tuple(kv_cache.shape)}")
