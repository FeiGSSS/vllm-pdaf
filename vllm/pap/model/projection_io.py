# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side QKV layout and Attention output placement."""

from __future__ import annotations

import math
from typing import Any

import torch


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
