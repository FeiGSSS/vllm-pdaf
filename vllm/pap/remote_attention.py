# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small PAP remote-attention data-plane helpers."""

from __future__ import annotations

import base64
from typing import Any, Literal

import torch

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unsupported PAP tensor dtype: {name}") from exc


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().contiguous().cpu()
    raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": _dtype_name(cpu_tensor.dtype),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def deserialize_tensor(payload: dict[str, Any]) -> torch.Tensor:
    dtype = _dtype_from_name(str(payload["dtype"]))
    raw = base64.b64decode(str(payload["data"]).encode("ascii"))
    byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return byte_tensor.view(dtype).reshape(tuple(int(dim) for dim in payload["shape"]))


def serialize_attention_result(tensor: torch.Tensor) -> dict[str, Any]:
    return serialize_tensor(tensor)


def deserialize_attention_result(payload: dict[str, Any]) -> torch.Tensor:
    return deserialize_tensor(payload)


def gather_paged_kv(
    *,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    layout: Literal["NHD", "HND"],
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("PAP prototype supports one decode request per call")
    if kv_cache.shape[0] != 2:
        raise ValueError("expected KV cache first dimension to contain K and V")

    blocks = block_table[0].to(device="cpu", dtype=torch.long).tolist()
    if layout == "NHD":
        logical_nhd = True
    elif layout == "HND":
        # vLLM allocates HND physically, then binds a permuted logical view back
        # to Attention. In that common case shape[2] is still block_size and
        # shape[3] is num_kv_heads; direct physical-HND tensors have the reverse.
        logical_nhd = int(kv_cache.shape[3]) == int(num_kv_heads)
    else:
        raise ValueError(f"unsupported KV cache layout: {layout}")
    block_size = int(kv_cache.shape[2] if logical_nhd else kv_cache.shape[3])

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    remaining = int(seq_len)
    for block_id in blocks:
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        if logical_nhd:
            keys.append(kv_cache[0, block_id, :take, :num_kv_heads, :])
            values.append(kv_cache[1, block_id, :take, :num_kv_heads, :])
        else:
            keys.append(kv_cache[0, block_id, :num_kv_heads, :take, :].transpose(0, 1))
            values.append(
                kv_cache[1, block_id, :num_kv_heads, :take, :].transpose(0, 1)
            )
        remaining -= take

    if remaining > 0:
        raise ValueError("block table does not cover requested sequence length")
    if not keys:
        empty_shape = (0, num_kv_heads, int(kv_cache.shape[-1]))
        return (
            torch.empty(empty_shape, dtype=kv_cache.dtype, device=kv_cache.device),
            torch.empty(empty_shape, dtype=kv_cache.dtype, device=kv_cache.device),
        )
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


def compute_attention_output(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "expected query/key/value tensors with shape [tokens, heads, dim]"
        )
    if query.shape[0] != 1:
        raise ValueError("PAP prototype supports decode attention with one query token")
    if key.shape[1] == 0 or value.shape[1] == 0:
        raise ValueError("key/value tensors must have at least one KV head")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")

    compute_dtype = torch.float32
    q = query.to(compute_dtype)
    k = key.to(compute_dtype)
    v = value.to(compute_dtype)
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q, k) * float(scale)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("qhk,khd->qhd", probs, v)
    return out.to(query.dtype)
