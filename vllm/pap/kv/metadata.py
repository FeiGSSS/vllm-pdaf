# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FlashAttention metadata construction and caching."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.pap.kv.state import PAPUnifiedPagedKVState


@dataclass(frozen=True)
class PAPPagedFlashMetadata:
    """Batched metadata tensors consumed by paged FlashAttention."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seq_len: int


_UNIFIED_MD_CACHE: OrderedDict[tuple[Any, ...], PAPPagedFlashMetadata] = (
    OrderedDict()
)
_UNIFIED_MD_CU_SEQLENS_Q: dict[tuple[str, int], torch.Tensor] = {}
_UNIFIED_MD_CACHE_HITS = 0
_UNIFIED_MD_CACHE_MISSES = 0
_UNIFIED_MD_FAST_KEY_LOOKUPS = 0
_UNIFIED_MD_FAST_KEY_HITS = 0
_UNIFIED_MD_FULL_KEY_SCANS = 0
_UNIFIED_MD_BLOCK_IDS_SCANNED = 0
_UNIFIED_MD_CACHE_LOCK = Lock()


def reset_unified_paged_flash_metadata_cache() -> None:
    """Reset unified paged FlashAttention metadata cache and counters."""

    global _UNIFIED_MD_BLOCK_IDS_SCANNED
    global _UNIFIED_MD_CACHE_HITS, _UNIFIED_MD_CACHE_MISSES
    global _UNIFIED_MD_FAST_KEY_HITS, _UNIFIED_MD_FAST_KEY_LOOKUPS
    global _UNIFIED_MD_FULL_KEY_SCANS
    with _UNIFIED_MD_CACHE_LOCK:
        _UNIFIED_MD_CACHE.clear()
        _UNIFIED_MD_CU_SEQLENS_Q.clear()
        _UNIFIED_MD_CACHE_HITS = 0
        _UNIFIED_MD_CACHE_MISSES = 0
        _UNIFIED_MD_FAST_KEY_LOOKUPS = 0
        _UNIFIED_MD_FAST_KEY_HITS = 0
        _UNIFIED_MD_FULL_KEY_SCANS = 0
        _UNIFIED_MD_BLOCK_IDS_SCANNED = 0


def unified_paged_flash_metadata_cache_stats() -> dict[str, int]:
    """Return cache counters for tests and trace-time diagnostics."""

    with _UNIFIED_MD_CACHE_LOCK:
        return {
            "hits": int(_UNIFIED_MD_CACHE_HITS),
            "misses": int(_UNIFIED_MD_CACHE_MISSES),
            "entries": len(_UNIFIED_MD_CACHE),
            "fast_key_lookups": int(_UNIFIED_MD_FAST_KEY_LOOKUPS),
            "fast_key_hits": int(_UNIFIED_MD_FAST_KEY_HITS),
            "full_key_scans": int(_UNIFIED_MD_FULL_KEY_SCANS),
            "block_ids_scanned": int(_UNIFIED_MD_BLOCK_IDS_SCANNED),
        }


def _unified_paged_flash_metadata_cache_limit() -> int:
    return int(os.environ.get("PAP_UNIFIED_MD_CACHE_LIMIT", "256"))


def _unified_paged_flash_metadata_fast_key(
    *,
    states: Sequence[PAPUnifiedPagedKVState],
    device: torch.device,
) -> tuple[Any, ...] | None:
    rows: list[tuple[int, int]] = []
    for state in states:
        topology_id = int(state.slot_topology_id)
        if topology_id <= 0:
            return None
        rows.append((topology_id, int(state.seq_len)))
    return ("topology", str(torch.device(device)), tuple(rows))


def _coerce_block_id(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _cached_decode_cu_seqlens_q(
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    key = (str(torch.device(device)), int(batch_size))
    with _UNIFIED_MD_CACHE_LOCK:
        cached = _UNIFIED_MD_CU_SEQLENS_Q.get(key)
        if cached is not None:
            return cached
    value = torch.arange(
        0,
        int(batch_size) + 1,
        dtype=torch.int32,
        device=device,
    )
    with _UNIFIED_MD_CACHE_LOCK:
        cached = _UNIFIED_MD_CU_SEQLENS_Q.get(key)
        if cached is not None:
            return cached
        _UNIFIED_MD_CU_SEQLENS_Q[key] = value
        return value


def _lookup_unified_paged_flash_metadata(
    key: tuple[Any, ...],
    *,
    fast_key: bool,
) -> PAPPagedFlashMetadata | None:
    global _UNIFIED_MD_CACHE_HITS, _UNIFIED_MD_FAST_KEY_HITS
    global _UNIFIED_MD_FAST_KEY_LOOKUPS
    with _UNIFIED_MD_CACHE_LOCK:
        if fast_key:
            _UNIFIED_MD_FAST_KEY_LOOKUPS += 1
        cached = _UNIFIED_MD_CACHE.get(key)
        if cached is not None:
            _UNIFIED_MD_CACHE_HITS += 1
            if fast_key:
                _UNIFIED_MD_FAST_KEY_HITS += 1
            _UNIFIED_MD_CACHE.move_to_end(key)
        return cached


def _record_unified_paged_flash_metadata_scan(block_ids: int) -> None:
    global _UNIFIED_MD_BLOCK_IDS_SCANNED, _UNIFIED_MD_FULL_KEY_SCANS
    with _UNIFIED_MD_CACHE_LOCK:
        _UNIFIED_MD_FULL_KEY_SCANS += 1
        _UNIFIED_MD_BLOCK_IDS_SCANNED += int(block_ids)


def _store_unified_paged_flash_metadata(
    key: tuple[Any, ...],
    metadata: PAPPagedFlashMetadata,
) -> PAPPagedFlashMetadata:
    global _UNIFIED_MD_CACHE_HITS, _UNIFIED_MD_CACHE_MISSES
    limit = _unified_paged_flash_metadata_cache_limit()
    with _UNIFIED_MD_CACHE_LOCK:
        cached = _UNIFIED_MD_CACHE.get(key)
        if cached is not None:
            _UNIFIED_MD_CACHE_HITS += 1
            _UNIFIED_MD_CACHE.move_to_end(key)
            return cached
        _UNIFIED_MD_CACHE_MISSES += 1
        if limit <= 0:
            return metadata
        _UNIFIED_MD_CACHE[key] = metadata
        _UNIFIED_MD_CACHE.move_to_end(key)
        while len(_UNIFIED_MD_CACHE) > limit:
            _UNIFIED_MD_CACHE.popitem(last=False)
        return metadata


def build_unified_paged_flash_metadata(
    *,
    states: list[PAPUnifiedPagedKVState],
    device: torch.device,
) -> PAPPagedFlashMetadata:
    """Build or reuse FA metadata for a decode batch signature."""

    batch_size = len(states)
    if batch_size <= 0:
        raise ValueError(
            "unified paged FlashAttention metadata requires at least one state"
        )
    cache_key = _unified_paged_flash_metadata_fast_key(
        states=states,
        device=device,
    )
    if cache_key is not None:
        cached = _lookup_unified_paged_flash_metadata(
            cache_key,
            fast_key=True,
        )
        if cached is not None:
            return cached

    block_rows: list[tuple[int, ...]] = []
    seq_lens_list: list[int] = []
    max_blocks = 0
    max_seq_len = 0
    for state in states:
        block_row = tuple(_coerce_block_id(raw) for raw in state.block_ids)
        if not block_row:
            raise ValueError("unified state has no blocks")
        seq_len = int(state.seq_len)
        block_rows.append(block_row)
        seq_lens_list.append(seq_len)
        max_blocks = max(max_blocks, len(block_row))
        max_seq_len = max(max_seq_len, seq_len)
    if max_blocks <= 0:
        raise ValueError(
            "unified paged FlashAttention metadata requires non-empty blocks"
        )
    _record_unified_paged_flash_metadata_scan(
        sum(len(block_row) for block_row in block_rows)
    )
    if cache_key is None:
        cache_key = (
            str(torch.device(device)),
            tuple(block_rows),
            tuple(seq_lens_list),
        )
        cached = _lookup_unified_paged_flash_metadata(
            cache_key,
            fast_key=False,
        )
        if cached is not None:
            return cached

    padded_block_rows = [
        block_row + (block_row[-1],) * (max_blocks - len(block_row))
        for block_row in block_rows
    ]
    block_table = torch.tensor(
        padded_block_rows,
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(
        seq_lens_list,
        dtype=torch.int32,
        device=device,
    )
    metadata = PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=_cached_decode_cu_seqlens_q(
            batch_size=batch_size,
            device=device,
        ),
        max_seq_len=max_seq_len,
    )
    return _store_unified_paged_flash_metadata(cache_key, metadata)


__all__ = [
    "PAPPagedFlashMetadata",
    "build_unified_paged_flash_metadata",
    "reset_unified_paged_flash_metadata_cache",
    "unified_paged_flash_metadata_cache_stats",
]
