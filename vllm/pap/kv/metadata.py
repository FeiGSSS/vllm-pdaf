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
    from vllm.pap.kv.registry import PAPUnifiedPagedKVState


@dataclass(frozen=True)
class PAPPagedFlashMetadata:
    """Batched metadata tensors consumed by paged FlashAttention."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seq_len: int


class PAPPagedBlockTableBuffer:
    """Per-peer mutable backing storage for paged decode block tables."""

    def __init__(self, *, row_capacity: int = 256) -> None:
        if row_capacity <= 0:
            raise ValueError("PAP block table row capacity must be positive")
        self.row_capacity = int(row_capacity)
        self._host: torch.Tensor | None = None
        self._target: torch.Tensor | None = None
        self._identity: tuple[Any, ...] | None = None
        self._shape: tuple[int, int] | None = None
        self._lock = Lock()

    def lookup(
        self,
        identity: tuple[Any, ...],
    ) -> torch.Tensor | None:
        """Return the current view when its topology identity is unchanged."""
        with self._lock:
            if (
                self._identity != identity
                or self._shape is None
                or self._target is None
            ):
                return None
            rows, columns = self._shape
            return self._target[:rows, :columns]

    def update(
        self,
        *,
        identity: tuple[Any, ...],
        rows: tuple[tuple[int, ...], ...],
        device: torch.device,
        column_capacity: int,
    ) -> torch.Tensor:
        """Asynchronously update the active view without device allocation."""
        row_count = len(rows)
        column_count = len(rows[0]) if rows else 0
        if (
            row_count <= 0
            or column_count <= 0
            or any(len(row) != column_count for row in rows)
        ):
            raise ValueError("PAP block table buffer requires a non-empty matrix")
        if row_count > self.row_capacity:
            raise ValueError("PAP block table row capacity exceeded")
        if column_count > int(column_capacity):
            raise ValueError("PAP block table column capacity exceeded")

        normalized_device = torch.device(device)
        with self._lock:
            capacity_shape = (self.row_capacity, int(column_capacity))
            if (
                self._target is None
                or tuple(self._target.shape) != capacity_shape
                or self._target.device.type != normalized_device.type
                or (
                    normalized_device.index is not None
                    and self._target.device.index != normalized_device.index
                )
            ):
                self._host = torch.empty(
                    capacity_shape,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=normalized_device.type == "cuda",
                )
                self._target = torch.empty(
                    capacity_shape,
                    dtype=torch.int32,
                    device=normalized_device,
                )
            assert self._host is not None
            source = torch.tensor(rows, dtype=torch.int32)
            host_view = self._host[:row_count, :column_count]
            target_view = self._target[:row_count, :column_count]
            host_view.copy_(source)
            target_view.copy_(
                host_view,
                non_blocking=normalized_device.type == "cuda",
            )
            self._identity = identity
            self._shape = (row_count, column_count)
            return target_view


_UNIFIED_STATIC_BLOCK_TABLE_CACHE: OrderedDict[tuple[Any, ...], torch.Tensor] = (
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
        _UNIFIED_STATIC_BLOCK_TABLE_CACHE.clear()
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
            "entries": len(_UNIFIED_STATIC_BLOCK_TABLE_CACHE),
            "fast_key_lookups": int(_UNIFIED_MD_FAST_KEY_LOOKUPS),
            "fast_key_hits": int(_UNIFIED_MD_FAST_KEY_HITS),
            "full_key_scans": int(_UNIFIED_MD_FULL_KEY_SCANS),
            "block_ids_scanned": int(_UNIFIED_MD_BLOCK_IDS_SCANNED),
        }


def _unified_paged_flash_metadata_cache_limit() -> int:
    return int(os.environ.get("PAP_UNIFIED_MD_CACHE_LIMIT", "256"))


def _unified_static_block_table_fast_key(
    *,
    states: Sequence[PAPUnifiedPagedKVState],
    device: torch.device,
) -> tuple[Any, ...] | None:
    topology_ids = tuple(int(state.slot_topology_id) for state in states)
    if any(topology_id <= 0 for topology_id in topology_ids):
        return None
    return ("static_topology", str(torch.device(device)), topology_ids)


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


def _record_unified_paged_flash_metadata_scan(block_ids: int) -> None:
    global _UNIFIED_MD_BLOCK_IDS_SCANNED, _UNIFIED_MD_FULL_KEY_SCANS
    with _UNIFIED_MD_CACHE_LOCK:
        _UNIFIED_MD_FULL_KEY_SCANS += 1
        _UNIFIED_MD_BLOCK_IDS_SCANNED += int(block_ids)


def _lookup_unified_static_block_table(
    key: tuple[Any, ...],
    *,
    fast_key: bool,
) -> torch.Tensor | None:
    global _UNIFIED_MD_CACHE_HITS, _UNIFIED_MD_FAST_KEY_HITS
    global _UNIFIED_MD_FAST_KEY_LOOKUPS
    with _UNIFIED_MD_CACHE_LOCK:
        if fast_key:
            _UNIFIED_MD_FAST_KEY_LOOKUPS += 1
        cached = _UNIFIED_STATIC_BLOCK_TABLE_CACHE.get(key)
        if cached is not None:
            _UNIFIED_MD_CACHE_HITS += 1
            if fast_key:
                _UNIFIED_MD_FAST_KEY_HITS += 1
            _UNIFIED_STATIC_BLOCK_TABLE_CACHE.move_to_end(key)
        return cached


def _store_unified_static_block_table(
    key: tuple[Any, ...],
    block_table: torch.Tensor,
) -> torch.Tensor:
    global _UNIFIED_MD_CACHE_HITS, _UNIFIED_MD_CACHE_MISSES
    limit = _unified_paged_flash_metadata_cache_limit()
    with _UNIFIED_MD_CACHE_LOCK:
        cached = _UNIFIED_STATIC_BLOCK_TABLE_CACHE.get(key)
        if cached is not None:
            _UNIFIED_MD_CACHE_HITS += 1
            _UNIFIED_STATIC_BLOCK_TABLE_CACHE.move_to_end(key)
            return cached
        _UNIFIED_MD_CACHE_MISSES += 1
        if limit <= 0:
            return block_table
        _UNIFIED_STATIC_BLOCK_TABLE_CACHE[key] = block_table
        _UNIFIED_STATIC_BLOCK_TABLE_CACHE.move_to_end(key)
        while len(_UNIFIED_STATIC_BLOCK_TABLE_CACHE) > limit:
            _UNIFIED_STATIC_BLOCK_TABLE_CACHE.popitem(last=False)
        return block_table


def build_unified_paged_flash_step_metadata(
    *,
    states: Sequence[PAPUnifiedPagedKVState],
    seq_lens: Sequence[int],
    device: torch.device,
    seq_lens_tensor: torch.Tensor | None = None,
    block_table_buffer: PAPPagedBlockTableBuffer | None = None,
) -> PAPPagedFlashMetadata:
    """Build dynamic decode metadata over a reusable static block table."""

    batch_size = len(states)
    normalized_seq_lens = tuple(int(seq_len) for seq_len in seq_lens)
    if batch_size <= 0:
        raise ValueError(
            "unified paged FlashAttention metadata requires at least one state"
        )
    if len(normalized_seq_lens) != batch_size:
        raise ValueError("PAP Attention step metadata row count mismatch")

    cache_key = _unified_static_block_table_fast_key(
        states=states,
        device=device,
    )
    block_table = None
    if cache_key is not None and block_table_buffer is not None:
        block_table = block_table_buffer.lookup(cache_key)
    elif cache_key is not None:
        block_table = _lookup_unified_static_block_table(
            cache_key,
            fast_key=True,
        )

    if block_table is None:
        block_rows = [
            tuple(_coerce_block_id(raw) for raw in state.block_ids) for state in states
        ]
        if any(not block_row for block_row in block_rows):
            raise ValueError("unified state has no blocks")
        _record_unified_paged_flash_metadata_scan(
            sum(len(block_row) for block_row in block_rows)
        )
        if cache_key is None:
            cache_key = (
                "static_blocks",
                str(torch.device(device)),
                tuple(block_rows),
            )
            block_table = _lookup_unified_static_block_table(
                cache_key,
                fast_key=False,
            )
        if block_table is None:
            max_blocks = max(len(block_row) for block_row in block_rows)
            padded_rows = [
                block_row + (block_row[-1],) * (max_blocks - len(block_row))
                for block_row in block_rows
            ]
            if block_table_buffer is not None:
                block_table = block_table_buffer.update(
                    identity=cache_key,
                    rows=tuple(padded_rows),
                    device=device,
                    column_capacity=int(states[0].kv_cache.shape[0]),
                )
            else:
                block_table = torch.tensor(
                    padded_rows,
                    dtype=torch.int32,
                    device=device,
                )
                block_table = _store_unified_static_block_table(
                    cache_key,
                    block_table,
                )

    if seq_lens_tensor is None:
        seq_lens_tensor = torch.tensor(
            normalized_seq_lens,
            dtype=torch.int32,
            device=device,
        )
    else:
        expected_device = torch.device(device)
        actual_device = seq_lens_tensor.device
        device_mismatch = actual_device.type != expected_device.type or (
            expected_device.index is not None
            and actual_device.index != expected_device.index
        )
        if (
            tuple(seq_lens_tensor.shape) != (batch_size,)
            or seq_lens_tensor.dtype != torch.int32
            or device_mismatch
        ):
            raise ValueError("PAP Attention step seq_lens tensor is incompatible")
    return PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens_tensor,
        cu_seqlens_q=_cached_decode_cu_seqlens_q(
            batch_size=batch_size,
            device=device,
        ),
        max_seq_len=max(normalized_seq_lens),
    )


__all__ = [
    "PAPPagedBlockTableBuffer",
    "PAPPagedFlashMetadata",
    "build_unified_paged_flash_step_metadata",
    "reset_unified_paged_flash_metadata_cache",
    "unified_paged_flash_metadata_cache_stats",
]
