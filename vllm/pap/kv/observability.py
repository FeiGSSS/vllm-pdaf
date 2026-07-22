# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional PAP KV profiling and trace helpers."""

from __future__ import annotations

import logging
import os

import torch

from vllm.pap.kv.models import PAPUnifiedPagedKVState

logger = logging.getLogger("pap_attention")

_KV_LOCALITY_PROFILE_SEEN: set[tuple[str, str]] = set()


def pap_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def pap_kv_lease_profile_enabled() -> bool:
    return pap_env_flag("PAP_KV_LEASE_PROFILE", False)


def pap_kv_locality_profile_enabled() -> bool:
    return pap_env_flag("PAP_KV_LOCALITY_PROFILE", False)


def block_locality_stats(
    *,
    block_ids: tuple[int, ...],
    seq_len: int,
    block_size: int,
) -> dict[str, float | int | list[int]]:
    """Summarize physical block locality for diagnostics."""
    live_blocks = min(
        len(block_ids),
        max(0, (int(seq_len) + int(block_size) - 1) // int(block_size)),
    )
    live = [int(block_id) for block_id in block_ids[:live_blocks]]
    if not live:
        return {
            "seq_len": int(seq_len),
            "live_blocks": 0,
            "total_blocks": len(block_ids),
            "reserved_blocks": len(block_ids),
            "span": 0,
            "density": 0.0,
            "contiguous_pair_frac": 0.0,
            "mean_abs_delta": 0.0,
            "max_abs_delta": 0,
            "runs": 0,
            "first_blocks": [],
            "first_deltas": [],
        }
    deltas = [live[index + 1] - live[index] for index in range(len(live) - 1)]
    span = max(live) - min(live) + 1
    contiguous_pairs = sum(1 for delta in deltas if delta == 1)
    abs_deltas = [abs(delta) for delta in deltas]
    return {
        "seq_len": int(seq_len),
        "live_blocks": live_blocks,
        "total_blocks": len(block_ids),
        "reserved_blocks": len(block_ids) - live_blocks,
        "span": span,
        "density": float(live_blocks) / float(span) if span > 0 else 0.0,
        "contiguous_pair_frac": (
            float(contiguous_pairs) / float(len(deltas)) if deltas else 1.0
        ),
        "mean_abs_delta": (
            float(sum(abs_deltas)) / float(len(abs_deltas)) if abs_deltas else 0.0
        ),
        "max_abs_delta": max(abs_deltas) if abs_deltas else 0,
        "runs": 1 + sum(1 for delta in deltas if delta != 1),
        "first_blocks": live[:16],
        "first_deltas": deltas[:15],
    }


def log_kv_locality_profile(
    *,
    mode: str,
    layer_name: str,
    states: list[PAPUnifiedPagedKVState],
    kv_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layout: str,
) -> None:
    """Emit one locality profile per mode and layer when enabled."""
    if not pap_kv_locality_profile_enabled():
        return
    min_batch = int(os.environ.get("PAP_KV_LOCALITY_PROFILE_MIN_BATCH", "1"))
    if len(states) < min_batch:
        return
    profile_key = (str(mode), str(layer_name))
    if profile_key in _KV_LOCALITY_PROFILE_SEEN:
        return
    _KV_LOCALITY_PROFILE_SEEN.add(profile_key)

    rows = [
        block_locality_stats(
            block_ids=tuple(int(block_id) for block_id in state.block_ids),
            seq_len=int(state.seq_len),
            block_size=int(state.block_size),
        )
        for state in states
    ]
    first = rows[0]

    def avg(name: str) -> float:
        values = [float(row[name]) for row in rows]  # type: ignore[arg-type]
        return sum(values) / float(len(values)) if values else 0.0

    seq_lens = [int(row["seq_len"]) for row in rows]
    logger.info(
        "PAP KV locality profile mode=%s layer=%s batch=%d layout=%s "
        "kv_shape=%s kv_stride=%s key_stride=%s value_stride=%s dtype=%s "
        "device=%s kv_contiguous=%s seq_len_min=%d seq_len_max=%d "
        "live_blocks_avg=%.2f total_blocks_avg=%.2f reserved_blocks_avg=%.2f "
        "span_avg=%.2f density_avg=%.3f contiguous_pair_frac_avg=%.3f "
        "mean_abs_delta_avg=%.2f max_abs_delta_avg=%.2f runs_avg=%.2f "
        "first_live_blocks=%s first_deltas=%s",
        mode,
        layer_name,
        len(states),
        layout,
        tuple(kv_cache.shape),
        tuple(kv_cache.stride()),
        tuple(key_cache.stride()),
        tuple(value_cache.stride()),
        kv_cache.dtype,
        kv_cache.device,
        kv_cache.is_contiguous(),
        min(seq_lens) if seq_lens else 0,
        max(seq_lens) if seq_lens else 0,
        avg("live_blocks"),
        avg("total_blocks"),
        avg("reserved_blocks"),
        avg("span"),
        avg("density"),
        avg("contiguous_pair_frac"),
        avg("mean_abs_delta"),
        avg("max_abs_delta"),
        avg("runs"),
        first["first_blocks"],
        first["first_deltas"],
    )
