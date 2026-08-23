# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safe runtime state for PAP prefix-cache debugging."""

from __future__ import annotations

import os
from typing import Any

_TRUE_VALUES = {"1", "true", "yes", "on"}


def pap_prefix_cache_audit_enabled() -> bool:
    """Return whether PAP prefix-cache audit logging is enabled."""
    return os.environ.get("PAP_PREFIX_CACHE_AUDIT", "").lower() in _TRUE_VALUES


def _short_hash(value: bytes, *, has_group_id: bool = False) -> str:
    raw = bytes(value)
    if has_group_id and len(raw) >= 4:
        raw = raw[:-4]
    return raw.hex()[:16]


def build_prefix_cache_audit_state(
    kv_cache_manager: Any,
    request: Any,
) -> dict[str, Any]:
    """Build token-safe PAP prefix-cache state for one request.

    The state contains only counts and truncated hashes. It intentionally
    excludes token IDs and prompt content.
    """
    request_id = str(request.request_id)
    request_hashes = list(getattr(request, "block_hashes", ()))
    groups: list[dict[str, Any]] = []
    coordinator = getattr(kv_cache_manager, "coordinator", None)
    managers = getattr(coordinator, "single_type_managers", ())
    for fallback_group_id, manager in enumerate(managers):
        blocks = list(manager.req_to_blocks.get(request_id, ()))
        block_hashes = [
            block.block_hash
            for block in blocks
            if getattr(block, "block_hash", None) is not None
        ]
        groups.append(
            {
                "group_id": int(
                    getattr(manager, "kv_cache_group_id", fallback_group_id)
                ),
                "allocated_blocks": len(blocks),
                "cached_blocks": int(manager.num_cached_block.get(request_id, 0)),
                "hashed_blocks": len(block_hashes),
                "allocated_hash_tail": [
                    _short_hash(block_hash, has_group_id=True)
                    for block_hash in block_hashes[-3:]
                ],
            }
        )

    return {
        "request_id": request_id,
        "num_tokens": int(request.num_tokens),
        "num_computed_tokens": int(request.num_computed_tokens),
        "request_hash_count": len(request_hashes),
        "request_hash_tail": [
            _short_hash(block_hash) for block_hash in request_hashes[-3:]
        ],
        "groups": groups,
    }
