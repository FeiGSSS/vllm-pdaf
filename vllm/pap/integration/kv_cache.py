# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP metadata and diagnostics at the vLLM KV-cache boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.prefix_cache_audit import (
    build_prefix_cache_audit_state,
    pap_prefix_cache_audit_enabled,
)


class PAPKVCacheAdapter:
    """Keep PAP metadata and audit glue out of KV-cache mechanics."""

    @staticmethod
    def should_cache_locally(params: Mapping[str, Any] | None) -> bool:
        """Return whether the vLLM cache owns this request's local prefix."""
        return not PAPRequestMetadata.from_mapping(params).projection_kv_unaware

    @staticmethod
    def projection_scratch_config(config: Any, *, enabled: bool) -> Any:
        """Shrink Projection KV allocation to one structural scratch block."""
        if not enabled or int(config.num_blocks) <= 1:
            return config
        num_blocks = int(config.num_blocks)
        scratch_tensors = []
        for tensor in config.kv_cache_tensors:
            if tensor.block_stride > 0:
                scratch_size = int(tensor.block_stride)
            else:
                if int(tensor.size) % num_blocks:
                    raise ValueError("Projection KV tensor is not block-aligned")
                scratch_size = int(tensor.size) // num_blocks
            scratch_tensors.append(replace(tensor, size=scratch_size))
        return replace(config, num_blocks=1, kv_cache_tensors=scratch_tensors)

    @staticmethod
    def log_prefix_lookup(
        logger: Any,
        manager: Any,
        request: Any,
        hit_tokens: int,
    ) -> None:
        """Emit an optional token-safe prefix lookup audit."""
        if not pap_prefix_cache_audit_enabled():
            return
        audit_state = build_prefix_cache_audit_state(manager, request)
        audit_state["hit_tokens"] = int(hit_tokens)
        logger.info("PAP prefix cache lookup audit %s", audit_state)

    @staticmethod
    def log_decode_commit(logger: Any, manager: Any, request: Any) -> None:
        """Emit an optional token-safe decode commit audit."""
        if pap_prefix_cache_audit_enabled():
            logger.info(
                "PAP prefix cache commit audit %s",
                build_prefix_cache_audit_state(manager, request),
            )
