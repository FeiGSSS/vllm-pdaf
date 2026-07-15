# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable Attention runtime owned by the PAP service."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vllm.pap.attention.dispatcher import PAPAttentionDispatcher
from vllm.pap.attention.execution import (
    _execute_offload_exec_work_items,
    _offload_exec_work_item_compatibility_key,
)
from vllm.pap.config import PAPRuntimeConfig
from vllm.pap.deferred_cuda_trace import (
    deferred_cuda_trace_enabled,
    deferred_cuda_trace_snapshot,
)
from vllm.pap.kv.metadata import unified_paged_flash_metadata_cache_stats
from vllm.pap.kv.models import PAPAttentionSession, PAPPrefillLayerReadiness
from vllm.pap.kv.registry import PAPAttentionRegistry
from vllm.pap.protocol import PAPAttentionRegistration


class PAPAttentionRuntime:
    """Compose Attention registry, dispatch, lifecycle, and observability."""

    def __init__(
        self,
        *,
        config: PAPRuntimeConfig,
        registry: PAPAttentionRegistry | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or PAPAttentionRegistry(runtime_config=config)
        attention_config = config.attention
        self.dispatch_mode = attention_config.dispatch_mode.value
        self.active_peer_tracking = attention_config.active_peer_tracking
        self.deferred_trace_enabled = deferred_cuda_trace_enabled()
        if self.dispatch_mode == "central_combine":
            self.dispatcher: PAPAttentionDispatcher | None = PAPAttentionDispatcher(
                batch_handler=lambda items: _execute_offload_exec_work_items(
                    registry=self.registry,
                    items=items,
                ),
                compatibility_key=_offload_exec_work_item_compatibility_key,
                max_queue_size=attention_config.dispatch_queue_size,
                coalesce_timeout_s=(
                    attention_config.combine_wait_us / 1_000_000.0
                ),
            )
        else:
            self.dispatcher = None

    def sync_dispatcher_membership(self, source_ids: Iterable[str]) -> None:
        """Update expected combine membership from active Projection peers."""
        if self.dispatcher is None:
            return
        normalized_ids = set(source_ids)
        self.dispatcher.set_expected_group_size(max(1, len(normalized_ids)))
        self.dispatcher.set_preferred_peer_id(
            min(normalized_ids) if normalized_ids else None
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "role": "attention",
            "sessions": self.registry.size(),
            "dispatch_mode": self.dispatch_mode,
        }

    def stats(
        self,
        membership: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        stats = {
            "attention_dispatch_mode": self.dispatch_mode,
            "attention_active_peer_tracking": self.active_peer_tracking,
            **dict(membership or {}),
            **self.registry.decode_append_fast_path_stats(),
            **self.registry.decode_token_stats(),
            **self.registry.offload_exec_dispatch_stats(),
        }
        stats.update(
            {
                f"unified_md_{key}": value
                for key, value in (
                    unified_paged_flash_metadata_cache_stats().items()
                )
            }
        )
        if self.dispatcher is not None:
            stats.update(self.dispatcher.stats())
        if self.deferred_trace_enabled:
            active_sessions = self.registry.size()
            if active_sessions == 0:
                trace_snapshot = deferred_cuda_trace_snapshot(blocking=True)
                trace_snapshot["scope"] = "attention_process_critical_chain"
                stats["deferred_cuda_trace"] = trace_snapshot
            else:
                stats["deferred_cuda_trace"] = {
                    "enabled": True,
                    "scope": "attention_process_critical_chain",
                    "status": "waiting_for_session_drain",
                    "active_sessions": active_sessions,
                }
        return stats

    def register_prefill_kv(
        self,
        registration: PAPAttentionRegistration,
    ) -> dict[str, Any]:
        return self.registry.register_prefill_kv(registration).__dict__

    def record_decode_token(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        token_id: int,
    ) -> str:
        return self.registry.record_decode_token(
            request_id=request_id,
            new_seq_len=new_seq_len,
            token_id=token_id,
        )

    def get_session(self, request_id: str) -> PAPAttentionSession | None:
        return self.registry.get_session(request_id)

    def active_session_count(self) -> int:
        return self.registry.active_session_count()

    def get_prefill_readiness(
        self,
        request_id: str,
    ) -> list[PAPPrefillLayerReadiness]:
        return self.registry.get_prefill_readiness(request_id)

    def release_session(self, request_id: str) -> bool:
        return self.registry.release_session(request_id)

    def stop(self) -> None:
        """Drain and stop the optional central dispatcher."""
        if self.dispatcher is not None:
            self.dispatcher.stop(drain=True, timeout=5.0)
