# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable Attention runtime owned by the PAP service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import torch

from vllm.pap.attention.triton_backend import paged_decode_kernel_config_for_sms
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
        self.deferred_trace_enabled = deferred_cuda_trace_enabled()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "role": "attention",
            "sessions": self.registry.size(),
            "offload_exec_transport": "nvshmem_graph",
        }

    def stats(
        self,
        membership: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        storage_device = self.registry.storage_device
        visible_sms = (
            torch.cuda.get_device_properties(storage_device).multi_processor_count
            if storage_device.type == "cuda"
            else 0
        )
        stats = {
            "attention_dispatch_mode": "nvshmem_graph",
            "paged_decode_visible_sms": visible_sms,
            "paged_decode_kernel_config": asdict(
                paged_decode_kernel_config_for_sms(visible_sms)
            ),
            **dict(membership or {}),
            **self.registry.slot_topology_stats(),
            **self.registry.attention_step_context_stats(),
            **self.registry.decode_token_stats(),
            **self.registry.decode_capacity_stats(),
            **self.registry.offload_exec_dispatch_stats(),
        }
        stats.update(
            {
                f"unified_md_{key}": value
                for key, value in (unified_paged_flash_metadata_cache_stats().items())
            }
        )
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

    def get_prefill_readiness_snapshot(
        self,
        request_id: str,
        *,
        include_layers: bool = True,
    ) -> dict[str, Any]:
        return self.registry.get_prefill_readiness_snapshot(
            request_id,
            include_layers=include_layers,
        )

    def prefetch_decode_capacity(self, request_id: str, required_tokens: int) -> None:
        self.registry.prefetch_decode_capacity(request_id, required_tokens)

    def release_session(self, request_id: str) -> bool:
        return self.registry.release_session(request_id)

    def stop(self) -> None:
        """Stop the runtime after its peer loops have drained."""
