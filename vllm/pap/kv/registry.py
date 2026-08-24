# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention KV registry and lifecycle ownership."""

from __future__ import annotations

import os
from collections import Counter, OrderedDict
from threading import Condition, Lock
from typing import Any

import torch

from vllm.pap.config import PAPRuntimeConfig
from vllm.pap.kv.decode_state import _PAPDecodeStateMixin
from vllm.pap.kv.models import (
    PAPAttentionSession,
    PAPAttentionStepContext,
    PAPOffloadExecSessionEntry,
    PAPPrefillKVCacheCatalogEntry,
    PAPPrefillLayerReadiness,
    PAPUnifiedPagedKVState,
    PAPUnifiedSlotActivation,
)
from vllm.pap.kv.session_registry import (
    _get_commit_client,
    _PAPSessionRegistryMixin,
)
from vllm.pap.kv.step_context_registry import _PAPAttentionStepContextMixin
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)


class PAPAttentionRegistry(
    _PAPDecodeStateMixin,
    _PAPSessionRegistryMixin,
    _PAPAttentionStepContextMixin,
):
    """Thread-safe in-memory registry for PAP Attention control-plane state."""

    def __init__(
        self,
        storage_device: str | torch.device | None = None,
        *,
        runtime_config: PAPRuntimeConfig | None = None,
    ) -> None:
        self.runtime_config = runtime_config
        self._lock = Lock()
        self._decode_append_lock = Lock()
        self._prefill_condition = Condition(self._lock)
        if runtime_config is None:
            self._decode_slot_plan_cache_limit_value = int(
                os.environ.get("PAP_DECODE_SLOT_PLAN_CACHE_LIMIT", "256")
            )
            self._decode_token_flush_timeout_s = float(
                os.environ.get("PAP_DECODE_TOKEN_FLUSH_TIMEOUT", "5.0")
            )
            self._prefill_wait_timeout_s = float(
                os.environ.get("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "5.0")
            )
        else:
            self._decode_slot_plan_cache_limit_value = (
                runtime_config.features.decode_slot_plan_cache_limit
            )
            self._decode_token_flush_timeout_s = (
                runtime_config.decode_token.flush_timeout_s
            )
            self._prefill_wait_timeout_s = (
                runtime_config.attention.prefill_wait_timeout_s
            )
            if storage_device is None:
                storage_device = runtime_config.attention.storage_device
        self._offload_exec_shape_defaults = (
            int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0")),
            int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0")),
            int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0")),
            int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0")),
            int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0")),
        )
        self._storage_device = self._resolve_storage_device(storage_device)
        self._sessions: dict[str, PAPAttentionSession] = {}
        self._prefill_kv_catalog_id: str | None = None
        self._prefill_kv_catalog: dict[str, PAPPrefillKVCacheCatalogEntry] = {}
        self._session_manifest_prefix_lens: dict[str, int] = {}
        self._session_manifest_ready_prefix_lens: dict[str, int] = {}
        self._session_manifest_events: dict[str, Any] = {}
        self._session_manifest_claimed: set[str] = set()
        self._prefill_readiness: dict[str, dict[str, PAPPrefillLayerReadiness]] = {}
        self._request_id_resolution_cache: dict[str, str] = {}
        self._released_session_aliases: OrderedDict[str, str] = OrderedDict()
        self._session_lease_ids: dict[str, str] = {}
        self._session_leased_block_ids: dict[str, tuple[int, ...]] = {}
        self._session_lease_capacity_tokens: dict[str, int] = {}
        self._unified_paged_kv: dict[str, dict[str, PAPUnifiedPagedKVState]] = {}
        self._session_epochs: dict[str, int] = {}
        self._next_session_epoch = 1
        self._unified_slot_activations: dict[str, PAPUnifiedSlotActivation] = {}
        self._decode_slot_plan_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = (
            OrderedDict()
        )
        self._offload_exec_session_entry_cache: dict[
            tuple[str, int, int, int, int, int, int], PAPOffloadExecSessionEntry
        ] = {}
        self._attention_step_context_cache: OrderedDict[
            tuple[Any, ...], PAPAttentionStepContext
        ] = OrderedDict()
        self._attention_step_context_hits = 0
        self._attention_step_context_misses = 0
        self._attention_step_slot_plan_builds = 0
        self._attention_step_metadata_builds = 0
        self._attention_step_kv_ready_publishes = 0
        self._reshape_cache_scales: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._decode_append_fast_path_hits = 0
        self._decode_append_fallbacks = 0
        self._decode_slot_plan_cache_hits = 0
        self._decode_slot_plan_cache_misses = 0
        self._decode_slot_topology_mismatches = 0
        self._offload_exec_stats_lock = Lock()
        self._offload_exec_peer_batches = 0
        self._offload_exec_peer_rows = 0
        self._offload_exec_compute_calls = 0
        self._offload_exec_compute_rows = 0
        self._offload_exec_source_batches_per_compute_sum = 0
        self._offload_exec_max_source_batches_per_compute = 0
        self._offload_exec_peer_batches_by_source: Counter[str] = Counter()
        self._offload_exec_peer_rows_by_source: Counter[str] = Counter()
        self._offload_exec_compute_calls_by_layer: Counter[str] = Counter()
        self._paged_decode_warmup_started = False
        self._paged_decode_warmup_done = False
        self._paged_decode_warmup_failed = False
        self._decode_token_committer = DeferredDecodeTokenCommitter(
            self._dispatch_deferred_decode_commit
        )

    @staticmethod
    def _dispatch_deferred_decode_commit(commit: DeferredDecodeCommit) -> None:
        _get_commit_client().commit(
            request_id=commit.commit_request_id or commit.request_id,
            session_request_id=commit.request_id,
            new_seq_len=commit.new_seq_len,
            new_token_ids=commit.token_ids,
            endpoint=commit.endpoint,
        )

    def record_decode_token(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        token_id: int,
    ) -> str:
        request_id = str(request_id)
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                if request_id in self._released_session_aliases:
                    self._released_session_aliases.move_to_end(request_id)
                    return "released"
                raise KeyError(request_id)
        status = self._decode_token_committer.record_token(
            request_id=session_request_id,
            new_seq_len=new_seq_len,
            token_ids=(token_id,),
        )
        with self._lock:
            if session_request_id not in self._sessions:
                self._decode_token_committer.forget_request(session_request_id)
                return "released"
        return status

    def decode_token_stats(self) -> dict[str, int]:
        return self._decode_token_committer.stats()

    def record_decode_kv_ready(
        self,
        *,
        request_id: str,
        session_epoch: int,
        new_seq_len: int,
        endpoint: str,
        commit_request_id: str | None = None,
    ) -> str:
        """Join KV readiness only while the originating session is active."""
        request_id = str(request_id)
        with self._lock:
            if self._session_epochs.get(request_id) != int(session_epoch):
                return "released"
            return self._decode_token_committer.record_kv_ready(
                request_id=request_id,
                new_seq_len=new_seq_len,
                endpoint=endpoint,
                commit_request_id=commit_request_id,
            )

    @staticmethod
    def _resolve_storage_device(
        storage_device: str | torch.device | None,
    ) -> torch.device:
        if storage_device is None:
            storage_device = os.environ.get("PAP_ATTENTION_STORAGE_DEVICE")
        if storage_device is None:
            storage_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(storage_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PAP attention storage requested CUDA without CUDA")
        return device

    @property
    def storage_device(self) -> torch.device:
        return self._storage_device

    @property
    def offload_exec_shape_defaults(self) -> tuple[int, int, int, int, int]:
        """Return process-lifetime OFFLOAD_EXEC shape defaults."""
        return self._offload_exec_shape_defaults

    def _decode_slot_plan_cache_limit(self) -> int:
        return self._decode_slot_plan_cache_limit_value

    def record_offload_exec_peer_batch(
        self,
        *,
        peer_id: str,
        rows: int,
    ) -> None:
        with self._offload_exec_stats_lock:
            self._offload_exec_peer_batches += 1
            self._offload_exec_peer_rows += int(rows)
            self._offload_exec_peer_batches_by_source[str(peer_id)] += 1
            self._offload_exec_peer_rows_by_source[str(peer_id)] += int(rows)

    def record_offload_exec_compute(
        self,
        *,
        layer_name: str,
        rows: int,
        source_batches: int,
    ) -> None:
        with self._offload_exec_stats_lock:
            self._offload_exec_compute_calls += 1
            self._offload_exec_compute_rows += int(rows)
            self._offload_exec_source_batches_per_compute_sum += int(source_batches)
            self._offload_exec_max_source_batches_per_compute = max(
                self._offload_exec_max_source_batches_per_compute,
                int(source_batches),
            )
            self._offload_exec_compute_calls_by_layer[str(layer_name)] += 1

    def offload_exec_dispatch_stats(self) -> dict[str, Any]:
        with self._offload_exec_stats_lock:
            return {
                "offload_exec_peer_batches": self._offload_exec_peer_batches,
                "offload_exec_peer_rows": self._offload_exec_peer_rows,
                "offload_exec_compute_calls": self._offload_exec_compute_calls,
                "offload_exec_compute_rows": self._offload_exec_compute_rows,
                "offload_exec_source_batches_per_compute_sum": (
                    self._offload_exec_source_batches_per_compute_sum
                ),
                "offload_exec_max_source_batches_per_compute": (
                    self._offload_exec_max_source_batches_per_compute
                ),
                "offload_exec_peer_batches_by_source": dict(
                    sorted(self._offload_exec_peer_batches_by_source.items())
                ),
                "offload_exec_peer_rows_by_source": dict(
                    sorted(self._offload_exec_peer_rows_by_source.items())
                ),
                "offload_exec_compute_calls_by_layer": dict(
                    sorted(self._offload_exec_compute_calls_by_layer.items())
                ),
                "paged_decode_warmup_started": (self._paged_decode_warmup_started),
                "paged_decode_warmup_done": self._paged_decode_warmup_done,
                "paged_decode_warmup_failed": self._paged_decode_warmup_failed,
            }
