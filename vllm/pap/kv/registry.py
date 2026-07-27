# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention KV registry and lifecycle ownership."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, OrderedDict
from threading import Condition, Lock, Thread
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
from vllm.pap.kv.observability import (
    pap_kv_lease_profile_enabled as _pap_kv_lease_profile_enabled,
)
from vllm.pap.lifecycle.commit import DecodeCommitClient as _DecodeCommitClient
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)
from vllm.pap.lifecycle.lease_release import (
    LeaseReleaseClient as _LeaseReleaseClient,
)
from vllm.pap.protocol import (
    PAPAttentionRegistration,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)

logger = logging.getLogger("pap_attention")

_commit_client: _DecodeCommitClient | None = None

_lease_release_client: _LeaseReleaseClient | None = None

_DECODE_COMMIT_PATH = "/v1/pap/prefill/decode-commit"

_LEASE_RELEASE_PATH = "/v1/pap/prefill/lease-release"

_RELEASED_SESSION_ALIAS_LIMIT = 4096


def _prefill_control_endpoint(prefill_endpoint: str, path: str) -> str:
    return f"{str(prefill_endpoint).rstrip('/')}{path}"


def _get_commit_client() -> _DecodeCommitClient:
    global _commit_client
    if _commit_client is None:
        _commit_client = _DecodeCommitClient()
    return _commit_client


def _get_lease_release_client() -> _LeaseReleaseClient:
    global _lease_release_client
    if _lease_release_client is None:
        _lease_release_client = _LeaseReleaseClient()
    return _lease_release_client


class PAPAttentionRegistry(_PAPDecodeStateMixin):
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
        self._session_manifest_events: dict[str, Any] = {}
        self._session_manifest_event_waited: set[str] = set()
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
            tuple[str, int, int, int, int, int], PAPOffloadExecSessionEntry
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

    def _release_session_locked(
        self, request_id: str
    ) -> tuple[bool, str | None, str | None]:
        session = self._sessions.pop(request_id, None)
        existed = session is not None
        prefill_endpoint = None if session is None else session.prefill_endpoint
        if session is not None:
            released_aliases = {request_id, session.prefill_kv_handle}
            released_aliases.update(
                cached_request_id
                for cached_request_id, cached_session_id in (
                    self._request_id_resolution_cache.items()
                )
                if cached_session_id == request_id
            )
            for alias in released_aliases:
                self._released_session_aliases[alias] = request_id
                self._released_session_aliases.move_to_end(alias)
            while len(self._released_session_aliases) > _RELEASED_SESSION_ALIAS_LIMIT:
                self._released_session_aliases.popitem(last=False)
        self._session_manifest_prefix_lens.pop(request_id, None)
        self._session_manifest_events.pop(request_id, None)
        self._session_manifest_event_waited.discard(request_id)
        self._session_manifest_claimed.discard(request_id)
        self._prefill_readiness.pop(request_id, None)
        self._unified_paged_kv.pop(request_id, None)
        self._session_epochs.pop(request_id, None)
        self._unified_slot_activations.pop(request_id, None)
        self._drop_offload_exec_session_entry_cache_locked(request_id)
        self._drop_attention_step_contexts_locked(request_id)
        lease_id = self._session_lease_ids.pop(request_id, None)
        leased_blocks = self._session_leased_block_ids.pop(request_id, None)
        self._session_lease_capacity_tokens.pop(request_id, None)
        if lease_id is not None and _pap_kv_lease_profile_enabled():
            logger.info(
                "PAP Attention release session lease request_id=%s "
                "lease_id=%s leased_blocks=%d",
                request_id,
                lease_id,
                len(leased_blocks or ()),
            )
        for cached_request_id, cached_session_id in list(
            self._request_id_resolution_cache.items()
        ):
            if cached_request_id == request_id or cached_session_id == request_id:
                self._request_id_resolution_cache.pop(cached_request_id, None)
        return existed, lease_id, prefill_endpoint

    def _replace_existing_session_locked(
        self, request_id: str
    ) -> tuple[str | None, str | None]:
        if request_id in self._sessions:
            _existed, lease_id, prefill_endpoint = self._release_session_locked(
                request_id
            )
            return lease_id, prefill_endpoint

        self._session_manifest_prefix_lens.pop(request_id, None)
        self._session_manifest_events.pop(request_id, None)
        self._session_manifest_event_waited.discard(request_id)
        self._session_manifest_claimed.discard(request_id)
        self._prefill_readiness.pop(request_id, None)
        self._unified_paged_kv.pop(request_id, None)
        self._session_epochs.pop(request_id, None)
        self._unified_slot_activations.pop(request_id, None)
        self._drop_offload_exec_session_entry_cache_locked(request_id)
        self._drop_attention_step_contexts_locked(request_id)
        lease_id = self._session_lease_ids.pop(request_id, None)
        self._session_leased_block_ids.pop(request_id, None)
        self._session_lease_capacity_tokens.pop(request_id, None)
        for cached_request_id, cached_session_id in list(
            self._request_id_resolution_cache.items()
        ):
            if cached_request_id == request_id or cached_session_id == request_id:
                self._request_id_resolution_cache.pop(cached_request_id, None)
        self._sessions.pop(request_id, None)
        self._request_id_resolution_cache.pop(request_id, None)
        return lease_id, None

    def release_session(
        self,
        request_id: str,
        *,
        retain_lease: bool = False,
    ) -> bool:
        request_id = self.resolve_session_request_id(request_id) or str(request_id)
        deferred_flushed = self._decode_token_committer.flush_request(
            request_id,
            timeout_s=self._decode_token_flush_timeout_s,
        )
        if not deferred_flushed:
            logger.warning(
                "PAP decode-token join flush timed out before session release "
                "request_id=%s",
                request_id,
            )
        with self._lock:
            existed, lease_id, prefill_endpoint = self._release_session_locked(
                request_id
            )
        commit_client = _get_commit_client()
        if not commit_client.flush_request(request_id):
            logger.warning(
                "PAP decode commit flush timed out before lease release request_id=%s",
                request_id,
            )
            self._decode_token_committer.forget_request(request_id)
            return existed
        if lease_id is not None:
            release_endpoint = (
                None
                if prefill_endpoint is None
                else _prefill_control_endpoint(
                    prefill_endpoint,
                    _LEASE_RELEASE_PATH,
                )
            )
            release_client = _get_lease_release_client()
            if retain_lease:
                release_client.release(
                    request_id=request_id,
                    lease_id=lease_id,
                    endpoint=release_endpoint,
                    retain=True,
                )
            else:
                release_client.release(
                    request_id=request_id,
                    lease_id=lease_id,
                    endpoint=release_endpoint,
                )
        if lease_id is not None and retain_lease:
            logger.info(
                "PAP Attention retained Prefill lease request_id=%s lease_id=%s",
                request_id,
                lease_id,
            )
        commit_client.forget_request(request_id)
        self._decode_token_committer.forget_request(request_id)
        return existed

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
                "paged_decode_warmup_started": (
                    self._paged_decode_warmup_started
                ),
                "paged_decode_warmup_done": self._paged_decode_warmup_done,
                "paged_decode_warmup_failed": self._paged_decode_warmup_failed,
            }

    def register_prefill_kv_catalog(
        self,
        *,
        descriptor: PAPPrefillKVCacheCatalogDescriptor,
        kv_cache: torch.Tensor,
    ) -> bool:
        """Register one immutable process-lifetime Prefill KV-cache layer."""
        entry = PAPPrefillKVCacheCatalogEntry(
            catalog_id=descriptor.catalog_id,
            layer_name=descriptor.layer_name,
            kv_cache=kv_cache.detach(),
            block_size=descriptor.block_size,
            num_kv_heads=descriptor.num_kv_heads,
            layout=descriptor.layout,
        )
        with self._lock:
            catalog_id = self._prefill_kv_catalog_id
            if catalog_id is None:
                self._prefill_kv_catalog_id = descriptor.catalog_id
            elif catalog_id != descriptor.catalog_id:
                raise RuntimeError(
                    "PAP Prefill KV catalog epoch changed while Attention is "
                    f"running current={catalog_id} incoming={descriptor.catalog_id}"
                )
            existing = self._prefill_kv_catalog.get(descriptor.layer_name)
            if existing is not None:
                if (
                    existing.catalog_id != entry.catalog_id
                    or existing.block_size != entry.block_size
                    or existing.num_kv_heads != entry.num_kv_heads
                    or existing.layout != entry.layout
                    or tuple(existing.kv_cache.shape) != tuple(entry.kv_cache.shape)
                    or existing.kv_cache.dtype != entry.kv_cache.dtype
                ):
                    raise RuntimeError(
                        "PAP Prefill KV catalog layer changed after registration "
                        f"layer={descriptor.layer_name}"
                    )
                return False
            self._prefill_kv_catalog[descriptor.layer_name] = entry
            self._prefill_condition.notify_all()
        logger.info(
            "PAP Prefill KV catalog registered catalog_id=%s layer=%s shape=%s",
            descriptor.catalog_id,
            descriptor.layer_name,
            tuple(kv_cache.shape),
        )
        self._start_paged_decode_warmup(
            kv_cache=entry.kv_cache,
            block_size=entry.block_size,
        )
        return True

    def _start_paged_decode_warmup(
        self,
        *,
        kv_cache: torch.Tensor,
        block_size: int,
    ) -> None:
        if kv_cache.device.type != "cuda":
            return
        with self._lock:
            if self._paged_decode_warmup_started:
                return
            self._paged_decode_warmup_started = True
        num_heads = int(self._offload_exec_shape_defaults[2])
        head_dim = int(self._offload_exec_shape_defaults[4])

        def _warm() -> None:
            try:
                from vllm.pap.attention.kernels import (
                    warm_paged_decode_attention,
                )

                warm_paged_decode_attention(
                    kv_cache=kv_cache,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                )
            except Exception:
                with self._lock:
                    self._paged_decode_warmup_failed = True
                logger.exception("PAP paged decode kernel warmup failed")
                return
            with self._lock:
                self._paged_decode_warmup_done = True
            logger.info("PAP paged decode kernel warmup complete")

        Thread(
            target=_warm,
            name="pap-paged-decode-warmup",
            daemon=True,
        ).start()

    def install_prefill_kv_session_manifest(
        self,
        *,
        manifest: PAPPrefillKVSessionManifest,
        ready_event: Any | None,
    ) -> int:
        """Atomically publish one complete request-level KV layout snapshot."""
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(
                manifest.session_handle
            )
            if session_request_id is None:
                raise KeyError(manifest.session_handle)
            request_session_id = self._resolve_session_request_id_locked(
                manifest.request_id
            )
            if request_session_id != session_request_id:
                raise RuntimeError(
                    "PAP Prefill KV manifest session generation mismatch "
                    f"request_id={manifest.request_id} "
                    f"session_handle={manifest.session_handle}"
                )
            if self._prefill_kv_catalog_id != manifest.catalog_id:
                raise RuntimeError(
                    "PAP Prefill KV manifest catalog mismatch "
                    f"current={self._prefill_kv_catalog_id} "
                    f"incoming={manifest.catalog_id}"
                )
            if len(self._prefill_kv_catalog) != manifest.expected_layer_count:
                raise RuntimeError(
                    "PAP Prefill KV manifest layer count mismatch "
                    f"catalog={len(self._prefill_kv_catalog)} "
                    f"expected={manifest.expected_layer_count}"
                )
            previous_prefix_len = self._session_manifest_prefix_lens.get(
                session_request_id
            )
            if session_request_id in self._session_manifest_claimed:
                if previous_prefix_len == manifest.prefix_len:
                    return manifest.prefix_len
                raise RuntimeError(
                    "PAP Prefill KV manifest changed after Decode claimed layout "
                    f"request_id={session_request_id} "
                    f"current={previous_prefix_len} incoming={manifest.prefix_len}"
                )
            if (
                previous_prefix_len is not None
                and manifest.prefix_len < previous_prefix_len
            ):
                raise RuntimeError(
                    "PAP Prefill KV manifest prefix regressed "
                    f"request_id={session_request_id} "
                    f"current={previous_prefix_len} incoming={manifest.prefix_len}"
                )
            existing_lease = self._session_lease_ids.get(session_request_id)
            if existing_lease not in (None, manifest.lease_id):
                raise RuntimeError(
                    "PAP Prefill KV manifest lease changed "
                    f"request_id={session_request_id}"
                )
            self._session_lease_ids[session_request_id] = manifest.lease_id
            self._session_leased_block_ids[session_request_id] = (
                manifest.leased_block_ids
            )
            self._session_lease_capacity_tokens[session_request_id] = (
                manifest.lease_capacity_tokens
            )

            layer_states: dict[str, PAPUnifiedPagedKVState] = {}
            for layer_name, entry in sorted(self._prefill_kv_catalog.items()):
                if entry.block_size != manifest.block_size:
                    raise RuntimeError(
                        "PAP Prefill KV manifest block size differs from catalog "
                        f"layer={layer_name}"
                    )
                state = PAPUnifiedPagedKVState(
                    kv_cache=entry.kv_cache,
                    block_ids=manifest.block_ids,
                    prefix_len=manifest.prefix_len,
                    seq_len=manifest.prefix_len,
                    capacity_tokens=manifest.lease_capacity_tokens,
                    writable_start_token=manifest.writable_start_token,
                    writable_end_token=manifest.writable_end_token,
                    lease_id=manifest.lease_id,
                    block_size=entry.block_size,
                    num_kv_heads=entry.num_kv_heads,
                    layout=entry.layout,
                )
                self._record_unified_slot_topology_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                    state=state,
                )
                layer_states[layer_name] = state
            activation = self._unified_slot_activations[session_request_id]
            activation.expected_layers = frozenset(layer_states)
            activation.complete = True
            self._unified_paged_kv[session_request_id] = layer_states

            session = self._sessions[session_request_id]
            session.prefix_len = manifest.prefix_len
            session.seq_len = max(int(session.seq_len), manifest.prefix_len)
            session.block_ids = manifest.block_ids
            readiness_by_layer = self._prefill_readiness.setdefault(
                session_request_id,
                {},
            )
            readiness_by_layer.clear()
            for layer_name in layer_states:
                self._mark_prefill_ready_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                )
            self._session_manifest_prefix_lens[session_request_id] = manifest.prefix_len
            self._session_manifest_events[session_request_id] = ready_event
            self._session_manifest_event_waited.discard(session_request_id)
            self._prefill_condition.notify_all()
        logger.info(
            "PAP Prefill KV manifest installed request_id=%s catalog_id=%s "
            "prefix_len=%d layers=%d blocks=%d",
            session_request_id,
            manifest.catalog_id,
            manifest.prefix_len,
            manifest.expected_layer_count,
            len(manifest.block_ids),
        )
        return manifest.prefix_len

    def register_prefill_kv(
        self, registration: PAPAttentionRegistration
    ) -> PAPAttentionSession:
        session = PAPAttentionSession(
            request_id=registration.request_id,
            conversation_id=registration.conversation_id,
            prefill_endpoint=registration.prefill_endpoint,
            kv_transfer_params=dict(registration.kv_transfer_params),
            prefix_len=registration.prefix_len,
            block_size=registration.block_size,
            max_seq_len=registration.max_seq_len,
            prefill_kv_handle=registration.request_id,
            q_size=registration.q_size,
            kv_size=registration.kv_size,
        )
        if self.resolve_session_request_id(registration.request_id) is not None:
            deferred_flushed = self._decode_token_committer.flush_request(
                registration.request_id,
                timeout_s=self._decode_token_flush_timeout_s,
            )
            if not deferred_flushed:
                logger.warning(
                    "PAP decode-token join flush timed out before replaced "
                    "session request_id=%s",
                    registration.request_id,
                )
        with self._lock:
            replaced_lease_id, replaced_prefill_endpoint = (
                self._replace_existing_session_locked(registration.request_id)
            )
            for alias, released_request_id in list(
                self._released_session_aliases.items()
            ):
                if released_request_id == registration.request_id:
                    self._released_session_aliases.pop(alias, None)
            session_epoch = self._next_session_epoch
            session.prefill_kv_handle = (
                f"{registration.request_id}@pap-session-{session_epoch}"
            )
            self._session_epochs[registration.request_id] = session_epoch
            self._next_session_epoch += 1
            self._sessions[registration.request_id] = session
            self._prefill_readiness.setdefault(registration.request_id, {})
            self._request_id_resolution_cache[registration.request_id] = (
                registration.request_id
            )
            self._request_id_resolution_cache[session.prefill_kv_handle] = (
                registration.request_id
            )
        if replaced_lease_id is not None:
            commit_client = _get_commit_client()
            commits_acked = commit_client.flush_request(registration.request_id)
            if not commits_acked:
                logger.warning(
                    "PAP decode commit flush timed out before replaced "
                    "lease release request_id=%s",
                    registration.request_id,
                )
            if commits_acked:
                commit_client.forget_request(registration.request_id)
            if commits_acked:
                release_endpoint = (
                    None
                    if replaced_prefill_endpoint is None
                    else _prefill_control_endpoint(
                        replaced_prefill_endpoint,
                        _LEASE_RELEASE_PATH,
                    )
                )
                _get_lease_release_client().release(
                    request_id=registration.request_id,
                    lease_id=replaced_lease_id,
                    endpoint=release_endpoint,
                )
            self._decode_token_committer.forget_request(registration.request_id)
        logger.info(
            "registered PAP attention session request_id=%s "
            "conversation_id=%s kv_keys=%s",
            registration.request_id,
            registration.conversation_id,
            sorted(registration.kv_transfer_params.keys()),
        )
        return session.copy()

    def get_session(self, request_id: str) -> PAPAttentionSession | None:
        with self._lock:
            session = self._sessions.get(request_id)
            return None if session is None else session.copy()

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def resolve_session_request_id(self, request_id: str) -> str | None:
        """Map vLLM-wrapped request ids back to the proxy-level PAP id."""
        with self._lock:
            return self._resolve_session_request_id_locked(request_id)

    def get_or_create_attention_step_context(
        self,
        *,
        request_ids: tuple[str, ...],
        decode_seq_lens: tuple[int, ...],
        scales: tuple[float, ...],
        layer_name: str,
        default_q_size: int,
        default_kv_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> PAPAttentionStepContext:
        """Return one decode-step context shared by all model layers."""
        cache_key = (
            tuple(str(request_id) for request_id in request_ids),
            tuple(int(seq_len) for seq_len in decode_seq_lens),
            tuple(float(scale) for scale in scales),
            int(default_q_size),
            int(default_kv_size),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            str(self.storage_device),
        )
        with self._lock:
            cached = self._attention_step_context_cache.get(cache_key)
            if cached is not None:
                self._attention_step_context_hits += 1
                self._attention_step_context_cache.move_to_end(cache_key)
                return cached

        session_entries = tuple(
            self.offload_exec_batch_session_entries(
                request_ids,
                default_q_size=default_q_size,
                default_kv_size=default_kv_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        )
        session_request_ids = tuple(
            entry.session_request_id for entry in session_entries
        )
        self.get_unified_paged_states(
            session_request_ids=session_request_ids,
            layer_name=layer_name,
        )

        with self._lock:
            cached = self._attention_step_context_cache.get(cache_key)
            if cached is not None:
                self._attention_step_context_hits += 1
                self._attention_step_context_cache.move_to_end(cache_key)
                return cached

            expected_layers: frozenset[str] | None = None
            for session_request_id in session_request_ids:
                states = self._unified_paged_kv.get(session_request_id, {})
                activation = self._unified_slot_activations.get(session_request_id)
                session_layers = (
                    activation.expected_layers
                    if activation is not None and activation.expected_layers is not None
                    else frozenset(states)
                )
                if not session_layers or layer_name not in session_layers:
                    raise RuntimeError(
                        "PAP Attention step has no complete layer set for "
                        f"request_id={session_request_id} layer={layer_name}"
                    )
                if expected_layers is None:
                    expected_layers = frozenset(session_layers)
                elif expected_layers != session_layers:
                    raise RuntimeError(
                        "PAP Attention step mixes incompatible layer sets"
                    )
            assert expected_layers is not None

            layer_states: dict[str, tuple[PAPUnifiedPagedKVState, ...]] = {}
            for expected_layer in expected_layers:
                row_states: list[PAPUnifiedPagedKVState] = []
                for session_request_id in session_request_ids:
                    state = self._unified_paged_kv.get(session_request_id, {}).get(
                        expected_layer
                    )
                    if state is None:
                        raise RuntimeError(
                            "PAP Attention step is missing sealed layer state "
                            f"request_id={session_request_id} "
                            f"layer={expected_layer}"
                        )
                    row_states.append(state)
                layer_states[expected_layer] = tuple(row_states)

            prior_seq_lens: list[int] = []
            topology_ids: list[int] = []
            for row_index, session_request_id in enumerate(session_request_ids):
                row_seq_lens = {
                    int(layer_states[name][row_index].seq_len)
                    for name in expected_layers
                }
                if len(row_seq_lens) != 1:
                    raise RuntimeError(
                        "PAP Attention step observed divergent layer sequence "
                        f"lengths request_id={session_request_id}"
                    )
                prior_seq_lens.append(next(iter(row_seq_lens)))
                row_topologies = {
                    (
                        state.block_ids,
                        int(state.block_size),
                        str(state.kv_cache.device),
                        str(state.layout),
                    )
                    for state in (
                        layer_states[name][row_index] for name in expected_layers
                    )
                }
                if len(row_topologies) != 1:
                    raise RuntimeError(
                        "PAP sealed step topology differs across layers "
                        f"request_id={session_request_id}"
                    )
                activation = self._unified_slot_activations.get(session_request_id)
                if activation is None:
                    topology_ids.append(0)
                    continue
                if activation.conflict_latched or not activation.complete:
                    raise RuntimeError(
                        "PAP sealed step topology activation is incomplete "
                        f"request_id={session_request_id}"
                    )
                topology_id = int(activation.canonical_topology_id)
                topology_ids.append(topology_id)
                if any(
                    state.slot_generation != activation.generation
                    or state.slot_topology_id != topology_id
                    for state in (
                        layer_states[name][row_index] for name in expected_layers
                    )
                ):
                    raise RuntimeError(
                        "PAP sealed step topology identity differs across "
                        f"layers request_id={session_request_id}"
                    )

            result_seq_lens: list[int] = []
            commit_new_seq_lens: list[int | None] = []
            for request_id, prior_seq_len, decode_seq_len in zip(
                request_ids,
                prior_seq_lens,
                decode_seq_lens,
            ):
                if int(decode_seq_len) > int(prior_seq_len) + 1:
                    raise RuntimeError(
                        "PAP unified KV append out of order request_id="
                        f"{request_id} current_seq_len={prior_seq_len} "
                        f"decode_seq_len={decode_seq_len}"
                    )
                result_seq_lens.append(max(int(prior_seq_len), int(decode_seq_len)))
                commit_new_seq_lens.append(
                    int(decode_seq_len)
                    if int(decode_seq_len) > int(prior_seq_len)
                    else None
                )

            shapes = {
                (
                    entry.q_size,
                    entry.kv_size,
                    entry.num_heads,
                    entry.num_kv_heads,
                    entry.head_dim,
                )
                for entry in session_entries
            }
            if len(shapes) != 1 or not scales:
                raise RuntimeError("PAP OFFLOAD_EXEC batch has mixed shapes or scales")
            shape = next(iter(shapes))
            scale = float(scales[0])
            if any(float(value) != scale for value in scales):
                raise RuntimeError("PAP OFFLOAD_EXEC batch has mixed shapes or scales")

            context = PAPAttentionStepContext(
                cache_key=cache_key,
                request_ids=tuple(request_ids),
                decode_seq_lens=tuple(decode_seq_lens),
                session_entries=session_entries,
                prior_seq_lens=tuple(prior_seq_lens),
                result_seq_lens=tuple(result_seq_lens),
                commit_new_seq_lens=tuple(commit_new_seq_lens),
                active_indices=tuple(
                    index
                    for index, seq_len in enumerate(commit_new_seq_lens)
                    if seq_len is not None
                ),
                expected_layers=expected_layers,
                layer_states=layer_states,
                topology_ids=tuple(topology_ids),
                q_size=int(shape[0]),
                kv_size=int(shape[1]),
                num_heads=int(shape[2]),
                num_kv_heads=int(shape[3]),
                head_dim=int(shape[4]),
                scale=scale,
            )
            self._attention_step_context_misses += 1
            limit = self._decode_slot_plan_cache_limit()
            if limit > 0:
                self._attention_step_context_cache[cache_key] = context
                self._attention_step_context_cache.move_to_end(cache_key)
                while len(self._attention_step_context_cache) > limit:
                    self._attention_step_context_cache.popitem(last=False)
            return context

    def record_attention_step_slot_plan_build(self) -> None:
        """Record construction of one step-owned decode slot tensor."""
        with self._lock:
            self._attention_step_slot_plan_builds += 1

    def record_attention_step_metadata_build(self) -> None:
        """Record construction of one step-owned paged-attention plan."""
        with self._lock:
            self._attention_step_metadata_builds += 1

    def complete_attention_step_layer(
        self,
        *,
        context: PAPAttentionStepContext,
        layer_name: str,
    ) -> int:
        """Publish KV readiness once every layer in the step has completed."""
        with context.lock:
            if layer_name not in context.expected_layers:
                raise RuntimeError(
                    f"PAP Attention step received unexpected layer {layer_name}"
                )
            context.completed_layers.add(layer_name)
            if (
                context.kv_ready_published
                or len(context.completed_layers) != len(context.expected_layers)
            ):
                return 0

            published = 0
            for request_id, entry, new_seq_len in zip(
                context.request_ids,
                context.session_entries,
                context.commit_new_seq_lens,
            ):
                if new_seq_len is None:
                    continue
                endpoint = _prefill_control_endpoint(
                    entry.prefill_endpoint,
                    _DECODE_COMMIT_PATH,
                )
                self._decode_token_committer.record_kv_ready(
                    request_id=entry.session_request_id,
                    new_seq_len=new_seq_len,
                    endpoint=endpoint,
                    commit_request_id=request_id,
                )
                published += 1
            context.kv_ready_published = True
        if published:
            with self._lock:
                self._attention_step_kv_ready_publishes += published
        return published

    def attention_step_context_stats(self) -> dict[str, int]:
        """Return step-context reuse counters."""
        with self._lock:
            return {
                "step_context_hits": self._attention_step_context_hits,
                "step_context_misses": self._attention_step_context_misses,
                "step_context_entries": len(self._attention_step_context_cache),
                "step_slot_plan_builds": (self._attention_step_slot_plan_builds),
                "step_metadata_builds": self._attention_step_metadata_builds,
                "step_kv_ready_publishes": (self._attention_step_kv_ready_publishes),
            }

    def offload_exec_batch_session_entries(
        self,
        request_ids: tuple[str, ...],
        *,
        default_q_size: int,
        default_kv_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> list[PAPOffloadExecSessionEntry]:
        if int(num_heads) <= 0 or int(num_kv_heads) <= 0 or int(head_dim) <= 0:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC requires PAP_OFFLOAD_EXEC_NUM_HEADS, "
                "PAP_OFFLOAD_EXEC_NUM_KV_HEADS, and PAP_OFFLOAD_EXEC_HEAD_DIM"
            )
        with self._lock:
            entries: list[PAPOffloadExecSessionEntry] = []
            for request_id in request_ids:
                session_request_id = self._resolve_session_request_id_locked(request_id)
                if session_request_id is None:
                    raise KeyError(request_id)
                session = self._sessions[session_request_id]
                q_size = int(session.q_size or default_q_size)
                kv_size = int(session.kv_size or default_kv_size)
                if q_size <= 0 or kv_size <= 0:
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC requires q_size and kv_size in "
                        "attention registration or PAP_OFFLOAD_EXEC_Q_SIZE/"
                        "PAP_OFFLOAD_EXEC_KV_SIZE"
                    )
                cache_key = (
                    session_request_id,
                    q_size,
                    kv_size,
                    int(num_heads),
                    int(num_kv_heads),
                    int(head_dim),
                )
                session_entry = self._offload_exec_session_entry_cache.get(cache_key)
                if session_entry is None:
                    session_entry = PAPOffloadExecSessionEntry(
                        session_request_id=session_request_id,
                        prefill_endpoint=session.prefill_endpoint,
                        q_size=q_size,
                        kv_size=kv_size,
                        num_heads=cache_key[3],
                        num_kv_heads=cache_key[4],
                        head_dim=cache_key[5],
                    )
                    self._offload_exec_session_entry_cache[cache_key] = session_entry
                entries.append(session_entry)
            return entries

    def _drop_offload_exec_session_entry_cache_locked(
        self, session_request_id: str
    ) -> None:
        for cache_key in list(self._offload_exec_session_entry_cache):
            if cache_key[0] == session_request_id:
                self._offload_exec_session_entry_cache.pop(cache_key, None)

    def _drop_attention_step_contexts_locked(self, session_request_id: str) -> None:
        for cache_key, context in list(self._attention_step_context_cache.items()):
            if session_request_id in context.session_request_ids:
                self._attention_step_context_cache.pop(cache_key, None)

    def _resolve_session_request_id_locked(self, request_id: str) -> str | None:
        cached = self._request_id_resolution_cache.get(request_id)
        if cached is not None:
            if cached in self._sessions:
                return cached
            self._request_id_resolution_cache.pop(request_id, None)

        if request_id in self._sessions:
            self._request_id_resolution_cache[request_id] = request_id
            return request_id

        candidates = [request_id]
        for prefix in ("cmpl-", "chatcmpl-"):
            if request_id.startswith(prefix):
                candidates.append(request_id[len(prefix) :])

        for candidate in candidates:
            if candidate in self._sessions:
                self._request_id_resolution_cache[request_id] = candidate
                return candidate
            for session_request_id in self._sessions:
                if candidate.startswith(
                    f"{session_request_id}-"
                ) or candidate.startswith(f"{session_request_id}_"):
                    self._request_id_resolution_cache[request_id] = session_request_id
                    return session_request_id
        return None

    def _prefill_readiness_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
    ) -> PAPPrefillLayerReadiness:
        return self._prefill_readiness.setdefault(session_request_id, {}).setdefault(
            layer_name,
            PAPPrefillLayerReadiness(
                request_id=session_request_id,
                layer_name=layer_name,
            ),
        )

    def _mark_prefill_ready_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
    ) -> PAPPrefillLayerReadiness:
        readiness = self._prefill_readiness_locked(
            session_request_id=session_request_id,
            layer_name=layer_name,
        )
        readiness.descriptor_received = True
        readiness.descriptor_opened = True
        readiness.ready = True
        readiness.failed = False
        readiness.error = ""
        readiness.ready_at = time.perf_counter()
        self._prefill_condition.notify_all()
        return readiness

    def get_prefill_readiness(self, request_id: str) -> list[PAPPrefillLayerReadiness]:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                return []
            readiness_by_layer = self._prefill_readiness.get(session_request_id, {})
            return [
                readiness.copy()
                for _layer_name, readiness in sorted(readiness_by_layer.items())
            ]

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)
