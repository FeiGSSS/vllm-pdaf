# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP KV ownership, lifecycle state, and registry."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Sequence
from threading import Condition, Lock
from typing import Any

import torch

from vllm.pap.config import PAPRuntimeConfig
from vllm.pap.kv.metadata import (
    PAPPagedFlashMetadata,
    _coerce_block_id,
    _UNIFIED_MD_CACHE,
    build_unified_paged_flash_metadata,
    reset_unified_paged_flash_metadata_cache,
    unified_paged_flash_metadata_cache_stats,
)
from vllm.pap.kv.ipc import (
    open_ipc_tensor_handle,
    open_prefill_manifest_event,
)
from vllm.pap.kv.models import (
    PAPAttentionSession,
    PAPOffloadExecSessionEntry,
    PAPPrefillKVCacheCatalogEntry,
    PAPPrefillLayerReadiness,
    PAPUnifiedPagedKVState,
    PAPUnifiedSlotActivation,
    PAPUnifiedSlotTopology,
    allocate_unified_slot_topology_id as _allocate_unified_slot_topology_id,
)
from vllm.pap.kv.observability import (
    _KV_LOCALITY_PROFILE_SEEN,
    block_locality_stats as _block_locality_stats,
    log_kv_locality_profile as _log_kv_locality_profile,
    pap_attention_pool_profile_enabled as _pap_attention_pool_profile_enabled,
    pap_env_flag as _pap_env_flag,
    pap_kv_lease_profile_enabled as _pap_kv_lease_profile_enabled,
    pap_kv_locality_profile_enabled as _pap_kv_locality_profile_enabled,
    trace_add_elapsed_ms as _trace_add_elapsed_ms,
)
from vllm.pap.lifecycle.commit import DecodeCommitClient as _DecodeCommitClient
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    deferred_cuda_trace_enabled,
    end_deferred_cuda_span,
)
from vllm.pap.lease_release_client import LeaseReleaseClient as _LeaseReleaseClient
from vllm.pap.protocol import (
    PAPAttentionRegistration,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)

logger = logging.getLogger("pap_attention")

_commit_client: _DecodeCommitClient | None = None

_lease_release_client: _LeaseReleaseClient | None = None

_DEFERRED_CUDA_TRACE_ENABLED = deferred_cuda_trace_enabled()

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


class PAPAttentionRegistry:
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
        self._decode_token_committer = DeferredDecodeTokenCommitter(
            self._dispatch_deferred_decode_commit
        )

    @staticmethod
    def _dispatch_deferred_decode_commit(commit: DeferredDecodeCommit) -> None:
        _get_commit_client().commit(
            request_id=commit.commit_request_id or commit.request_id,
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

    def record_decode_kv_ready(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        endpoint: str,
    ) -> str:
        commit_request_id = str(request_id)
        session_request_id = self.resolve_session_request_id(commit_request_id)
        if session_request_id is None:
            raise KeyError(request_id)
        return self._decode_token_committer.record_kv_ready(
            request_id=session_request_id,
            new_seq_len=new_seq_len,
            endpoint=endpoint,
            commit_request_id=commit_request_id,
        )

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


    def _decode_slot_plan_cache_limit(self) -> int:
        return self._decode_slot_plan_cache_limit_value

    def _record_unified_slot_topology_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        state: PAPUnifiedPagedKVState,
    ) -> None:
        topology = (
            state.block_ids,
            int(state.block_size),
            str(state.kv_cache.device),
        )
        prefix_len = int(state.prefix_len)
        activation = self._unified_slot_activations.get(session_request_id)
        if activation is None:
            topology_id = _allocate_unified_slot_topology_id()
            activation = PAPUnifiedSlotActivation(
                prefix_len=prefix_len,
                generation=1,
                canonical_topology=topology,
                canonical_topology_id=topology_id,
                topology_ids={topology: topology_id},
                layer_observations={layer_name: topology_id},
            )
            self._unified_slot_activations[session_request_id] = activation
            state.slot_generation = activation.generation
            state.slot_topology_id = topology_id
            return

        if prefix_len < activation.prefix_len:
            raise RuntimeError(
                "PAP unified KV stale slot activation "
                f"request_id={session_request_id} layer={layer_name} "
                f"prefix_len={prefix_len} current={activation.prefix_len}"
            )

        if prefix_len > activation.prefix_len:
            expected_layers = activation.expected_layers
            if expected_layers is None:
                expected_layers = frozenset(activation.layer_observations)
            elif not activation.complete:
                raise RuntimeError(
                    "PAP unified KV slot activation advanced before all layers "
                    f"request_id={session_request_id} "
                    f"prefix_len={activation.prefix_len} next={prefix_len}"
                )
            if expected_layers and layer_name not in expected_layers:
                raise RuntimeError(
                    "PAP unified KV slot activation contains unexpected layer "
                    f"request_id={session_request_id} layer={layer_name}"
                )
            topology_id = _allocate_unified_slot_topology_id()
            activation = PAPUnifiedSlotActivation(
                prefix_len=prefix_len,
                generation=activation.generation + 1,
                canonical_topology=topology,
                canonical_topology_id=topology_id,
                topology_ids={topology: topology_id},
                layer_observations={layer_name: topology_id},
                expected_layers=expected_layers,
                complete=not expected_layers or expected_layers == {layer_name},
            )
            self._unified_slot_activations[session_request_id] = activation
            state.slot_generation = activation.generation
            state.slot_topology_id = topology_id
            return

        if activation.expected_layers and layer_name not in activation.expected_layers:
            raise RuntimeError(
                "PAP unified KV slot activation contains unexpected layer "
                f"request_id={session_request_id} layer={layer_name}"
            )
        topology_id = activation.topology_ids.get(topology)
        if topology_id is None:
            topology_id = _allocate_unified_slot_topology_id()
            activation.topology_ids[topology] = topology_id
        previous_topology_id = activation.layer_observations.get(layer_name)
        has_conflict = (
            topology_id != activation.canonical_topology_id
            or previous_topology_id not in (None, topology_id)
        )
        if has_conflict and not activation.conflict_latched:
            self._decode_slot_topology_mismatches += 1
            logger.warning(
                "PAP unified KV slot topology conflicts within one activation; "
                "disabling cross-layer slot-plan cache request_id=%s",
                session_request_id,
            )
        activation.conflict_latched = activation.conflict_latched or has_conflict
        activation.layer_observations[layer_name] = topology_id
        if activation.expected_layers:
            activation.complete = activation.expected_layers.issubset(
                activation.layer_observations
            )
        state.slot_generation = activation.generation
        state.slot_topology_id = topology_id

    def _decode_slot_plan_key_locked(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
        decode_seq_lens: Sequence[int],
        device: torch.device,
    ) -> tuple[Any, ...] | None:
        session_generations: list[tuple[str, int, int, int, int]] = []
        for session_request_id in session_request_ids:
            activation = self._unified_slot_activations.get(session_request_id)
            if (
                activation is None
                or activation.conflict_latched
                or not activation.complete
            ):
                return None
            epoch = self._session_epochs.get(session_request_id)
            if epoch is None:
                return None
            state = self._unified_paged_kv.get(session_request_id, {}).get(layer_name)
            if (
                state is None
                or state.slot_generation != activation.generation
                or state.slot_topology_id != activation.canonical_topology_id
            ):
                return None
            session_generations.append(
                (
                    session_request_id,
                    int(epoch),
                    activation.prefix_len,
                    activation.generation,
                    activation.canonical_topology_id,
                )
            )
        return (
            tuple(session_generations),
            tuple(int(seq_len) for seq_len in decode_seq_lens),
            str(device),
        )

    def _store_decode_slot_plan_locked(
        self,
        key: tuple[Any, ...],
        slot_tensor: torch.Tensor,
    ) -> None:
        limit = self._decode_slot_plan_cache_limit()
        if limit <= 0:
            return
        self._decode_slot_plan_cache[key] = slot_tensor
        self._decode_slot_plan_cache.move_to_end(key)
        while len(self._decode_slot_plan_cache) > limit:
            self._decode_slot_plan_cache.popitem(last=False)

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
            while (
                len(self._released_session_aliases)
                > _RELEASED_SESSION_ALIAS_LIMIT
            ):
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

    def release_session(self, request_id: str) -> bool:
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
            _get_lease_release_client().release(
                request_id=request_id,
                lease_id=lease_id,
                endpoint=release_endpoint,
            )
        commit_client.forget_request(request_id)
        self._decode_token_committer.forget_request(request_id)
        return existed

    def append_decode_kv_to_unified_prefill_cache(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
        key_batch: torch.Tensor,
        value_batch: torch.Tensor,
        decode_seq_lens: Sequence[int],
        trace_stats: dict[str, float] | None = None,
    ) -> int:
        """Write a same-layer decode batch into Prefill-owned unified KV blocks.

        Uses a single ``reshape_and_cache_flash`` call over the batch (one GPU
        kernel launch), mirroring the legacy local-pool fast path.
        """
        if len(session_request_ids) != len(decode_seq_lens):
            raise ValueError(
                "PAP unified KV append session_request_ids/decode_seq_lens "
                "length mismatch"
            )

        decode_lock_start = time.perf_counter() if trace_stats is not None else 0.0
        with self._decode_append_lock:
            decode_lock_acquired = (
                time.perf_counter() if trace_stats is not None else 0.0
            )
            registry_lock_start = (
                time.perf_counter() if trace_stats is not None else 0.0
            )
            with self._lock:
                if trace_stats is not None:
                    trace_stats["append_lock_wait_ms"] += (
                        decode_lock_acquired - decode_lock_start
                    ) * 1000.0
                    trace_stats["append_lock_wait_ms"] += (
                        time.perf_counter() - registry_lock_start
                    ) * 1000.0
                prepare_start = (
                    time.perf_counter() if trace_stats is not None else 0.0
                )
                active_indices: list[int] = []
                active_states: list[PAPUnifiedPagedKVState] = []
                expected_positions: list[int] = []
                base_v_cache: torch.Tensor | None = None
                for index, session_request_id in enumerate(session_request_ids):
                    decode_len = int(decode_seq_lens[index])
                    if decode_len <= 0:
                        continue
                    layer_states = self._unified_paged_kv.get(session_request_id, {})
                    state = layer_states.get(layer_name)
                    if state is None:
                        raise RuntimeError(
                            "PAP unified KV state missing for request_id="
                            f"{session_request_id} layer={layer_name}"
                        )
                    base_v_cache = state.kv_cache
                    position = int(state.seq_len)
                    if decode_len <= position:
                        continue
                    if decode_len != position + 1:
                        raise RuntimeError(
                            "PAP unified KV append out of order request_id="
                            f"{session_request_id} layer={layer_name} "
                            f"current_seq_len={position} "
                            f"decode_seq_len={decode_len}"
                        )
                    if position < int(state.writable_start_token) or position >= int(
                        state.writable_end_token
                    ):
                        raise RuntimeError(
                            "PAP unified KV append out of range request_id="
                            f"{session_request_id} layer={layer_name} "
                            f"position={position} writable=["
                            f"{state.writable_start_token},"
                            f"{state.writable_end_token})"
                        )
                    active_indices.append(index)
                    active_states.append(state)
                    expected_positions.append(position)

                if not active_indices or base_v_cache is None:
                    if trace_stats is not None:
                        trace_stats["append_prepare_ms"] += (
                            time.perf_counter() - prepare_start
                        ) * 1000.0
                    return 0

                all_rows_active = len(active_indices) == len(session_request_ids)
                if all_rows_active:
                    self._decode_append_fast_path_hits += 1
                else:
                    self._decode_append_fallbacks += 1

                slot_plan_key = None
                slot_tensor = None
                if all_rows_active:
                    slot_plan_key = self._decode_slot_plan_key_locked(
                        session_request_ids=session_request_ids,
                        layer_name=layer_name,
                        decode_seq_lens=decode_seq_lens,
                        device=base_v_cache.device,
                    )
                    if slot_plan_key is not None:
                        slot_tensor = self._decode_slot_plan_cache.get(slot_plan_key)
                        if slot_tensor is not None:
                            self._decode_slot_plan_cache_hits += 1
                            self._decode_slot_plan_cache.move_to_end(slot_plan_key)
                        else:
                            self._decode_slot_plan_cache_misses += 1
                slots: list[int] = []
                if slot_tensor is None:
                    for state, position in zip(
                        active_states,
                        expected_positions,
                    ):
                        block_size = int(state.block_size)
                        logical_block = position // block_size
                        if logical_block >= len(state.block_ids):
                            raise RuntimeError(
                                f"PAP unified KV logical_block={logical_block} "
                                f"exceeds block_ids len={len(state.block_ids)} "
                                f"(layer={layer_name})"
                            )
                        physical_block = _coerce_block_id(
                            state.block_ids[logical_block]
                        )
                        slots.append(
                            physical_block * block_size + position % block_size
                        )
                scale_key = str(base_v_cache.device)
                scales = self._reshape_cache_scales.get(scale_key)
                if trace_stats is not None:
                    trace_stats["append_prepare_ms"] += (
                        time.perf_counter() - prepare_start
                    ) * 1000.0

            tensor_start = time.perf_counter() if trace_stats is not None else 0.0
            if all_rows_active:
                kb = key_batch
                vb = value_batch
            else:
                kb = key_batch[active_indices]
                vb = value_batch[active_indices]
            if slot_tensor is None:
                slot_tensor = torch.tensor(
                    slots,
                    dtype=torch.int64,
                    device=base_v_cache.device,
                )
                if slot_plan_key is not None:
                    with self._lock:
                        self._store_decode_slot_plan_locked(
                            slot_plan_key,
                            slot_tensor,
                        )
            if scales is None:
                scales = (
                    torch.ones(
                        1,
                        dtype=torch.float32,
                        device=base_v_cache.device,
                    ),
                    torch.ones(
                        1,
                        dtype=torch.float32,
                        device=base_v_cache.device,
                    ),
                )
                with self._lock:
                    self._reshape_cache_scales[scale_key] = scales
            if trace_stats is not None:
                trace_stats["append_tensor_ms"] += (
                    time.perf_counter() - tensor_start
                ) * 1000.0

            copy_start = time.perf_counter() if trace_stats is not None else 0.0
            if kb.device != base_v_cache.device or kb.dtype != base_v_cache.dtype:
                kb = kb.to(device=base_v_cache.device, dtype=base_v_cache.dtype)
            if vb.device != base_v_cache.device or vb.dtype != base_v_cache.dtype:
                vb = vb.to(device=base_v_cache.device, dtype=base_v_cache.dtype)
            if trace_stats is not None:
                trace_stats["append_copy_ms"] += (
                    time.perf_counter() - copy_start
                ) * 1000.0

            record_start = time.perf_counter() if trace_stats is not None else 0.0
            k_scale, v_scale = scales
            key_cache, value_cache = base_v_cache.unbind(1)
            if (
                _DEFERRED_CUDA_TRACE_ENABLED
                and base_v_cache.is_cuda
                and torch.cuda.is_available()
            ):
                append_trace = begin_deferred_cuda_span(
                    "kv_append_gpu_ms",
                    torch.cuda.current_stream(base_v_cache.device),
                )
                try:
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        kb,
                        vb,
                        key_cache,
                        value_cache,
                        slot_tensor,
                        "auto",
                        k_scale,
                        v_scale,
                    )
                finally:
                    end_deferred_cuda_span(append_trace)
            else:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    kb,
                    vb,
                    key_cache,
                    value_cache,
                    slot_tensor,
                    "auto",
                    k_scale,
                    v_scale,
                )
            if trace_stats is not None:
                trace_stats["append_record_ms"] += (
                    time.perf_counter() - record_start
                ) * 1000.0

            state_start = time.perf_counter() if trace_stats is not None else 0.0
            with self._lock:
                for index, expected_state, expected_position in zip(
                    active_indices,
                    active_states,
                    expected_positions,
                ):
                    session_request_id = session_request_ids[index]
                    current_state = self._unified_paged_kv.get(
                        session_request_id,
                        {},
                    ).get(layer_name)
                    if current_state is not expected_state:
                        raise RuntimeError(
                            "PAP unified KV state changed during decode append "
                            f"request_id={session_request_id} layer={layer_name}"
                        )
                    if int(current_state.seq_len) != expected_position:
                        raise RuntimeError(
                            "PAP unified KV seq_len changed during decode append "
                            f"request_id={session_request_id} layer={layer_name}"
                        )
                for state in active_states:
                    state.seq_len = int(state.seq_len) + 1
            if trace_stats is not None:
                trace_stats["append_state_ms"] += (
                    time.perf_counter() - state_start
                ) * 1000.0
            return len(active_indices)

    def decode_append_fast_path_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "fast_path_hits": self._decode_append_fast_path_hits,
                "fallbacks": self._decode_append_fallbacks,
                "scale_cache_entries": len(self._reshape_cache_scales),
                "slot_plan_hits": self._decode_slot_plan_cache_hits,
                "slot_plan_misses": self._decode_slot_plan_cache_misses,
                "slot_plan_entries": len(self._decode_slot_plan_cache),
                "slot_topology_mismatches": (self._decode_slot_topology_mismatches),
            }

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
            }

    def get_unified_paged_states(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
    ) -> list[PAPUnifiedPagedKVState] | None:
        """Return per-row unified states if every row has unified state."""
        events_to_wait: list[tuple[Any, torch.device]] = []
        with self._lock:
            deadline = time.monotonic() + self._prefill_wait_timeout_s
            while True:
                states: list[PAPUnifiedPagedKVState] = []
                pending_request_id = ""
                pending_prefix_len = 0
                pending_state_len = -1
                for session_request_id in session_request_ids:
                    session = self._sessions.get(session_request_id)
                    if session is None:
                        raise KeyError(session_request_id)
                    readiness = self._prefill_readiness.get(
                        session_request_id,
                        {},
                    ).get(layer_name)
                    if readiness is not None and readiness.failed:
                        raise RuntimeError(
                            "prefill KV import failed before unified decode "
                            f"attention: {readiness.error}"
                        )
                    state = self._unified_paged_kv.get(
                        session_request_id,
                        {},
                    ).get(layer_name)
                    required_prefix_len = int(session.prefix_len or 0)
                    if state is None or int(state.seq_len) < required_prefix_len:
                        pending_request_id = session_request_id
                        pending_prefix_len = required_prefix_len
                        pending_state_len = -1 if state is None else int(state.seq_len)
                        break
                    states.append(state)
                if not pending_request_id:
                    for session_request_id, state in zip(
                        session_request_ids,
                        states,
                    ):
                        if session_request_id not in self._session_manifest_prefix_lens:
                            continue
                        self._session_manifest_claimed.add(session_request_id)
                        if session_request_id in self._session_manifest_event_waited:
                            continue
                        event = self._session_manifest_events.get(session_request_id)
                        self._session_manifest_event_waited.add(session_request_id)
                        if event is not None:
                            events_to_wait.append((event, state.kv_cache.device))
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "prefill KV must reach the registered prefix before "
                        "unified decode attention "
                        f"request_id={pending_request_id} layer={layer_name} "
                        f"state_seq_len={pending_state_len} "
                        f"required_prefix_len={pending_prefix_len}"
                    )
                self._prefill_condition.wait(timeout=remaining)
        for event, device in events_to_wait:
            if device.type != "cuda":
                continue
            with torch.cuda.device(device):
                torch.cuda.current_stream(device).wait_event(event)
        return states

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
        return True

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
            self._session_manifest_prefix_lens[session_request_id] = (
                manifest.prefix_len
            )
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


    def prefill_layer_readiness(
        self,
        *,
        request_id: str,
        layer_name: str,
    ) -> PAPPrefillLayerReadiness | None:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                return None
            readiness = self._prefill_readiness.setdefault(session_request_id, {}).get(
                layer_name
            )
            return None if readiness is None else readiness.copy()

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
