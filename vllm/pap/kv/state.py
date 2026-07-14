# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP KV ownership, lifecycle state, and registry."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Condition, Lock
from typing import Any

import torch

from vllm.pap.config import PAPRuntimeConfig
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
    PAPCudaIPCTensorHandle,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)

logger = logging.getLogger("pap_attention")

_commit_client: _DecodeCommitClient | None = None

_lease_release_client: _LeaseReleaseClient | None = None

_DEFERRED_CUDA_TRACE_ENABLED = deferred_cuda_trace_enabled()

_DECODE_COMMIT_PATH = "/v1/pap/prefill/decode-commit"

_LEASE_RELEASE_PATH = "/v1/pap/prefill/lease-release"


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


def _pap_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _pap_attention_pool_profile_enabled() -> bool:
    return _pap_env_flag("PAP_ATTENTION_POOL_PROFILE", False)


def _pap_kv_lease_profile_enabled() -> bool:
    return _pap_env_flag("PAP_KV_LEASE_PROFILE", False)


def _pap_kv_locality_profile_enabled() -> bool:
    return _pap_env_flag("PAP_KV_LOCALITY_PROFILE", False)

_KV_LOCALITY_PROFILE_SEEN: set[tuple[str, str]] = set()


def _block_locality_stats(
    *,
    block_ids: tuple[int, ...],
    seq_len: int,
    block_size: int,
) -> dict[str, float | int | list[int]]:
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


def _log_kv_locality_profile(
    *,
    mode: str,
    layer_name: str,
    states: list[PAPUnifiedPagedKVState],
    kv_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layout: str,
) -> None:
    if not _pap_kv_locality_profile_enabled():
        return
    min_batch = int(os.environ.get("PAP_KV_LOCALITY_PROFILE_MIN_BATCH", "1"))
    if len(states) < min_batch:
        return
    profile_key = (str(mode), str(layer_name))
    if profile_key in _KV_LOCALITY_PROFILE_SEEN:
        return
    _KV_LOCALITY_PROFILE_SEEN.add(profile_key)

    rows = []
    for state in states:
        rows.append(
            _block_locality_stats(
                block_ids=tuple(int(block_id) for block_id in state.block_ids),
                seq_len=int(state.seq_len),
                block_size=int(state.block_size),
            )
        )
    first = rows[0]

    def avg(name: str) -> float:
        values = [float(row[name]) for row in rows]
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


def _trace_add_elapsed_ms(
    trace_stats: dict[str, float] | None,
    key: str,
    start: float,
) -> None:
    if trace_stats is not None:
        trace_stats[key] = (
            trace_stats.get(key, 0.0) + (time.perf_counter() - start) * 1000.0
        )

@dataclass
class PAPAttentionSession:
    """Snapshot of one PAP request known by the Attention executor."""

    request_id: str
    conversation_id: str
    prefill_endpoint: str
    kv_transfer_params: dict[str, Any]
    prefix_len: int | None
    block_size: int
    max_seq_len: int
    block_ids: tuple[int, ...] = ()
    seq_len: int = 0
    created_at: float = field(default_factory=time.time)
    role: str = "attention"
    prefill_kv_handle: str = ""
    decode_seq_lens: dict[str, int] = field(default_factory=dict)
    prefill_seq_lens: dict[str, int] = field(default_factory=dict)
    q_size: int | None = None
    kv_size: int | None = None

    def copy(self) -> PAPAttentionSession:
        return PAPAttentionSession(
            request_id=self.request_id,
            conversation_id=self.conversation_id,
            prefill_endpoint=self.prefill_endpoint,
            kv_transfer_params=dict(self.kv_transfer_params),
            prefix_len=self.prefix_len,
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            block_ids=tuple(self.block_ids),
            seq_len=self.seq_len,
            created_at=self.created_at,
            role=self.role,
            prefill_kv_handle=self.prefill_kv_handle,
            decode_seq_lens=dict(self.decode_seq_lens),
            prefill_seq_lens=dict(self.prefill_seq_lens),
            q_size=self.q_size,
            kv_size=self.kv_size,
        )

@dataclass
class PAPPrefillLayerReadiness:
    """Import readiness for one Prefill KV descriptor."""

    request_id: str
    layer_name: str
    descriptor_received: bool = False
    descriptor_opened: bool = False
    ready: bool = False
    failed: bool = False
    error: str = ""
    received_at: float = field(default_factory=time.perf_counter)
    opened_at: float = 0.0
    ready_at: float = 0.0
    failed_at: float = 0.0

    def copy(self) -> PAPPrefillLayerReadiness:
        return PAPPrefillLayerReadiness(
            request_id=self.request_id,
            layer_name=self.layer_name,
            descriptor_received=self.descriptor_received,
            descriptor_opened=self.descriptor_opened,
            ready=self.ready,
            failed=self.failed,
            error=self.error,
            received_at=self.received_at,
            opened_at=self.opened_at,
            ready_at=self.ready_at,
            failed_at=self.failed_at,
        )

@dataclass(frozen=True)
class PAPPrefillKVCacheCatalogEntry:
    """Opened process-lifetime Prefill KV-cache backing for one layer."""

    catalog_id: str
    layer_name: str
    kv_cache: torch.Tensor
    block_size: int
    num_kv_heads: int
    layout: str

@dataclass
class PAPUnifiedPagedKVState:
    """Unified Prefill-owned KV state for unified-KV mode.

    The same physical KV cache backs prefix and decode suffix. Attention writes
    decode K/V into the writable range [writable_start_token, writable_end_token)
    via leased block IDs, and reads the full [0, seq_len) range for FA compute.
    """

    kv_cache: torch.Tensor
    block_ids: tuple[int, ...]
    prefix_len: int
    seq_len: int
    capacity_tokens: int
    writable_start_token: int
    writable_end_token: int
    lease_id: str
    block_size: int
    num_kv_heads: int
    layout: str
    slot_generation: int = 0
    slot_topology_id: int = 0

PAPUnifiedSlotTopology = tuple[tuple[int, ...], int, str]

@dataclass
class PAPUnifiedSlotActivation:
    """Generation-bound topology observations for one unified-KV session."""

    prefix_len: int
    generation: int
    canonical_topology: PAPUnifiedSlotTopology
    canonical_topology_id: int
    topology_ids: dict[PAPUnifiedSlotTopology, int]
    layer_observations: dict[str, int]
    expected_layers: frozenset[str] | None = None
    conflict_latched: bool = False
    complete: bool = True

@dataclass(frozen=True)
class PAPPagedFlashMetadata:
    """Batched metadata tensors consumed by paged FlashAttention."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seq_len: int

@dataclass(frozen=True)
class PAPOffloadExecSessionEntry:
    """Shape metadata for one OFFLOAD_EXEC request in a batch."""

    session_request_id: str
    prefill_endpoint: str
    q_size: int
    kv_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

_UNIFIED_MD_CACHE: OrderedDict[tuple[Any, ...], PAPPagedFlashMetadata] = OrderedDict()

_UNIFIED_MD_CU_SEQLENS_Q: dict[tuple[str, int], torch.Tensor] = {}

_UNIFIED_MD_CACHE_HITS = 0

_UNIFIED_MD_CACHE_MISSES = 0

_UNIFIED_MD_FAST_KEY_LOOKUPS = 0

_UNIFIED_MD_FAST_KEY_HITS = 0

_UNIFIED_MD_FULL_KEY_SCANS = 0

_UNIFIED_MD_BLOCK_IDS_SCANNED = 0

_UNIFIED_MD_CACHE_LOCK = Lock()

_UNIFIED_SLOT_TOPOLOGY_ID_LOCK = Lock()

_UNIFIED_SLOT_TOPOLOGY_ID_NEXT = 1


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


def _allocate_unified_slot_topology_id() -> int:
    global _UNIFIED_SLOT_TOPOLOGY_ID_NEXT
    with _UNIFIED_SLOT_TOPOLOGY_ID_LOCK:
        topology_id = _UNIFIED_SLOT_TOPOLOGY_ID_NEXT
        _UNIFIED_SLOT_TOPOLOGY_ID_NEXT += 1
    return topology_id


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
        session_request_id = self.resolve_session_request_id(request_id)
        if session_request_id is None:
            raise KeyError(request_id)
        return self._decode_token_committer.record_token(
            request_id=session_request_id,
            new_seq_len=new_seq_len,
            token_ids=(token_id,),
        )

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


    @staticmethod
    def _decode_slot_plan_cache_limit() -> int:
        return int(os.environ.get("PAP_DECODE_SLOT_PLAN_CACHE_LIMIT", "256"))

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
            timeout_s=float(
                os.environ.get("PAP_DECODE_TOKEN_FLUSH_TIMEOUT", "5.0")
            ),
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
            deadline = time.monotonic() + float(
                os.environ.get("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "5.0")
            )
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
                manifest.request_id
            )
            if session_request_id is None:
                raise KeyError(manifest.request_id)
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
                timeout_s=float(
                    os.environ.get("PAP_DECODE_TOKEN_FLUSH_TIMEOUT", "5.0")
                ),
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
            self._session_epochs[registration.request_id] = self._next_session_epoch
            self._next_session_epoch += 1
            self._sessions[registration.request_id] = session
            self._prefill_readiness.setdefault(registration.request_id, {})
            self._request_id_resolution_cache[registration.request_id] = (
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


def open_ipc_tensor_handle(handle: PAPCudaIPCTensorHandle) -> torch.Tensor:
    """Open one CUDA IPC tensor handle on the current physical GPU."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    device_index = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device_index)
    physical_gpu_id = str(props.uuid)
    ipc_handle = handle.ipc_handle
    if physical_gpu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for GPU UUID {physical_gpu_id}. "
            f"Available UUIDs: {list(ipc_handle.keys())}"
        )
    args = list(ipc_handle[physical_gpu_id])
    args[6] = device_index
    return rebuild_cuda_tensor(*args)


def open_prefill_manifest_event(
    manifest: PAPPrefillKVSessionManifest,
) -> Any | None:
    """Open an interprocess CUDA event carried by a Prefill manifest."""
    if manifest.ready_event_handle is None:
        return None
    device_index = torch.accelerator.current_device_index()
    return torch.cuda.Event.from_ipc_handle(
        device_index,
        manifest.ready_event_handle,
    )
