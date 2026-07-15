# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention KV and session data models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any

import torch


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
    """Unified Prefill-owned KV state for one Attention layer."""

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
class PAPOffloadExecSessionEntry:
    """Shape metadata for one OFFLOAD_EXEC request in a batch."""

    session_request_id: str
    prefill_endpoint: str
    q_size: int
    kv_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass
class PAPAttentionStepContext:
    """Decode-step state shared by every remote Attention layer."""

    cache_key: tuple[Any, ...]
    request_ids: tuple[str, ...]
    decode_seq_lens: tuple[int, ...]
    session_entries: tuple[PAPOffloadExecSessionEntry, ...]
    prior_seq_lens: tuple[int, ...]
    result_seq_lens: tuple[int, ...]
    commit_new_seq_lens: tuple[int | None, ...]
    active_indices: tuple[int, ...]
    expected_layers: frozenset[str]
    layer_states: dict[str, tuple[PAPUnifiedPagedKVState, ...]]
    topology_ids: tuple[int, ...]
    q_size: int
    kv_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    slot_tensor: torch.Tensor | None = None
    metadata: Any | None = None
    completed_layers: set[str] = field(default_factory=set)
    kv_ready_published: bool = False
    lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def session_request_ids(self) -> tuple[str, ...]:
        """Resolved registry request ids in batch row order."""
        return tuple(entry.session_request_id for entry in self.session_entries)


_UNIFIED_SLOT_TOPOLOGY_ID_LOCK = Lock()
_UNIFIED_SLOT_TOPOLOGY_ID_NEXT = 1


def allocate_unified_slot_topology_id() -> int:
    """Return a process-local identity for one observed slot topology."""
    global _UNIFIED_SLOT_TOPOLOGY_ID_NEXT
    with _UNIFIED_SLOT_TOPOLOGY_ID_LOCK:
        topology_id = _UNIFIED_SLOT_TOPOLOGY_ID_NEXT
        _UNIFIED_SLOT_TOPOLOGY_ID_NEXT += 1
    return topology_id
