# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention internal executor.

This first PAP slice keeps Attention as an internal compute endpoint. The
process is intentionally not an OpenAI-compatible vLLM server: it records which
Prefill KV handle belongs to which PAP request so the proxy and remote-attention
path have a stable control-plane contract.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import socket
import socketserver
import time
from dataclasses import dataclass, field
from queue import Queue
from threading import Condition, Lock, Thread
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from vllm.pap.attention_session import (
    AttentionDecodeDescriptor,
    AttentionSessionStore,
)
from vllm.pap.data_plane import (
    PAPOffloadKVIPCDescriptor,
    PAPOffloadKVPagedIPCDescriptor,
    build_local_fast_offload_exec_transport,
    build_nixl_mailbox_offload_exec_transport,
    pap_offload_exec_trace_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_attention")


def _pap_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _pap_attention_pool_profile_enabled() -> bool:
    return _pap_env_flag("PAP_ATTENTION_POOL_PROFILE", False)


def _pap_attention_copy_prefix_kv() -> bool:
    return _pap_env_flag("PAP_ATTENTION_COPY_PREFIX_KV", False)


def _pap_prefill_kv_async_enabled() -> bool:
    return _pap_env_flag("PAP_PREFILL_KV_ASYNC", False)


def _pap_prefill_ipc_profile_enabled() -> bool:
    return _pap_env_flag("PAP_PREFILL_IPC_PROFILE", False)


def _pap_kv_lease_profile_enabled() -> bool:
    return _pap_env_flag("PAP_KV_LEASE_PROFILE", False)


def _trace_add_elapsed_ms(
    trace_stats: dict[str, float] | None,
    key: str,
    start: float,
) -> None:
    if trace_stats is not None:
        trace_stats[key] = (
            trace_stats.get(key, 0.0) + (time.perf_counter() - start) * 1000.0
        )


def _local_paged_copy_source(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    return tensor.to(device=device, dtype=dtype, non_blocking=True)


def _local_paged_slot_mapping_tensor(
    *,
    local_blocks: list[int],
    block_offsets: list[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    if len(local_blocks) != len(block_offsets):
        raise ValueError("local paged slot mapping length mismatch")
    slots = [
        int(local_block) * int(block_size) + int(block_offset)
        for local_block, block_offset in zip(local_blocks, block_offsets)
    ]
    return torch.tensor(slots, dtype=torch.long, device=device)


class PAPAttentionRegistration(BaseModel):
    """Request metadata registered after Prefill completes."""

    request_id: str
    conversation_id: str = ""
    prefill_endpoint: str
    kv_transfer_params: dict[str, Any] = Field(default_factory=dict)
    prefix_len: int | None = None
    block_size: int = 16
    max_seq_len: int = 32768
    q_size: int | None = None
    kv_size: int | None = None


class PAPAttentionComputeRequest(BaseModel):
    """Tensor payload for the first PAP remote-attention compute path."""

    request_id: str
    layer_name: str
    query: dict[str, Any]
    key: dict[str, Any]
    value: dict[str, Any]
    scale: float


class PAPAttentionAppendAndComputeRequest(BaseModel):
    """Stateful decode attention request carrying only the current Q/K/V."""

    request_id: str
    layer_name: str
    query: dict[str, Any]
    key: dict[str, Any]
    value: dict[str, Any]
    scale: float
    block_id: int | None = None
    slot: int | None = None
    seq_len: int | None = None


class PAPAttentionImportPrefillKVRequest(BaseModel):
    """Prompt KV import for the stateful PAP attention path."""

    request_id: str
    layer_name: str
    key: dict[str, Any]
    value: dict[str, Any]
    seq_len: int
    block_ids: list[int] | None = None


class PAPOffloadExecRequest(BaseModel):
    """Control-plane trigger for one OFFLOAD_EXEC tensor exchange."""

    request_id: str
    layer_name: str
    step: int
    scale: float
    remote_address: str


class PAPOffloadExecMailboxBindRequest(BaseModel):
    """Projection NIXL mailbox metadata used for one-time OFFLOAD_EXEC bind."""

    agent_metadata_b64: str


class PAPAttentionLayerEventRequest(BaseModel):
    """Shape-only event emitted at Projection's q/k/v -> attention boundary."""

    request_id: str
    layer_name: str
    query_shape: list[int]
    key_shape: list[int]
    value_shape: list[int]
    dtype: str
    device: str
    is_decode: bool
    num_reqs: int | None = None
    num_actual_tokens: int | None = None
    max_seq_len: int | None = None


@dataclass
class PAPAttentionLayerEvent:
    """Trace event for one Projection-side layer attention invocation."""

    request_id: str
    session_request_id: str
    layer_name: str
    query_shape: list[int]
    key_shape: list[int]
    value_shape: list[int]
    dtype: str
    device: str
    is_decode: bool
    num_reqs: int | None
    num_actual_tokens: int | None
    max_seq_len: int | None
    created_at: float = field(default_factory=time.time)

    def copy(self) -> PAPAttentionLayerEvent:
        return PAPAttentionLayerEvent(
            request_id=self.request_id,
            session_request_id=self.session_request_id,
            layer_name=self.layer_name,
            query_shape=list(self.query_shape),
            key_shape=list(self.key_shape),
            value_shape=list(self.value_shape),
            dtype=self.dtype,
            device=self.device,
            is_decode=self.is_decode,
            num_reqs=self.num_reqs,
            num_actual_tokens=self.num_actual_tokens,
            max_seq_len=self.max_seq_len,
            created_at=self.created_at,
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
class PAPDecodeKVBuffer:
    """Growable per-request/layer decode KV storage."""

    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    @property
    def capacity(self) -> int:
        return int(self.key.shape[0])

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key[: self.length], self.value[: self.length]


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


@dataclass
class PAPPrefillPagedKV:
    """Paged KV backing opened from the Prefill-owned cache via IPC."""

    kv_cache: torch.Tensor
    block_ids: list[int]
    seq_len: int
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


@dataclass
class PAPPagedFlashMetadataScratch:
    """Reusable metadata buffers for paged FlashAttention."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    block_table_host: torch.Tensor | None = None
    seq_lens_host: torch.Tensor | None = None
    block_signature: tuple[tuple[int, ...], ...] | None = None


@dataclass
class PAPLocalPagedKVPool:
    """Attention-local paged KV cache shared by requests for one layer."""

    layer_name: str
    kv_cache: torch.Tensor
    block_size: int
    num_kv_heads: int
    head_dim: int
    next_block: int = 0
    free_blocks: list[int] = field(default_factory=list)
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None
    paged_metadata_scratch: PAPPagedFlashMetadataScratch | None = None


@dataclass(frozen=True)
class PAPLocalPagedAttentionState:
    """Per-request logical block table into an attention-local paged KV pool."""

    pool: PAPLocalPagedKVPool
    block_ids: tuple[int, ...]
    seq_len: int
    base_seq_len: int = 0
    prefix_state: PAPPrefillPagedKV | None = None

    @property
    def kv_cache(self) -> torch.Tensor:
        return self.pool.kv_cache

    @property
    def local_seq_len(self) -> int:
        return max(0, int(self.seq_len) - int(self.base_seq_len))

    @property
    def has_descriptor_prefix(self) -> bool:
        return self.prefix_state is not None and int(self.base_seq_len) > 0

    @property
    def block_size(self) -> int:
        return self.pool.block_size

    @property
    def num_kv_heads(self) -> int:
        return self.pool.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.pool.head_dim


@dataclass(frozen=True)
class PAPPagedFlashMetadata:
    """Batched metadata tensors consumed by paged FlashAttention."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seq_len: int


@dataclass(frozen=True)
class PAPLocalPagedDecodeAppend:
    """One decode-token append into the attention-local paged KV cache."""

    request_id: str
    layer_name: str
    key: torch.Tensor
    value: torch.Tensor
    seq_len: int


@dataclass(frozen=True)
class PAPOffloadExecSessionEntry:
    """Shape metadata for one OFFLOAD_EXEC request in a batch."""

    session_request_id: str
    q_size: int
    kv_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def _local_paged_cache_scale(pool: PAPLocalPagedKVPool, attr: str) -> torch.Tensor:
    scale = getattr(pool, attr)
    if (
        scale is None
        or scale.device != pool.kv_cache.device
        or scale.dtype != torch.float32
        or scale.numel() != 1
    ):
        scale = torch.ones((), dtype=torch.float32, device=pool.kv_cache.device)
        setattr(pool, attr, scale)
    return scale


def _try_local_paged_native_cache_append(
    *,
    pool: PAPLocalPagedKVPool,
    key_batch: torch.Tensor,
    value_batch: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    if slot_mapping.numel() <= 0:
        return True
    if pool.kv_cache.device.type != "cuda":
        return False
    if key_batch.device != pool.kv_cache.device:
        return False
    if value_batch.device != pool.kv_cache.device:
        return False
    if key_batch.dtype != pool.kv_cache.dtype:
        return False
    if value_batch.dtype != pool.kv_cache.dtype:
        return False
    if slot_mapping.device != pool.kv_cache.device:
        return False

    key_cache, value_cache = pool.kv_cache.unbind(1)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key_batch,
        value_batch,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        _local_paged_cache_scale(pool, "k_scale"),
        _local_paged_cache_scale(pool, "v_scale"),
    )
    return True


def _device_matches_request(actual: torch.device, requested: torch.device) -> bool:
    requested = torch.device(requested)
    if actual.type != requested.type:
        return False
    if requested.index is None:
        return True
    return actual.index == requested.index


def _get_or_create_paged_flash_metadata_scratch(
    *,
    pool: PAPLocalPagedKVPool,
    batch_size: int,
    max_blocks: int,
    device: torch.device,
) -> PAPPagedFlashMetadataScratch:
    scratch = pool.paged_metadata_scratch
    if (
        scratch is not None
        and _device_matches_request(scratch.block_table.device, device)
        and _device_matches_request(scratch.seq_lens.device, device)
        and _device_matches_request(scratch.cu_seqlens_q.device, device)
        and tuple(scratch.block_table.shape) == (batch_size, max_blocks)
        and tuple(scratch.seq_lens.shape) == (batch_size,)
        and tuple(scratch.cu_seqlens_q.shape) == (batch_size + 1,)
    ):
        return scratch

    block_table = torch.empty(
        (batch_size, max_blocks),
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(
        0,
        batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    block_table_host = None
    seq_lens_host = None
    if device.type == "cuda":
        block_table_host = torch.empty(
            (batch_size, max_blocks),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        seq_lens_host = torch.empty(
            (batch_size,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
    scratch = PAPPagedFlashMetadataScratch(
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        block_table_host=block_table_host,
        seq_lens_host=seq_lens_host,
    )
    pool.paged_metadata_scratch = scratch
    return scratch


def _paged_flash_effective_seq_len(
    state: PAPLocalPagedAttentionState | PAPPrefillPagedKV,
    *,
    use_base_seq_len: bool,
) -> int:
    seq_len = int(state.seq_len)
    if use_base_seq_len:
        seq_len -= int(getattr(state, "base_seq_len", 0))
    if seq_len <= 0:
        raise ValueError("paged FlashAttention effective seq_len must be positive")
    return seq_len


def _fill_paged_flash_metadata_scratch(
    *,
    scratch: PAPPagedFlashMetadataScratch,
    states: list[PAPLocalPagedAttentionState],
    max_blocks: int,
    use_base_seq_len: bool,
) -> int:
    if scratch.seq_lens_host is not None:
        seq_target = scratch.seq_lens_host
    else:
        seq_target = scratch.seq_lens

    signature_matches = scratch.block_signature is not None and (
        len(scratch.block_signature) == len(states)
    )
    max_seq_len = 0
    for index, state in enumerate(states):
        block_ids = state.block_ids
        if not block_ids:
            raise ValueError("paged FlashAttention state has no blocks")
        seq_len = _paged_flash_effective_seq_len(
            state,
            use_base_seq_len=use_base_seq_len,
        )
        seq_target[index] = seq_len
        if seq_len > max_seq_len:
            max_seq_len = seq_len
        if signature_matches and scratch.block_signature[index] != block_ids:
            signature_matches = False
    if seq_target is not scratch.seq_lens:
        scratch.seq_lens.copy_(seq_target, non_blocking=True)

    if signature_matches:
        return max_seq_len

    if scratch.block_table_host is not None:
        block_target = scratch.block_table_host
    else:
        block_target = scratch.block_table

    next_signature: list[tuple[int, ...]] = []
    for row_index, state in enumerate(states):
        block_ids = state.block_ids
        next_signature.append(block_ids)
        last_block = int(block_ids[-1])
        for column_index in range(max_blocks):
            if column_index < len(block_ids):
                block_target[row_index, column_index] = int(block_ids[column_index])
            else:
                block_target[row_index, column_index] = last_block
    if block_target is not scratch.block_table:
        scratch.block_table.copy_(block_target, non_blocking=True)
    scratch.block_signature = tuple(next_signature)
    return max_seq_len


def build_paged_flash_metadata(
    *,
    states: list[PAPLocalPagedAttentionState | PAPPrefillPagedKV],
    device: torch.device,
    use_base_seq_len: bool = False,
) -> PAPPagedFlashMetadata:
    batch_size = len(states)
    if batch_size <= 0:
        raise ValueError("paged FlashAttention metadata requires at least one state")
    max_blocks = max(len(state.block_ids) for state in states)
    if max_blocks <= 0:
        raise ValueError("paged FlashAttention metadata requires non-empty blocks")

    same_local_pool = all(isinstance(state, PAPLocalPagedAttentionState) for state in states)
    use_scratch = False
    if same_local_pool:
        local_states = [state for state in states if isinstance(state, PAPLocalPagedAttentionState)]
        pool = local_states[0].pool
        same_pool = all(state.pool is pool for state in local_states)
        use_scratch = same_pool and (
            device.type != "cuda"
            or _pap_env_flag("PAP_ATTENTION_PAGED_METADATA_SCRATCH_CUDA", False)
        )
        if use_scratch:
            scratch = _get_or_create_paged_flash_metadata_scratch(
                pool=pool,
                batch_size=batch_size,
                max_blocks=max_blocks,
                device=device,
            )
            max_seq_len = _fill_paged_flash_metadata_scratch(
                scratch=scratch,
                states=local_states,
                max_blocks=max_blocks,
                use_base_seq_len=use_base_seq_len,
            )
            return PAPPagedFlashMetadata(
                block_table=scratch.block_table,
                seq_lens=scratch.seq_lens,
                cu_seqlens_q=scratch.cu_seqlens_q,
                max_seq_len=max_seq_len,
            )

    block_rows: list[list[int]] = []
    seq_lens: list[int] = []
    max_seq_len = 0
    for state in states:
        if not state.block_ids:
            raise ValueError("paged FlashAttention state has no blocks")
        row = list(state.block_ids)
        if len(row) < max_blocks:
            row.extend([row[-1]] * (max_blocks - len(row)))
        block_rows.append(row)
        seq_len = _paged_flash_effective_seq_len(
            state,
            use_base_seq_len=use_base_seq_len,
        )
        seq_lens.append(seq_len)
        if seq_len > max_seq_len:
            max_seq_len = seq_len

    return PAPPagedFlashMetadata(
        block_table=torch.tensor(block_rows, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.arange(
            0,
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        ),
        max_seq_len=max_seq_len,
    )


def build_unified_paged_flash_metadata(
    *,
    states: list[PAPUnifiedPagedKVState],
    device: torch.device,
) -> PAPPagedFlashMetadata:
    """Build FA metadata for a single-source Prefill-owned paged KV batch.

    Mirrors build_paged_flash_metadata but consumes PAPUnifiedPagedKVState and
    skips scratch optimization (Stage 4 scaffolding: simple per-call tensors).
    """
    batch_size = len(states)
    if batch_size <= 0:
        raise ValueError(
            "unified paged FlashAttention metadata requires at least one state"
        )
    max_blocks = max(len(state.block_ids) for state in states)
    if max_blocks <= 0:
        raise ValueError(
            "unified paged FlashAttention metadata requires non-empty blocks"
        )
    block_table = torch.zeros(
        (batch_size, max_blocks), dtype=torch.int32, device=device
    )
    seq_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    max_seq_len = 0
    for row_index, state in enumerate(states):
        if not state.block_ids:
            raise ValueError("unified state has no blocks")
        last_block = int(state.block_ids[-1])
        for column_index in range(max_blocks):
            if column_index < len(state.block_ids):
                block_table[row_index, column_index] = int(
                    state.block_ids[column_index]
                )
            else:
                block_table[row_index, column_index] = last_block
        seq_lens[row_index] = int(state.seq_len)
        if int(state.seq_len) > max_seq_len:
            max_seq_len = int(state.seq_len)
    cu_seqlens_q = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    )
    return PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_seq_len=max_seq_len,
    )


class PAPAttentionRegistry:
    """Thread-safe in-memory registry for PAP Attention control-plane state."""

    def __init__(self, storage_device: str | torch.device | None = None) -> None:
        self._lock = Lock()
        self._prefill_condition = Condition(self._lock)
        self._storage_device = self._resolve_storage_device(storage_device)
        self._sessions: dict[str, PAPAttentionSession] = {}
        self._layer_events: dict[str, list[PAPAttentionLayerEvent]] = {}
        self._decode_kv: dict[str, dict[str, PAPDecodeKVBuffer]] = {}
        self._prefill_kv: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor]]]
        ] = {}
        self._prefill_paged_kv: dict[str, dict[str, PAPPrefillPagedKV]] = {}
        self._prefill_readiness: dict[
            str, dict[str, PAPPrefillLayerReadiness]
        ] = {}
        self._prefill_async_queue: Queue[
            tuple[PAPOffloadKVPagedIPCDescriptor, float]
        ] = Queue()
        self._prefill_async_worker_started = False
        self._local_paged_kv_pools: dict[str, PAPLocalPagedKVPool] = {}
        self._local_paged_kv: dict[str, dict[str, PAPLocalPagedAttentionState]] = {}
        self._request_id_resolution_cache: dict[str, str] = {}
        self._session_lease_ids: dict[str, str] = {}
        self._session_leased_block_ids: dict[str, tuple[int, ...]] = {}
        self._session_lease_capacity_tokens: dict[str, int] = {}
        self._unified_paged_kv: dict[str, dict[str, PAPUnifiedPagedKVState]] = {}
        self._offload_exec_session_entry_cache: dict[
            tuple[str, int, int, int, int, int], PAPOffloadExecSessionEntry
        ] = {}
        self._attention_sessions = AttentionSessionStore()

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
    def _initial_decode_capacity(num_tokens: int) -> int:
        configured = int(
            os.environ.get("PAP_ATTENTION_DECODE_KV_INITIAL_CAPACITY", 128)
        )
        return max(int(num_tokens), configured)

    @staticmethod
    def _make_decode_buffer(
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        capacity: int,
    ) -> PAPDecodeKVBuffer:
        return PAPDecodeKVBuffer(
            key=torch.empty(
                (capacity, *key.shape[1:]),
                dtype=key.dtype,
                device=key.device,
            ),
            value=torch.empty(
                (capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            ),
        )

    def _ensure_decode_capacity(
        self,
        buffer: PAPDecodeKVBuffer,
        *,
        required: int,
    ) -> PAPDecodeKVBuffer:
        if required <= buffer.capacity:
            return buffer
        new_capacity = max(required, buffer.capacity * 2)
        grown = self._make_decode_buffer(
            key=buffer.key,
            value=buffer.value,
            capacity=new_capacity,
        )
        if buffer.length > 0:
            grown.key[: buffer.length].copy_(buffer.key[: buffer.length])
            grown.value[: buffer.length].copy_(buffer.value[: buffer.length])
            grown.length = buffer.length
        return grown

    @staticmethod
    @staticmethod
    def _local_paged_cache_enabled() -> bool:
        return True

    def _get_or_create_local_paged_pool_locked(
        self,
        *,
        layer_name: str,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        min_blocks: int,
    ) -> PAPLocalPagedKVPool:
        pool = self._local_paged_kv_pools.get(layer_name)
        if pool is not None:
            if (
                pool.block_size != int(block_size)
                or pool.num_kv_heads != int(num_kv_heads)
                or pool.head_dim != int(head_dim)
                or pool.kv_cache.dtype != dtype
                or pool.kv_cache.device != device
            ):
                raise ValueError("local paged KV pool shape does not match layer")
            additional_blocks = max(0, int(min_blocks) - len(pool.free_blocks))
            self._ensure_local_paged_pool_capacity_locked(
                pool,
                required_blocks=pool.next_block + additional_blocks,
            )
            return pool

        configured_blocks = int(
            os.environ.get("PAP_ATTENTION_LOCAL_PAGED_POOL_INITIAL_BLOCKS", "512")
        )
        initial_blocks = max(int(min_blocks), configured_blocks)
        kv_cache = torch.empty(
            (initial_blocks, 2, int(block_size), int(num_kv_heads), int(head_dim)),
            dtype=dtype,
            device=device,
        )
        pool = PAPLocalPagedKVPool(
            layer_name=layer_name,
            kv_cache=kv_cache,
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
        )
        self._local_paged_kv_pools[layer_name] = pool
        return pool

    @staticmethod
    def _ensure_local_paged_pool_capacity_locked(
        pool: PAPLocalPagedKVPool,
        *,
        required_blocks: int,
    ) -> None:
        if int(required_blocks) <= int(pool.kv_cache.shape[0]):
            return
        old_capacity = int(pool.kv_cache.shape[0])
        new_capacity = max(int(required_blocks), old_capacity * 2)
        grown = torch.empty(
            (new_capacity, *pool.kv_cache.shape[1:]),
            dtype=pool.kv_cache.dtype,
            device=pool.kv_cache.device,
        )
        if pool.next_block > 0:
            grown[: pool.next_block].copy_(pool.kv_cache[: pool.next_block])
        pool.kv_cache = grown
        if _pap_attention_pool_profile_enabled():
            logger.info(
                "PAP local paged KV pool grow layer=%s old_capacity=%d "
                "new_capacity=%d next_block=%d free_count=%d",
                pool.layer_name,
                old_capacity,
                new_capacity,
                int(pool.next_block),
                len(pool.free_blocks),
            )

    def _allocate_local_paged_blocks_locked(
        self,
        pool: PAPLocalPagedKVPool,
        *,
        count: int,
    ) -> tuple[int, ...]:
        count = int(count)
        if count <= 0:
            return ()

        reused_count = min(count, len(pool.free_blocks))
        reused = []
        if reused_count > 0:
            reused = pool.free_blocks[-reused_count:]
            del pool.free_blocks[-reused_count:]

        new_count = count - reused_count
        if new_count <= 0:
            return tuple(reused)

        start = pool.next_block
        end = start + new_count
        self._ensure_local_paged_pool_capacity_locked(pool, required_blocks=end)
        pool.next_block = end
        return (*reused, *range(start, end))

    def _release_local_paged_blocks_locked(self, session_request_id: str) -> None:
        layer_states = self._local_paged_kv.get(session_request_id, {})
        if not layer_states:
            return

        released_layers = 0
        released_blocks = 0
        free_count_by_layer: list[str] = []
        for layer_name, state in list(layer_states.items()):
            released_layers += 1
            block_ids = tuple(int(block_id) for block_id in state.block_ids)
            if block_ids:
                state.pool.free_blocks.extend(block_ids)
                released_blocks += len(block_ids)
            free_count_by_layer.append(
                f"{layer_name}:{len(state.pool.free_blocks)}"
            )

        self._local_paged_kv.pop(session_request_id, None)
        if _pap_attention_pool_profile_enabled():
            logger.info(
                "PAP local paged KV pool release request_id=%s layers=%d "
                "blocks=%d free_counts=%s",
                session_request_id,
                released_layers,
                released_blocks,
                ",".join(free_count_by_layer) if free_count_by_layer else "none",
            )

    def _release_session_locked(self, request_id: str) -> bool:
        existed = self._sessions.pop(request_id, None) is not None
        self._layer_events.pop(request_id, None)
        self._decode_kv.pop(request_id, None)
        self._prefill_kv.pop(request_id, None)
        self._prefill_paged_kv.pop(request_id, None)
        self._prefill_readiness.pop(request_id, None)
        self._unified_paged_kv.pop(request_id, None)
        self._release_local_paged_blocks_locked(request_id)
        self._drop_offload_exec_session_entry_cache_locked(request_id)
        lease_id = self._session_lease_ids.pop(request_id, None)
        leased_blocks = self._session_leased_block_ids.pop(request_id, None)
        self._session_lease_capacity_tokens.pop(request_id, None)
        if lease_id is not None and _pap_kv_lease_profile_enabled():
            logger.info(
                "PAP Attention release session lease request_id=%s "
                "lease_id=%s leased_blocks=%d (Prefill TTL sweep will free)",
                request_id,
                lease_id,
                len(leased_blocks or ()),
            )
        for cached_request_id, cached_session_id in list(
            self._request_id_resolution_cache.items()
        ):
            if cached_request_id == request_id or cached_session_id == request_id:
                self._request_id_resolution_cache.pop(cached_request_id, None)
        self._attention_sessions.free_session(request_id)
        return existed

    def _replace_existing_session_locked(self, request_id: str) -> None:
        if request_id in self._sessions:
            self._release_session_locked(request_id)
            return

        self._release_local_paged_blocks_locked(request_id)
        self._layer_events.pop(request_id, None)
        self._decode_kv.pop(request_id, None)
        self._prefill_kv.pop(request_id, None)
        self._prefill_paged_kv.pop(request_id, None)
        self._prefill_readiness.pop(request_id, None)
        self._unified_paged_kv.pop(request_id, None)
        self._drop_offload_exec_session_entry_cache_locked(request_id)
        for cached_request_id, cached_session_id in list(
            self._request_id_resolution_cache.items()
        ):
            if cached_request_id == request_id or cached_session_id == request_id:
                self._request_id_resolution_cache.pop(cached_request_id, None)
        self._attention_sessions.free_session(request_id)
        self._sessions.pop(request_id, None)
        self._request_id_resolution_cache.pop(request_id, None)

    def release_session(self, request_id: str) -> bool:
        with self._lock:
            return self._release_session_locked(request_id)

    @staticmethod
    def _copy_segments_to_local_paged_blocks(
        *,
        pool: PAPLocalPagedKVPool,
        block_ids: tuple[int, ...],
        segments: list[tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
    ) -> None:
        copied = 0
        for segment_key, segment_value in segments:
            segment_len = int(segment_key.shape[0])
            segment_offset = 0
            while segment_offset < segment_len and copied < int(seq_len):
                position = copied
                logical_block = position // pool.block_size
                block_offset = position % pool.block_size
                take = min(
                    segment_len - segment_offset,
                    pool.block_size - block_offset,
                    int(seq_len) - copied,
                )
                local_block = block_ids[logical_block]
                pool.kv_cache[
                    local_block,
                    0,
                    block_offset : block_offset + take,
                    : pool.num_kv_heads,
                    :,
                ].copy_(
                    segment_key[segment_offset : segment_offset + take].to(
                        device=pool.kv_cache.device,
                        dtype=pool.kv_cache.dtype,
                    )
                )
                pool.kv_cache[
                    local_block,
                    1,
                    block_offset : block_offset + take,
                    : pool.num_kv_heads,
                    :,
                ].copy_(
                    segment_value[segment_offset : segment_offset + take].to(
                        device=pool.kv_cache.device,
                        dtype=pool.kv_cache.dtype,
                    )
                )
                segment_offset += take
                copied += take
            if copied >= int(seq_len):
                break

    def _install_local_paged_prefill_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        segments: list[tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        block_size: int,
        num_kv_heads: int,
    ) -> None:
        if int(seq_len) <= 0:
            return
        non_empty_segments = [
            (segment_key, segment_value)
            for segment_key, segment_value in segments
            if segment_key.numel() > 0
        ]
        if not non_empty_segments:
            return
        first_key = non_empty_segments[0][0]
        head_dim = int(first_key.shape[-1])
        required_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
        state = self._local_paged_kv.setdefault(session_request_id, {}).get(layer_name)
        min_blocks = required_blocks
        if state is not None:
            min_blocks = max(required_blocks, len(state.block_ids))
        pool = self._get_or_create_local_paged_pool_locked(
            layer_name=layer_name,
            block_size=int(block_size),
            num_kv_heads=int(num_kv_heads),
            head_dim=head_dim,
            dtype=first_key.dtype,
            device=first_key.device,
            min_blocks=min_blocks,
        )
        if state is not None and state.pool is pool:
            block_ids = state.block_ids
            if len(block_ids) < required_blocks:
                block_ids = (
                    *block_ids,
                    *self._allocate_local_paged_blocks_locked(
                        pool,
                        count=required_blocks - len(block_ids),
                    ),
                )
        else:
            block_ids = self._allocate_local_paged_blocks_locked(
                pool,
                count=required_blocks,
            )

        self._copy_segments_to_local_paged_blocks(
            pool=pool,
            block_ids=block_ids,
            segments=non_empty_segments,
            seq_len=int(seq_len),
        )
        state_seq_len = max(int(seq_len), int(state.seq_len) if state else 0)
        self._local_paged_kv.setdefault(session_request_id, {})[layer_name] = (
            PAPLocalPagedAttentionState(
                pool=pool,
                block_ids=tuple(block_ids),
                seq_len=state_seq_len,
            )
        )

    def _local_prefix_state_locked(
        self,
        session_request_id: str,
        layer_name: str,
    ) -> PAPPrefillPagedKV | None:
        return self._prefill_paged_kv.setdefault(session_request_id, {}).get(layer_name)

    def _canonicalize_local_paged_state_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        state: PAPLocalPagedAttentionState,
    ) -> PAPLocalPagedAttentionState:
        prefix_state = self._local_prefix_state_locked(session_request_id, layer_name)
        if prefix_state is None:
            return state
        base_seq_len = int(prefix_state.seq_len)
        if (
            state.prefix_state is prefix_state
            and int(state.base_seq_len) == base_seq_len
        ):
            return state
        canonical_state = PAPLocalPagedAttentionState(
            pool=state.pool,
            block_ids=tuple(state.block_ids),
            seq_len=int(state.seq_len),
            base_seq_len=base_seq_len,
            prefix_state=prefix_state,
        )
        self._local_paged_kv.setdefault(session_request_id, {})[layer_name] = (
            canonical_state
        )
        return canonical_state

    def _ensure_local_paged_decode_state_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        key: torch.Tensor,
        seq_len: int,
    ) -> PAPLocalPagedAttentionState:
        state = self._local_paged_kv.setdefault(session_request_id, {}).get(layer_name)
        prefix_state = self._local_prefix_state_locked(session_request_id, layer_name)
        if state is not None:
            state = self._canonicalize_local_paged_state_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
                state=state,
            )
            if prefix_state is not None:
                base_seq_len = int(prefix_state.seq_len)
                if int(state.seq_len) < max(int(seq_len), base_seq_len):
                    state = PAPLocalPagedAttentionState(
                        pool=state.pool,
                        block_ids=tuple(state.block_ids),
                        seq_len=max(int(seq_len), int(state.seq_len), base_seq_len),
                        base_seq_len=base_seq_len,
                        prefix_state=prefix_state,
                    )
                    self._local_paged_kv.setdefault(session_request_id, {})[
                        layer_name
                    ] = state
            return state

        base_seq_len = int(prefix_state.seq_len) if prefix_state is not None else 0
        block_size = (
            int(prefix_state.block_size)
            if prefix_state is not None
            else max(1, int(key.shape[0]))
        )
        num_kv_heads = (
            int(prefix_state.num_kv_heads)
            if prefix_state is not None
            else int(key.shape[1])
        )
        head_dim = int(key.shape[-1])
        local_seq_len = max(0, int(seq_len) - base_seq_len)
        required_blocks = (
            (local_seq_len + block_size - 1) // block_size if local_seq_len > 0 else 0
        )
        pool: PAPLocalPagedKVPool | None = None
        block_ids: tuple[int, ...] = ()
        if required_blocks > 0:
            pool = self._get_or_create_local_paged_pool_locked(
                layer_name=layer_name,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=key.dtype,
                device=key.device,
                min_blocks=required_blocks,
            )
            block_ids = self._allocate_local_paged_blocks_locked(
                pool,
                count=required_blocks,
            )
        else:
            pool = self._get_or_create_local_paged_pool_locked(
                layer_name=layer_name,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=key.dtype,
                device=key.device,
                min_blocks=0,
            )
        state = PAPLocalPagedAttentionState(
            pool=pool,
            block_ids=tuple(block_ids),
            seq_len=max(int(seq_len), base_seq_len),
            base_seq_len=base_seq_len,
            prefix_state=prefix_state,
        )
        self._local_paged_kv.setdefault(session_request_id, {})[layer_name] = state
        return state

    def _append_local_paged_decode_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> None:
        state = self._ensure_local_paged_decode_state_locked(
            session_request_id=session_request_id,
            layer_name=layer_name,
            key=key,
            seq_len=int(seq_len),
        )
        pool = state.pool
        num_tokens = int(key.shape[0])
        if num_tokens <= 0:
            return
        start_position = int(seq_len) - int(state.base_seq_len) - num_tokens
        if start_position < 0:
            raise ValueError("decode seq_len is shorter than decode KV tensor")
        local_seq_len = max(0, int(seq_len) - int(state.base_seq_len))
        required_blocks = (
            (local_seq_len + pool.block_size - 1) // pool.block_size
            if local_seq_len > 0
            else 0
        )
        block_ids = state.block_ids
        if len(block_ids) < required_blocks:
            block_ids = (
                *block_ids,
                *self._allocate_local_paged_blocks_locked(
                    pool,
                    count=required_blocks - len(block_ids),
                ),
            )

        for token_offset in range(num_tokens):
            position = start_position + token_offset
            logical_block = position // pool.block_size
            block_offset = position % pool.block_size
            local_block = block_ids[logical_block]
            pool.kv_cache[local_block, 0, block_offset, : pool.num_kv_heads, :].copy_(
                key[token_offset].to(
                    device=pool.kv_cache.device,
                    dtype=pool.kv_cache.dtype,
                )
            )
            pool.kv_cache[local_block, 1, block_offset, : pool.num_kv_heads, :].copy_(
                value[token_offset].to(
                    device=pool.kv_cache.device,
                    dtype=pool.kv_cache.dtype,
                )
            )
        self._local_paged_kv.setdefault(session_request_id, {})[layer_name] = (
            PAPLocalPagedAttentionState(
                pool=pool,
                block_ids=tuple(block_ids),
                seq_len=max(int(seq_len), state.seq_len),
                base_seq_len=int(state.base_seq_len),
                prefix_state=state.prefix_state,
            )
        )

    def local_paged_attention_state(
        self,
        *,
        request_id: str,
        layer_name: str,
    ) -> PAPLocalPagedAttentionState | None:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            state = self._local_paged_kv.setdefault(session_request_id, {}).get(
                layer_name
            )
            if state is None:
                return None
            return PAPLocalPagedAttentionState(
                pool=state.pool,
                block_ids=tuple(state.block_ids),
                seq_len=int(state.seq_len),
                base_seq_len=int(state.base_seq_len),
                prefix_state=state.prefix_state,
            )

    def has_descriptor_prefix_for_local_paged_attention(
        self,
        *,
        session_request_ids: tuple[str, ...],
        layer_name: str,
    ) -> bool:
        with self._lock:
            for session_request_id in session_request_ids:
                prefix_state = self._local_prefix_state_locked(
                    session_request_id,
                    layer_name,
                )
                if prefix_state is None:
                    continue
                state = self._local_paged_kv.setdefault(session_request_id, {}).get(
                    layer_name
                )
                if state is None or state.has_descriptor_prefix:
                    return True
            return False

    def materialize_descriptor_prefix_for_local_paged_attention(
        self,
        *,
        session_request_ids: tuple[str, ...],
        layer_name: str,
    ) -> None:
        with self._lock:
            for session_request_id in session_request_ids:
                prefix_state = self._local_prefix_state_locked(
                    session_request_id,
                    layer_name,
                )
                if prefix_state is None:
                    continue
                state = self._local_paged_kv.setdefault(session_request_id, {}).get(
                    layer_name
                )
                if state is not None and state.local_seq_len > 0:
                    raise RuntimeError(
                        "PAP cannot switch a live descriptor-prefix decode session "
                        "to copied-prefix fallback after local suffix blocks exist; "
                        "set PAP_ATTENTION_COPY_PREFIX_KV=1 before the session starts"
                    )
                segments = self._prefill_kv.setdefault(session_request_id, {}).get(
                    layer_name
                )
                if not segments:
                    raise RuntimeError(
                        "PAP copied-prefix fallback requires imported prefill KV "
                        "segments"
                    )
                self._install_local_paged_prefill_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                    segments=segments,
                    seq_len=int(prefix_state.seq_len),
                    block_size=int(prefix_state.block_size),
                    num_kv_heads=int(prefix_state.num_kv_heads),
                )
                self._prefill_paged_kv.setdefault(session_request_id, {}).pop(
                    layer_name, None
                )

    def append_decode_kv_to_unified_prefill_cache(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
        key_batch: torch.Tensor,
        value_batch: torch.Tensor,
        decode_seq_lens: Sequence[int],
    ) -> int:
        """Write a same-layer decode batch into Prefill-owned unified KV blocks.

        Each row writes into its own session's unified state. Returns the
        number of rows successfully appended.
        """
        unified_profile = _pap_env_flag("PAP_UNIFIED_KV_PROFILE", False)
        if len(session_request_ids) != len(decode_seq_lens):
            raise ValueError(
                "PAP unified KV append session_request_ids/decode_seq_lens length mismatch"
            )
        append_start = time.perf_counter() if unified_profile else 0.0
        written = 0
        with self._lock:
            for index, session_request_id in enumerate(session_request_ids):
                decode_len = int(decode_seq_lens[index])
                if decode_len <= 0:
                    continue
                layer_states = self._unified_paged_kv.get(session_request_id, {})
                state = layer_states.get(layer_name)
                if state is None:
                    raise RuntimeError(
                        f"PAP unified KV state missing for request_id="
                        f"{session_request_id} layer={layer_name}"
                    )
                block_size = int(state.block_size)
                kv_cache = state.kv_cache
                position = int(state.seq_len)
                if (
                    position < int(state.writable_start_token)
                    or position >= int(state.writable_end_token)
                ):
                    raise RuntimeError(
                        f"PAP unified KV append out of range request_id="
                        f"{session_request_id} layer={layer_name} position={position} "
                        f"writable=[{state.writable_start_token},"
                        f"{state.writable_end_token})"
                    )
                logical_block = position // block_size
                block_offset = position % block_size
                if logical_block >= len(state.block_ids):
                    raise RuntimeError(
                        f"PAP unified KV logical_block={logical_block} exceeds "
                        f"block_ids len={len(state.block_ids)} (request_id="
                        f"{session_request_id} layer={layer_name})"
                    )
                physical_block = int(state.block_ids[logical_block])
                key_row = key_batch[index].to(
                    device=kv_cache.device, dtype=kv_cache.dtype
                )
                value_row = value_batch[index].to(
                    device=kv_cache.device, dtype=kv_cache.dtype
                )
                try:
                    key_cache, value_cache = kv_cache.unbind(1)
                    slot = physical_block * block_size + block_offset
                    slot_tensor = torch.tensor(
                        [slot],
                        dtype=torch.int64,
                        device=kv_cache.device,
                    )
                    key_row_batch = key_row.unsqueeze(0).contiguous()
                    value_row_batch = value_row.unsqueeze(0).contiguous()
                    k_scale = torch.ones(
                        1, dtype=torch.float32, device=kv_cache.device
                    )
                    v_scale = torch.ones(
                        1, dtype=torch.float32, device=kv_cache.device
                    )
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        key_row_batch,
                        value_row_batch,
                        key_cache,
                        value_cache,
                        slot_tensor,
                        "auto",
                        k_scale,
                        v_scale,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"PAP unified KV append via reshape_and_cache_flash failed: {exc}"
                    )
                written += 1
                new_seq_len = max(int(state.seq_len), position + 1)
                layer_states[layer_name] = PAPUnifiedPagedKVState(
                    kv_cache=state.kv_cache,
                    block_ids=state.block_ids,
                    prefix_len=state.prefix_len,
                    seq_len=new_seq_len,
                    capacity_tokens=state.capacity_tokens,
                    writable_start_token=state.writable_start_token,
                    writable_end_token=state.writable_end_token,
                    lease_id=state.lease_id,
                    block_size=state.block_size,
                    num_kv_heads=state.num_kv_heads,
                    layout=state.layout,
                )
                if unified_profile:
                    logger.info(
                        "PAP unified KV append request_id=%s layer=%s "
                        "position=%d block=%d offset=%d seq_len=%d append_ms=%.3f",
                        session_request_id,
                        layer_name,
                        position,
                        physical_block,
                        block_offset,
                        new_seq_len,
                        (time.perf_counter() - append_start) * 1000.0,
                    )
        return written

    def get_unified_paged_states(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
    ) -> list[PAPUnifiedPagedKVState] | None:
        """Return per-row unified states if every row has unified state."""
        with self._lock:
            states: list[PAPUnifiedPagedKVState] = []
            for session_request_id in session_request_ids:
                layer_states = self._unified_paged_kv.get(session_request_id, {})
                state = layer_states.get(layer_name)
                if state is None:
                    return None
                states.append(state)
            return states

    def append_decode_kv_batch_for_local_paged_attention(
        self,
        *,
        entries: list[PAPLocalPagedDecodeAppend],
    ) -> list[PAPLocalPagedAttentionState] | None:
        """Append a same-layer decode batch directly to local paged KV.

        This is the paged FlashAttention fast path. It intentionally does not
        update the legacy growable decode segment buffer, which is only needed
        by the padded-SDPA fallback.
        """

        if not entries:
            return None

        with self._lock:
            prepared: list[
                tuple[
                    str,
                    PAPAttentionSession,
                    PAPLocalPagedDecodeAppend,
                    AttentionDecodeDescriptor,
                ]
            ] = []
            for entry in entries:
                session_request_id = self._resolve_session_request_id_locked(
                    entry.request_id
                )
                if session_request_id is None:
                    raise KeyError(entry.request_id)
                session = self._sessions[session_request_id]
                seq_len = int(entry.seq_len)
                if seq_len <= 0:
                    raise ValueError("decode seq_len must be positive")
                if int(entry.key.shape[0]) != 1 or int(entry.value.shape[0]) != 1:
                    return None

                self._wait_for_prefill_layer_locked(
                    session_request_id=session_request_id,
                    session=session,
                    layer_name=entry.layer_name,
                    decode_seq_len=seq_len,
                )
                local_state = self._local_paged_kv.setdefault(
                    session_request_id, {}
                ).get(entry.layer_name)
                if local_state is None:
                    local_state = self._ensure_local_paged_decode_state_locked(
                        session_request_id=session_request_id,
                        layer_name=entry.layer_name,
                        key=entry.key,
                        seq_len=seq_len,
                    )
                else:
                    local_state = self._canonicalize_local_paged_state_locked(
                        session_request_id=session_request_id,
                        layer_name=entry.layer_name,
                        state=local_state,
                    )

                block_id = (seq_len - 1) // session.block_size
                slot = block_id * session.block_size + (
                    (seq_len - 1) % session.block_size
                )
                prepared.append(
                    (
                        session_request_id,
                        session,
                        entry,
                        AttentionDecodeDescriptor(
                            request_id=session_request_id,
                            block_id=block_id,
                            slot=slot,
                            seq_len=seq_len,
                        ),
                    )
                )

            write_groups: dict[
                tuple[int, int],
                tuple[
                    PAPLocalPagedKVPool,
                    list[int],
                    list[torch.Tensor],
                    list[torch.Tensor],
                ],
            ] = {}
            state_order: list[tuple[str, PAPAttentionSession, str]] = []
            for session_request_id, session, entry, descriptor in prepared:
                should_append = self._record_layer_decode_descriptor(
                    session, entry.layer_name, descriptor
                )
                if should_append:
                    state = self._local_paged_kv[session_request_id][entry.layer_name]
                    pool = state.pool
                    seq_len = int(entry.seq_len)
                    start_position = seq_len - int(state.base_seq_len) - 1
                    if start_position < 0:
                        raise RuntimeError(
                            "PAP local paged decode append start position is negative"
                        )
                    local_seq_len = max(0, seq_len - int(state.base_seq_len))
                    required_blocks = (
                        (local_seq_len + pool.block_size - 1) // pool.block_size
                        if local_seq_len > 0
                        else 0
                    )
                    block_ids = state.block_ids
                    if len(block_ids) < required_blocks:
                        block_ids = (
                            *block_ids,
                            *self._allocate_local_paged_blocks_locked(
                                pool,
                                count=required_blocks - len(block_ids),
                            ),
                        )
                    if local_seq_len > 0:
                        logical_block = start_position // pool.block_size
                        block_offset = start_position % pool.block_size
                        local_block = block_ids[logical_block]
                        key_state = (
                            entry.key.detach()
                            .contiguous()
                            .to(
                                device=pool.kv_cache.device,
                                dtype=pool.kv_cache.dtype,
                            )
                        )
                        value_state = (
                            entry.value.detach()
                            .contiguous()
                            .to(
                                device=pool.kv_cache.device,
                                dtype=pool.kv_cache.dtype,
                            )
                        )
                        group_key = (id(pool), block_offset)
                        group = write_groups.setdefault(
                            group_key,
                            (pool, [], [], []),
                        )
                        group[1].append(local_block)
                        group[2].append(key_state)
                        group[3].append(value_state)
                    self._local_paged_kv[session_request_id][entry.layer_name] = (
                        PAPLocalPagedAttentionState(
                            pool=pool,
                            block_ids=tuple(block_ids),
                            seq_len=max(seq_len, state.seq_len),
                            base_seq_len=int(state.base_seq_len),
                            prefix_state=state.prefix_state,
                        )
                    )

                state_order.append((session_request_id, session, entry.layer_name))

            for (
                _pool_id,
                block_offset,
            ), (pool, local_blocks, keys, values) in write_groups.items():
                key_batch = torch.cat(keys, dim=0)
                value_batch = torch.cat(values, dim=0)
                first_block = local_blocks[0]
                contiguous_blocks = local_blocks == list(
                    range(first_block, first_block + len(local_blocks))
                )
                if contiguous_blocks:
                    block_slice = slice(first_block, first_block + len(local_blocks))
                    pool.kv_cache[
                        block_slice,
                        0,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ].copy_(key_batch)
                    pool.kv_cache[
                        block_slice,
                        1,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ].copy_(value_batch)
                else:
                    block_index = torch.tensor(
                        local_blocks,
                        dtype=torch.long,
                        device=pool.kv_cache.device,
                    )
                    pool.kv_cache[
                        block_index,
                        0,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ] = key_batch
                    pool.kv_cache[
                        block_index,
                        1,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ] = value_batch

            states: list[PAPLocalPagedAttentionState] = []
            for session_request_id, session, layer_name in state_order:
                state = self._local_paged_kv[session_request_id][layer_name]
                session.decode_seq_lens[layer_name] = int(state.seq_len)
                states.append(
                    PAPLocalPagedAttentionState(
                        pool=state.pool,
                        block_ids=tuple(state.block_ids),
                        seq_len=int(state.seq_len),
                        base_seq_len=int(state.base_seq_len),
                        prefix_state=state.prefix_state,
                    )
                )
            return states

    def append_decode_kv_tensor_batch_for_local_paged_attention(
        self,
        *,
        session_request_ids: tuple[str, ...],
        layer_name: str,
        key_batch: torch.Tensor,
        value_batch: torch.Tensor,
        seq_lens: tuple[int, ...],
        trace_stats: dict[str, float] | None = None,
    ) -> list[PAPLocalPagedAttentionState] | None:
        """Append a same-layer decode batch from batched K/V tensors."""

        if not session_request_ids:
            return None
        if (
            int(key_batch.shape[0]) != len(session_request_ids)
            or int(value_batch.shape[0]) != len(session_request_ids)
            or len(seq_lens) != len(session_request_ids)
        ):
            raise RuntimeError("local paged tensor batch append shape mismatch")
        if key_batch.ndim != 3 or value_batch.ndim != 3:
            return None

        lock_wait_start = time.perf_counter() if trace_stats is not None else 0.0
        with self._lock:
            _trace_add_elapsed_ms(
                trace_stats,
                "append_lock_wait_ms",
                lock_wait_start,
            )
            prepare_start = time.perf_counter() if trace_stats is not None else 0.0
            prepared: list[
                tuple[
                    int,
                    str,
                    PAPAttentionSession,
                    int,
                    PAPLocalPagedAttentionState,
                ]
            ] = []
            for index, request_id in enumerate(session_request_ids):
                if request_id in self._sessions:
                    session_request_id = request_id
                else:
                    session_request_id = self._resolve_session_request_id_locked(
                        request_id
                    )
                    if session_request_id is None:
                        raise KeyError(request_id)
                session = self._sessions[session_request_id]
                seq_len = int(seq_lens[index])
                if seq_len <= 0:
                    raise ValueError("decode seq_len must be positive")
                local_state = self._local_paged_kv.setdefault(
                    session_request_id, {}
                ).get(layer_name)
                if local_state is None:
                    self._wait_for_prefill_layer_locked(
                        session_request_id=session_request_id,
                        session=session,
                        layer_name=layer_name,
                        decode_seq_len=seq_len,
                    )
                    local_state = self._local_paged_kv.setdefault(
                        session_request_id, {}
                    ).get(layer_name)
                    if local_state is None:
                        local_state = self._ensure_local_paged_decode_state_locked(
                            session_request_id=session_request_id,
                            layer_name=layer_name,
                            key=key_batch[index],
                            seq_len=seq_len,
                        )
                    else:
                        local_state = self._canonicalize_local_paged_state_locked(
                            session_request_id=session_request_id,
                            layer_name=layer_name,
                            state=local_state,
                        )
                else:
                    local_state = self._canonicalize_local_paged_state_locked(
                        session_request_id=session_request_id,
                        layer_name=layer_name,
                        state=local_state,
                    )
                prepared.append(
                    (
                        index,
                        session_request_id,
                        session,
                        seq_len,
                        local_state,
                    )
                )
            _trace_add_elapsed_ms(trace_stats, "append_prepare_ms", prepare_start)

            record_start = time.perf_counter() if trace_stats is not None else 0.0
            write_groups: dict[
                tuple[int, int],
                tuple[PAPLocalPagedKVPool, list[int], list[int]],
            ] = {}
            native_write_batches: dict[
                int,
                tuple[PAPLocalPagedKVPool, list[int], list[int], list[int]],
            ] = {}
            state_order: list[tuple[str, PAPAttentionSession]] = []
            for index, session_request_id, session, seq_len, state in prepared:
                should_append = self._record_layer_decode_without_descriptor(
                    session, layer_name, seq_len
                )
                if should_append:
                    pool = state.pool
                    start_position = seq_len - int(state.base_seq_len) - 1
                    if start_position < 0:
                        raise RuntimeError(
                            "PAP local paged tensor append start position is negative"
                        )
                    local_seq_len = max(0, seq_len - int(state.base_seq_len))
                    required_blocks = (
                        (local_seq_len + pool.block_size - 1) // pool.block_size
                        if local_seq_len > 0
                        else 0
                    )
                    block_ids = state.block_ids
                    if len(block_ids) < required_blocks:
                        block_ids = (
                            *block_ids,
                            *self._allocate_local_paged_blocks_locked(
                                pool,
                                count=required_blocks - len(block_ids),
                            ),
                        )
                    if local_seq_len > 0:
                        logical_block = start_position // pool.block_size
                        block_offset = start_position % pool.block_size
                        local_block = block_ids[logical_block]
                        native_group = native_write_batches.setdefault(
                            id(pool),
                            (pool, [], [], []),
                        )
                        native_group[1].append(local_block)
                        native_group[2].append(block_offset)
                        native_group[3].append(index)
                        group_key = (id(pool), block_offset)
                        group = write_groups.setdefault(group_key, (pool, [], []))
                        group[1].append(local_block)
                        group[2].append(index)
                    self._local_paged_kv[session_request_id][layer_name] = (
                        PAPLocalPagedAttentionState(
                            pool=pool,
                            block_ids=tuple(block_ids),
                            seq_len=max(seq_len, state.seq_len),
                            base_seq_len=int(state.base_seq_len),
                            prefix_state=state.prefix_state,
                        )
                    )

                state_order.append((session_request_id, session))
            _trace_add_elapsed_ms(trace_stats, "append_record_ms", record_start)

            key_source = key_batch.detach()
            value_source = value_batch.detach()
            native_written_pool_ids: set[int] = set()
            for pool_id, (
                pool,
                local_blocks,
                block_offsets,
                row_indices,
            ) in native_write_batches.items():
                tensor_start = (
                    time.perf_counter() if trace_stats is not None else 0.0
                )
                first_row = row_indices[0]
                contiguous_rows = row_indices == list(
                    range(first_row, first_row + len(row_indices))
                )
                if contiguous_rows:
                    row_slice = slice(first_row, first_row + len(row_indices))
                    key_group = key_source[row_slice]
                    value_group = value_source[row_slice]
                else:
                    row_index = torch.tensor(
                        row_indices,
                        dtype=torch.long,
                        device=key_source.device,
                    )
                    key_group = key_source.index_select(0, row_index)
                    value_group = value_source.index_select(0, row_index)
                slot_mapping = _local_paged_slot_mapping_tensor(
                    local_blocks=local_blocks,
                    block_offsets=block_offsets,
                    block_size=pool.block_size,
                    device=pool.kv_cache.device,
                )
                _trace_add_elapsed_ms(
                    trace_stats,
                    "append_tensor_ms",
                    tensor_start,
                )
                copy_start = time.perf_counter() if trace_stats is not None else 0.0
                native_written = _try_local_paged_native_cache_append(
                    pool=pool,
                    key_batch=key_group,
                    value_batch=value_group,
                    slot_mapping=slot_mapping,
                )
                if native_written:
                    native_written_pool_ids.add(pool_id)
                    _trace_add_elapsed_ms(
                        trace_stats,
                        "append_copy_ms",
                        copy_start,
                    )

            for (
                _pool_id,
                block_offset,
            ), (pool, local_blocks, row_indices) in write_groups.items():
                if id(pool) in native_written_pool_ids:
                    continue
                tensor_start = time.perf_counter() if trace_stats is not None else 0.0
                first_row = row_indices[0]
                contiguous_rows = row_indices == list(
                    range(first_row, first_row + len(row_indices))
                )
                if contiguous_rows:
                    row_slice = slice(first_row, first_row + len(row_indices))
                    key_group = key_source[row_slice]
                    value_group = value_source[row_slice]
                else:
                    row_index = torch.tensor(
                        row_indices,
                        dtype=torch.long,
                        device=key_source.device,
                    )
                    key_group = key_source.index_select(0, row_index)
                    value_group = value_source.index_select(0, row_index)

                key_group = _local_paged_copy_source(
                    key_group,
                    device=pool.kv_cache.device,
                    dtype=pool.kv_cache.dtype,
                )
                value_group = _local_paged_copy_source(
                    value_group,
                    device=pool.kv_cache.device,
                    dtype=pool.kv_cache.dtype,
                )
                _trace_add_elapsed_ms(trace_stats, "append_tensor_ms", tensor_start)

                copy_start = time.perf_counter() if trace_stats is not None else 0.0
                first_block = local_blocks[0]
                contiguous_blocks = local_blocks == list(
                    range(first_block, first_block + len(local_blocks))
                )
                if contiguous_blocks:
                    block_slice = slice(first_block, first_block + len(local_blocks))
                    pool.kv_cache[
                        block_slice,
                        0,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ].copy_(key_group)
                    pool.kv_cache[
                        block_slice,
                        1,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ].copy_(value_group)
                else:
                    block_index = torch.tensor(
                        local_blocks,
                        dtype=torch.long,
                        device=pool.kv_cache.device,
                    )
                    pool.kv_cache[
                        block_index,
                        0,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ] = key_group
                    pool.kv_cache[
                        block_index,
                        1,
                        block_offset,
                        : pool.num_kv_heads,
                        :,
                    ] = value_group
                _trace_add_elapsed_ms(trace_stats, "append_copy_ms", copy_start)

            state_start = time.perf_counter() if trace_stats is not None else 0.0
            states: list[PAPLocalPagedAttentionState] = []
            for session_request_id, session in state_order:
                state = self._local_paged_kv[session_request_id][layer_name]
                session.decode_seq_lens[layer_name] = int(state.seq_len)
                states.append(
                    PAPLocalPagedAttentionState(
                        pool=state.pool,
                        block_ids=tuple(state.block_ids),
                        seq_len=int(state.seq_len),
                        base_seq_len=int(state.base_seq_len),
                        prefix_state=state.prefix_state,
                    )
                )
            _trace_add_elapsed_ms(trace_stats, "append_state_ms", state_start)
            return states

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
        with self._lock:
            self._replace_existing_session_locked(registration.request_id)
            self._sessions[registration.request_id] = session
            self._layer_events.setdefault(registration.request_id, [])
            self._decode_kv.setdefault(registration.request_id, {})
            self._prefill_kv.setdefault(registration.request_id, {})
            self._prefill_paged_kv.setdefault(registration.request_id, {})
            self._prefill_readiness.setdefault(registration.request_id, {})
            self._local_paged_kv.setdefault(registration.request_id, {})
            self._request_id_resolution_cache[registration.request_id] = (
                registration.request_id
            )
            self._attention_sessions.create_session(
                registration.request_id,
                registration.conversation_id,
                block_size=registration.block_size,
                max_seq_len=registration.max_seq_len,
            )
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

    def reserve_decode_slot(
        self,
        *,
        request_id: str,
        layer_name: str,
        seq_len: int,
    ) -> tuple[int, int]:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            seq_len = int(seq_len)
            if seq_len <= 0:
                raise ValueError("decode seq_len must be positive")

            block_id = (seq_len - 1) // session.block_size
            slot = block_id * session.block_size + ((seq_len - 1) % session.block_size)
            return block_id, slot

    def append_decode_kv_at_seq_len(
        self,
        *,
        request_id: str,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        """Reserve the decode slot and append KV under one registry lock."""

        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            seq_len = int(seq_len)
            if seq_len <= 0:
                raise ValueError("decode seq_len must be positive")

            self._wait_for_prefill_layer_locked(
                session_request_id=session_request_id,
                session=session,
                layer_name=layer_name,
                decode_seq_len=seq_len,
            )
            prefill_layer_kv = self._prefill_kv.setdefault(session_request_id, {})
            block_id = (seq_len - 1) // session.block_size
            slot = block_id * session.block_size + ((seq_len - 1) % session.block_size)
            descriptor = AttentionDecodeDescriptor(
                request_id=session_request_id,
                block_id=block_id,
                slot=slot,
                seq_len=seq_len,
            )
            should_append = self._record_layer_decode_descriptor(
                session, layer_name, descriptor
            )

            layer_kv = self._decode_kv.setdefault(session_request_id, {})
            decode_buffer = layer_kv.get(layer_name)
            if not should_append:
                if decode_buffer is None:
                    decode_key = torch.empty(
                        (0, *key.shape[1:]),
                        dtype=key.dtype,
                        device=key.device,
                    )
                    decode_value = torch.empty(
                        (0, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                else:
                    decode_key, decode_value = decode_buffer.view()
            else:
                key_state = key.detach().contiguous().to(self._storage_device)
                value_state = value.detach().contiguous().to(self._storage_device)
                if decode_buffer is None:
                    decode_buffer = self._make_decode_buffer(
                        key=key_state,
                        value=value_state,
                        capacity=self._initial_decode_capacity(key_state.shape[0]),
                    )
                required = decode_buffer.length + int(key_state.shape[0])
                decode_buffer = self._ensure_decode_capacity(
                    decode_buffer,
                    required=required,
                )
                start = decode_buffer.length
                end = start + int(key_state.shape[0])
                decode_buffer.key[start:end].copy_(key_state)
                decode_buffer.value[start:end].copy_(value_state)
                decode_buffer.length = end
                layer_kv[layer_name] = decode_buffer
                decode_key, decode_value = decode_buffer.view()
                self._append_local_paged_decode_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                    key=key_state,
                    value=value_state,
                    seq_len=seq_len,
                )

            segments: list[tuple[torch.Tensor, torch.Tensor]] = []
            if layer_name in prefill_layer_kv:
                segments.extend(prefill_layer_kv[layer_name])
            if decode_key.numel() > 0:
                segments.append((decode_key, decode_value))

            full_seq_len = sum(int(segment_key.shape[0]) for segment_key, _ in segments)
            session.decode_seq_lens[layer_name] = full_seq_len
            return segments, session.decode_seq_lens[layer_name]

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

    def record_layer_event(
        self,
        *,
        request_id: str,
        layer_name: str,
        query_shape: list[int],
        key_shape: list[int],
        value_shape: list[int],
        dtype: str,
        device: str,
        is_decode: bool,
        num_reqs: int | None = None,
        num_actual_tokens: int | None = None,
        max_seq_len: int | None = None,
    ) -> PAPAttentionLayerEvent:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            event = PAPAttentionLayerEvent(
                request_id=request_id,
                session_request_id=session_request_id,
                layer_name=layer_name,
                query_shape=list(query_shape),
                key_shape=list(key_shape),
                value_shape=list(value_shape),
                dtype=dtype,
                device=device,
                is_decode=is_decode,
                num_reqs=num_reqs,
                num_actual_tokens=num_actual_tokens,
                max_seq_len=max_seq_len,
            )
            self._layer_events.setdefault(session_request_id, []).append(event)
        logger.debug(
            "recorded PAP attention layer event request_id=%s "
            "session=%s layer=%s decode=%s",
            request_id,
            session_request_id,
            layer_name,
            is_decode,
        )
        return event.copy()

    def get_layer_events(self, request_id: str) -> list[PAPAttentionLayerEvent]:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                return []
            return [
                event.copy() for event in self._layer_events.get(session_request_id, [])
            ]

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

    def _mark_prefill_descriptor_received_locked(
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
        readiness.received_at = time.perf_counter()
        self._prefill_condition.notify_all()
        return readiness

    def _mark_prefill_descriptor_opened_locked(
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
        readiness.opened_at = time.perf_counter()
        self._prefill_condition.notify_all()
        return readiness

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

    def _mark_prefill_failed_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        error: BaseException,
    ) -> PAPPrefillLayerReadiness:
        readiness = self._prefill_readiness_locked(
            session_request_id=session_request_id,
            layer_name=layer_name,
        )
        readiness.failed = True
        readiness.ready = False
        readiness.error = str(error)
        readiness.failed_at = time.perf_counter()
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

    def import_prefill_kv(
        self,
        *,
        request_id: str,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
        block_ids: list[int] | None = None,
        copy: bool = True,
    ) -> int:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            self._mark_prefill_descriptor_opened_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
            )
            if copy:
                key_state = key.detach().contiguous().to(self._storage_device)
                value_state = value.detach().contiguous().to(self._storage_device)
            else:
                key_state = key.detach()
                value_state = value.detach()
            seq_len = int(seq_len)
            if seq_len < 0:
                raise ValueError("seq_len must be non-negative")
            if key_state.shape[0] != seq_len or value_state.shape[0] != seq_len:
                raise ValueError("prefill KV seq_len must match tensor length")
            expected_prefix_len = session.prefix_len
            if expected_prefix_len is not None and int(expected_prefix_len) != seq_len:
                raise ValueError(
                    f"prefill KV seq_len {seq_len} does not match "
                    f"registered prefix_len {expected_prefix_len}"
                )
            self._prefill_kv.setdefault(session_request_id, {})[layer_name] = [
                (key_state, value_state)
            ]
            self._prefill_paged_kv.setdefault(session_request_id, {}).pop(
                layer_name, None
            )
            imported_session = self._attention_sessions.import_prefill_kv(
                session_request_id,
                block_ids=list(block_ids)
                if block_ids is not None
                else list(
                    range((seq_len + session.block_size - 1) // session.block_size)
                ),
                seq_len=seq_len,
            )
            session.block_ids = tuple(imported_session.block_ids)
            session.seq_len = imported_session.seq_len
            session.prefill_seq_lens[layer_name] = seq_len
            session.decode_seq_lens[layer_name] = seq_len
            self._mark_prefill_ready_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
            )
            if os.environ.get("PAP_ATTENTION_KV_DEBUG", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                logger.info(
                    "PAP prefill KV imported request_id=%s layer=%s seq_len=%s "
                    "key_shape=%s value_shape=%s key_norm=%.6f value_norm=%.6f",
                    session_request_id,
                    layer_name,
                    seq_len,
                    tuple(key_state.shape),
                    tuple(value_state.shape),
                    float(key_state.float().norm().item()),
                    float(value_state.float().norm().item()),
                )
            return seq_len

    def import_prefill_paged_kv(
        self,
        *,
        request_id: str,
        layer_name: str,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        seq_len: int,
        block_size: int,
        num_kv_heads: int,
        layout: str,
        lease_id: str | None = None,
        leased_block_ids: tuple[int, ...] | None = None,
        lease_capacity_tokens: int | None = None,
        unified_kv_mode: bool = False,
        prefix_len: int | None = None,
        writable_start_token: int | None = None,
        writable_end_token: int | None = None,
    ) -> int:
        from vllm.pap.remote_attention import paged_kv_segments

        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            if lease_id:
                existing = self._session_lease_ids.get(session_request_id)
                if existing is None:
                    self._session_lease_ids[session_request_id] = lease_id
                    if leased_block_ids is not None:
                        self._session_leased_block_ids[session_request_id] = (
                            tuple(int(b) for b in leased_block_ids)
                        )
                    if lease_capacity_tokens is not None:
                        self._session_lease_capacity_tokens[session_request_id] = (
                            int(lease_capacity_tokens)
                        )
                    if _pap_kv_lease_profile_enabled():
                        logger.info(
                            "PAP Attention captured lease request_id=%s "
                            "lease_id=%s leased_blocks=%d capacity_tokens=%s",
                            session_request_id,
                            lease_id,
                            len(leased_block_ids or ()),
                            lease_capacity_tokens,
                        )
            self._mark_prefill_descriptor_opened_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
            )
            seq_len = int(seq_len)
            if unified_kv_mode and lease_id is not None:
                prefix_value = (
                    int(prefix_len) if prefix_len is not None else seq_len
                )
                w_start = (
                    int(writable_start_token)
                    if writable_start_token is not None
                    else prefix_value
                )
                w_end = (
                    int(writable_end_token)
                    if writable_end_token is not None
                    else prefix_value
                )
                capacity = (
                    int(lease_capacity_tokens)
                    if lease_capacity_tokens is not None
                    else max(seq_len, w_end)
                )
                unified_state = PAPUnifiedPagedKVState(
                    kv_cache=kv_cache.detach(),
                    block_ids=tuple(int(b) for b in block_ids),
                    prefix_len=prefix_value,
                    seq_len=seq_len,
                    capacity_tokens=capacity,
                    writable_start_token=w_start,
                    writable_end_token=w_end,
                    lease_id=str(lease_id),
                    block_size=int(block_size),
                    num_kv_heads=int(num_kv_heads),
                    layout=str(layout),
                )
                self._unified_paged_kv.setdefault(session_request_id, {})[
                    layer_name
                ] = unified_state
                if _pap_kv_lease_profile_enabled():
                    logger.info(
                        "PAP unified KV state stored request_id=%s layer=%s "
                        "prefix_len=%d seq_len=%d capacity=%d writable=%d..%d",
                        session_request_id,
                        layer_name,
                        prefix_value,
                        seq_len,
                        capacity,
                        w_start,
                        w_end,
                    )
                self._mark_prefill_descriptor_received_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                )
                self._mark_prefill_descriptor_opened_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                )
                self._mark_prefill_ready_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                )
                return seq_len
            if seq_len < 0:
                raise ValueError("seq_len must be non-negative")
            expected_prefix_len = session.prefix_len
            if expected_prefix_len is not None and int(expected_prefix_len) != seq_len:
                raise ValueError(
                    f"prefill KV seq_len {seq_len} does not match "
                    f"registered prefix_len {expected_prefix_len}"
                )
            if int(block_size) != int(session.block_size):
                raise ValueError(
                    f"prefill KV block_size {block_size} does not match "
                    f"registered block_size {session.block_size}"
                )
            segments = paged_kv_segments(
                kv_cache=kv_cache.detach(),
                block_ids=[int(block_id) for block_id in block_ids],
                seq_len=seq_len,
                num_kv_heads=int(num_kv_heads),
                layout=layout,  # type: ignore[arg-type]
            )
            existing_prefill = self._prefill_paged_kv.get(session_request_id, {}).get(
                layer_name
            )
            prefill_block_ids = [int(block_id) for block_id in block_ids]
            prefill_seq_len = seq_len
            if existing_prefill is not None:
                prefill_block_ids = list(existing_prefill.block_ids)
                for block_id in block_ids:
                    block_id = int(block_id)
                    if block_id not in prefill_block_ids:
                        prefill_block_ids.append(block_id)
                prefill_seq_len = max(int(existing_prefill.seq_len), seq_len)

            existing_session_block_ids = [
                int(block_id) for block_id in session.block_ids
            ]
            existing_session_seq_len = int(session.seq_len)
            existing_decode_seq_len = int(
                session.decode_seq_lens.get(layer_name, existing_session_seq_len)
            )
            merged_session_block_ids = list(prefill_block_ids)
            for block_id in existing_session_block_ids:
                if block_id not in merged_session_block_ids:
                    merged_session_block_ids.append(block_id)
            merged_session_seq_len = max(existing_session_seq_len, prefill_seq_len)
            merged_decode_seq_len = max(
                existing_decode_seq_len,
                existing_session_seq_len,
                prefill_seq_len,
            )

            self._prefill_kv.setdefault(session_request_id, {})[layer_name] = segments
            prefix_state = PAPPrefillPagedKV(
                kv_cache=kv_cache.detach(),
                block_ids=prefill_block_ids,
                seq_len=prefill_seq_len,
                block_size=int(block_size),
                num_kv_heads=int(num_kv_heads),
                layout=str(layout),
            )
            self._prefill_paged_kv.setdefault(session_request_id, {})[layer_name] = (
                prefix_state
            )
            if _pap_attention_copy_prefix_kv():
                self._install_local_paged_prefill_locked(
                    session_request_id=session_request_id,
                    layer_name=layer_name,
                    segments=segments,
                    seq_len=prefill_seq_len,
                    block_size=int(block_size),
                    num_kv_heads=int(num_kv_heads),
                )
                self._prefill_paged_kv.setdefault(session_request_id, {}).pop(
                    layer_name, None
                )
            else:
                state = self._local_paged_kv.setdefault(session_request_id, {}).get(
                    layer_name
                )
                if state is None:
                    existing_pool = self._local_paged_kv_pools.get(layer_name)
                    if existing_pool is None:
                        logger.info(
                            "PAP descriptor-resident prefix KV deferred local pool allocation request_id=%s layer=%s seq_len=%d blocks=%d",
                            session_request_id,
                            layer_name,
                            prefill_seq_len,
                            len(prefill_block_ids),
                        )
                    else:
                        if (
                            existing_pool.block_size != int(block_size)
                            or existing_pool.num_kv_heads != int(num_kv_heads)
                            or existing_pool.head_dim != int(kv_cache.shape[-1])
                            or existing_pool.kv_cache.dtype != kv_cache.dtype
                            or existing_pool.kv_cache.device != kv_cache.device
                        ):
                            raise ValueError(
                                "local paged KV pool shape does not match layer"
                            )
                    if existing_pool is not None:
                        self._local_paged_kv.setdefault(session_request_id, {})[
                            layer_name
                        ] = PAPLocalPagedAttentionState(
                            pool=existing_pool,
                            block_ids=(),
                            seq_len=prefill_seq_len,
                            base_seq_len=prefill_seq_len,
                            prefix_state=prefix_state,
                        )
                else:
                    self._local_paged_kv.setdefault(session_request_id, {})[layer_name] = (
                        PAPLocalPagedAttentionState(
                            pool=state.pool,
                            block_ids=tuple(state.block_ids),
                            seq_len=max(int(state.seq_len), prefill_seq_len),
                            base_seq_len=prefill_seq_len,
                            prefix_state=prefix_state,
                        )
                    )
                logger.info(
                    "PAP descriptor-resident prefix KV enabled request_id=%s layer=%s seq_len=%d blocks=%d",
                    session_request_id,
                    layer_name,
                    prefill_seq_len,
                    len(prefill_block_ids),
                )






































































            imported_session = self._attention_sessions.import_prefill_kv(
                session_request_id,
                block_ids=merged_session_block_ids,
                seq_len=merged_session_seq_len,
            )
            session.block_ids = tuple(imported_session.block_ids)
            session.seq_len = max(imported_session.seq_len, existing_session_seq_len)
            session.prefill_seq_lens[layer_name] = prefill_seq_len
            session.decode_seq_lens[layer_name] = merged_decode_seq_len
            self._mark_prefill_ready_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
            )
            return prefill_seq_len

    def enqueue_prefill_paged_kv_descriptor(
        self,
        descriptor: PAPOffloadKVPagedIPCDescriptor,
    ) -> int:
        queue_start = time.perf_counter()
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(
                descriptor.request_id
            )
            if session_request_id is None:
                raise KeyError(descriptor.request_id)
            self._mark_prefill_descriptor_received_locked(
                session_request_id=session_request_id,
                layer_name=descriptor.layer_name,
            )
            if not self._prefill_async_worker_started:
                Thread(
                    target=self._prefill_async_worker_loop,
                    daemon=True,
                    name="pap-prefill-kv-import-worker",
                ).start()
                self._prefill_async_worker_started = True
        self._prefill_async_queue.put((descriptor, queue_start))
        if _pap_prefill_ipc_profile_enabled():
            logger.info(
                "PAP prefill IPC attention queued request_id=%s layer=%s "
                "seq_len=%d blocks=%d queue_ms=%.3f",
                descriptor.request_id,
                descriptor.layer_name,
                int(descriptor.seq_len),
                len(descriptor.block_ids),
                (time.perf_counter() - queue_start) * 1000.0,
            )
        return int(descriptor.seq_len)

    def _prefill_async_worker_loop(self) -> None:
        while True:
            descriptor, queued_at = self._prefill_async_queue.get()
            try:
                open_start = time.perf_counter()
                kv_cache = open_ipc_paged_kv_cache(descriptor)
                open_ms = (time.perf_counter() - open_start) * 1000.0
                ready_start = time.perf_counter()
                seq_len = self.import_prefill_paged_kv(
                    request_id=descriptor.request_id,
                    layer_name=descriptor.layer_name,
                    kv_cache=kv_cache,
                    block_ids=list(descriptor.block_ids),
                    seq_len=descriptor.seq_len,
                    block_size=descriptor.block_size,
                    num_kv_heads=descriptor.num_kv_heads,
                    layout=descriptor.layout,
                )
                ready_ms = (time.perf_counter() - ready_start) * 1000.0
                if _pap_prefill_ipc_profile_enabled():
                    logger.info(
                        "PAP prefill IPC attention ready request_id=%s layer=%s "
                        "seq_len=%d blocks=%d queue_to_open_ms=%.3f open_ms=%.3f "
                        "install_ms=%.3f total_ms=%.3f",
                        descriptor.request_id,
                        descriptor.layer_name,
                        seq_len,
                        len(descriptor.block_ids),
                        (open_start - queued_at) * 1000.0,
                        open_ms,
                        ready_ms,
                        (time.perf_counter() - queued_at) * 1000.0,
                    )
            except BaseException as exc:
                with self._lock:
                    session_request_id = self._resolve_session_request_id_locked(
                        descriptor.request_id
                    )
                    if session_request_id is not None:
                        self._mark_prefill_failed_locked(
                            session_request_id=session_request_id,
                            layer_name=descriptor.layer_name,
                            error=exc,
                        )
                logger.exception(
                    "PAP async prefill KV import failed request_id=%s layer=%s",
                    descriptor.request_id,
                    descriptor.layer_name,
                )

    def _wait_for_prefill_layer_locked(
        self,
        *,
        session_request_id: str,
        session: PAPAttentionSession,
        layer_name: str,
        decode_seq_len: int | None = None,
    ) -> None:
        has_registered_prefix = int(session.prefix_len or 0) > 0
        has_scheduler_prefix = decode_seq_len is not None and int(decode_seq_len) > 1
        if not has_registered_prefix and not has_scheduler_prefix:
            return
        deadline = time.monotonic() + float(
            os.environ.get("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "5.0")
        )
        prefill_layer_kv = self._prefill_kv.setdefault(session_request_id, {})
        wait_start = time.perf_counter()
        while True:
            readiness = self._prefill_readiness.setdefault(session_request_id, {}).get(
                layer_name
            )
            if readiness is not None and readiness.failed:
                raise RuntimeError(
                    "prefill KV import failed before stateful decode attention: "
                    f"{readiness.error}"
                )
            if readiness is not None and readiness.ready:
                return
            if layer_name in prefill_layer_kv:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                state = "missing"
                if readiness is not None:
                    state = (
                        "received=%s opened=%s ready=%s failed=%s"
                        % (
                            readiness.descriptor_received,
                            readiness.descriptor_opened,
                            readiness.ready,
                            readiness.failed,
                        )
                    )
                raise RuntimeError(
                    "prefill KV must be ready before stateful decode attention "
                    f"request_id={session_request_id} layer={layer_name} state={state}"
                )
            if _pap_prefill_ipc_profile_enabled():
                logger.info(
                    "PAP prefill IPC attention wait request_id=%s layer=%s "
                    "remaining_ms=%.3f waited_ms=%.3f",
                    session_request_id,
                    layer_name,
                    remaining * 1000.0,
                    (time.perf_counter() - wait_start) * 1000.0,
                )
            self._prefill_condition.wait(timeout=remaining)

    def attention_segments_before_decode(
        self,
        *,
        request_id: str,
        layer_name: str,
        seq_len: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            decode_seq_len = int(seq_len) if seq_len is not None else None
            self._wait_for_prefill_layer_locked(
                session_request_id=session_request_id,
                session=session,
                layer_name=layer_name,
                decode_seq_len=decode_seq_len,
            )
            segments: list[tuple[torch.Tensor, torch.Tensor]] = []
            prefill_layer_kv = self._prefill_kv.setdefault(session_request_id, {})
            if layer_name in prefill_layer_kv:
                segments.extend(prefill_layer_kv[layer_name])
            decode_buffer = self._decode_kv.setdefault(session_request_id, {}).get(
                layer_name
            )
            if decode_buffer is not None:
                decode_key, decode_value = decode_buffer.view()
                if decode_key.numel() > 0:
                    segments.append((decode_key, decode_value))
            return segments

    def append_decode_kv(
        self,
        *,
        request_id: str,
        layer_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        block_id: int | None = None,
        slot: int | None = None,
        seq_len: int | None = None,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            decode_seq_len = int(seq_len) if seq_len is not None else None
            self._wait_for_prefill_layer_locked(
                session_request_id=session_request_id,
                session=session,
                layer_name=layer_name,
                decode_seq_len=decode_seq_len,
            )
            prefill_layer_kv = self._prefill_kv.setdefault(session_request_id, {})

            if block_id is not None or slot is not None or seq_len is not None:
                if block_id is None or slot is None or seq_len is None:
                    raise ValueError(
                        "block_id, slot, and seq_len must be provided together"
                    )
                descriptor = AttentionDecodeDescriptor(
                    request_id=session_request_id,
                    block_id=int(block_id),
                    slot=int(slot),
                    seq_len=int(seq_len),
                )
                should_append = self._record_layer_decode_descriptor(
                    session, layer_name, descriptor
                )
            else:
                should_append = True

            layer_kv = self._decode_kv.setdefault(session_request_id, {})
            decode_buffer = layer_kv.get(layer_name)
            if not should_append:
                if decode_buffer is None:
                    decode_key = torch.empty(
                        (0, *key.shape[1:]),
                        dtype=key.dtype,
                        device=key.device,
                    )
                    decode_value = torch.empty(
                        (0, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                else:
                    decode_key, decode_value = decode_buffer.view()
            else:
                key_state = key.detach().contiguous().to(self._storage_device)
                value_state = value.detach().contiguous().to(self._storage_device)
                if decode_buffer is None:
                    decode_buffer = self._make_decode_buffer(
                        key=key_state,
                        value=value_state,
                        capacity=self._initial_decode_capacity(key_state.shape[0]),
                    )
                required = decode_buffer.length + int(key_state.shape[0])
                decode_buffer = self._ensure_decode_capacity(
                    decode_buffer,
                    required=required,
                )
                start = decode_buffer.length
                end = start + int(key_state.shape[0])
                decode_buffer.key[start:end].copy_(key_state)
                decode_buffer.value[start:end].copy_(value_state)
                decode_buffer.length = end
                layer_kv[layer_name] = decode_buffer
                decode_key, decode_value = decode_buffer.view()
                if seq_len is not None:
                    self._append_local_paged_decode_locked(
                        session_request_id=session_request_id,
                        layer_name=layer_name,
                        key=key_state,
                        value=value_state,
                        seq_len=int(seq_len),
                    )

            segments: list[tuple[torch.Tensor, torch.Tensor]] = []
            if layer_name in prefill_layer_kv:
                segments.extend(prefill_layer_kv[layer_name])
            if decode_key.numel() > 0:
                segments.append((decode_key, decode_value))

            full_seq_len = sum(int(segment_key.shape[0]) for segment_key, _ in segments)
            if block_id is None and slot is None and seq_len is None:
                self._record_layer_decode_without_descriptor(
                    session, layer_name, full_seq_len
                )
            session.decode_seq_lens[layer_name] = full_seq_len
            return segments, session.decode_seq_lens[layer_name]

    @staticmethod
    def _record_layer_decode_descriptor(
        session: PAPAttentionSession,
        layer_name: str,
        descriptor: AttentionDecodeDescriptor,
    ) -> bool:
        if descriptor.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if descriptor.slot < 0:
            raise ValueError("slot must be non-negative")
        if descriptor.seq_len <= 0:
            raise ValueError("decode descriptor seq_len must be positive")
        if descriptor.seq_len > session.max_seq_len:
            raise ValueError(
                f"seq_len {descriptor.seq_len} exceeds max_seq_len "
                f"{session.max_seq_len}"
            )

        expected_offset = (descriptor.seq_len - 1) % session.block_size
        expected_slot = descriptor.block_id * session.block_size + expected_offset
        if descriptor.slot != expected_slot:
            raise ValueError(
                f"slot {descriptor.slot} does not match block_id "
                f"{descriptor.block_id} and offset {expected_offset}"
            )

        layer_seq_len = session.decode_seq_lens.get(
            layer_name,
            session.prefill_seq_lens.get(layer_name, int(session.prefix_len or 0)),
        )
        if descriptor.seq_len < layer_seq_len:
            raise ValueError(
                f"decode descriptor seq_len {descriptor.seq_len} is behind "
                f"current layer seq_len {layer_seq_len}"
            )
        should_append = descriptor.seq_len > layer_seq_len
        if descriptor.seq_len > layer_seq_len + 1:
            raise ValueError(
                f"expected seq_len {layer_seq_len + 1}, got {descriptor.seq_len}"
            )

        if should_append:
            block_ids = session.block_ids
            if not block_ids or block_ids[-1] != descriptor.block_id:
                session.block_ids = (*block_ids, descriptor.block_id)
            session.seq_len = max(session.seq_len, descriptor.seq_len)
        return should_append

    @staticmethod
    def _record_layer_decode_without_descriptor(
        session: PAPAttentionSession,
        layer_name: str,
        seq_len: int,
    ) -> bool:
        if seq_len <= 0:
            raise ValueError("decode seq_len must be positive")
        if seq_len > session.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {session.max_seq_len}"
            )
        layer_seq_len = session.decode_seq_lens.get(
            layer_name,
            session.prefill_seq_lens.get(layer_name, int(session.prefix_len or 0)),
        )
        if seq_len < layer_seq_len:
            raise ValueError(
                f"decode seq_len {seq_len} is behind current layer seq_len "
                f"{layer_seq_len}"
            )
        if seq_len > layer_seq_len + 1:
            raise ValueError(f"expected seq_len {layer_seq_len + 1}, got {seq_len}")
        should_append = seq_len > layer_seq_len
        if should_append:
            block_id = (seq_len - 1) // session.block_size
            expected_offset = (seq_len - 1) % session.block_size
            block_ids = session.block_ids
            if ((not block_ids) or expected_offset == 0) and (
                not block_ids or block_ids[-1] != block_id
            ):
                session.block_ids = (*block_ids, block_id)
            session.seq_len = max(session.seq_len, seq_len)
        return should_append

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)


def _http_error_for_attention_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="unknown PAP request")
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _compute_single_binary_attention_response(
    registry: PAPAttentionRegistry,
    payload: bytes,
) -> bytes:
    from vllm.pap.remote_attention import (
        compute_segmented_attention_output,
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    metadata, tensors = deserialize_tensor_bundle(payload)
    request_id = str(metadata["request_id"])
    layer_name = str(metadata["layer_name"])
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    block_id = metadata.get("block_id")
    slot = metadata.get("slot")
    seq_len_meta = metadata.get("seq_len")
    segments, seq_len = registry.append_decode_kv(
        request_id=request_id,
        layer_name=layer_name,
        key=key,
        value=value,
        block_id=None if block_id is None else int(block_id),
        slot=None if slot is None else int(slot),
        seq_len=None if seq_len_meta is None else int(seq_len_meta),
    )
    if torch.cuda.is_available():
        query = query.to(registry.storage_device, non_blocking=True)
    output = compute_segmented_attention_output(
        query=query,
        segments=segments,
        scale=float(metadata["scale"]),
    )
    return serialize_tensor_bundle(
        {
            "request_id": request_id,
            "layer_name": layer_name,
            "seq_len": seq_len,
        },
        {"output": output},
    )


def open_ipc_prefill_kv(
    descriptor: PAPOffloadKVIPCDescriptor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Open CUDA IPC prefill KV tensors described by OFFLOAD_KV metadata."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    device_index = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device_index)
    physical_gpu_id = str(props.uuid)

    def rebuild(handle: dict[str, tuple[Any, ...]]) -> torch.Tensor:
        if physical_gpu_id not in handle:
            raise ValueError(
                f"IPC handle not found for GPU UUID {physical_gpu_id}. "
                f"Available UUIDs: {list(handle.keys())}"
            )
        args = list(handle[physical_gpu_id])
        args[6] = device_index
        return rebuild_cuda_tensor(*args)

    return rebuild(descriptor.key.ipc_handle), rebuild(descriptor.value.ipc_handle)


def open_ipc_paged_kv_cache(
    descriptor: PAPOffloadKVPagedIPCDescriptor,
) -> torch.Tensor:
    """Open CUDA IPC paged KV backing tensor described by OFFLOAD_KV metadata."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    device_index = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device_index)
    physical_gpu_id = str(props.uuid)
    handle = descriptor.kv_cache.ipc_handle
    if physical_gpu_id not in handle:
        raise ValueError(
            f"IPC handle not found for GPU UUID {physical_gpu_id}. "
            f"Available UUIDs: {list(handle.keys())}"
        )
    args = list(handle[physical_gpu_id])
    args[6] = device_index
    return rebuild_cuda_tensor(*args)


def compute_batch_binary_attention_response(
    registry: PAPAttentionRegistry,
    payload: bytes,
) -> bytes:
    from vllm.pap.remote_attention import (
        compute_segmented_attention_output,
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    trace_remote_attention = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    trace_total_start = time.perf_counter() if trace_remote_attention else 0.0
    trace_deserialize_start = time.perf_counter() if trace_remote_attention else 0.0
    metadata, tensors = deserialize_tensor_bundle(payload)
    trace_deserialize_ms = (
        (time.perf_counter() - trace_deserialize_start) * 1000.0
        if trace_remote_attention
        else 0.0
    )
    response_items: list[dict[str, Any]] = []
    response_tensors: dict[str, torch.Tensor] = {}
    append_ms = 0.0
    query_ms = 0.0
    compute_ms = 0.0
    for index, item in enumerate(metadata.get("items", [])):
        request_id = str(item["request_id"])
        layer_name = str(item["layer_name"])
        if f"qkv_{index}" in tensors:
            session = registry.get_session(
                registry.resolve_session_request_id(request_id) or request_id
            )
            q_size = (session.q_size if session is not None else None) or int(
                os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0")
            )
            kv_size = (session.kv_size if session is not None else None) or int(
                os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0")
            )
            num_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0"))
            num_kv_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0"))
            head_dim = int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0"))
            if (
                q_size <= 0
                or kv_size <= 0
                or num_heads <= 0
                or num_kv_heads <= 0
                or head_dim <= 0
            ):
                raise RuntimeError(
                    "PAP packed QKV requires q_size, kv_size, num_heads, "
                    "num_kv_heads, and head_dim"
                )
            qkv = tensors[f"qkv_{index}"]
            query_flat, key_flat, value_flat = qkv.split(
                [q_size, kv_size, kv_size],
                dim=-1,
            )
            query = query_flat.reshape(1, num_heads, head_dim)
            key = key_flat.reshape(1, num_kv_heads, head_dim)
            value = value_flat.reshape(1, num_kv_heads, head_dim)
        else:
            query = tensors[f"query_{index}"]
            key = tensors[f"key_{index}"]
            value = tensors[f"value_{index}"]
        block_id = item.get("block_id")
        slot = item.get("slot")
        seq_len_meta = item.get("seq_len")
        trace_append_start = time.perf_counter() if trace_remote_attention else 0.0
        segments, seq_len = registry.append_decode_kv(
            request_id=request_id,
            layer_name=layer_name,
            key=key,
            value=value,
            block_id=None if block_id is None else int(block_id),
            slot=None if slot is None else int(slot),
            seq_len=None if seq_len_meta is None else int(seq_len_meta),
        )
        if trace_remote_attention:
            append_ms += (time.perf_counter() - trace_append_start) * 1000.0
        if torch.cuda.is_available():
            trace_query_start = time.perf_counter() if trace_remote_attention else 0.0
            query = query.to(registry.storage_device, non_blocking=True)
            if trace_remote_attention:
                query_ms += (time.perf_counter() - trace_query_start) * 1000.0
        trace_compute_start = time.perf_counter() if trace_remote_attention else 0.0
        output = compute_segmented_attention_output(
            query=query,
            segments=segments,
            scale=float(item["scale"]),
        )
        if trace_remote_attention:
            compute_ms += (time.perf_counter() - trace_compute_start) * 1000.0
        response_items.append(
            {
                "request_id": request_id,
                "layer_name": layer_name,
                "seq_len": seq_len,
            }
        )
        response_tensors[f"output_{index}"] = output
    trace_serialize_start = time.perf_counter() if trace_remote_attention else 0.0
    response_body = serialize_tensor_bundle({"items": response_items}, response_tensors)
    if trace_remote_attention:
        trace_serialize_ms = (time.perf_counter() - trace_serialize_start) * 1000.0
        trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
        layer_name = (
            str(metadata["items"][0]["layer_name"]) if metadata["items"] else ""
        )
        logger.info(
            "PAP remote attention batch server trace layer=%s calls=%d "
            "deserialize_ms=%.3f append_ms=%.3f query_ms=%.3f compute_ms=%.3f "
            "serialize_ms=%.3f total_ms=%.3f request_bytes=%d response_bytes=%d",
            layer_name,
            len(metadata.get("items", [])),
            trace_deserialize_ms,
            append_ms,
            query_ms,
            compute_ms,
            trace_serialize_ms,
            trace_total_ms,
            len(payload),
            len(response_body),
        )
    return response_body


def compute_compact_attention_response(
    registry: PAPAttentionRegistry,
    payload: bytes,
) -> bytes:
    from vllm.pap.remote_attention import (
        compute_segmented_attention_output,
        deserialize_compact_attention_batch,
        serialize_compact_attention_response,
    )

    trace_remote_attention = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    trace_total_start = time.perf_counter() if trace_remote_attention else 0.0
    trace_deserialize_start = time.perf_counter() if trace_remote_attention else 0.0
    items, qkv_tensors = deserialize_compact_attention_batch(payload)
    trace_deserialize_ms = (
        (time.perf_counter() - trace_deserialize_start) * 1000.0
        if trace_remote_attention
        else 0.0
    )
    outputs: list[torch.Tensor] = []
    append_ms = 0.0
    query_ms = 0.0
    compute_ms = 0.0
    for item, qkv in zip(items, qkv_tensors):
        q_size = int(item["q_size"])
        kv_size = int(item["kv_size"])
        num_heads = int(item["num_heads"])
        num_kv_heads = int(item["num_kv_heads"])
        head_dim = int(item["head_dim"])
        query_flat, key_flat, value_flat = qkv.split(
            [q_size, kv_size, kv_size],
            dim=-1,
        )
        query = query_flat.reshape(1, num_heads, head_dim)
        key = key_flat.reshape(1, num_kv_heads, head_dim)
        value = value_flat.reshape(1, num_kv_heads, head_dim)
        trace_append_start = time.perf_counter() if trace_remote_attention else 0.0
        segments, _seq_len = registry.append_decode_kv(
            request_id=str(item["request_id"]),
            layer_name=str(item["layer_name"]),
            key=key,
            value=value,
            block_id=item.get("block_id"),
            slot=item.get("slot"),
            seq_len=item.get("seq_len"),
        )
        if trace_remote_attention:
            append_ms += (time.perf_counter() - trace_append_start) * 1000.0
        if torch.cuda.is_available():
            trace_query_start = time.perf_counter() if trace_remote_attention else 0.0
            query = query.to(registry.storage_device, non_blocking=True)
            if trace_remote_attention:
                query_ms += (time.perf_counter() - trace_query_start) * 1000.0
        trace_compute_start = time.perf_counter() if trace_remote_attention else 0.0
        outputs.append(
            compute_segmented_attention_output(
                query=query,
                segments=segments,
                scale=float(item["scale"]),
            )
        )
        if trace_remote_attention:
            compute_ms += (time.perf_counter() - trace_compute_start) * 1000.0
    trace_serialize_start = time.perf_counter() if trace_remote_attention else 0.0
    response_body = serialize_compact_attention_response(outputs)
    if trace_remote_attention:
        trace_serialize_ms = (time.perf_counter() - trace_serialize_start) * 1000.0
        trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
        layer_name = str(items[0]["layer_name"]) if items else ""
        logger.info(
            "PAP remote attention compact server trace layer=%s calls=%d "
            "deserialize_ms=%.3f append_ms=%.3f query_ms=%.3f compute_ms=%.3f "
            "serialize_ms=%.3f total_ms=%.3f request_bytes=%d response_bytes=%d",
            layer_name,
            len(items),
            trace_deserialize_ms,
            append_ms,
            query_ms,
            compute_ms,
            trace_serialize_ms,
            trace_total_ms,
            len(payload),
            len(response_body),
        )
    return response_body


def compute_binary_attention_response(
    registry: PAPAttentionRegistry,
    payload: bytes,
    *,
    offload_exec_transport: Any | None = None,
    offload_exec_lock: Any | None = None,
) -> bytes:
    from vllm.pap.data_plane import (
        PAPOffloadExecBatchDescriptor,
        PAPOffloadExecDescriptor,
    )
    from vllm.pap.remote_attention import (
        COMPACT_ATTENTION_REQUEST_MAGIC,
        COMPACT_OFFLOAD_EXEC_BATCH_MAGIC,
        COMPACT_OFFLOAD_EXEC_MAGIC,
        deserialize_compact_offload_exec_batch_command,
        deserialize_compact_offload_exec_command,
        deserialize_tensor_bundle,
        serialize_compact_offload_exec_ack,
        serialize_tensor_bundle,
    )

    if payload.startswith(COMPACT_ATTENTION_REQUEST_MAGIC):
        return compute_compact_attention_response(registry, payload)
    if payload.startswith(COMPACT_OFFLOAD_EXEC_MAGIC):
        if offload_exec_transport is None:
            raise RuntimeError("PAP OFFLOAD_EXEC transport is not initialized")
        metadata = deserialize_compact_offload_exec_command(payload)
        descriptor = PAPOffloadExecDescriptor(
            request_id=str(metadata["request_id"]),
            layer_name=str(metadata["layer_name"]),
            step=int(metadata["step"]),
            scale=float(metadata["scale"]),
        )
        if offload_exec_lock is None:
            run_offload_exec_once(
                registry=registry,
                transport=offload_exec_transport,
                remote_address=str(metadata["remote_address"]),
                descriptor=descriptor,
            )
        else:
            with offload_exec_lock:
                run_offload_exec_once(
                    registry=registry,
                    transport=offload_exec_transport,
                    remote_address=str(metadata["remote_address"]),
                    descriptor=descriptor,
                )
        return serialize_compact_offload_exec_ack()
    if payload.startswith(COMPACT_OFFLOAD_EXEC_BATCH_MAGIC):
        if offload_exec_transport is None:
            raise RuntimeError("PAP OFFLOAD_EXEC transport is not initialized")
        metadata = deserialize_compact_offload_exec_batch_command(payload)
        descriptor = PAPOffloadExecBatchDescriptor(
            layer_name=str(metadata["layer_name"]),
            items=tuple(
                PAPOffloadExecDescriptor(
                    request_id=str(item["request_id"]),
                    layer_name=str(metadata["layer_name"]),
                    step=int(item["step"]),
                    scale=float(item["scale"]),
                )
                for item in metadata["items"]
            ),
        )
        if offload_exec_lock is None:
            run_offload_exec_batch_once(
                registry=registry,
                transport=offload_exec_transport,
                remote_address=str(metadata["remote_address"]),
                descriptor=descriptor,
            )
        else:
            with offload_exec_lock:
                run_offload_exec_batch_once(
                    registry=registry,
                    transport=offload_exec_transport,
                    remote_address=str(metadata["remote_address"]),
                    descriptor=descriptor,
                )
        return serialize_compact_offload_exec_ack()

    metadata, tensors = deserialize_tensor_bundle(payload)
    if metadata.get("command") == "import_prefill_kv":
        seq_len = registry.import_prefill_kv(
            request_id=str(metadata["request_id"]),
            layer_name=str(metadata["layer_name"]),
            key=tensors["key"],
            value=tensors["value"],
            seq_len=int(metadata["seq_len"]),
            block_ids=[int(block_id) for block_id in metadata.get("block_ids", [])],
        )
        return serialize_tensor_bundle(
            {
                "request_id": str(metadata["request_id"]),
                "layer_name": str(metadata["layer_name"]),
                "seq_len": seq_len,
            },
            {},
        )
    if metadata.get("command") == "import_prefill_kv_ipc":
        descriptor = PAPOffloadKVIPCDescriptor.from_dict(metadata["descriptor"])
        key, value = open_ipc_prefill_kv(descriptor)
        seq_len = registry.import_prefill_kv(
            request_id=descriptor.request_id,
            layer_name=descriptor.layer_name,
            key=key,
            value=value,
            seq_len=descriptor.seq_len,
            block_ids=list(descriptor.block_ids),
            copy=False,
        )
        logger.info(
            "PAP prefill KV imported via IPC descriptor request_id=%s "
            "layer=%s seq_len=%s blocks=%s",
            descriptor.request_id,
            descriptor.layer_name,
            seq_len,
            len(descriptor.block_ids),
        )
        return serialize_tensor_bundle(
            {
                "request_id": descriptor.request_id,
                "layer_name": descriptor.layer_name,
                "seq_len": seq_len,
            },
            {},
        )
    if metadata.get("command") == "import_prefill_paged_kv_ipc":
        descriptor = PAPOffloadKVPagedIPCDescriptor.from_dict(metadata["descriptor"])
        async_import = bool(metadata.get("async", False)) or _pap_prefill_kv_async_enabled()
        if async_import:
            seq_len = registry.enqueue_prefill_paged_kv_descriptor(descriptor)
            return serialize_tensor_bundle(
                {
                    "request_id": descriptor.request_id,
                    "layer_name": descriptor.layer_name,
                    "seq_len": seq_len,
                    "status": "queued",
                },
                {},
            )
        kv_cache = open_ipc_paged_kv_cache(descriptor)
        seq_len = registry.import_prefill_paged_kv(
            request_id=descriptor.request_id,
            layer_name=descriptor.layer_name,
            kv_cache=kv_cache,
            block_ids=list(descriptor.block_ids),
            seq_len=descriptor.seq_len,
            block_size=descriptor.block_size,
            num_kv_heads=descriptor.num_kv_heads,
            layout=descriptor.layout,
            lease_id=descriptor.lease_id,
            leased_block_ids=descriptor.leased_block_ids,
            lease_capacity_tokens=descriptor.lease_capacity_tokens,
            unified_kv_mode=descriptor.unified_kv_mode,
            prefix_len=descriptor.prefix_len,
            writable_start_token=descriptor.writable_start_token,
            writable_end_token=descriptor.writable_end_token,
        )
        logger.info(
            "PAP prefill paged KV imported via IPC descriptor request_id=%s "
            "layer=%s seq_len=%s blocks=%s",
            descriptor.request_id,
            descriptor.layer_name,
            seq_len,
            len(descriptor.block_ids),
        )
        return serialize_tensor_bundle(
            {
                "request_id": descriptor.request_id,
                "layer_name": descriptor.layer_name,
                "seq_len": seq_len,
                "status": "ready",
            },
            {},
        )
    if metadata.get("command") == "offload_exec":
        if offload_exec_transport is None:
            raise RuntimeError("PAP OFFLOAD_EXEC transport is not initialized")
        descriptor = PAPOffloadExecDescriptor(
            request_id=str(metadata["request_id"]),
            layer_name=str(metadata["layer_name"]),
            step=int(metadata["step"]),
            scale=float(metadata["scale"]),
        )
        if offload_exec_lock is None:
            run_offload_exec_once(
                registry=registry,
                transport=offload_exec_transport,
                remote_address=str(metadata["remote_address"]),
                descriptor=descriptor,
            )
        else:
            with offload_exec_lock:
                run_offload_exec_once(
                    registry=registry,
                    transport=offload_exec_transport,
                    remote_address=str(metadata["remote_address"]),
                    descriptor=descriptor,
                )
        return serialize_tensor_bundle(
            {
                "request_id": descriptor.request_id,
                "layer_name": descriptor.layer_name,
                "step": descriptor.step,
                "remote_address": str(metadata["remote_address"]),
            },
            {},
        )
    if "items" in metadata:
        return compute_batch_binary_attention_response(registry, payload)
    return _compute_single_binary_attention_response(registry, payload)


def _offload_exec_attention_shapes(
    *,
    session: PAPAttentionSession,
) -> tuple[int, int, int, int, int]:
    q_size = session.q_size or int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0"))
    kv_size = session.kv_size or int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0"))
    num_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0"))
    num_kv_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0"))
    head_dim = int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0"))
    if q_size <= 0 or kv_size <= 0:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC requires q_size and kv_size in attention "
            "registration or PAP_OFFLOAD_EXEC_Q_SIZE/PAP_OFFLOAD_EXEC_KV_SIZE"
        )
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC requires PAP_OFFLOAD_EXEC_NUM_HEADS, "
            "PAP_OFFLOAD_EXEC_NUM_KV_HEADS, and PAP_OFFLOAD_EXEC_HEAD_DIM"
        )
    return q_size, kv_size, num_heads, num_kv_heads, head_dim


def _offload_exec_session(
    *,
    registry: PAPAttentionRegistry,
    request_id: str,
) -> tuple[str, PAPAttentionSession]:
    session_request_id = registry.resolve_session_request_id(request_id)
    if session_request_id is None:
        raise KeyError(request_id)
    session = registry.get_session(session_request_id)
    if session is None:
        raise KeyError(request_id)
    return session_request_id, session


def compute_offload_exec_query_partial(
    *,
    registry: PAPAttentionRegistry,
    request_id: str,
    layer_name: str,
    query_flat: torch.Tensor,
    scale: float,
    step: int,
) -> Any | None:
    from vllm.pap.remote_attention import compute_segmented_attention_partial_state

    session_request_id, session = _offload_exec_session(
        registry=registry,
        request_id=request_id,
    )
    q_size, _kv_size, num_heads, _num_kv_heads, head_dim = (
        _offload_exec_attention_shapes(session=session)
    )
    if int(query_flat.shape[-1]) != q_size:
        raise ValueError(
            f"query width {query_flat.shape[-1]} does not match q_size={q_size}"
        )
    if query_flat.shape[0] != 1:
        raise RuntimeError("PAP OFFLOAD_EXEC currently supports one token per call")
    query = query_flat.view(1, num_heads, head_dim)
    segments = registry.attention_segments_before_decode(
        request_id=session_request_id,
        layer_name=layer_name,
        seq_len=int(step),
    )
    non_empty_segments = [(key, value) for key, value in segments if key.numel() > 0]
    if not non_empty_segments:
        return None
    if torch.cuda.is_available():
        query = query.to(registry.storage_device, non_blocking=True)
    return compute_segmented_attention_partial_state(
        query=query,
        segments=non_empty_segments,
        scale=float(scale),
    )


def compute_offload_exec_output_from_kv_and_partial(
    *,
    registry: PAPAttentionRegistry,
    request_id: str,
    layer_name: str,
    query_flat: torch.Tensor,
    kv_flat: torch.Tensor,
    scale: float,
    step: int,
    partial_state: Any | None,
) -> torch.Tensor:
    from vllm.pap.remote_attention import (
        combine_segmented_attention_partial_states,
        compute_segmented_attention_output,
        compute_segmented_attention_partial_state,
    )

    session_request_id, session = _offload_exec_session(
        registry=registry,
        request_id=request_id,
    )
    q_size, kv_size, num_heads, num_kv_heads, head_dim = _offload_exec_attention_shapes(
        session=session
    )
    if int(query_flat.shape[-1]) != q_size:
        raise ValueError(
            f"query width {query_flat.shape[-1]} does not match q_size={q_size}"
        )
    if int(kv_flat.shape[-1]) != kv_size + kv_size:
        raise ValueError(
            f"packed kv width {kv_flat.shape[-1]} does not match kv_size={kv_size}"
        )
    if query_flat.shape[0] != 1 or kv_flat.shape[0] != 1:
        raise RuntimeError("PAP OFFLOAD_EXEC currently supports one token per call")

    query = query_flat.view(1, num_heads, head_dim)
    key_flat, value_flat = kv_flat.split([kv_size, kv_size], dim=-1)
    key = key_flat.view(1, num_kv_heads, head_dim)
    value = value_flat.view(1, num_kv_heads, head_dim)
    seq_len = int(step)
    if seq_len <= 0:
        raise ValueError("PAP OFFLOAD_EXEC step must be positive")
    block_id, slot = registry.reserve_decode_slot(
        request_id=session_request_id,
        layer_name=layer_name,
        seq_len=seq_len,
    )
    registry.append_decode_kv(
        request_id=session_request_id,
        layer_name=layer_name,
        key=key,
        value=value,
        block_id=block_id,
        slot=slot,
        seq_len=seq_len,
    )
    if torch.cuda.is_available():
        query = query.to(registry.storage_device, non_blocking=True)
    if partial_state is None:
        output = compute_segmented_attention_output(
            query=query,
            segments=[(key, value)],
            scale=float(scale),
        )
    else:
        current_state = compute_segmented_attention_partial_state(
            query=query,
            segments=[(key, value)],
            scale=float(scale),
        )
        output = combine_segmented_attention_partial_states(
            [partial_state, current_state]
        )
    return output.reshape(1, -1)


def compute_offload_exec_output(
    *,
    registry: PAPAttentionRegistry,
    request_id: str,
    layer_name: str,
    qkv: torch.Tensor,
    scale: float,
    step: int,
) -> torch.Tensor:
    """Compute one OFFLOAD_EXEC attention output from a packed QKV tensor."""

    from vllm.pap.remote_attention import compute_segmented_attention_output

    session_request_id = registry.resolve_session_request_id(request_id)
    if session_request_id is None:
        raise KeyError(request_id)
    session = registry.get_session(session_request_id)
    if session is None:
        raise KeyError(request_id)
    q_size = session.q_size or int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0"))
    kv_size = session.kv_size or int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0"))
    if q_size <= 0 or kv_size <= 0:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC requires q_size and kv_size in attention "
            "registration or PAP_OFFLOAD_EXEC_Q_SIZE/PAP_OFFLOAD_EXEC_KV_SIZE"
        )
    if int(qkv.shape[-1]) != q_size + kv_size + kv_size:
        raise ValueError(
            f"packed qkv width {qkv.shape[-1]} does not match "
            f"q_size={q_size} kv_size={kv_size}"
        )
    query_flat, key_flat, value_flat = qkv.split([q_size, kv_size, kv_size], dim=-1)
    if query_flat.shape[0] != 1:
        raise RuntimeError("PAP OFFLOAD_EXEC currently supports one token per call")
    num_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0"))
    num_kv_heads = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0"))
    head_dim = int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0"))
    if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC requires PAP_OFFLOAD_EXEC_NUM_HEADS, "
            "PAP_OFFLOAD_EXEC_NUM_KV_HEADS, and PAP_OFFLOAD_EXEC_HEAD_DIM"
        )
    query = query_flat.view(1, num_heads, head_dim)
    key = key_flat.view(1, num_kv_heads, head_dim)
    value = value_flat.view(1, num_kv_heads, head_dim)
    seq_len = int(step)
    if seq_len <= 0:
        raise ValueError("PAP OFFLOAD_EXEC step must be positive")
    block_id, slot = registry.reserve_decode_slot(
        request_id=session_request_id,
        layer_name=layer_name,
        seq_len=seq_len,
    )
    segments, _ = registry.append_decode_kv(
        request_id=session_request_id,
        layer_name=layer_name,
        key=key,
        value=value,
        block_id=block_id,
        slot=slot,
        seq_len=seq_len,
    )
    if torch.cuda.is_available():
        query = query.to(registry.storage_device, non_blocking=True)
    output = compute_segmented_attention_output(
        query=query,
        segments=segments,
        scale=float(scale),
    )
    return output.reshape(1, -1)


def _paged_flash_prefix_cache_group_key(prefix_state: PAPPrefillPagedKV) -> tuple[Any, ...]:
    kv_cache = prefix_state.kv_cache
    return (
        int(kv_cache.data_ptr()),
        tuple(int(dim) for dim in kv_cache.shape),
        tuple(int(stride) for stride in kv_cache.stride()),
        kv_cache.dtype,
        kv_cache.device,
        str(prefix_state.layout),
        int(prefix_state.num_kv_heads),
    )



def _paged_flash_kv_cache_views(
    prefix_state: PAPPrefillPagedKV,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_cache = prefix_state.kv_cache
    if kv_cache.ndim != 5:
        raise RuntimeError("PAP zero-copy prefix KV requires a 5D paged KV cache")
    if int(kv_cache.shape[0]) == 2:
        kv_dim = 0
    elif int(kv_cache.shape[1]) == 2:
        kv_dim = 1
    else:
        raise RuntimeError(
            "PAP zero-copy prefix KV requires a paged KV cache with a K/V dimension"
        )

    if prefix_state.layout == "NHD":
        logical_nhd = True
    elif prefix_state.layout == "HND":
        logical_nhd = int(kv_cache.shape[3]) == int(prefix_state.num_kv_heads)
    else:
        raise RuntimeError(
            f"PAP zero-copy prefix KV does not support layout {prefix_state.layout!r}"
        )
    if not logical_nhd:
        raise RuntimeError(
            "PAP zero-copy prefix KV requires logical NHD paged layout; "
            "set PAP_ATTENTION_COPY_PREFIX_KV=1 to fall back to copied prefix KV"
        )
    if kv_dim == 0:
        return kv_cache[0], kv_cache[1]
    return kv_cache[:, 0], kv_cache[:, 1]



def _run_paged_flash_varlen(
    *,
    flash_attn_varlen_func: Any,
    fa_version: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: PAPPagedFlashMetadata,
    scale: float,
    causal: bool,
    return_softmax_lse: bool,
) -> Any:
    output = torch.empty_like(query)
    return flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=metadata.cu_seqlens_q,
        seqused_k=metadata.seq_lens,
        max_seqlen_q=1,
        max_seqlen_k=metadata.max_seq_len,
        softmax_scale=float(scale),
        causal=causal,
        block_table=metadata.block_table,
        softcap=0.0,
        return_softmax_lse=return_softmax_lse,
        fa_version=fa_version,
    )



def _compute_unified_paged_flash_batch(
    *,
    query_batch: torch.Tensor,
    states: list[PAPUnifiedPagedKVState],
    scale: float,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor | None:
    """Single-source Prefill-owned paged FA compute (Stage 4 unified path)."""
    if not states:
        return None
    if not query_batch.is_cuda:
        return None
    base_kv = states[0].kv_cache
    if any(
        state.kv_cache.device != base_kv.device
        or state.kv_cache.shape != base_kv.shape
        or state.kv_cache.dtype != base_kv.dtype
        for state in states
    ):
        return None
    if base_kv.device != query_batch.device:
        return None

    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            get_flash_attn_version,
            is_flash_attn_varlen_func_available,
        )
    except Exception:
        return None
    if not is_flash_attn_varlen_func_available():
        return None

    metadata = build_unified_paged_flash_metadata(
        states=states, device=query_batch.device
    )
    if metadata.max_seq_len <= 0:
        return None

    fa_version = get_flash_attn_version(head_size=int(query_batch.shape[-1]))
    key_cache, value_cache = base_kv.unbind(1)
    metadata_start = time.perf_counter() if trace_stats is not None else 0.0
    paged_start = time.perf_counter() if trace_stats is not None else 0.0
    result = _run_paged_flash_varlen(
        flash_attn_varlen_func=flash_attn_varlen_func,
        fa_version=fa_version,
        query=query_batch,
        key_cache=key_cache,
        value_cache=value_cache,
        metadata=metadata,
        scale=float(scale),
        causal=True,
        return_softmax_lse=False,
    )
    output = result[0] if isinstance(result, tuple) else result
    if trace_stats is not None:
        trace_stats["unified_metadata_ms"] = (
            trace_stats.get("unified_metadata_ms", 0.0)
            + (time.perf_counter() - metadata_start) * 1000.0
        )
        trace_stats["unified_paged_flash_ms"] = (
            trace_stats.get("unified_paged_flash_ms", 0.0)
            + (time.perf_counter() - paged_start) * 1000.0
        )
    return output


def _compute_offload_exec_paged_flash_batch(
    *,
    query_batch: torch.Tensor,
    states: list[PAPLocalPagedAttentionState | None],
    scale: float,
    allow_descriptor_prefix: bool = True,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor | None:
    if not states or any(state is None for state in states):
        return None
    if not query_batch.is_cuda:
        return None

    first_state = states[0]
    assert first_state is not None
    pool = first_state.pool
    if pool.kv_cache.device != query_batch.device:
        return None
    if any(state is None or state.pool is not pool for state in states):
        return None

    try:
        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
            get_flash_attn_version,
            is_flash_attn_varlen_func_available,
        )
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
    except Exception:
        return None
    if not is_flash_attn_varlen_func_available():
        return None

    typed_states = [state for state in states if state is not None]
    has_descriptor_prefix = any(state.has_descriptor_prefix for state in typed_states)
    if trace_stats is not None:
        trace_stats["pre_compute_done_ns"] = float(time.perf_counter_ns())
    fa_version = get_flash_attn_version(head_size=int(query_batch.shape[-1]))

    if not has_descriptor_prefix:
        metadata_start = time.perf_counter() if trace_stats is not None else 0.0
        metadata = build_paged_flash_metadata(
            states=typed_states,
            device=query_batch.device,
        )
        if trace_stats is not None:
            trace_stats["paged_metadata_ms"] = (
                trace_stats.get("paged_metadata_ms", 0.0)
                + (time.perf_counter() - metadata_start) * 1000.0
            )
        if metadata.max_seq_len <= 0:
            return None

        paged_start = time.perf_counter() if trace_stats is not None else 0.0
        result = _run_paged_flash_varlen(
            flash_attn_varlen_func=flash_attn_varlen_func,
            fa_version=fa_version,
            query=query_batch,
            key_cache=pool.kv_cache[:, 0],
            value_cache=pool.kv_cache[:, 1],
            metadata=metadata,
            scale=float(scale),
            causal=True,
            return_softmax_lse=False,
        )
        if trace_stats is not None:
            trace_stats["paged_flash_ms"] = (
                trace_stats.get("paged_flash_ms", 0.0)
                + (time.perf_counter() - paged_start) * 1000.0
            )
            trace_stats["paged_flash_done_ns"] = float(time.perf_counter_ns())
        return result

    if not allow_descriptor_prefix:
        return None

    if fa_version not in (3, 4):
        return None

    suffix_states = [state for state in typed_states if state.local_seq_len > 0]
    suffix_output: torch.Tensor | None = None
    suffix_lse: torch.Tensor | None = None
    if suffix_states:
        suffix_metadata_start = time.perf_counter() if trace_stats is not None else 0.0
        suffix_metadata = build_paged_flash_metadata(
            states=typed_states,
            device=query_batch.device,
            use_base_seq_len=True,
        )
        if trace_stats is not None:
            trace_stats["paged_metadata_ms"] = (
                trace_stats.get("paged_metadata_ms", 0.0)
                + (time.perf_counter() - suffix_metadata_start) * 1000.0
            )
        if suffix_metadata.max_seq_len <= 0:
            return None

        paged_start = time.perf_counter() if trace_stats is not None else 0.0
        suffix_result = _run_paged_flash_varlen(
            flash_attn_varlen_func=flash_attn_varlen_func,
            fa_version=fa_version,
            query=query_batch,
            key_cache=pool.kv_cache[:, 0],
            value_cache=pool.kv_cache[:, 1],
            metadata=suffix_metadata,
            scale=float(scale),
            causal=True,
            return_softmax_lse=True,
        )
        suffix_output, suffix_lse = suffix_result
        if trace_stats is not None:
            trace_stats["paged_flash_ms"] = (
                trace_stats.get("paged_flash_ms", 0.0)
                + (time.perf_counter() - paged_start) * 1000.0
            )

    output = torch.empty_like(query_batch)
    prefix_row_indices = [
        index for index, state in enumerate(typed_states) if state.has_descriptor_prefix
    ]
    prefix_groups: dict[tuple[Any, ...], list[int]] = {}
    for index in prefix_row_indices:
        prefix_state = typed_states[index].prefix_state
        assert prefix_state is not None
        prefix_groups.setdefault(
            _paged_flash_prefix_cache_group_key(prefix_state),
            [],
        ).append(index)

    merge_start = time.perf_counter() if trace_stats is not None else 0.0
    for row_indices in prefix_groups.values():
        row_index_tensor = torch.tensor(
            row_indices,
            dtype=torch.long,
            device=query_batch.device,
        )
        group_query = query_batch.index_select(0, row_index_tensor)
        group_prefix_states = []
        for row_index in row_indices:
            prefix_state = typed_states[row_index].prefix_state
            assert prefix_state is not None
            group_prefix_states.append(prefix_state)
        prefix_metadata_start = time.perf_counter() if trace_stats is not None else 0.0
        prefix_metadata = build_paged_flash_metadata(
            states=group_prefix_states,
            device=query_batch.device,
        )
        if trace_stats is not None:
            trace_stats["paged_metadata_ms"] = (
                trace_stats.get("paged_metadata_ms", 0.0)
                + (time.perf_counter() - prefix_metadata_start) * 1000.0
            )
        if prefix_metadata.max_seq_len <= 0:
            continue
        prefix_key_cache, prefix_value_cache = _paged_flash_kv_cache_views(
            group_prefix_states[0]
        )
        if (
            prefix_key_cache.device != query_batch.device
            or prefix_value_cache.device != query_batch.device
        ):
            raise RuntimeError(
                "PAP zero-copy prefix KV device does not match attention query device"
            )
        prefix_paged_start = time.perf_counter() if trace_stats is not None else 0.0
        prefix_result = _run_paged_flash_varlen(
            flash_attn_varlen_func=flash_attn_varlen_func,
            fa_version=fa_version,
            query=group_query,
            key_cache=prefix_key_cache,
            value_cache=prefix_value_cache,
            metadata=prefix_metadata,
            scale=float(scale),
            causal=False,
            return_softmax_lse=True,
        )
        prefix_output, prefix_lse = prefix_result
        if trace_stats is not None:
            trace_stats["paged_flash_ms"] = (
                trace_stats.get("paged_flash_ms", 0.0)
                + (time.perf_counter() - prefix_paged_start) * 1000.0
            )
        if suffix_output is None or suffix_lse is None:
            output.index_copy_(0, row_index_tensor, prefix_output)
            continue
        suffix_group_output = suffix_output.index_select(0, row_index_tensor)
        suffix_group_lse = suffix_lse.index_select(1, row_index_tensor)
        merged_group_output = torch.empty_like(prefix_output)
        merge_attn_states(
            merged_group_output,
            prefix_output,
            prefix_lse,
            suffix_group_output,
            suffix_group_lse,
        )
        output.index_copy_(0, row_index_tensor, merged_group_output)
    if trace_stats is not None:
        trace_stats["paged_flash_done_ns"] = float(time.perf_counter_ns())
        trace_stats["reshape_ms"] = trace_stats.get("reshape_ms", 0.0) + 0.0
        trace_stats["fallback_ms"] = trace_stats.get("fallback_ms", 0.0) + (
            (time.perf_counter() - merge_start) * 1000.0
        )
    return output


def compute_offload_exec_batch_output(
    *,
    registry: PAPAttentionRegistry,
    descriptor: Any,
    qkv_batch: torch.Tensor,
    trace_stats: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute one OFFLOAD_EXEC attention output batch via paged FlashAttention."""

    items = tuple(descriptor.items)
    if int(qkv_batch.shape[0]) != len(items):
        raise RuntimeError(
            "PAP OFFLOAD_EXEC batch QKV row count does not match descriptor"
        )
    if trace_stats is not None:
        trace_stats["pre_compute_start_ns"] = float(time.perf_counter_ns())

    shape_lookup_start = time.perf_counter() if trace_stats is not None else 0.0
    num_heads_default = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_HEADS", "0"))
    num_kv_heads_default = int(os.environ.get("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "0"))
    head_dim_default = int(os.environ.get("PAP_OFFLOAD_EXEC_HEAD_DIM", "0"))
    session_entries = registry.offload_exec_batch_session_entries(
        tuple(item.request_id for item in items),
        default_q_size=int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0")),
        default_kv_size=int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0")),
        num_heads=num_heads_default,
        num_kv_heads=num_kv_heads_default,
        head_dim=head_dim_default,
    )

    common_shape: tuple[int, int, int, int, int] | None = None
    common_scale: float | None = None
    for item, session_entry in zip(items, session_entries):
        shape = (
            session_entry.q_size, session_entry.kv_size,
            session_entry.num_heads, session_entry.num_kv_heads, session_entry.head_dim,
        )
        scale = float(item.scale)
        if common_shape is None:
            common_shape = shape
            common_scale = scale
        elif shape != common_shape or scale != common_scale:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC batch has mixed shapes or scales"
            )
    if trace_stats is not None:
        trace_stats["shape_lookup_ms"] += (
            time.perf_counter() - shape_lookup_start
        ) * 1000.0

    assert common_shape is not None
    _q_size, kv_size, num_heads, num_kv_heads, head_dim = common_shape
    batch_size = len(items)

    qkv_split_start = time.perf_counter() if trace_stats is not None else 0.0
    query_flat, key_flat, value_flat = qkv_batch.split(
        [_q_size, kv_size, kv_size], dim=-1,
    )
    query_batch = query_flat.view(batch_size, num_heads, head_dim)
    key_batch = key_flat.view(batch_size, num_kv_heads, head_dim)
    value_batch = value_flat.view(batch_size, num_kv_heads, head_dim)
    if trace_stats is not None:
        trace_stats["qkv_split_ms"] += (
            time.perf_counter() - qkv_split_start
        ) * 1000.0

    query_move_start = time.perf_counter() if trace_stats is not None else 0.0
    if torch.cuda.is_available():
        query_batch = query_batch.to(registry.storage_device, non_blocking=True)
    if trace_stats is not None:
        trace_stats["query_move_ms"] += (
            time.perf_counter() - query_move_start
        ) * 1000.0

    append_start = time.perf_counter() if trace_stats is not None else 0.0
    decode_seq_lens = [int(item.step) for item in items]
    session_request_ids = tuple(
        session_entry.session_request_id for session_entry in session_entries
    )

    unified_states = registry.get_unified_paged_states(
        session_request_ids=session_request_ids,
        layer_name=descriptor.layer_name,
    )
    if unified_states is not None:
        written = registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=session_request_ids,
            layer_name=descriptor.layer_name,
            key_batch=key_batch,
            value_batch=value_batch,
            decode_seq_lens=decode_seq_lens,
        )
        if written <= 0:
            raise RuntimeError(
                "PAP unified KV append wrote no rows"
            )
        if trace_stats is not None:
            trace_stats["append_kv_ms"] += (
                time.perf_counter() - append_start
            ) * 1000.0
        unified_output = _compute_unified_paged_flash_batch(
            query_batch=query_batch,
            states=unified_states,
            scale=common_scale,
            trace_stats=trace_stats,
        )
        if unified_output is None:
            raise RuntimeError("PAP unified paged FlashAttention failed")
        if unified_output.ndim == 3:
            unified_output = unified_output.reshape(
                batch_size, num_heads * head_dim
            )
        if trace_stats is not None:
            trace_stats["reshape_ms"] = (
                trace_stats.get("reshape_ms", 0.0)
                + (time.perf_counter() - append_start) * 0.0
            )
        return unified_output

    if _pap_attention_copy_prefix_kv():
        registry.materialize_descriptor_prefix_for_local_paged_attention(
            session_request_ids=session_request_ids,
            layer_name=descriptor.layer_name,
        )
    paged_states = registry.append_decode_kv_tensor_batch_for_local_paged_attention(
        session_request_ids=session_request_ids,
        layer_name=descriptor.layer_name,
        key_batch=key_batch,
        value_batch=value_batch,
        seq_lens=tuple(decode_seq_lens),
        trace_stats=trace_stats,
    )
    if trace_stats is not None:
        trace_stats["append_kv_ms"] += (
            time.perf_counter() - append_start
        ) * 1000.0

    if paged_states is None:
        raise RuntimeError("PAP local paged cache not available for decode attention")

    paged_output = _compute_offload_exec_paged_flash_batch(
        query_batch=query_batch,
        states=list(paged_states),
        scale=common_scale,
        allow_descriptor_prefix=True,
        trace_stats=trace_stats,
    )
    if paged_output is None:
        raise RuntimeError("PAP paged FlashAttention failed")

    reshape_start = time.perf_counter() if trace_stats is not None else 0.0
    if paged_output.ndim == 3:
        paged_output = paged_output.reshape(batch_size, num_heads * head_dim)
    elif paged_output.ndim != 2:
        raise RuntimeError("PAP paged FlashAttention output has invalid shape")
    if trace_stats is not None:
        trace_stats["reshape_ms"] += (
            time.perf_counter() - reshape_start
        ) * 1000.0
        trace_stats["compute_done_ns"] = float(time.perf_counter_ns())
        trace_stats["post_compute_done_ns"] = float(time.perf_counter_ns())
    return paged_output


def run_offload_exec_once(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
    remote_address: str,
    descriptor: Any,
) -> None:
    """Receive packed QKV over OFFLOAD_EXEC and send attention output back."""

    trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
    trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
    logger.debug(
        "PAP OFFLOAD_EXEC recv_qkv request_id=%s layer=%s step=%s remote=%s",
        descriptor.request_id,
        descriptor.layer_name,
        descriptor.step,
        remote_address,
    )
    qkv = transport.recv_qkv(descriptor, remote_address=remote_address)
    trace_recv_ms = (
        (time.perf_counter() - trace_recv_start) * 1000.0 if trace_offload_exec else 0.0
    )
    trace_compute_start = time.perf_counter() if trace_offload_exec else 0.0
    logger.debug(
        "PAP OFFLOAD_EXEC compute request_id=%s layer=%s step=%s qkv_shape=%s",
        descriptor.request_id,
        descriptor.layer_name,
        descriptor.step,
        tuple(qkv.shape),
    )
    output = compute_offload_exec_output(
        registry=registry,
        request_id=descriptor.request_id,
        layer_name=descriptor.layer_name,
        qkv=qkv,
        scale=descriptor.scale,
        step=descriptor.step,
    )
    trace_compute_ms = (
        (time.perf_counter() - trace_compute_start) * 1000.0
        if trace_offload_exec
        else 0.0
    )
    trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
    logger.debug(
        "PAP OFFLOAD_EXEC send_output request_id=%s layer=%s step=%s "
        "output_shape=%s remote=%s",
        descriptor.request_id,
        descriptor.layer_name,
        descriptor.step,
        tuple(output.shape),
        remote_address,
    )
    transport.send_output(descriptor, output, remote_address=remote_address)
    if trace_offload_exec:
        trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
        trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
        logger.info(
            "PAP OFFLOAD_EXEC attention trace request_id=%s layer=%s step=%s "
            "recv_qkv_ms=%.3f compute_ms=%.3f send_output_ms=%.3f "
            "total_ms=%.3f qkv_shape=%s output_shape=%s",
            descriptor.request_id,
            descriptor.layer_name,
            descriptor.step,
            trace_recv_ms,
            trace_compute_ms,
            trace_send_ms,
            trace_total_ms,
            tuple(qkv.shape),
            tuple(output.shape),
        )
    logger.debug(
        "PAP OFFLOAD_EXEC complete request_id=%s layer=%s step=%s",
        descriptor.request_id,
        descriptor.layer_name,
        descriptor.step,
    )


def _combine_offload_exec_outputs(outputs: list[torch.Tensor]) -> torch.Tensor:
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=0)


def run_offload_exec_batch_once(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
    remote_address: str,
    descriptor: Any,
) -> None:
    """Receive one batched QKV tensor and send one batched attention output."""

    trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
    trace_recv_start_ns = 0
    trace_recv_done_ns = 0
    trace_compute_done_ns = 0
    trace_send_start_ns = 0
    trace_send_done_ns = 0
    trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
    if trace_offload_exec:
        trace_recv_start_ns = time.perf_counter_ns()
    qkv_batch = transport.recv_qkv_batch(
        descriptor,
        remote_address=remote_address,
    )
    trace_recv_ms = (
        (time.perf_counter() - trace_recv_start) * 1000.0 if trace_offload_exec else 0.0
    )
    if trace_offload_exec:
        trace_recv_done_ns = time.perf_counter_ns()
    if int(qkv_batch.shape[0]) != len(descriptor.items):
        raise RuntimeError(
            "PAP OFFLOAD_EXEC batch QKV row count does not match descriptor"
        )
    trace_compute_start = time.perf_counter() if trace_offload_exec else 0.0
    trace_compute_stats = (
        {
            "append_kv_ms": 0.0,
            "pack_ms": 0.0,
            "sdpa_ms": 0.0,
            "reshape_ms": 0.0,
            "paged_metadata_ms": 0.0,
            "paged_flash_ms": 0.0,
            "fallback_ms": 0.0,
            "shape_lookup_ms": 0.0,
            "qkv_split_ms": 0.0,
            "query_move_ms": 0.0,
            "query_cat_ms": 0.0,
            "append_lock_wait_ms": 0.0,
            "append_prepare_ms": 0.0,
            "append_record_ms": 0.0,
            "append_tensor_ms": 0.0,
            "append_copy_ms": 0.0,
            "append_state_ms": 0.0,
            "pre_compute_start_ns": 0.0,
            "pre_compute_done_ns": 0.0,
            "paged_flash_done_ns": 0.0,
            "post_compute_done_ns": 0.0,
        }
        if trace_offload_exec
        else None
    )
    output_batch = compute_offload_exec_batch_output(
        registry=registry,
        descriptor=descriptor,
        qkv_batch=qkv_batch,
        trace_stats=trace_compute_stats,
    )
    trace_compute_ms = (
        (time.perf_counter() - trace_compute_start) * 1000.0
        if trace_offload_exec
        else 0.0
    )
    if trace_offload_exec:
        trace_compute_done_ns = time.perf_counter_ns()
    trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
    if trace_offload_exec:
        trace_send_start_ns = time.perf_counter_ns()
    transport.send_output_batch(
        descriptor,
        output_batch,
        remote_address=remote_address,
    )
    if trace_offload_exec:
        trace_send_done_ns = time.perf_counter_ns()
        trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
        trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
        logger.info(
            "PAP OFFLOAD_EXEC attention batch trace layer=%s calls=%d "
            "recv_qkv_ms=%.3f compute_ms=%.3f send_output_ms=%.3f "
            "total_ms=%.3f append_kv_ms=%.3f pack_ms=%.3f "
            "sdpa_ms=%.3f reshape_ms=%.3f paged_metadata_ms=%.3f "
            "paged_flash_ms=%.3f fallback_ms=%.3f shape_lookup_ms=%.3f "
            "qkv_split_ms=%.3f query_move_ms=%.3f query_cat_ms=%.3f "
            "append_lock_wait_ms=%.3f append_prepare_ms=%.3f "
            "append_record_ms=%.3f append_tensor_ms=%.3f "
            "append_copy_ms=%.3f append_state_ms=%.3f "
            "qkv_shape=%s output_shape=%s batch_key=%s "
            "recv_done_ns=%d compute_done_ns=%d send_done_ns=%d "
            "recv_start_ns=%d pre_compute_start_ns=%d "
            "pre_compute_done_ns=%d paged_flash_done_ns=%d reshape_done_ns=%d "
            "send_start_ns=%d",
            descriptor.layer_name,
            len(descriptor.items),
            trace_recv_ms,
            trace_compute_ms,
            trace_send_ms,
            trace_total_ms,
            trace_compute_stats["append_kv_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["pack_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["sdpa_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["reshape_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["paged_metadata_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["paged_flash_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["fallback_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["shape_lookup_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["qkv_split_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["query_move_ms"] if trace_compute_stats else 0.0,
            trace_compute_stats["query_cat_ms"] if trace_compute_stats else 0.0,
            (
                trace_compute_stats["append_lock_wait_ms"]
                if trace_compute_stats
                else 0.0
            ),
            (trace_compute_stats["append_prepare_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_record_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_tensor_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_copy_ms"] if trace_compute_stats else 0.0),
            (trace_compute_stats["append_state_ms"] if trace_compute_stats else 0.0),
            tuple(qkv_batch.shape),
            tuple(output_batch.shape),
            pap_offload_exec_trace_id(descriptor.output_tensor_id),
            trace_recv_done_ns,
            trace_compute_done_ns,
            trace_send_done_ns,
            trace_recv_start_ns,
            int(trace_compute_stats.get("pre_compute_start_ns", 0.0)) if trace_compute_stats else 0,
            int(trace_compute_stats.get("pre_compute_done_ns", 0.0)) if trace_compute_stats else 0,
            int(trace_compute_stats.get("paged_flash_done_ns", 0.0)) if trace_compute_stats else 0,
            int(trace_compute_stats.get("post_compute_done_ns", 0.0)) if trace_compute_stats else 0,
            trace_send_start_ns,
        )


def _recv_next_qkv_batch_message_or_tensor(
    transport: Any,
) -> tuple[Any, Any | None, torch.Tensor]:
    recv_message_fn = getattr(transport, "recv_next_qkv_batch_message", None)
    if callable(recv_message_fn):
        descriptor, qkv_message = recv_message_fn()
        return descriptor, qkv_message, qkv_message.tensor
    descriptor, qkv_batch = transport.recv_next_qkv_batch()
    return descriptor, None, qkv_batch


class _QKVBatchMessagePrefetcher:
    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self._requests: Queue[object] = Queue()
        self._results: Queue[tuple[bool, Any]] = Queue(maxsize=1)
        self._stop = object()
        self._thread = Thread(
            target=self._run,
            name="pap-attention-mailbox-prefetch",
            daemon=True,
        )
        self._thread.start()

    def prefetch(self) -> None:
        self._requests.put(None)

    def result(self) -> tuple[Any, Any | None, torch.Tensor]:
        ok, payload = self._results.get()
        if ok:
            return payload
        raise payload

    def close(self) -> None:
        self._requests.put(self._stop)

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is self._stop:
                return
            try:
                payload = _recv_next_qkv_batch_message_or_tensor(self._transport)
            except BaseException as exc:
                self._results.put((False, exc))
            else:
                self._results.put((True, payload))


def run_offload_exec_mailbox_loop(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
) -> None:
    """Consume mailbox QKV messages and publish mailbox attention outputs."""

    trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    prefetch_enabled = (
        _pap_env_flag("PAP_ATTENTION_MAILBOX_PREFETCH", False)
        and callable(getattr(transport, "recv_next_qkv_batch_message", None))
    )
    prefetcher = _QKVBatchMessagePrefetcher(transport) if prefetch_enabled else None
    if prefetcher is not None:
        prefetcher.prefetch()
    while True:
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_recv_start_ns = 0
        trace_recv_done_ns = 0
        trace_compute_done_ns = 0
        trace_send_start_ns = 0
        trace_send_done_ns = 0
        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
        if trace_offload_exec:
            trace_recv_start_ns = time.perf_counter_ns()
        qkv_message = None
        if prefetcher is not None:
            descriptor, qkv_message, qkv_batch = prefetcher.result()
            prefetcher.prefetch()
        else:
            descriptor, qkv_message, qkv_batch = _recv_next_qkv_batch_message_or_tensor(
                transport
            )
        trace_recv_ms = (
            (time.perf_counter() - trace_recv_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        if trace_offload_exec:
            trace_recv_done_ns = time.perf_counter_ns()
        try:
            if int(qkv_batch.shape[0]) != len(descriptor.items):
                raise RuntimeError(
                    "PAP OFFLOAD_EXEC mailbox batch QKV row count does not "
                    "match descriptor"
                )
            trace_compute_start = time.perf_counter() if trace_offload_exec else 0.0
            trace_compute_stats = (
                {
                    "append_kv_ms": 0.0,
                    "pack_ms": 0.0,
                    "sdpa_ms": 0.0,
                    "reshape_ms": 0.0,
                    "paged_metadata_ms": 0.0,
                    "paged_flash_ms": 0.0,
                    "fallback_ms": 0.0,
                    "shape_lookup_ms": 0.0,
                    "qkv_split_ms": 0.0,
                    "query_move_ms": 0.0,
                    "query_cat_ms": 0.0,
                    "append_lock_wait_ms": 0.0,
                    "append_prepare_ms": 0.0,
                    "append_record_ms": 0.0,
                    "append_tensor_ms": 0.0,
                    "append_copy_ms": 0.0,
                    "append_state_ms": 0.0,
                    "pre_compute_start_ns": 0.0,
                    "pre_compute_done_ns": 0.0,
                    "compute_done_ns": 0.0,
                    "post_compute_done_ns": 0.0,
                }
                if trace_offload_exec
                else None
            )
            output_batch = compute_offload_exec_batch_output(
                registry=registry,
                descriptor=descriptor,
                qkv_batch=qkv_batch,
                trace_stats=trace_compute_stats,
            )
        finally:
            if qkv_message is not None:
                qkv_message.release()
        trace_compute_ms = (
            (time.perf_counter() - trace_compute_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        if trace_offload_exec:
            trace_compute_done_ns = time.perf_counter_ns()
        trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
        if trace_offload_exec:
            trace_send_start_ns = time.perf_counter_ns()
        transport.send_output_batch(
            descriptor,
            output_batch,
            remote_address="",
        )
        if trace_offload_exec:
            trace_send_done_ns = time.perf_counter_ns()
            trace_send_ms = (time.perf_counter() - trace_send_start) * 1000.0
            trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
            logger.info(
                "PAP OFFLOAD_EXEC attention mailbox batch trace layer=%s "
                "calls=%d recv_qkv_ms=%.3f compute_ms=%.3f "
                "send_output_ms=%.3f total_ms=%.3f append_kv_ms=%.3f "
                "pack_ms=%.3f sdpa_ms=%.3f reshape_ms=%.3f "
                "paged_metadata_ms=%.3f paged_flash_ms=%.3f fallback_ms=%.3f "
                "shape_lookup_ms=%.3f qkv_split_ms=%.3f query_move_ms=%.3f "
                "query_cat_ms=%.3f append_lock_wait_ms=%.3f "
                "append_prepare_ms=%.3f append_record_ms=%.3f "
                "append_tensor_ms=%.3f append_copy_ms=%.3f "
                "append_state_ms=%.3f qkv_shape=%s output_shape=%s batch_key=%s "
                "recv_done_ns=%d compute_done_ns=%d send_done_ns=%d "
                "recv_start_ns=%d pre_compute_start_ns=%d "
                "pre_compute_done_ns=%d paged_flash_done_ns=%d reshape_done_ns=%d "
                "send_start_ns=%d",
                descriptor.layer_name,
                len(descriptor.items),
                trace_recv_ms,
                trace_compute_ms,
                trace_send_ms,
                trace_total_ms,
                trace_compute_stats["append_kv_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["pack_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["sdpa_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["reshape_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["paged_metadata_ms"]
                if trace_compute_stats
                else 0.0,
                trace_compute_stats["paged_flash_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["fallback_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["shape_lookup_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["qkv_split_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["query_move_ms"] if trace_compute_stats else 0.0,
                trace_compute_stats["query_cat_ms"] if trace_compute_stats else 0.0,
                (
                    trace_compute_stats["append_lock_wait_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (
                    trace_compute_stats["append_prepare_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (
                    trace_compute_stats["append_record_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (
                    trace_compute_stats["append_tensor_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                (trace_compute_stats["append_copy_ms"] if trace_compute_stats else 0.0),
                (
                    trace_compute_stats["append_state_ms"]
                    if trace_compute_stats
                    else 0.0
                ),
                tuple(qkv_batch.shape),
                tuple(output_batch.shape),
                pap_offload_exec_trace_id(descriptor.output_tensor_id),
                trace_recv_done_ns,
                trace_compute_done_ns,
                trace_send_done_ns,
                trace_recv_start_ns,
                int(trace_compute_stats.get("pre_compute_start_ns", 0.0)) if trace_compute_stats else 0,
                int(trace_compute_stats.get("pre_compute_done_ns", 0.0)) if trace_compute_stats else 0,
                int(trace_compute_stats.get("paged_flash_done_ns", 0.0)) if trace_compute_stats else 0,
                int(trace_compute_stats.get("post_compute_done_ns", 0.0)) if trace_compute_stats else 0,
                trace_send_start_ns,
            )


def _recv_exact(sock: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("PAP attention TCP peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _AttentionTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_attention_tcp_server(
    registry: PAPAttentionRegistry,
    *,
    host: str,
    port: int,
    app: FastAPI | None = None,
) -> socketserver.TCPServer:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    header = _recv_exact(self.request, 8)
                    request_len = int.from_bytes(header, byteorder="little")
                    if request_len <= 0:
                        raise ValueError("PAP attention TCP request length <= 0")
                    payload = _recv_exact(self.request, request_len)
                    response = compute_binary_attention_response(
                        registry,
                        payload,
                        offload_exec_transport=(
                            None if app is None else app.state.offload_exec_transport
                        ),
                        offload_exec_lock=(
                            None if app is None else app.state.offload_exec_lock
                        ),
                    )
                    self.request.sendall(
                        len(response).to_bytes(8, byteorder="little") + response
                    )
                except EOFError:
                    return
                except Exception:
                    logger.exception("PAP attention TCP request failed")
                    return

    server = _AttentionTCPServer((host, port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("PAP Attention TCP data plane listening on %s:%d", host, port)
    return server


def create_app(registry: PAPAttentionRegistry | None = None) -> FastAPI:
    registry = registry or PAPAttentionRegistry()
    app = FastAPI(title="PAP Attention Executor")
    app.state.registry = registry
    app.state.offload_exec_transport = None
    app.state.offload_exec_lock = Lock()
    app.state.offload_exec_mailbox_loop_started = False

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "role": "attention",
            "sessions": registry.size(),
        }

    @app.post("/v1/pap/attention/register")
    async def register(
        registration: PAPAttentionRegistration,
    ) -> dict[str, Any]:
        return registry.register_prefill_kv(registration).__dict__

    @app.get("/v1/pap/attention/sessions/{request_id}")
    async def get_session(request_id: str) -> dict[str, Any]:
        session = registry.get_session(request_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown PAP request")
        return session.__dict__

    @app.post("/v1/pap/attention/offload-exec")
    async def offload_exec(request: PAPOffloadExecRequest) -> dict[str, Any]:
        from vllm.pap.data_plane import PAPOffloadExecDescriptor

        transport = app.state.offload_exec_transport
        if transport is None:
            raise HTTPException(
                status_code=409,
                detail="PAP OFFLOAD_EXEC transport is not initialized",
            )
        descriptor = PAPOffloadExecDescriptor(
            request_id=request.request_id,
            layer_name=request.layer_name,
            step=int(request.step),
            scale=float(request.scale),
        )
        try:
            with app.state.offload_exec_lock:
                run_offload_exec_once(
                    registry=registry,
                    transport=transport,
                    remote_address=request.remote_address,
                    descriptor=descriptor,
                )
        except Exception as exc:
            raise _http_error_for_attention_exception(exc) from exc
        return {
            "request_id": request.request_id,
            "layer_name": request.layer_name,
            "step": int(request.step),
            "remote_address": request.remote_address,
        }

    @app.post("/v1/pap/attention/offload-exec-mailbox/bind")
    async def bind_offload_exec_mailbox(
        request: PAPOffloadExecMailboxBindRequest,
    ) -> dict[str, Any]:
        transport = app.state.offload_exec_transport
        if transport is None or not hasattr(transport, "local_agent_metadata"):
            raise HTTPException(
                status_code=409,
                detail="PAP OFFLOAD_EXEC mailbox transport is not initialized",
            )
        peer_metadata = base64.b64decode(request.agent_metadata_b64.encode("ascii"))
        with app.state.offload_exec_lock:
            if not getattr(transport, "_pap_mailbox_bound", False):
                transport.bind_peer(peer_metadata)
                transport._pap_mailbox_bound = True
            if not app.state.offload_exec_mailbox_loop_started:
                Thread(
                    target=run_offload_exec_mailbox_loop,
                    kwargs={
                        "registry": registry,
                        "transport": transport,
                    },
                    daemon=True,
                    name="pap-offload-exec-mailbox-loop",
                ).start()
                app.state.offload_exec_mailbox_loop_started = True
        return {
            "agent_metadata_b64": base64.b64encode(
                transport.local_agent_metadata
            ).decode("ascii")
        }

    @app.post("/v1/pap/attention/import-prefill-kv")
    async def import_prefill_kv(
        request: PAPAttentionImportPrefillKVRequest,
    ) -> dict[str, Any]:
        from vllm.pap.remote_attention import deserialize_tensor

        key = deserialize_tensor(request.key)
        value = deserialize_tensor(request.value)
        try:
            seq_len = registry.import_prefill_kv(
                request_id=request.request_id,
                layer_name=request.layer_name,
                key=key,
                value=value,
                seq_len=request.seq_len,
                block_ids=request.block_ids,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown PAP request") from exc
        except RuntimeError as exc:
            logger.warning(
                "rejected stateful PAP attention request_id=%s layer=%s "
                "block_id=%s slot=%s seq_len=%s reason=%s",
                request.request_id,
                request.layer_name,
                request.block_id,
                request.slot,
                request.seq_len,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            logger.warning(
                "rejected stateful PAP attention request_id=%s layer=%s "
                "block_id=%s slot=%s seq_len=%s reason=%s",
                request.request_id,
                request.layer_name,
                request.block_id,
                request.slot,
                request.seq_len,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.debug(
            "imported PAP prefill KV request_id=%s layer=%s seq_len=%s",
            request.request_id,
            request.layer_name,
            seq_len,
        )
        return {
            "request_id": request.request_id,
            "layer_name": request.layer_name,
            "seq_len": seq_len,
        }

    @app.post("/v1/pap/attention/append-and-compute")
    async def append_and_compute_attention(
        request: PAPAttentionAppendAndComputeRequest,
    ) -> dict[str, Any]:
        from vllm.pap.remote_attention import (
            compute_segmented_attention_output,
            deserialize_tensor,
            serialize_attention_result,
        )

        query = deserialize_tensor(request.query)
        key = deserialize_tensor(request.key)
        value = deserialize_tensor(request.value)
        try:
            segments, seq_len = registry.append_decode_kv(
                request_id=request.request_id,
                layer_name=request.layer_name,
                key=key,
                value=value,
                block_id=request.block_id,
                slot=request.slot,
                seq_len=request.seq_len,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown PAP request") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if torch.cuda.is_available():
            query = query.to(registry.storage_device, non_blocking=True)
        output = compute_segmented_attention_output(
            query=query,
            segments=segments,
            scale=request.scale,
        )
        logger.debug(
            "computed stateful PAP attention output request_id=%s layer=%s seq_len=%s",
            request.request_id,
            request.layer_name,
            seq_len,
        )
        return {
            "request_id": request.request_id,
            "layer_name": request.layer_name,
            "seq_len": seq_len,
            "output": serialize_attention_result(output),
        }

    @app.post("/v1/pap/attention/append-and-compute-binary")
    async def append_and_compute_attention_binary(
        request: Request,
    ) -> Response:
        try:
            content = _compute_single_binary_attention_response(
                registry,
                await request.body(),
            )
        except Exception as exc:
            logger.warning("rejected binary PAP attention request", exc_info=True)
            raise _http_error_for_attention_exception(exc) from exc
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/v1/pap/attention/import-prefill-kv-binary")
    async def import_prefill_kv_binary(
        request: Request,
    ) -> Response:
        try:
            content = compute_binary_attention_response(
                registry,
                await request.body(),
            )
        except Exception as exc:
            logger.warning("rejected binary PAP prefill import", exc_info=True)
            raise _http_error_for_attention_exception(exc) from exc
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/v1/pap/attention/append-and-compute-batch-binary")
    async def append_and_compute_attention_batch_binary(
        request: Request,
    ) -> Response:
        try:
            content = compute_batch_binary_attention_response(
                registry,
                await request.body(),
            )
        except Exception as exc:
            logger.warning("rejected batch binary PAP attention request", exc_info=True)
            raise _http_error_for_attention_exception(exc) from exc
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/v1/pap/attention/compute")
    async def compute_attention(
        request: PAPAttentionComputeRequest,
    ) -> dict[str, Any]:
        from vllm.pap.remote_attention import (
            compute_attention_output,
            deserialize_tensor,
            serialize_attention_result,
        )

        query = deserialize_tensor(request.query)
        key = deserialize_tensor(request.key)
        value = deserialize_tensor(request.value)
        if torch.cuda.is_available():
            query = query.cuda(non_blocking=True)
            key = key.cuda(non_blocking=True)
            value = value.cuda(non_blocking=True)
        output = compute_attention_output(
            query=query,
            key=key,
            value=value,
            scale=request.scale,
        )
        logger.debug(
            "computed PAP attention output request_id=%s layer=%s query_shape=%s",
            request.request_id,
            request.layer_name,
            list(query.shape),
        )
        return {
            "request_id": request.request_id,
            "layer_name": request.layer_name,
            "output": serialize_attention_result(output),
        }

    @app.post("/v1/pap/attention/layer-event")
    async def record_layer_event(
        event: PAPAttentionLayerEventRequest,
    ) -> dict[str, Any]:
        try:
            recorded = registry.record_layer_event(**event.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown PAP request") from exc
        return recorded.__dict__

    @app.get("/v1/pap/attention/sessions/{request_id}/layer-events")
    async def get_layer_events(request_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "events": [
                event.__dict__ for event in registry.get_layer_events(request_id)
            ],
        }

    @app.delete("/v1/pap/attention/sessions/{request_id}")
    async def release_session(request_id: str) -> dict[str, Any]:
        return {"released": registry.release_session(request_id)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAP Attention executor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument(
        "--offload-exec-zmq-port",
        type=int,
        default=None,
        help=(
            "Reserved PAP OFFLOAD_EXEC ZMQ control port for the "
            "Projection<->Attention data plane."
        ),
    )
    return parser.parse_args()


def maybe_start_offload_exec_transport(
    *,
    app: FastAPI,
    host: str,
    zmq_port: int | None,
) -> None:
    """Initialize the optional OFFLOAD_EXEC data plane."""

    if zmq_port is None:
        return
    local_rank = int(os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0"))
    transport = os.environ.get("PAP_OFFLOAD_EXEC_TRANSPORT", "nixl_mailbox").lower()
    if transport in {"nixl", "nixl_mailbox"}:
        app.state.offload_exec_transport = build_nixl_mailbox_offload_exec_transport(
            actor_id=os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "attention"),
            local_rank=local_rank,
        )
        logger.info(
            "PAP Attention OFFLOAD_EXEC NIXL mailbox initialized local_rank=%d",
            local_rank,
        )
        return
    if transport in {"local_fast", "local-fast", "cuda_ipc_fast"}:
        app.state.offload_exec_transport = build_local_fast_offload_exec_transport(
            actor_id=os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "attention"),
            local_rank=local_rank,
        )
        logger.info(
            "PAP Attention OFFLOAD_EXEC local_fast (CUDA IPC + spin doorbell) "
            "initialized local_rank=%d",
            local_rank,
        )
        return
    raise RuntimeError(
        f"PAP OFFLOAD_EXEC transport {transport!r} is not supported; use "
        "nixl_mailbox or local_fast"
    )


app = create_app()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.tcp_port is not None:
        start_attention_tcp_server(
            app.state.registry,
            host=args.host,
            port=args.tcp_port,
            app=app,
        )
    if args.offload_exec_zmq_port is not None:
        logger.info(
            "PAP Attention OFFLOAD_EXEC ZMQ endpoint reserved at %s:%d",
            args.host,
            args.offload_exec_zmq_port,
        )
    maybe_start_offload_exec_transport(
        app=app,
        host=args.host,
        zmq_port=args.offload_exec_zmq_port,
    )
    uvicorn.run(app, host=args.host, port=args.port)
