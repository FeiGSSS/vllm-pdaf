# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention internal executor.

This first PAP slice keeps the data-plane in vLLM's NIXL connector and exposes
the Attention role as an internal state owner. The process is intentionally not
an OpenAI-compatible vLLM server: it records which prefill KV handle belongs to
which PAP request so the proxy and future remote-attention backend have a stable
control-plane contract.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import socketserver
import time
from dataclasses import dataclass, field
from threading import Condition, Lock, Thread
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from vllm.pap.attention_session import (
    AttentionDecodeDescriptor,
    AttentionSessionStore,
)
from vllm.pap.kv_owner import PAKVOwner
from vllm.pap.data_plane import (
    PAPOffloadKVIPCDescriptor,
    PAPOffloadKVPagedIPCDescriptor,
    build_p2p_nccl_offload_exec_transport,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_attention")


class PAPAttentionRegistration(BaseModel):
    """KV ownership metadata registered after Prefill completes."""

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
    """Control-plane trigger for one OFFLOAD_EXEC NCCL/P2P tensor exchange."""

    request_id: str
    layer_name: str
    step: int
    scale: float
    remote_address: str


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
class PAPResidentPagedKV:
    """Resident paged KV backing attached from the Prefill-owned cache."""

    kv_cache: torch.Tensor
    block_ids: list[int]
    seq_len: int
    block_size: int
    num_kv_heads: int
    layout: str


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
        self._resident_paged_kv: dict[str, dict[str, PAPResidentPagedKV]] = {}
        self._attention_sessions = AttentionSessionStore()
        self._pa_kv_owner = PAKVOwner()

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
            self._sessions[registration.request_id] = session
            self._layer_events.setdefault(registration.request_id, [])
            self._decode_kv.setdefault(registration.request_id, {})
            self._prefill_kv.setdefault(registration.request_id, {})
            self._resident_paged_kv.setdefault(registration.request_id, {})
            self._attention_sessions.create_session(
                registration.request_id,
                registration.conversation_id,
                block_size=registration.block_size,
                max_seq_len=registration.max_seq_len,
            )
            self._pa_kv_owner.register_session(
                session_id=registration.request_id,
                block_size=registration.block_size,
                max_seq_len=registration.max_seq_len,
            )
            self._pa_kv_owner.acquire_lease(registration.request_id)
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

    def release_session(self, request_id: str) -> bool:
        with self._lock:
            existed = self._sessions.pop(request_id, None) is not None
            self._layer_events.pop(request_id, None)
            self._decode_kv.pop(request_id, None)
            self._prefill_kv.pop(request_id, None)
            self._resident_paged_kv.pop(request_id, None)
            self._attention_sessions.free_session(request_id)
            try:
                self._pa_kv_owner.release_lease(request_id)
            except KeyError:
                pass
            return existed

    def resolve_session_request_id(self, request_id: str) -> str | None:
        """Map vLLM-wrapped request ids back to the proxy-level PAP id."""
        with self._lock:
            return self._resolve_session_request_id_locked(request_id)

    def _resolve_session_request_id_locked(self, request_id: str) -> str | None:
        if request_id in self._sessions:
            return request_id

        candidates = [request_id]
        for prefix in ("cmpl-", "chatcmpl-"):
            if request_id.startswith(prefix):
                candidates.append(request_id[len(prefix) :])

        for candidate in candidates:
            if candidate in self._sessions:
                return candidate
            for session_request_id in self._sessions:
                if candidate.startswith(
                    f"{session_request_id}-"
                ) or candidate.startswith(f"{session_request_id}_"):
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
            self._prefill_kv.setdefault(session_request_id, {})[layer_name] = (
                [(key_state, value_state)]
            )
            self._resident_paged_kv.setdefault(session_request_id, {}).pop(
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
            self._prefill_condition.notify_all()
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
    ) -> int:
        from vllm.pap.remote_attention import paged_kv_segments

        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            session = self._sessions[session_request_id]
            seq_len = int(seq_len)
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
            self._prefill_kv.setdefault(session_request_id, {})[layer_name] = segments
            self._resident_paged_kv.setdefault(session_request_id, {})[layer_name] = (
                PAPResidentPagedKV(
                    kv_cache=kv_cache.detach(),
                    block_ids=[int(block_id) for block_id in block_ids],
                    seq_len=seq_len,
                    block_size=int(block_size),
                    num_kv_heads=int(num_kv_heads),
                    layout=str(layout),
                )
            )
            self._pa_kv_owner.register_layer_blocks(
                session_id=session_request_id,
                layer_name=layer_name,
                block_ids=[int(block_id) for block_id in block_ids],
                seq_len=seq_len,
                num_blocks=int(kv_cache.shape[1]),
            )
            imported_session = self._attention_sessions.import_prefill_kv(
                session_request_id,
                block_ids=[int(block_id) for block_id in block_ids],
                seq_len=seq_len,
            )
            session.block_ids = tuple(imported_session.block_ids)
            session.seq_len = imported_session.seq_len
            session.prefill_seq_lens[layer_name] = seq_len
            session.decode_seq_lens[layer_name] = seq_len
            self._prefill_condition.notify_all()
            return seq_len

    @staticmethod
    def _resident_paged_logical_nhd(
        resident: PAPResidentPagedKV,
    ) -> bool:
        if resident.layout == "NHD":
            return True
        if resident.layout == "HND":
            return int(resident.kv_cache.shape[3]) == int(resident.num_kv_heads)
        raise ValueError(f"unsupported KV cache layout: {resident.layout}")

    @classmethod
    def _resident_paged_block_capacity(
        cls,
        resident: PAPResidentPagedKV,
    ) -> int:
        if cls._resident_paged_logical_nhd(resident):
            return int(resident.kv_cache.shape[2])
        return int(resident.kv_cache.shape[3])

    @staticmethod
    def _resident_paged_num_blocks(resident: PAPResidentPagedKV) -> int:
        return int(resident.kv_cache.shape[1])

    def _try_write_decode_to_resident_paged_kv(
        self,
        *,
        resident: PAPResidentPagedKV,
        key: torch.Tensor,
        value: torch.Tensor,
        block_id: int,
        slot: int,
        seq_len: int,
    ) -> bool:
        if int(key.shape[0]) != 1 or int(value.shape[0]) != 1:
            return False
        if int(block_id) not in resident.block_ids:
            if int(block_id) < 0 or int(block_id) >= self._resident_paged_num_blocks(
                resident
            ):
                return False
            resident.block_ids.append(int(block_id))

        if int(block_id) not in resident.block_ids:
            return False

        logical_nhd = self._resident_paged_logical_nhd(resident)
        block_capacity = self._resident_paged_block_capacity(resident)
        if int(seq_len) > len(resident.block_ids) * block_capacity:
            return False
        block_offset = int(slot) - int(block_id) * int(resident.block_size)
        if block_offset >= block_capacity:
            return False
        if block_offset < 0:
            return False

        key_state = key.detach().to(
            device=resident.kv_cache.device,
            dtype=resident.kv_cache.dtype,
        )
        value_state = value.detach().to(
            device=resident.kv_cache.device,
            dtype=resident.kv_cache.dtype,
        )
        if logical_nhd:
            resident.kv_cache[
                0, int(block_id), block_offset, : resident.num_kv_heads, :
            ].copy_(key_state[0])
            resident.kv_cache[
                1, int(block_id), block_offset, : resident.num_kv_heads, :
            ].copy_(value_state[0])
        else:
            resident.kv_cache[
                0, int(block_id), : resident.num_kv_heads, block_offset, :
            ].copy_(key_state[0])
            resident.kv_cache[
                1, int(block_id), : resident.num_kv_heads, block_offset, :
            ].copy_(value_state[0])

        resident.seq_len = max(int(resident.seq_len), int(seq_len))
        return True

    def _wait_for_prefill_layer_locked(
        self,
        *,
        session_request_id: str,
        session: PAPAttentionSession,
        layer_name: str,
        decode_seq_len: int | None = None,
    ) -> None:
        has_registered_prefix = int(session.prefix_len or 0) > 0
        has_scheduler_prefix = (
            decode_seq_len is not None and int(decode_seq_len) > 1
        )
        if not has_registered_prefix and not has_scheduler_prefix:
            return
        deadline = time.monotonic() + float(
            os.environ.get("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "5.0")
        )
        prefill_layer_kv = self._prefill_kv.setdefault(session_request_id, {})
        while layer_name not in prefill_layer_kv:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "prefill KV must be imported before stateful decode attention"
                )
            self._prefill_condition.wait(timeout=remaining)

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
            resident = self._resident_paged_kv.get(session_request_id, {}).get(
                layer_name
            )
            wrote_resident = False
            if (
                should_append
                and resident is not None
                and block_id is not None
                and seq_len is not None
            ):
                from vllm.pap.remote_attention import paged_kv_segments

                wrote_resident = self._try_write_decode_to_resident_paged_kv(
                    resident=resident,
                    key=key,
                    value=value,
                    block_id=int(block_id),
                    slot=int(slot),
                    seq_len=int(seq_len),
                )
                if wrote_resident:
                    self._pa_kv_owner.materialize_decode_slot(
                        session_id=session_request_id,
                        layer_name=layer_name,
                        block_id=int(block_id),
                        seq_len=int(seq_len),
                    )
                    prefill_layer_kv[layer_name] = paged_kv_segments(
                        kv_cache=resident.kv_cache,
                        block_ids=resident.block_ids,
                        seq_len=resident.seq_len,
                        num_kv_heads=resident.num_kv_heads,
                        layout=resident.layout,  # type: ignore[arg-type]
                    )

            if not should_append or wrote_resident:
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
    ) -> None:
        if seq_len > session.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {session.max_seq_len}"
            )
        layer_seq_len = session.decode_seq_lens.get(
            layer_name,
            session.prefill_seq_lens.get(layer_name, int(session.prefix_len or 0)),
        )
        if seq_len > layer_seq_len + 1:
            raise ValueError(f"expected seq_len {layer_seq_len + 1}, got {seq_len}")
        if seq_len > layer_seq_len:
            block_id = (seq_len - 1) // session.block_size
            expected_offset = (seq_len - 1) % session.block_size
            block_ids = session.block_ids
            if ((not block_ids) or expected_offset == 0) and (
                not block_ids or block_ids[-1] != block_id
            ):
                session.block_ids = (*block_ids, block_id)
            session.seq_len = max(session.seq_len, seq_len)

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

    trace_remote_attention = os.environ.get(
        "PAP_OFFLOAD_EXEC_TRACE", ""
    ).lower() in ("1", "true", "yes", "on")
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
            q_size = (
                (session.q_size if session is not None else None)
                or int(os.environ.get("PAP_OFFLOAD_EXEC_Q_SIZE", "0"))
            )
            kv_size = (
                (session.kv_size if session is not None else None)
                or int(os.environ.get("PAP_OFFLOAD_EXEC_KV_SIZE", "0"))
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
        layer_name = str(metadata["items"][0]["layer_name"]) if metadata["items"] else ""
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

    trace_remote_attention = os.environ.get(
        "PAP_OFFLOAD_EXEC_TRACE", ""
    ).lower() in ("1", "true", "yes", "on")
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
    from vllm.pap.remote_attention import (
        COMPACT_ATTENTION_REQUEST_MAGIC,
        COMPACT_OFFLOAD_EXEC_MAGIC,
        deserialize_compact_offload_exec_command,
        deserialize_tensor_bundle,
        serialize_compact_offload_exec_ack,
        serialize_tensor_bundle,
    )
    from vllm.pap.data_plane import PAPOffloadExecDescriptor

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

    metadata, tensors = deserialize_tensor_bundle(payload)
    if metadata.get("command") == "import_prefill_kv":
        seq_len = registry.import_prefill_kv(
            request_id=str(metadata["request_id"]),
            layer_name=str(metadata["layer_name"]),
            key=tensors["key"],
            value=tensors["value"],
            seq_len=int(metadata["seq_len"]),
            block_ids=[
                int(block_id) for block_id in metadata.get("block_ids", [])
            ],
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
    block_id = (seq_len - 1) // session.block_size
    slot = block_id * session.block_size + ((seq_len - 1) % session.block_size)
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
        (time.perf_counter() - trace_recv_start) * 1000.0
        if trace_offload_exec
        else 0.0
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
                            None
                            if app is None
                            else app.state.offload_exec_transport
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
            "Reserved PAP OFFLOAD_EXEC ZMQ control port for the NCCL/P2P "
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
    """Initialize the optional NCCL/P2P OFFLOAD_EXEC data plane."""

    if zmq_port is None:
        return
    local_rank = int(os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0"))
    app.state.offload_exec_transport = build_p2p_nccl_offload_exec_transport(
        local_rank=local_rank,
        kv_port=int(zmq_port),
        hostname=host,
    )
    logger.info(
        "PAP Attention OFFLOAD_EXEC NCCL/P2P data plane listening at %s:%d "
        "local_rank=%d",
        host,
        zmq_port,
        local_rank,
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
            "PAP Attention OFFLOAD_EXEC NCCL/P2P ZMQ endpoint reserved at %s:%d",
            args.host,
            args.offload_exec_zmq_port,
        )
    maybe_start_offload_exec_transport(
        app=app,
        host=args.host,
        zmq_port=args.offload_exec_zmq_port,
    )
    uvicorn.run(app, host=args.host, port=args.port)
