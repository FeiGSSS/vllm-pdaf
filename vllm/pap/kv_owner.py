# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock


@dataclass(frozen=True)
class PAKVDecodeSlot:
    session_id: str
    layer_name: str
    block_id: int
    slot: int
    seq_len: int
    materialized: bool = False


@dataclass(frozen=True)
class PAKVLayerState:
    layer_name: str
    block_ids: tuple[int, ...]
    seq_len: int
    num_blocks: int
    materialized_slots: tuple[PAKVDecodeSlot, ...] = ()


@dataclass(frozen=True)
class PAKVSessionState:
    session_id: str
    block_size: int
    max_seq_len: int
    lease_count: int = 0
    layers: dict[str, PAKVLayerState] = field(default_factory=dict)


class PAKVOwner:
    """PA-local metadata owner for Prefill-owned resident KV blocks."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, PAKVSessionState] = {}

    def register_session(
        self,
        *,
        session_id: str,
        block_size: int,
        max_seq_len: int,
    ) -> PAKVSessionState:
        session_id = str(session_id)
        block_size = int(block_size)
        max_seq_len = int(max_seq_len)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"PA KV session already exists: {session_id}")
            session = PAKVSessionState(
                session_id=session_id,
                block_size=block_size,
                max_seq_len=max_seq_len,
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> PAKVSessionState | None:
        with self._lock:
            return self._sessions.get(str(session_id))

    def register_layer_blocks(
        self,
        *,
        session_id: str,
        layer_name: str,
        block_ids: list[int],
        seq_len: int,
        num_blocks: int,
    ) -> PAKVLayerState:
        session_id = str(session_id)
        layer_name = str(layer_name)
        block_ids_tuple = tuple(int(block_id) for block_id in block_ids)
        seq_len = int(seq_len)
        num_blocks = int(num_blocks)
        with self._lock:
            session = self._require_session(session_id)
            self._validate_seq_len(session, seq_len)
            if num_blocks < 0:
                raise ValueError("num_blocks must be non-negative")
            for block_id in block_ids_tuple:
                self._validate_backed_block(block_id, num_blocks)
            layer = PAKVLayerState(
                layer_name=layer_name,
                block_ids=block_ids_tuple,
                seq_len=seq_len,
                num_blocks=num_blocks,
            )
            layers = dict(session.layers)
            layers[layer_name] = layer
            self._sessions[session_id] = replace(session, layers=layers)
            return layer

    def get_layer_state(
        self,
        session_id: str,
        layer_name: str,
    ) -> PAKVLayerState | None:
        with self._lock:
            session = self._sessions.get(str(session_id))
            if session is None:
                return None
            return session.layers.get(str(layer_name))

    def reserve_decode_slot(
        self,
        *,
        session_id: str,
        layer_name: str,
        block_id: int,
        seq_len: int,
    ) -> PAKVDecodeSlot:
        return self._record_decode_slot(
            session_id=str(session_id),
            layer_name=str(layer_name),
            block_id=int(block_id),
            seq_len=int(seq_len),
            materialized=False,
        )

    def reserve_next_decode_slot(
        self,
        *,
        session_id: str,
        layer_name: str,
    ) -> PAKVDecodeSlot:
        with self._lock:
            session = self._require_session(str(session_id))
            layer = self._require_layer(session, str(layer_name))
            seq_len = layer.seq_len + 1
            self._validate_seq_len(session, seq_len)
            block_id = (seq_len - 1) // session.block_size
        return self.reserve_decode_slot(
            session_id=str(session_id),
            layer_name=str(layer_name),
            block_id=block_id,
            seq_len=seq_len,
        )

    def materialize_decode_slot(
        self,
        *,
        session_id: str,
        layer_name: str,
        block_id: int,
        seq_len: int,
    ) -> PAKVDecodeSlot:
        return self._record_decode_slot(
            session_id=str(session_id),
            layer_name=str(layer_name),
            block_id=int(block_id),
            seq_len=int(seq_len),
            materialized=True,
        )

    def acquire_lease(self, session_id: str) -> PAKVSessionState:
        with self._lock:
            session = self._require_session(str(session_id))
            updated = replace(session, lease_count=session.lease_count + 1)
            self._sessions[str(session_id)] = updated
            return updated

    def release_lease(self, session_id: str) -> bool:
        session_id = str(session_id)
        with self._lock:
            session = self._require_session(session_id)
            if session.lease_count <= 1:
                self._sessions.pop(session_id, None)
                return True
            self._sessions[session_id] = replace(
                session, lease_count=session.lease_count - 1
            )
            return False

    def _record_decode_slot(
        self,
        *,
        session_id: str,
        layer_name: str,
        block_id: int,
        seq_len: int,
        materialized: bool,
    ) -> PAKVDecodeSlot:
        with self._lock:
            session = self._require_session(session_id)
            layer = self._require_layer(session, layer_name)
            self._validate_seq_len(session, seq_len)
            if seq_len < layer.seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} is behind layer seq_len "
                    f"{layer.seq_len}"
                )
            if seq_len > layer.seq_len + 1:
                raise ValueError(f"expected seq_len {layer.seq_len + 1}, got {seq_len}")
            self._validate_backed_block(block_id, layer.num_blocks)

            block_ids = layer.block_ids
            if block_id not in block_ids:
                block_ids = (*block_ids, block_id)
            slot = block_id * session.block_size + ((seq_len - 1) % session.block_size)
            decode_slot = PAKVDecodeSlot(
                session_id=session_id,
                layer_name=layer_name,
                block_id=block_id,
                slot=slot,
                seq_len=seq_len,
                materialized=bool(materialized),
            )
            materialized_slots = layer.materialized_slots
            if materialized:
                materialized_slots = (*materialized_slots, decode_slot)
            updated_layer = replace(
                layer,
                block_ids=block_ids,
                seq_len=max(layer.seq_len, seq_len),
                materialized_slots=materialized_slots,
            )
            layers = dict(session.layers)
            layers[layer_name] = updated_layer
            self._sessions[session_id] = replace(session, layers=layers)
            return decode_slot

    def _require_session(self, session_id: str) -> PAKVSessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown PA KV session: {session_id}") from exc

    @staticmethod
    def _require_layer(
        session: PAKVSessionState,
        layer_name: str,
    ) -> PAKVLayerState:
        try:
            return session.layers[layer_name]
        except KeyError as exc:
            raise KeyError(f"unknown PA KV layer: {layer_name}") from exc

    @staticmethod
    def _validate_seq_len(session: PAKVSessionState, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if seq_len > session.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {session.max_seq_len}"
            )

    @staticmethod
    def _validate_backed_block(block_id: int, num_blocks: int) -> None:
        if block_id < 0:
            raise ValueError("block_id must be non-negative")
        if block_id >= num_blocks:
            raise ValueError(
                f"block_id {block_id} is not backed by resident KV tensor "
                f"with {num_blocks} blocks"
            )
