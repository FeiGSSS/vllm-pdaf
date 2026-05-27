# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock


@dataclass(frozen=True)
class AttentionDecodeDescriptor:
    request_id: str
    block_id: int
    slot: int
    seq_len: int


@dataclass(frozen=True)
class AttentionSession:
    request_id: str
    conversation_id: str
    block_size: int
    max_seq_len: int
    block_ids: tuple[int, ...] = ()
    seq_len: int = 0


class AttentionSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, AttentionSession] = {}
        self._lock = Lock()

    def create_session(
        self,
        request_id: str,
        conversation_id: str,
        *,
        block_size: int,
        max_seq_len: int,
    ) -> AttentionSession:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        with self._lock:
            if request_id in self._sessions:
                raise ValueError(f"attention session already exists: {request_id}")
            session = AttentionSession(
                request_id=request_id,
                conversation_id=conversation_id,
                block_size=block_size,
                max_seq_len=max_seq_len,
            )
            self._sessions[request_id] = session
            return session

    def get_session(self, request_id: str) -> AttentionSession | None:
        with self._lock:
            return self._sessions.get(request_id)

    def import_prefill_kv(
        self, request_id: str, *, block_ids: list[int], seq_len: int
    ) -> AttentionSession:
        with self._lock:
            session = self._require_session(request_id)
            self._validate_seq_len(session, seq_len)
            updated = replace(
                session,
                block_ids=tuple(block_ids),
                seq_len=seq_len,
            )
            self._sessions[request_id] = updated
            return updated

    def append_decode_token(
        self, request_id: str, *, block_id: int, seq_len: int
    ) -> AttentionSession:
        with self._lock:
            session = self._require_session(request_id)
            slot = block_id * session.block_size + (
                (seq_len - 1) % session.block_size
            )
            updated, appended = self._record_decode_descriptor_unlocked(
                AttentionDecodeDescriptor(
                    request_id=request_id,
                    block_id=block_id,
                    slot=slot,
                    seq_len=seq_len,
                )
            )
            if not appended:
                raise ValueError(
                    f"expected seq_len {updated.seq_len + 1}, got {seq_len}"
                )
            return updated

    def append_decode_descriptor(
        self, descriptor: AttentionDecodeDescriptor
    ) -> AttentionSession:
        with self._lock:
            updated, appended = self._record_decode_descriptor_unlocked(descriptor)
            if not appended:
                raise ValueError(
                    f"expected seq_len {updated.seq_len + 1}, got {descriptor.seq_len}"
                )
            return updated

    def record_decode_descriptor(
        self, descriptor: AttentionDecodeDescriptor
    ) -> tuple[AttentionSession, bool]:
        with self._lock:
            return self._record_decode_descriptor_unlocked(descriptor)

    def free_session(self, request_id: str) -> AttentionSession | None:
        with self._lock:
            return self._sessions.pop(request_id, None)

    def _require_session(self, request_id: str) -> AttentionSession:
        try:
            return self._sessions[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown attention session: {request_id}") from exc

    def _record_decode_descriptor_unlocked(
        self, descriptor: AttentionDecodeDescriptor
    ) -> tuple[AttentionSession, bool]:
        session = self._require_session(descriptor.request_id)
        appended = self._validate_decode_descriptor(session, descriptor)
        if not appended:
            return session, False
        block_ids = session.block_ids
        if not block_ids or block_ids[-1] != descriptor.block_id:
            block_ids = (*block_ids, descriptor.block_id)
        updated = replace(session, block_ids=block_ids, seq_len=descriptor.seq_len)
        self._sessions[descriptor.request_id] = updated
        return updated, True

    @staticmethod
    def _validate_seq_len(session: AttentionSession, seq_len: int) -> None:
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if seq_len > session.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {session.max_seq_len}"
            )

    @classmethod
    def _validate_decode_descriptor(
        cls, session: AttentionSession, descriptor: AttentionDecodeDescriptor
    ) -> bool:
        cls._validate_seq_len(session, descriptor.seq_len)
        if descriptor.block_id < 0:
            raise ValueError("block_id must be non-negative")
        if descriptor.slot < 0:
            raise ValueError("slot must be non-negative")
        if descriptor.seq_len <= 0:
            raise ValueError("decode descriptor seq_len must be positive")

        expected_offset = (descriptor.seq_len - 1) % session.block_size
        expected_slot = descriptor.block_id * session.block_size + expected_offset
        if descriptor.slot != expected_slot:
            raise ValueError(
                f"slot {descriptor.slot} does not match block_id "
                f"{descriptor.block_id} and offset {expected_offset}"
            )

        block_ids = session.block_ids
        if descriptor.seq_len == session.seq_len:
            if not block_ids or block_ids[-1] != descriptor.block_id:
                raise ValueError(
                    f"existing seq_len {descriptor.seq_len} must reference current "
                    f"block {block_ids[-1] if block_ids else None}, got "
                    f"block {descriptor.block_id}"
                )
            return False

        expected_seq_len = session.seq_len + 1
        if descriptor.seq_len != expected_seq_len:
            raise ValueError(
                f"expected seq_len {expected_seq_len}, got {descriptor.seq_len}"
            )
        if block_ids and expected_offset != 0 and block_ids[-1] != descriptor.block_id:
            raise ValueError(
                f"slot for seq_len {descriptor.seq_len} must append to current "
                f"block {block_ids[-1]}, got block {descriptor.block_id}"
            )
        if block_ids and expected_offset == 0 and block_ids[-1] == descriptor.block_id:
            raise ValueError(
                f"seq_len {descriptor.seq_len} starts a new block but reused "
                f"current block {descriptor.block_id}"
            )
        return True
