# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join asynchronously copied decode tokens with PAP KV readiness."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Condition, Lock


@dataclass(frozen=True)
class DeferredDecodeCommit:
    request_id: str
    new_seq_len: int
    token_ids: tuple[int, ...]
    endpoint: str
    commit_request_id: str | None = None


class DeferredDecodeTokenCommitter:
    """Match token-ready and KV-ready notifications for one decode position."""

    def __init__(
        self,
        dispatch: Callable[[DeferredDecodeCommit], None],
    ) -> None:
        self._dispatch = dispatch
        self._lock = Lock()
        self._done = Condition(self._lock)
        self._tokens: dict[tuple[str, int], tuple[int, ...]] = {}
        self._kv_targets: dict[tuple[str, int], tuple[str, str | None]] = {}
        self._committed_tokens: dict[tuple[str, int], tuple[int, ...]] = {}
        self._dispatching_by_request: dict[str, int] = {}
        self._failure_by_request: dict[str, str] = {}
        self._received = 0
        self._kv_ready = 0
        self._matched = 0
        self._duplicates = 0
        self._mismatches = 0
        self._dispatch_failures = 0
        self._token_only_dropped = 0

    def record_token(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        token_ids: Iterable[int],
    ) -> str:
        key = (str(request_id), int(new_seq_len))
        new_token_ids = tuple(int(token_id) for token_id in token_ids)
        if not new_token_ids:
            raise ValueError("PAP deferred decode token IDs must not be empty")
        with self._done:
            existing = self._committed_tokens.get(key)
            if existing is None:
                existing = self._tokens.get(key)
            if existing is not None:
                if existing != new_token_ids:
                    self._mismatches += 1
                    raise ValueError(
                        "PAP deferred decode token changed token IDs for "
                        f"request_id={key[0]} new_seq_len={key[1]}"
                    )
                self._duplicates += 1
                return "duplicate"
            self._tokens[key] = new_token_ids
            self._received += 1
            commit = self._take_commit_locked(key)
        if commit is None:
            return "pending"
        self._dispatch_commit(commit)
        return "matched"

    def record_kv_ready(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        endpoint: str,
        commit_request_id: str | None = None,
    ) -> str:
        key = (str(request_id), int(new_seq_len))
        target_endpoint = str(endpoint)
        if not target_endpoint:
            raise ValueError("PAP deferred decode commit endpoint must not be empty")
        target_request_id = (
            None if commit_request_id is None else str(commit_request_id)
        )
        target = (target_endpoint, target_request_id)
        with self._done:
            if key in self._committed_tokens:
                self._duplicates += 1
                return "duplicate"
            existing_target = self._kv_targets.get(key)
            if existing_target is not None:
                if existing_target != target:
                    self._mismatches += 1
                    raise ValueError(
                        "PAP deferred decode KV changed commit target for "
                        f"request_id={key[0]} new_seq_len={key[1]}"
                    )
                self._duplicates += 1
                return "duplicate"
            self._kv_targets[key] = target
            self._kv_ready += 1
            commit = self._take_commit_locked(key)
        if commit is None:
            return "pending"
        self._dispatch_commit(commit)
        return "matched"

    def _take_commit_locked(
        self,
        key: tuple[str, int],
    ) -> DeferredDecodeCommit | None:
        token_ids = self._tokens.get(key)
        target = self._kv_targets.get(key)
        if token_ids is None or target is None:
            return None
        endpoint, commit_request_id = target
        self._tokens.pop(key)
        self._kv_targets.pop(key)
        self._committed_tokens[key] = token_ids
        request_id = key[0]
        self._dispatching_by_request[request_id] = (
            self._dispatching_by_request.get(request_id, 0) + 1
        )
        return DeferredDecodeCommit(
            request_id=request_id,
            new_seq_len=key[1],
            token_ids=token_ids,
            endpoint=endpoint,
            commit_request_id=commit_request_id,
        )

    def _dispatch_commit(self, commit: DeferredDecodeCommit) -> None:
        try:
            self._dispatch(commit)
        except Exception as exc:
            with self._done:
                self._dispatch_failures += 1
                self._failure_by_request[commit.request_id] = str(exc)
                self._finish_dispatch_locked(commit.request_id)
            raise
        with self._done:
            self._matched += 1
            self._finish_dispatch_locked(commit.request_id)

    def _finish_dispatch_locked(self, request_id: str) -> None:
        count = self._dispatching_by_request.get(request_id, 0) - 1
        if count > 0:
            self._dispatching_by_request[request_id] = count
        else:
            self._dispatching_by_request.pop(request_id, None)
        self._done.notify_all()

    def flush_request(self, request_id: str, timeout_s: float = 5.0) -> bool:
        """Wait until every KV-ready position is dispatched for a request."""
        request_id = str(request_id)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._done:
            while self._request_has_pending_kv_locked(request_id):
                if request_id in self._failure_by_request:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._done.wait(timeout=remaining)
            return request_id not in self._failure_by_request

    def _request_has_pending_kv_locked(self, request_id: str) -> bool:
        return any(key[0] == request_id for key in self._kv_targets) or (
            self._dispatching_by_request.get(request_id, 0) > 0
        )

    def forget_request(self, request_id: str) -> None:
        """Drop bounded duplicate state and the expected final token-only item."""
        request_id = str(request_id)
        with self._done:
            token_keys = [key for key in self._tokens if key[0] == request_id]
            self._token_only_dropped += len(token_keys)
            for mapping in (
                self._tokens,
                self._kv_targets,
                self._committed_tokens,
            ):
                for key in [key for key in mapping if key[0] == request_id]:
                    mapping.pop(key, None)
            self._failure_by_request.pop(request_id, None)
            self._done.notify_all()

    def stats(self) -> dict[str, int]:
        with self._done:
            return {
                "decode_token_received": self._received,
                "decode_kv_ready": self._kv_ready,
                "decode_token_matched": self._matched,
                "decode_token_duplicates": self._duplicates,
                "decode_token_mismatches": self._mismatches,
                "decode_token_dispatch_failures": self._dispatch_failures,
                "decode_token_pending_tokens": len(self._tokens),
                "decode_token_pending_kv": len(self._kv_targets),
                "decode_token_dispatching": sum(
                    self._dispatching_by_request.values()
                ),
                "decode_token_only_dropped": self._token_only_dropped,
            }
