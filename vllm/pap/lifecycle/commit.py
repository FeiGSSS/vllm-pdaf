# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reliable asynchronous HTTP client for PAP decode commit submissions.

The Attention executor uses this client to notify the Prefill engine about
decode KV commits that are ready on the remote side.  The client is designed
as a process-level singleton --- instantiate once, call commit() per batch.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Iterable

import httpx

from vllm.pap.config import reject_removed_pap_flags

logger = logging.getLogger(__name__)


class DecodeCommitClient:
    """Asynchronously deliver PAP decode commits with accepted ordering.

    The caller only enqueues payloads. A daemon worker preserves queue order,
    retries failed POSTs, and advances a per-request watermark after the
    Prefill endpoint accepts the commit into its EngineCore input queue.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float | None = None,
        queue_size: int | None = None,
        max_attempts: int | None = None,
        retry_initial_s: float | None = None,
        retry_max_s: float | None = None,
    ) -> None:
        """Create a client that POSTs to *endpoint* (or
        ``PAP_DECODE_COMMIT_ENDPOINT`` env var)."""
        reject_removed_pap_flags(os.environ)
        self.endpoint = endpoint or os.environ.get("PAP_DECODE_COMMIT_ENDPOINT", "")
        self.timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(os.environ.get("PAP_DECODE_COMMIT_TIMEOUT", "5.0"))
        )
        self.queue_size = (
            int(queue_size)
            if queue_size is not None
            else int(os.environ.get("PAP_DECODE_COMMIT_QUEUE_SIZE", "1024"))
        )
        self.max_attempts = max(
            1,
            int(max_attempts)
            if max_attempts is not None
            else int(os.environ.get("PAP_DECODE_COMMIT_MAX_ATTEMPTS", "8")),
        )
        self.retry_initial_s = max(
            0.0,
            float(retry_initial_s)
            if retry_initial_s is not None
            else float(
                os.environ.get("PAP_DECODE_COMMIT_RETRY_INITIAL_SECONDS", "0.05")
            ),
        )
        self.retry_max_s = max(
            self.retry_initial_s,
            float(retry_max_s)
            if retry_max_s is not None
            else float(os.environ.get("PAP_DECODE_COMMIT_RETRY_MAX_SECONDS", "0.5")),
        )
        self._queue: queue.Queue[dict] | None = None
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_done = threading.Condition(self._pending_lock)
        self._pending_by_request: dict[str, int] = {}
        self._queued_item_by_request: dict[str, dict] = {}
        self._latest_seen_seq_len_by_request: dict[str, int] = {}
        self._latest_commit_seq_by_request: dict[str, int] = {}
        self._acked_commit_seq_by_request: dict[str, int] = {}
        self._failed_commit_seq_by_request: dict[str, int] = {}
        self._failure_by_request: dict[str, str] = {}
        self._targets_by_session: dict[str, set[str]] = {}
        self._session_by_target: dict[str, str] = {}
        if self.enabled:
            self._ensure_worker()

    @property
    def enabled(self) -> bool:
        """Whether the client has a configured endpoint to talk to."""
        return bool(self.endpoint)

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        with self._worker_lock:
            if self._worker is not None:
                return
            self._queue = queue.Queue(maxsize=max(1, self.queue_size))
            self._worker = threading.Thread(
                target=self._run_worker,
                name="pap-decode-commit-client",
                daemon=True,
            )
            self._worker.start()

    def commit(
        self,
        *,
        request_id: str,
        session_request_id: str | None = None,
        new_seq_len: int,
        new_token_ids: Iterable[int],
        layer_complete: bool = True,
        endpoint: str | None = None,
    ) -> None:
        """Enqueue a decode-commit notification without blocking on HTTP.

        This method only raises when PAP_DECODE_COMMIT_FAIL_CLOSED is enabled
        and the local queue is full. Delivery failures are exposed by
        :meth:`flush_request`.
        """
        target_endpoint = str(endpoint or self.endpoint)
        if not target_endpoint:
            return
        self._ensure_worker()
        if self._queue is None:
            return
        with self._pending_done:
            request_id = str(request_id)
            session_request_id = str(session_request_id or request_id)
            existing_session = self._session_by_target.get(request_id)
            if existing_session is not None and existing_session != session_request_id:
                raise RuntimeError(
                    "PAP decode commit target changed logical session "
                    f"request_id={request_id} old_session={existing_session} "
                    f"new_session={session_request_id}"
                )
            self._session_by_target[request_id] = session_request_id
            self._targets_by_session.setdefault(session_request_id, set()).add(
                request_id
            )
            new_seq_len = int(new_seq_len)
            latest_seen = self._latest_seen_seq_len_by_request.get(request_id, -1)
            if new_seq_len <= latest_seen:
                return
            commit_seq = self._latest_commit_seq_by_request.get(request_id, 0) + 1
            payload = {
                "request_id": request_id,
                "session_request_id": session_request_id,
                "commit_seq": commit_seq,
                "new_seq_len": new_seq_len,
                "new_token_ids": [int(t) for t in new_token_ids],
                "layer_complete": bool(layer_complete),
                "submit_only": True,
            }
            queued_item = self._queued_item_by_request.get(request_id)
            if (
                queued_item is not None
                and queued_item["endpoint"] == target_endpoint
                and self._coalesce_payload(queued_item["payload"], payload)
            ):
                self._latest_seen_seq_len_by_request[request_id] = new_seq_len
                self._latest_commit_seq_by_request[request_id] = commit_seq
                return

            item = {"payload": payload, "endpoint": target_endpoint}
            try:
                self._queue.put_nowait(item)
            except queue.Full as exc:
                if os.environ.get("PAP_DECODE_COMMIT_FAIL_CLOSED", "").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }:
                    raise RuntimeError("PAP decode commit queue is full") from exc
                logger.warning(
                    "PAP decode commit queue full request_id=%s new_seq_len=%d",
                    request_id,
                    new_seq_len,
                )
                self._latest_seen_seq_len_by_request[request_id] = new_seq_len
                self._latest_commit_seq_by_request[request_id] = commit_seq
                return
            self._queued_item_by_request[request_id] = item
            self._pending_by_request[request_id] = (
                self._pending_by_request.get(request_id, 0) + 1
            )
            self._latest_seen_seq_len_by_request[request_id] = new_seq_len
            self._latest_commit_seq_by_request[request_id] = commit_seq

    def flush_submitted_request(
        self,
        request_id: str,
        timeout_s: float | None = None,
    ) -> bool:
        """Wait for Prefill to accept all commits issued for *request_id*."""
        if timeout_s is None:
            timeout_s = float(os.environ.get("PAP_DECODE_COMMIT_FLUSH_TIMEOUT", "15.0"))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        request_id = str(request_id)
        with self._pending_done:
            targets = self._targets_by_session.get(request_id)
            if targets is None and request_id in self._latest_commit_seq_by_request:
                targets = {request_id}
            target_sequences = {
                target: self._latest_commit_seq_by_request.get(target, 0)
                for target in targets or ()
            }
            while any(
                self._acked_commit_seq_by_request.get(target, 0) < target_seq
                for target, target_seq in target_sequences.items()
            ):
                failed = any(
                    self._acked_commit_seq_by_request.get(target, 0) < target_seq
                    and self._pending_by_request.get(target, 0) <= 0
                    for target, target_seq in target_sequences.items()
                )
                if failed:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_done.wait(timeout=remaining)
        return True

    def flush_request(self, request_id: str, timeout_s: float | None = None) -> bool:
        """Compatibility alias for :meth:`flush_submitted_request`."""
        return self.flush_submitted_request(request_id, timeout_s)

    def forget_request(self, request_id: str) -> None:
        """Drop duplicate-suppression state for a completed request."""
        with self._pending_done:
            request_id = str(request_id)
            session_request_id = self._session_by_target.get(request_id, request_id)
            targets = self._targets_by_session.pop(session_request_id, {request_id})
            for target in targets:
                self._session_by_target.pop(target, None)
                self._queued_item_by_request.pop(target, None)
                self._latest_seen_seq_len_by_request.pop(target, None)
                self._latest_commit_seq_by_request.pop(target, None)
                self._acked_commit_seq_by_request.pop(target, None)
                self._failed_commit_seq_by_request.pop(target, None)
                self._failure_by_request.pop(target, None)

    def _run_worker(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            payload = item["payload"]
            endpoint = str(item["endpoint"])
            request_id = str(payload["request_id"])
            with self._pending_done:
                if self._queued_item_by_request.get(request_id) is item:
                    self._queued_item_by_request.pop(request_id, None)
            try:
                acked_commit_seq = self._post_commit(payload, endpoint)
            except Exception as exc:
                logger.warning(
                    "PAP decode commit failed request_id=%s commit_seq=%d "
                    "new_seq_len=%d attempts=%d err=%s",
                    request_id,
                    int(payload["commit_seq"]),
                    int(payload["new_seq_len"]),
                    self.max_attempts,
                    exc,
                )
                self._mark_done(request_id, failed_payload=payload, error=exc)
            else:
                self._mark_done(
                    request_id,
                    acked_commit_seq=acked_commit_seq,
                )
            finally:
                self._queue.task_done()

    def _mark_done(
        self,
        request_id: str,
        *,
        acked_commit_seq: int | None = None,
        failed_payload: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        with self._pending_done:
            if acked_commit_seq is not None:
                previous_ack = self._acked_commit_seq_by_request.get(request_id, 0)
                self._acked_commit_seq_by_request[request_id] = max(
                    previous_ack, int(acked_commit_seq)
                )
                failed_seq = self._failed_commit_seq_by_request.get(request_id, 0)
                if int(acked_commit_seq) >= failed_seq:
                    self._failed_commit_seq_by_request.pop(request_id, None)
                    self._failure_by_request.pop(request_id, None)
            if failed_payload is not None:
                failed_seq = int(failed_payload["commit_seq"])
                self._failed_commit_seq_by_request[request_id] = max(
                    self._failed_commit_seq_by_request.get(request_id, 0),
                    failed_seq,
                )
                self._failure_by_request[request_id] = str(error)
            count = self._pending_by_request.get(str(request_id), 0) - 1
            if count > 0:
                self._pending_by_request[str(request_id)] = count
            else:
                self._pending_by_request.pop(str(request_id), None)
            self._pending_done.notify_all()

    @staticmethod
    def _coalesce_payload(existing: dict, payload: dict) -> bool:
        existing_seq_len = int(existing["new_seq_len"])
        new_seq_len = int(payload["new_seq_len"])
        if new_seq_len <= existing_seq_len:
            return True

        existing_tokens = list(existing["new_token_ids"])
        new_tokens = list(payload["new_token_ids"])
        existing_base_seq_len = existing_seq_len - len(existing_tokens)
        new_base_seq_len = new_seq_len - len(new_tokens)

        if new_base_seq_len == existing_seq_len:
            existing["commit_seq"] = int(payload["commit_seq"])
            existing["new_seq_len"] = new_seq_len
            existing["new_token_ids"] = existing_tokens + new_tokens
            existing["layer_complete"] = bool(payload["layer_complete"])
            return True
        if new_base_seq_len == existing_base_seq_len:
            existing["commit_seq"] = int(payload["commit_seq"])
            existing["new_seq_len"] = new_seq_len
            existing["new_token_ids"] = new_tokens
            existing["layer_complete"] = bool(payload["layer_complete"])
            return True
        return False

    def _post_commit(self, payload: dict, endpoint: str) -> int:
        delay_s = self.retry_initial_s
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = httpx.post(endpoint, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                body = resp.json()
                acked_commit_seq = int(
                    body.get(
                        "accepted_commit_seq",
                        body.get("acked_commit_seq", 0),
                    )
                )
                expected_commit_seq = int(payload["commit_seq"])
                if acked_commit_seq < expected_commit_seq:
                    raise RuntimeError(
                        "decode commit response did not acknowledge requested "
                        f"sequence: expected>={expected_commit_seq}, "
                        f"got={acked_commit_seq}"
                    )
                return acked_commit_seq
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                logger.debug(
                    "PAP decode commit retry request_id=%s commit_seq=%d "
                    "attempt=%d/%d err=%s",
                    payload.get("request_id"),
                    int(payload.get("commit_seq", 0)),
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if delay_s > 0:
                    time.sleep(delay_s)
                    delay_s = min(self.retry_max_s, delay_s * 2)
        assert last_error is not None
        raise last_error
