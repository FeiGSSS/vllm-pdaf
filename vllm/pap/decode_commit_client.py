# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fire-and-forget HTTP client for PAP decode commit notifications.

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
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


class DecodeCommitClient:
    """Fire-and-forget HTTP client for PAP decode commit notifications.

    POST errors are logged as warnings from a daemon worker. The caller only
    enqueues payloads unless fail-closed queue behavior is explicitly enabled.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float = 0.2,
        queue_size: int | None = None,
    ) -> None:
        """Create a client that POSTs to *endpoint* (or
        ``PAP_DECODE_COMMIT_ENDPOINT`` env var)."""
        self.endpoint = endpoint or os.environ.get(
            "PAP_DECODE_COMMIT_ENDPOINT", "")
        self.timeout_s = timeout_s
        self.queue_size = (
            int(queue_size)
            if queue_size is not None
            else int(os.environ.get("PAP_DECODE_COMMIT_QUEUE_SIZE", "1024"))
        )
        self._queue: queue.Queue[dict] | None = None
        self._worker: threading.Thread | None = None
        self._pending_lock = threading.Lock()
        self._pending_done = threading.Condition(self._pending_lock)
        self._pending_keys: set[tuple] = set()
        self._pending_by_request: dict[str, int] = {}
        self._known_keys_by_request: dict[str, set[tuple]] = {}
        if self.enabled:
            self._queue = queue.Queue(maxsize=max(1, self.queue_size))
            self._worker = threading.Thread(
                target=self._run_worker,
                name="pap-decode-commit-client",
                daemon=True,
            )
            self._worker.start()

    @property
    def enabled(self) -> bool:
        """Whether the client has a configured endpoint to talk to."""
        return bool(self.endpoint)

    def commit(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        new_token_ids: Iterable[int],
        layer_complete: bool = True,
    ) -> None:
        """Fire a decode-commit notification (fire-and-forget).

        This method only raises when PAP_DECODE_COMMIT_FAIL_CLOSED is enabled
        and the local queue is full; POST errors are logged by the worker.
        """
        if not self.enabled:
            return

        payload = {
            "request_id": str(request_id),
            "new_seq_len": int(new_seq_len),
            "new_token_ids": [int(t) for t in new_token_ids],
            "layer_complete": bool(layer_complete),
        }
        key = (
            payload["request_id"],
            payload["new_seq_len"],
            tuple(payload["new_token_ids"]),
            payload["layer_complete"],
        )

        if self._queue is None:
            return
        with self._pending_done:
            known_keys = self._known_keys_by_request.setdefault(
                payload["request_id"], set()
            )
            if key in known_keys:
                return
            known_keys.add(key)
            self._pending_keys.add(key)
            self._pending_by_request[payload["request_id"]] = (
                self._pending_by_request.get(payload["request_id"], 0) + 1
            )
        try:
            self._queue.put_nowait({"key": key, "payload": payload})
        except queue.Full as exc:
            self._mark_done(payload["request_id"], key, keep_known=False)
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

    def flush_request(self, request_id: str, timeout_s: float | None = None) -> bool:
        """Wait until all currently queued commits for *request_id* are sent."""
        if not self.enabled:
            return True
        if timeout_s is None:
            timeout_s = float(os.environ.get(
                "PAP_DECODE_COMMIT_FLUSH_TIMEOUT", "5.0"
            ))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        request_id = str(request_id)
        with self._pending_done:
            while self._pending_by_request.get(request_id, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_done.wait(timeout=remaining)
        return True

    def forget_request(self, request_id: str) -> None:
        """Drop duplicate-suppression state for a completed request."""
        with self._pending_done:
            self._known_keys_by_request.pop(str(request_id), None)

    def _run_worker(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            key = item["key"]
            payload = item["payload"]
            try:
                self._post_commit(payload)
            finally:
                self._mark_done(payload["request_id"], key, keep_known=True)
                self._queue.task_done()

    def _mark_done(
        self,
        request_id: str,
        key: tuple,
        *,
        keep_known: bool,
    ) -> None:
        with self._pending_done:
            if key in self._pending_keys:
                self._pending_keys.remove(key)
                count = self._pending_by_request.get(str(request_id), 0) - 1
                if count > 0:
                    self._pending_by_request[str(request_id)] = count
                else:
                    self._pending_by_request.pop(str(request_id), None)
                self._pending_done.notify_all()
            if not keep_known:
                known_keys = self._known_keys_by_request.get(str(request_id))
                if known_keys is not None:
                    known_keys.discard(key)
                    if not known_keys:
                        self._known_keys_by_request.pop(str(request_id), None)

    def _post_commit(self, payload: dict) -> None:
        try:
            resp = httpx.post(self.endpoint, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(
                "PAP decode commit failed request_id=%s new_seq_len=%d err=%s",
                payload.get("request_id"),
                int(payload.get("new_seq_len", 0)),
                exc,
            )
