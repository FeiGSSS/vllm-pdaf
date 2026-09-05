# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous Projection-to-Attention decode-token delivery."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from typing import SupportsIndex, SupportsInt, TypeAlias, TypedDict, cast

import httpx
from typing_extensions import Buffer

from vllm.pap.config import reject_removed_pap_flags

logger = logging.getLogger(__name__)

_DECODE_TOKEN_BATCH_PATH = "/v1/pap/attention/decode-tokens"

_IntInput: TypeAlias = str | Buffer | SupportsInt | SupportsIndex


class _DecodeTokenPayload(TypedDict):
    request_id: str
    new_seq_len: int
    token_id: int


class _DecodeTokenBatchBody(TypedDict):
    tokens: list[_DecodeTokenPayload]


class _DecodeTokenQueueItem(TypedDict):
    request_ids: tuple[str, ...]
    endpoint: str
    payload: _DecodeTokenBatchBody


def _coerce_int(value: object) -> int:
    return int(cast(_IntInput, value))


class DecodeTokenClient:
    """Reliably deliver sampled tokens without blocking the model thread."""

    def __init__(
        self,
        *,
        timeout_s: float | None = None,
        queue_size: int | None = None,
        max_attempts: int | None = None,
        retry_initial_s: float | None = None,
        retry_max_s: float | None = None,
    ) -> None:
        reject_removed_pap_flags(os.environ)
        self.timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(os.environ.get("PAP_DECODE_TOKEN_TIMEOUT", "0.2"))
        )
        self.max_attempts = max(
            1,
            int(max_attempts)
            if max_attempts is not None
            else int(os.environ.get("PAP_DECODE_TOKEN_MAX_ATTEMPTS", "8")),
        )
        self.retry_initial_s = max(
            0.0,
            float(retry_initial_s)
            if retry_initial_s is not None
            else float(
                os.environ.get(
                    "PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS",
                    "0.05",
                )
            ),
        )
        self.retry_max_s = max(
            self.retry_initial_s,
            float(retry_max_s)
            if retry_max_s is not None
            else float(os.environ.get("PAP_DECODE_TOKEN_RETRY_MAX_SECONDS", "0.5")),
        )
        capacity = max(
            1,
            int(queue_size)
            if queue_size is not None
            else int(os.environ.get("PAP_DECODE_TOKEN_QUEUE_SIZE", "1024")),
        )
        self._queue: queue.Queue[_DecodeTokenQueueItem | None] = queue.Queue(capacity)
        self._pending_lock = threading.Lock()
        self._pending_done = threading.Condition(self._pending_lock)
        self._pending_by_request: dict[str, int] = {}
        self._failure_by_request: dict[str, str] = {}
        self._worker = threading.Thread(
            target=self._run_worker,
            name="pap-decode-token-client",
            daemon=True,
        )
        self._worker.start()

    def publish(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        token_id: int,
        endpoint: str,
    ) -> None:
        self.publish_batch(
            (
                {
                    "request_id": request_id,
                    "new_seq_len": new_seq_len,
                    "token_id": token_id,
                    "endpoint": endpoint,
                },
            )
        )

    def publish_batch(
        self,
        tokens: Sequence[Mapping[str, object]],
    ) -> None:
        grouped: dict[str, list[_DecodeTokenPayload]] = {}
        for token in tokens:
            request_id = str(token["request_id"])
            endpoint = str(token["endpoint"]).rstrip("/")
            if not endpoint:
                raise ValueError("PAP decode-token endpoint must not be empty")
            grouped.setdefault(endpoint, []).append(
                {
                    "request_id": request_id,
                    "new_seq_len": _coerce_int(token["new_seq_len"]),
                    "token_id": _coerce_int(token["token_id"]),
                }
            )
        if not grouped:
            return

        items: list[_DecodeTokenQueueItem] = [
            {
                "request_ids": tuple(
                    str(payload["request_id"]) for payload in payloads
                ),
                "endpoint": f"{endpoint}{_DECODE_TOKEN_BATCH_PATH}",
                "payload": {"tokens": payloads},
            }
            for endpoint, payloads in grouped.items()
        ]
        with self._pending_done:
            available = self._queue.maxsize - self._queue.qsize()
            if available < len(items):
                raise RuntimeError("PAP decode-token queue is full")
            for item in items:
                self._queue.put_nowait(item)
                for request_id in item["request_ids"]:
                    request_id = str(request_id)
                    self._pending_by_request[request_id] = (
                        self._pending_by_request.get(request_id, 0) + 1
                    )

    def flush_request(self, request_id: str, timeout_s: float | None = None) -> bool:
        if timeout_s is None:
            timeout_s = float(os.environ.get("PAP_DECODE_TOKEN_FLUSH_TIMEOUT", "5.0"))
        request_id = str(request_id)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._pending_done:
            while self._pending_by_request.get(request_id, 0) > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._pending_done.wait(timeout=remaining)
            return request_id not in self._failure_by_request

    def forget_request(self, request_id: str) -> None:
        with self._pending_done:
            self._failure_by_request.pop(str(request_id), None)

    def shutdown(self, timeout_s: float = 1.0) -> None:
        if not self._worker.is_alive():
            return
        self._queue.put(None)
        self._worker.join(timeout=max(0.0, float(timeout_s)))

    def _run_worker(self) -> None:
        with httpx.Client() as client:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    return
                request_ids = tuple(
                    str(request_id) for request_id in item["request_ids"]
                )
                try:
                    self._post_token(
                        client=client,
                        endpoint=str(item["endpoint"]),
                        payload=item["payload"],
                    )
                except Exception as exc:
                    logger.warning(
                        "PAP decode-token delivery failed request_id=%s err=%s",
                        ",".join(request_ids),
                        exc,
                    )
                    self._mark_done(request_ids, error=exc)
                else:
                    self._mark_done(request_ids)
                finally:
                    self._queue.task_done()

    def _mark_done(
        self,
        request_ids: Sequence[str],
        error: Exception | None = None,
    ) -> None:
        with self._pending_done:
            for request_id in request_ids:
                if error is not None:
                    self._failure_by_request[request_id] = str(error)
                count = self._pending_by_request.get(request_id, 0) - 1
                if count > 0:
                    self._pending_by_request[request_id] = count
                else:
                    self._pending_by_request.pop(request_id, None)
            self._pending_done.notify_all()

    def _post_token(
        self,
        *,
        client: httpx.Client,
        endpoint: str,
        payload: _DecodeTokenBatchBody,
    ) -> None:
        delay_s = self.retry_initial_s
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = client.post(
                    endpoint,
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                body = response.json()
                if str(body.get("status", "")) not in {
                    "accepted",
                    "matched",
                    "pending",
                    "duplicate",
                }:
                    raise RuntimeError("decode-token response did not accept payload")
                return
            except Exception as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    response_body = exc.response.text.strip()
                    last_error = RuntimeError(
                        f"{exc}; response_body={response_body or '<empty>'}"
                    )
                else:
                    last_error = exc
                if attempt >= self.max_attempts:
                    break
                if delay_s > 0:
                    time.sleep(delay_s)
                    delay_s = min(self.retry_max_s, delay_s * 2)
        assert last_error is not None
        raise last_error
