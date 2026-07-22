# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention-to-Prefill KV lease-release client."""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)


class LeaseReleaseClient:
    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float | None = None,
        max_attempts: int | None = None,
        retry_initial_s: float | None = None,
        retry_max_s: float | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("PAP_LEASE_RELEASE_ENDPOINT")
        self.timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(os.environ.get("PAP_LEASE_RELEASE_TIMEOUT", "5.0"))
        )
        self.max_attempts = max(
            1,
            int(max_attempts)
            if max_attempts is not None
            else int(os.environ.get("PAP_LEASE_RELEASE_MAX_ATTEMPTS", "5")),
        )
        self.retry_initial_s = max(
            0.0,
            float(retry_initial_s)
            if retry_initial_s is not None
            else float(
                os.environ.get("PAP_LEASE_RELEASE_RETRY_INITIAL_SECONDS", "0.05")
            ),
        )
        self.retry_max_s = max(
            self.retry_initial_s,
            float(retry_max_s)
            if retry_max_s is not None
            else float(os.environ.get("PAP_LEASE_RELEASE_RETRY_MAX_SECONDS", "0.5")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)

    def release(
        self,
        *,
        request_id: str,
        lease_id: str,
        endpoint: str | None = None,
    ) -> bool:
        target_endpoint = str(endpoint or self.endpoint or "")
        if not target_endpoint:
            return True
        delay_s = self.retry_initial_s
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = httpx.post(
                    target_endpoint,
                    json={"request_id": request_id, "lease_id": lease_id},
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                body = response.json()
                if body.get("released", False) or body.get("reason") == (
                    "unknown_or_released_lease"
                ):
                    return True
                raise RuntimeError(f"lease release was not acknowledged: {body}")
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                logger.debug(
                    "PAP lease release retry request_id=%s lease_id=%s "
                    "attempt=%d/%d err=%s",
                    request_id,
                    lease_id,
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if delay_s > 0:
                    time.sleep(delay_s)
                    delay_s = min(self.retry_max_s, delay_s * 2)
        logger.warning(
            "PAP lease release failed request_id=%s lease_id=%s attempts=%d err=%s",
            request_id,
            lease_id,
            self.max_attempts,
            last_error,
        )
        return False
