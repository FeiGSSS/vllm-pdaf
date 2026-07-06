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
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


class DecodeCommitClient:
    """Fire-and-forget HTTP client for PAP decode commit notifications.

    All errors are logged as warnings; the caller is never blocked.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float = 0.2,
    ) -> None:
        """Create a client that POSTs to *endpoint* (or
        ``PAP_DECODE_COMMIT_ENDPOINT`` env var)."""
        self.endpoint = endpoint or os.environ.get(
            "PAP_DECODE_COMMIT_ENDPOINT", "")
        self.timeout_s = timeout_s

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

        This method never raises; errors are logged as warnings.
        """
        if not self.enabled:
            return

        payload = {
            "request_id": str(request_id),
            "new_seq_len": int(new_seq_len),
            "new_token_ids": [int(t) for t in new_token_ids],
            "layer_complete": bool(layer_complete),
        }

        try:
            resp = httpx.post(
                self.endpoint, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(
                "PAP decode commit failed request_id=%s new_seq_len=%d err=%s",
                request_id,
                new_seq_len,
                exc,
            )
