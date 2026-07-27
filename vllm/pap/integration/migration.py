# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-band PAP post-Prefill KV migration state."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PAPMigrationStatus(str, Enum):
    """Lifecycle states for one post-Prefill migration."""

    PENDING = "pending"
    TRANSFERRING = "transferring"
    READY = "ready"
    FAILED = "failed"


def validate_pap_migration_tp_size(tp_size: int) -> None:
    """Reject unverified tensor-parallel post-Prefill migrations."""
    if tp_size != 1:
        raise NotImplementedError(
            "PAP post-Prefill KV migration currently supports only TP=1; "
            f"TP={tp_size} requires rank-coordinated background completion "
            "and has not been implemented"
        )


@dataclass(slots=True)
class PAPMigrationJob:
    """Scheduler-owned state for one out-of-band KV migration."""

    request_id: str
    source_kv_params: dict[str, Any]
    prefix_len: int
    prefix_token_ids: tuple[int, ...]
    prefix_block_hashes: tuple[bytes, ...]
    decode_capacity_tokens: int
    session_handle: str
    attention_tcp_endpoint: str
    job_id: str = field(default_factory=lambda: f"pap-migration-{uuid.uuid4().hex}")
    status: PAPMigrationStatus = PAPMigrationStatus.PENDING
    block_ids: tuple[tuple[int, ...], ...] = ()
    created_at: float = field(default_factory=time.monotonic)
    submitted_at: float | None = None
    completed_at: float | None = None
    kv_transfer_params: dict[str, Any] | None = None
    error: str | None = None
    started: Future[dict[str, Any]] = field(
        default_factory=Future,
        repr=False,
    )
    completion: Future[dict[str, Any]] = field(
        default_factory=Future,
        repr=False,
    )

    @property
    def total_capacity_tokens(self) -> int:
        """Return prefix plus locally reserved Decode capacity."""
        return self.prefix_len + self.decode_capacity_tokens

    def manifest(self) -> dict[str, Any]:
        """Build worker-side manifest publication metadata."""
        if not self.block_ids:
            raise RuntimeError("PAP migration has no allocated blocks")
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "prefix_len": self.prefix_len,
            "decode_capacity_tokens": self.decode_capacity_tokens,
            "session_handle": self.session_handle,
            "attention_tcp_endpoint": self.attention_tcp_endpoint,
            "block_ids": [list(group) for group in self.block_ids],
        }

    def response(self) -> dict[str, Any]:
        """Return a JSON-serializable control-plane view."""
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "status": self.status.value,
            "prefix_len": self.prefix_len,
            "queue_ms": (
                int((self.submitted_at - self.created_at) * 1000)
                if self.submitted_at is not None
                else None
            ),
            "transfer_and_publish_ms": (
                int((self.completed_at - self.submitted_at) * 1000)
                if self.completed_at is not None and self.submitted_at is not None
                else None
            ),
            "total_ms": (
                int((self.completed_at - self.created_at) * 1000)
                if self.completed_at is not None
                else None
            ),
        }
        if self.kv_transfer_params is not None:
            result["kv_transfer_params"] = dict(self.kv_transfer_params)
        if self.error is not None:
            result["error"] = self.error
        return result


__all__ = [
    "PAPMigrationJob",
    "PAPMigrationStatus",
    "validate_pap_migration_tp_size",
]
