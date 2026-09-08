# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conversation-aware orchestration for same-host PA KV migration."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import pybase64 as base64

from vllm.pap.gateway.clients import PAPServiceClient, request_headers
from vllm.pap.integration.request import PAPKVCacheMigration

logger = logging.getLogger("pap_gateway")

_EXPORT_PATH = "/v1/pap/prefill/kv-migration/export"
_RELEASE_PATH = "/v1/pap/prefill/kv-migration/release"


@dataclass(slots=True)
class PAPKVCacheMigrationLease:
    """One pinned source export that must be released after target Prefill."""

    transfer_id: str
    source: PAPServiceClient
    metadata: dict[str, Any]
    request_id: str
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        response = await self.source.client.post(
            _RELEASE_PATH,
            json={"transfer_id": self.transfer_id},
            headers=request_headers(self.request_id),
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("released", False):
            raise RuntimeError(
                f"PAP KV migration source did not release {self.transfer_id}"
            )
        self.released = True


class PAPKVCacheMigrationCoordinator:
    """Track the previous PA and prepare a pinned same-host source prefix."""

    def __init__(self, *, enabled: bool, max_conversations: int) -> None:
        if max_conversations <= 0:
            raise ValueError("KV migration conversation limit must be positive")
        self.enabled = enabled
        self._max_conversations = max_conversations
        self._conversation_workers: dict[str, int | None] = {}
        self._active_conversations: dict[str, str] = {}
        self._new_requests: set[str] = set()
        self._attempts = 0
        self._exports = 0
        self._misses = 0
        self._failures = 0
        self._copied_prefix_tokens = 0
        self._release_tasks: set[asyncio.Task[None]] = set()

    def begin_request(self, conversation_id: str, request_id: str) -> None:
        if not self.enabled or not conversation_id:
            return
        if self._failures:
            raise RuntimeError("PAP KV migration coordinator is unhealthy")
        owner = self._active_conversations.get(conversation_id)
        if owner is not None:
            self._failures += 1
            raise RuntimeError(
                "PAP KV migration requires serialized conversation turns: "
                f"active_request={owner}"
            )
        if conversation_id not in self._conversation_workers:
            if len(self._conversation_workers) >= self._max_conversations:
                self._failures += 1
                raise RuntimeError("PAP KV migration conversation limit reached")
            self._conversation_workers[conversation_id] = None
            self._new_requests.add(request_id)
        self._active_conversations[conversation_id] = request_id

    async def prepare(
        self,
        *,
        conversation_id: str,
        request_id: str,
        target_worker_id: int,
        raw_block_hashes: tuple[bytes, ...],
        prefill_clients: list[PAPServiceClient],
    ) -> PAPKVCacheMigrationLease | None:
        if not self.enabled or not conversation_id:
            return None
        if self._active_conversations.get(conversation_id) != request_id:
            raise RuntimeError("PAP KV migration request was not registered")
        source_worker_id = self._conversation_workers[conversation_id]
        if source_worker_id is None or source_worker_id == target_worker_id:
            return None
        if not raw_block_hashes:
            self._misses += 1
            return None
        self._attempts += 1
        transfer_id = f"{request_id}:{source_worker_id}:{target_worker_id}"
        source = prefill_clients[source_worker_id]
        metadata: dict[str, Any] | None = None
        export_start = time.perf_counter()
        try:
            response = await source.client.post(
                _EXPORT_PATH,
                json={
                    "transfer_id": transfer_id,
                    "source_worker_id": source_worker_id,
                    "block_hashes_b64": [
                        base64.b64encode(value).decode("ascii")
                        for value in raw_block_hashes
                    ],
                },
                headers=request_headers(request_id),
            )
            response.raise_for_status()
            metadata = response.json()
            if metadata.get("status") == "miss":
                self._misses += 1
                logger.info(
                    "PAP KV migration miss request_id=%s conversation_id=%s "
                    "source_pa=%d target_pa=%d",
                    request_id,
                    conversation_id,
                    source_worker_id,
                    target_worker_id,
                )
                return None
            migration = PAPKVCacheMigration.from_mapping(metadata)
            self._exports += 1
            self._copied_prefix_tokens += migration.prefix_tokens
            logger.info(
                "PAP KV migration prepared request_id=%s conversation_id=%s "
                "source_pa=%d target_pa=%d prefix_tokens=%d blocks=%d",
                request_id,
                conversation_id,
                source_worker_id,
                target_worker_id,
                migration.prefix_tokens,
                len(migration.source_block_ids),
            )
            logger.info(
                "PAP KV migration export timing request_id=%s transfer_id=%s "
                "export_ms=%.3f",
                request_id,
                transfer_id,
                (time.perf_counter() - export_start) * 1000.0,
            )
            return PAPKVCacheMigrationLease(
                transfer_id=transfer_id,
                source=source,
                metadata=metadata,
                request_id=request_id,
            )
        except BaseException:
            self._failures += 1
            if metadata is not None and metadata.get("status") == "exported":
                try:
                    release = await source.client.post(
                        _RELEASE_PATH,
                        json={"transfer_id": transfer_id},
                        headers=request_headers(request_id),
                    )
                    release.raise_for_status()
                    if not release.json().get("released", False):
                        raise RuntimeError("source export was not released")
                except Exception:
                    logger.exception(
                        "PAP KV migration failed to unwind source export %s",
                        transfer_id,
                    )
            raise

    def record_prefill(
        self, conversation_id: str, request_id: str, worker_id: int
    ) -> None:
        if not self.enabled or not conversation_id:
            return
        if self._active_conversations.get(conversation_id) != request_id:
            raise RuntimeError("PAP KV migration Prefill has no active conversation")
        self._conversation_workers[conversation_id] = worker_id

    async def release(self, lease: PAPKVCacheMigrationLease) -> None:
        start = time.perf_counter()
        try:
            await lease.release()
        except BaseException:
            self._failures += 1
            raise
        logger.info(
            "PAP KV migration release timing request_id=%s transfer_id=%s "
            "release_ms=%.3f",
            lease.request_id,
            lease.transfer_id,
            (time.perf_counter() - start) * 1000.0,
        )

    def release_after_prefill(self, lease: PAPKVCacheMigrationLease) -> None:
        task = asyncio.create_task(
            self.release(lease),
            name=f"pap-kv-migration-release-{lease.transfer_id}",
        )
        self._release_tasks.add(task)
        task.add_done_callback(self._release_done)

    def _release_done(self, task: asyncio.Task[None]) -> None:
        self._release_tasks.discard(task)
        try:
            task.result()
        except Exception:
            logger.exception("PAP KV migration asynchronous source release failed")

    async def shutdown(self) -> None:
        if self._release_tasks:
            await asyncio.gather(*tuple(self._release_tasks), return_exceptions=True)

    def finish_request(self, conversation_id: str, request_id: str) -> None:
        if not self.enabled or not conversation_id:
            return
        if self._active_conversations.get(conversation_id) == request_id:
            self._active_conversations.pop(conversation_id, None)
        if request_id in self._new_requests:
            self._new_requests.remove(request_id)
            if self._conversation_workers.get(conversation_id) is None:
                self._conversation_workers.pop(conversation_id, None)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "conversation_bindings": len(self._conversation_workers),
            "active_conversations": len(self._active_conversations),
            "max_conversations": self._max_conversations,
            "attempts": self._attempts,
            "exports": self._exports,
            "misses": self._misses,
            "failures": self._failures,
            "exported_prefix_tokens": self._copied_prefix_tokens,
            "pending_releases": len(self._release_tasks),
            "healthy": self._failures == 0,
        }


__all__ = [
    "PAPKVCacheMigrationCoordinator",
    "PAPKVCacheMigrationLease",
]
