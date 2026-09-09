# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-after-write fence for conversation-local Dynamo KV routing."""

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

_ACQUIRE_PATH = "/v1/pap/prefill/kv-fence/acquire"
_RELEASE_PATH = "/v1/pap/prefill/kv-fence/release"


@dataclass(slots=True)
class PAPKVFenceLease:
    """Pinned prefix used while Dynamo makes a consistency-checked choice."""

    transfer_id: str
    source: PAPServiceClient
    request_id: str
    source_worker_id: int
    prefix_tokens: int
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
        if not response.json().get("released", False):
            raise RuntimeError(f"PAP KV fence did not release {self.transfer_id}")
        self.released = True


class PAPKVFenceCoordinator:
    """Track the previous PA and acquire its exact prefix before selection."""

    def __init__(self, *, enabled: bool, max_conversations: int) -> None:
        if max_conversations <= 0:
            raise ValueError("KV fence conversation limit must be positive")
        self.enabled = enabled
        self._max_conversations = max_conversations
        self._conversation_workers: dict[str, int | None] = {}
        self._active: dict[str, str] = {}
        self._new_requests: set[str] = set()
        self._checks = 0
        self._exports = 0
        self._misses = 0
        self._failures = 0
        self._export_ms = 0.0
        self._release_tasks: set[asyncio.Task[None]] = set()

    def begin_request(self, conversation_id: str, request_id: str) -> None:
        if not self.enabled or not conversation_id:
            return
        if self._failures:
            raise RuntimeError("PAP KV fence is unhealthy")
        owner = self._active.get(conversation_id)
        if owner is not None:
            self._failures += 1
            raise RuntimeError(
                "PAP KV fence requires serialized conversation turns: "
                f"active_request={owner}"
            )
        if conversation_id not in self._conversation_workers:
            if len(self._conversation_workers) >= self._max_conversations:
                self._failures += 1
                raise RuntimeError("PAP KV fence conversation limit reached")
            self._conversation_workers[conversation_id] = None
            self._new_requests.add(request_id)
        self._active[conversation_id] = request_id

    async def acquire(
        self,
        *,
        conversation_id: str,
        request_id: str,
        raw_block_hashes: tuple[bytes, ...],
        prefill_clients: list[PAPServiceClient],
    ) -> PAPKVFenceLease | None:
        if not self.enabled or not conversation_id:
            return None
        if self._active.get(conversation_id) != request_id:
            raise RuntimeError("PAP KV fence request was not registered")
        source_worker_id = self._conversation_workers[conversation_id]
        if source_worker_id is None or not raw_block_hashes:
            return None
        self._checks += 1
        transfer_id = f"fence:{request_id}:{source_worker_id}"
        source = prefill_clients[source_worker_id]
        start = time.perf_counter()
        exported = False
        try:
            response = await source.client.post(
                _ACQUIRE_PATH,
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
            metadata: dict[str, Any] = response.json()
            if metadata.get("status") == "miss":
                self._misses += 1
                return None
            exported = metadata.get("status") == "exported"
            migration = PAPKVCacheMigration.from_mapping(metadata)
            self._exports += 1
            return PAPKVFenceLease(
                transfer_id=transfer_id,
                source=source,
                request_id=request_id,
                source_worker_id=source_worker_id,
                prefix_tokens=migration.prefix_tokens,
            )
        except BaseException:
            self._failures += 1
            if exported:
                try:
                    release = await source.client.post(
                        _RELEASE_PATH,
                        json={"transfer_id": transfer_id},
                        headers=request_headers(request_id),
                    )
                    release.raise_for_status()
                except Exception:
                    logger.exception("PAP KV fence unwind failed: %s", transfer_id)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._export_ms += elapsed_ms
            logger.info(
                "PAP KV fence timing request_id=%s source_pa=%d export_ms=%.3f",
                request_id,
                source_worker_id,
                elapsed_ms,
            )

    def record_prefill(
        self, conversation_id: str, request_id: str, worker_id: int
    ) -> None:
        if not self.enabled or not conversation_id:
            return
        if self._active.get(conversation_id) != request_id:
            raise RuntimeError("PAP KV fence Prefill has no active conversation")
        self._conversation_workers[conversation_id] = worker_id

    async def release(self, lease: PAPKVFenceLease) -> None:
        try:
            await lease.release()
        except BaseException:
            self._failures += 1
            raise

    def release_after_prefill(self, lease: PAPKVFenceLease) -> None:
        task = asyncio.create_task(
            self.release(lease), name=f"pap-kv-fence-release-{lease.transfer_id}"
        )
        self._release_tasks.add(task)
        task.add_done_callback(self._release_done)

    def _release_done(self, task: asyncio.Task[None]) -> None:
        self._release_tasks.discard(task)
        try:
            task.result()
        except Exception:
            logger.exception("PAP KV fence asynchronous release failed")

    def finish_request(self, conversation_id: str, request_id: str) -> None:
        if not self.enabled or not conversation_id:
            return
        if self._active.get(conversation_id) == request_id:
            self._active.pop(conversation_id, None)
        if request_id in self._new_requests:
            self._new_requests.remove(request_id)
            if self._conversation_workers.get(conversation_id) is None:
                self._conversation_workers.pop(conversation_id, None)

    async def shutdown(self) -> None:
        if self._release_tasks:
            await asyncio.gather(*tuple(self._release_tasks), return_exceptions=True)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "conversation_bindings": len(self._conversation_workers),
            "active_conversations": len(self._active),
            "checks": self._checks,
            "exports": self._exports,
            "misses": self._misses,
            "failures": self._failures,
            "mean_export_ms": self._export_ms / self._checks if self._checks else 0.0,
            "pending_releases": len(self._release_tasks),
            "healthy": self._failures == 0,
        }


__all__ = ["PAPKVFenceCoordinator", "PAPKVFenceLease"]
