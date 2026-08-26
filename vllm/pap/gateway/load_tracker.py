# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gateway-owned PA load accounting for PAP routing."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from vllm.pap.gateway.clients import PAPServiceClient, get_prefill_kv_load
from vllm.pap.gateway.topology import PAPGroup

_BOOTSTRAP_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class _PACapacity:
    total_kv_tokens: int
    kv_block_size: int


@dataclass(slots=True)
class _RequestLoad:
    group: PAPGroup
    prefill_tokens: int
    decode_capacity_tokens: int
    prefix_tokens: int = 0
    prefill_complete: bool = False


class PAPLoadTracker:
    """Track active PA work locally from the Gateway request lifecycle."""

    def __init__(self, clients: dict[PAPGroup, PAPServiceClient]) -> None:
        self._clients = clients
        self._capacities: dict[PAPGroup, _PACapacity] = {}
        self._requests: dict[str, _RequestLoad] = {}
        self._started_at = 0.0
        self._lock = Lock()

    async def start(self) -> None:
        """Read immutable PA capacities once before the Gateway becomes ready."""
        groups = tuple(self._clients)
        snapshots = await asyncio.wait_for(
            asyncio.gather(
                *(get_prefill_kv_load(self._clients[group]) for group in groups)
            ),
            timeout=_BOOTSTRAP_TIMEOUT_S,
        )
        capacities = {
            group: _PACapacity(
                total_kv_tokens=max(0, int(snapshot["total_kv_tokens"])),
                kv_block_size=max(1, int(snapshot["kv_block_size"])),
            )
            for group, snapshot in zip(groups, snapshots)
        }
        if any(capacity.total_kv_tokens == 0 for capacity in capacities.values()):
            raise RuntimeError("PAP PA reported zero KV capacity")
        with self._lock:
            self._capacities = capacities
            self._started_at = time.monotonic()

    def begin_request(
        self,
        request_id: str,
        group: PAPGroup,
        *,
        prefill_tokens: int,
        decode_capacity_tokens: int,
    ) -> None:
        """Charge a routed request before any downstream await point."""
        load = _RequestLoad(
            group=group,
            prefill_tokens=max(1, int(prefill_tokens)),
            decode_capacity_tokens=max(0, int(decode_capacity_tokens)),
        )
        with self._lock:
            if request_id in self._requests:
                raise ValueError(f"duplicate PAP request load: {request_id}")
            self._requests[request_id] = load

    def mark_prefill_completed(self, request_id: str, prefix_tokens: int) -> None:
        """Replace estimated Prefill work with the exact Decode prefix length."""
        with self._lock:
            load = self._requests.get(request_id)
            if load is None:
                return
            load.prefix_tokens = max(1, int(prefix_tokens))
            load.prefill_complete = True

    def finish_request(self, request_id: str) -> None:
        """Release one active request from local PA accounting."""
        with self._lock:
            self._requests.pop(request_id, None)

    def snapshot(self) -> dict[PAPGroup, dict[str, int]]:
        """Return the current projected PA loads without performing I/O."""
        with self._lock:
            capacities = dict(self._capacities)
            requests = tuple(self._requests.values())

        snapshots: dict[PAPGroup, dict[str, int]] = {}
        for group, capacity in capacities.items():
            prefill = [
                load
                for load in requests
                if load.group == group and not load.prefill_complete
            ]
            decode = [
                load
                for load in requests
                if load.group == group and load.prefill_complete
            ]
            outstanding_prefill = sum(load.prefill_tokens for load in prefill)
            decode_reservations = sum(
                load.decode_capacity_tokens for load in (*prefill, *decode)
            )
            resident_prefix = sum(load.prefix_tokens for load in decode)
            projected_kv = resident_prefix + outstanding_prefill + decode_reservations
            block_size = capacity.kv_block_size
            non_evictable_blocks = (resident_prefix + block_size - 1) // block_size
            total_blocks = capacity.total_kv_tokens // block_size
            snapshots[group] = {
                "non_evictable_kv_blocks": non_evictable_blocks,
                "non_evictable_kv_tokens": resident_prefix,
                "running_prefill_tokens": 0,
                "queued_prefill_tokens": outstanding_prefill,
                "outstanding_prefill_tokens": outstanding_prefill,
                "running_decode_reservation_tokens": 0,
                "queued_decode_reservation_tokens": decode_reservations,
                "outstanding_decode_reservation_tokens": decode_reservations,
                "running_prefill_requests": 0,
                "queued_prefill_requests": len(prefill),
                "projected_kv_tokens": projected_kv,
                "routing_kv_tokens": projected_kv,
                "free_kv_blocks": max(0, total_blocks - non_evictable_blocks),
                "total_kv_blocks": total_blocks,
                "total_kv_tokens": capacity.total_kv_tokens,
                "kv_block_size": block_size,
            }
        return snapshots

    def stats(self) -> dict[str, Any]:
        """Return local accounting state for benchmark audits."""
        with self._lock:
            requests = tuple(self._requests.values())
            capacities = dict(self._capacities)
            started_at = self._started_at
        return {
            "source": "gateway_request_lifecycle",
            "bootstrap_rpc_count": len(capacities),
            "runtime_poll_count": 0,
            "tracked_requests": len(requests),
            "prefill_requests": sum(not load.prefill_complete for load in requests),
            "decode_requests": sum(load.prefill_complete for load in requests),
            "uptime_s": max(0.0, time.monotonic() - started_at),
        }


__all__ = ["PAPLoadTracker"]
