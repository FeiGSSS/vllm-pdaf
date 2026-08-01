# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway Prefill and Projection admission control."""

from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance


class PAPPrefillAdmission:
    """Bound in-flight Prefill requests independently on each PA."""

    def __init__(self, groups: list[PAPGroup], max_inflight_per_pa: int) -> None:
        if max_inflight_per_pa < 0:
            raise ValueError("max_inflight_per_pa must be >= 0")
        self._condition = asyncio.Condition()
        self._max_inflight_per_pa = max_inflight_per_pa
        self._active = {group: 0 for group in groups}
        self._waiters = {group: deque[object]() for group in groups}
        self._admitted = Counter[PAPGroup]()
        self._queued = Counter[PAPGroup]()
        self._wait_ms_total = {group: 0.0 for group in groups}
        self._wait_ms_max = {group: 0.0 for group in groups}
        self._group_indices = {group: index for index, group in enumerate(groups)}

    async def acquire(self, group: PAPGroup) -> float:
        """Wait for one FIFO Prefill slot; zero slots means unbounded."""
        if self._max_inflight_per_pa == 0:
            return 0.0
        started = time.perf_counter()
        async with self._condition:
            ticket = object()
            waiters = self._waiters[group]
            waiters.append(ticket)
            queued = (
                len(waiters) > 1 or self._active[group] >= self._max_inflight_per_pa
            )
            try:
                while (
                    waiters[0] is not ticket
                    or self._active[group] >= self._max_inflight_per_pa
                ):
                    await self._condition.wait()
                waiters.popleft()
                self._active[group] += 1
            except BaseException:
                waiters.remove(ticket)
                self._condition.notify_all()
                raise
            wait_ms = (time.perf_counter() - started) * 1000.0
            self._admitted[group] += 1
            if queued:
                self._queued[group] += 1
            self._wait_ms_total[group] += wait_ms
            self._wait_ms_max[group] = max(self._wait_ms_max[group], wait_ms)
            return wait_ms

    async def release(self, group: PAPGroup) -> None:
        """Release one bounded Prefill slot."""
        if self._max_inflight_per_pa == 0:
            return
        async with self._condition:
            if self._active[group] <= 0:
                raise RuntimeError("invalid PAP Prefill admission release")
            self._active[group] -= 1
            self._condition.notify_all()

    async def snapshot(self) -> dict[str, Any]:
        """Return the current Prefill admission state for audits."""
        async with self._condition:
            return {
                "max_inflight_per_pa": self._max_inflight_per_pa,
                "groups": [
                    {
                        "pa_index": self._group_indices[group],
                        "active_requests": self._active[group],
                        "waiting_requests": len(self._waiters[group]),
                        "admitted_requests": self._admitted[group],
                        "queued_requests": self._queued[group],
                        "wait_ms_total": self._wait_ms_total[group],
                        "wait_ms_max": self._wait_ms_max[group],
                    }
                    for group in self._active
                ],
            }


@dataclass
class _PAPProjectionAdmissionState:
    owner: ProjectionInstance | None = None
    active_requests: int = 0
    waiters: list[tuple[object, ProjectionInstance]] = field(default_factory=list)


class PAPProjectionAdmission:
    """Keep each PA on one Projection source for a complete request wave."""

    def __init__(self, groups: list[PAPGroup]) -> None:
        self._condition = asyncio.Condition()
        self._states = {group: _PAPProjectionAdmissionState() for group in groups}
        self._group_indices = {group: index for index, group in enumerate(groups)}

    async def acquire(
        self,
        group: PAPGroup,
        projection: ProjectionInstance,
    ) -> None:
        """Admit a request without changing the PA owner mid-wave."""
        ticket = object()
        async with self._condition:
            state = self._states[group]
            state.waiters.append((ticket, projection))
            try:
                while True:
                    if state.owner is None:
                        state.owner = state.waiters[0][1]
                        self._condition.notify_all()
                    if state.owner == projection and self._is_next_owner_ticket(
                        state,
                        ticket,
                    ):
                        state.waiters = [
                            item for item in state.waiters if item[0] is not ticket
                        ]
                        state.active_requests += 1
                        self._condition.notify_all()
                        return
                    await self._condition.wait()
            except BaseException:
                state.waiters = [
                    item for item in state.waiters if item[0] is not ticket
                ]
                if state.active_requests == 0 and not any(
                    waiting_projection == state.owner
                    for _, waiting_projection in state.waiters
                ):
                    state.owner = None
                self._condition.notify_all()
                raise

    @staticmethod
    def _is_next_owner_ticket(
        state: _PAPProjectionAdmissionState,
        ticket: object,
    ) -> bool:
        for waiting_ticket, waiting_projection in state.waiters:
            if waiting_ticket is ticket:
                return True
            if waiting_projection != state.owner:
                return False
        return False

    async def release(
        self,
        group: PAPGroup,
        projection: ProjectionInstance,
    ) -> None:
        """Release one request and hand the idle PA to the next source."""
        async with self._condition:
            state = self._states[group]
            if state.owner != projection or state.active_requests <= 0:
                raise RuntimeError("invalid PAP Projection admission release")
            state.active_requests -= 1
            if state.active_requests == 0:
                state.owner = None
            self._condition.notify_all()

    async def snapshot(self) -> list[dict[str, int | None]]:
        """Return the current PA admission state for audits."""
        async with self._condition:
            return [
                {
                    "pa_index": self._group_indices[group],
                    "projection_port": (
                        None if state.owner is None else state.owner.port
                    ),
                    "active_requests": state.active_requests,
                    "waiting_requests": len(state.waiters),
                }
                for group, state in self._states.items()
            ]
