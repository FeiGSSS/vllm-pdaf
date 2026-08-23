# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Projection admission control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance


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
