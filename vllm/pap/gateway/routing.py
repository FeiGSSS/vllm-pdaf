# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway conversation and static topology routing."""

from __future__ import annotations

from collections import Counter
from itertools import count
from typing import Any

from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance


class PAPConversationRouter:
    """Keep a conversation on one PA and round-robin new conversations."""

    def __init__(self, groups: list[PAPGroup]) -> None:
        if not groups:
            raise ValueError("PAP conversation routing requires a PA group")
        self._groups = groups
        self._group_indices = {group: index for index, group in enumerate(groups)}
        self._next_group = count()
        self._assignments: dict[str, PAPGroup] = {}
        self._request_counts: Counter[PAPGroup] = Counter()

    def select_group(
        self,
        conversation_id: str,
        *,
        request_number: int,
    ) -> PAPGroup:
        """Return the resident PA or round-robin a new conversation."""
        if conversation_id:
            group = self._assignments.get(conversation_id)
            if group is None:
                group = self._groups[next(self._next_group) % len(self._groups)]
                self._assignments[conversation_id] = group
        else:
            group = self._groups[request_number % len(self._groups)]
        self._request_counts[group] += 1
        return group

    def snapshot(self) -> dict[str, Any]:
        """Return token-free assignment and request counts by PA."""
        assignment_counts = Counter(self._assignments.values())
        return {
            "conversations": len(self._assignments),
            "pa_assignments": {
                str(self._group_indices[group]): assignment_counts[group]
                for group in self._groups
            },
            "pa_requests": {
                str(self._group_indices[group]): self._request_counts[group]
                for group in self._groups
            },
        }


def select_instances(
    request_number: int,
    groups: list[PAPGroup],
    projections: list[ProjectionInstance],
    *,
    routing_policy: str = "round_robin",
    conversation_id: str = "",
    conversation_router: PAPConversationRouter | None = None,
) -> tuple[PAPGroup, ProjectionInstance]:
    """Select one fixed PA owner and Projection endpoint."""
    group_index = request_number % len(groups)
    group = groups[group_index]
    if routing_policy == "round_robin":
        projection_index = request_number % len(projections)
    elif routing_policy == "crossbar_round_robin":
        projection_index = (request_number // len(groups) + group_index) % len(
            projections
        )
    elif routing_policy == "projection_affinity":
        groups_per_projection = (len(groups) + len(projections) - 1) // len(projections)
        projection_index = min(
            group_index // groups_per_projection,
            len(projections) - 1,
        )
    elif routing_policy == "projection_sticky":
        projection_index = request_number % len(projections)
        group = groups[projection_index % len(groups)]
    elif routing_policy == "conversation_affinity":
        if conversation_router is None:
            raise ValueError("conversation_affinity requires a PAPConversationRouter")
        group = conversation_router.select_group(
            conversation_id,
            request_number=request_number,
        )
        projection_index = request_number % len(projections)
    else:
        raise ValueError(f"unsupported PAP routing policy: {routing_policy}")
    return group, projections[projection_index]
