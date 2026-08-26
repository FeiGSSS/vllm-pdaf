# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway conversation and static topology routing."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance

_CHARACTERS_PER_TOKEN = 4
_KV_HEADROOM_TOKENS = 4096
_ROUTING_RESERVATION_TTL_S = 2.0


@dataclass(frozen=True, slots=True)
class _RoutingReservation:
    group: PAPGroup
    prefill_tokens: int
    kv_tokens: int
    expires_at: float


class PAPConversationRouter:
    """Keep a conversation on one initially load-balanced PA."""

    def __init__(self, groups: list[PAPGroup]) -> None:
        if not groups:
            raise ValueError("PAP conversation routing requires a PA group")
        self._groups = groups
        self._group_indices = {group: index for index, group in enumerate(groups)}
        self._assignments: dict[str, PAPGroup] = {}
        self._initial_loads: Counter[PAPGroup] = Counter()
        self._conversation_counts: Counter[PAPGroup] = Counter()
        self._request_counts: Counter[PAPGroup] = Counter()
        self._reservations: dict[str, _RoutingReservation] = {}
        self._lock = Lock()

    def select_group(
        self,
        conversation_id: str,
        *,
        request_number: int,
        initial_context_load: int = 1,
        initial_context_tokens: int = 1,
        decode_capacity_tokens: int = 0,
        request_id: str = "",
        current_pa_loads: Mapping[PAPGroup, Mapping[str, int]] | None = None,
    ) -> PAPGroup:
        """Return the resident PA or balance a new conversation once."""
        with self._lock:
            if conversation_id:
                group = self._assignments.get(conversation_id)
                if group is None:
                    load = max(1, int(initial_context_load))
                    if current_pa_loads is None:
                        group = min(
                            self._groups,
                            key=lambda candidate: (
                                self._initial_loads[candidate],
                                self._conversation_counts[candidate],
                                self._group_indices[candidate],
                            ),
                        )
                    else:
                        reserved_prefill, reserved_kv = (
                            self._active_reservations_locked()
                        )
                        incoming_prefill = max(1, int(initial_context_tokens))
                        candidates: list[tuple[PAPGroup, bool, int, int, int]] = []
                        for candidate in self._groups:
                            snapshot = current_pa_loads.get(candidate)
                            if snapshot is None:
                                continue
                            block_size = max(1, int(snapshot["kv_block_size"]))
                            incoming_kv = (
                                (
                                    incoming_prefill
                                    + max(0, int(decode_capacity_tokens))
                                    + block_size
                                    - 1
                                )
                                // block_size
                                * block_size
                            )
                            projected_kv = (
                                int(snapshot["projected_kv_tokens"])
                                + reserved_kv[candidate]
                                + incoming_kv
                            )
                            capacity_limit = max(
                                0,
                                int(snapshot["total_kv_tokens"]) - _KV_HEADROOM_TOKENS,
                            )
                            compute_score = (
                                int(snapshot["outstanding_prefill_tokens"])
                                + reserved_prefill[candidate]
                                + incoming_prefill
                            )
                            candidates.append(
                                (
                                    candidate,
                                    projected_kv <= capacity_limit,
                                    compute_score,
                                    projected_kv,
                                    incoming_kv,
                                )
                            )
                        group, _fits, _compute, _projected, incoming_kv = min(
                            candidates,
                            key=lambda item: (
                                not item[1],
                                item[2],
                                item[3],
                                self._initial_loads[item[0]],
                                self._conversation_counts[item[0]],
                                self._group_indices[item[0]],
                            ),
                        )
                        if request_id:
                            self._reservations[request_id] = _RoutingReservation(
                                group=group,
                                prefill_tokens=incoming_prefill,
                                kv_tokens=incoming_kv,
                                expires_at=(
                                    time.monotonic() + _ROUTING_RESERVATION_TTL_S
                                ),
                            )
                    self._assignments[conversation_id] = group
                    self._initial_loads[group] += load
                    self._conversation_counts[group] += 1
            else:
                group = self._groups[request_number % len(self._groups)]
            self._request_counts[group] += 1
            return group

    def _active_reservations_locked(
        self,
    ) -> tuple[Counter[PAPGroup], Counter[PAPGroup]]:
        now = time.monotonic()
        prefill: Counter[PAPGroup] = Counter()
        kv: Counter[PAPGroup] = Counter()
        for request_id, reservation in list(self._reservations.items()):
            if reservation.expires_at <= now:
                self._reservations.pop(request_id, None)
                continue
            prefill[reservation.group] += reservation.prefill_tokens
            kv[reservation.group] += reservation.kv_tokens
        return prefill, kv

    def release_reservation(self, request_id: str) -> None:
        """Release a first-turn reservation after Prefill owns the request."""
        if not request_id:
            return
        with self._lock:
            self._reservations.pop(request_id, None)

    def has_assignment(self, conversation_id: str) -> bool:
        """Return whether a conversation already has a sticky PA owner."""
        if not conversation_id:
            return False
        with self._lock:
            return conversation_id in self._assignments

    def snapshot(self) -> dict[str, Any]:
        """Return token-free assignment and request counts by PA."""
        with self._lock:
            reserved_prefill, reserved_kv = self._active_reservations_locked()
            return {
                "conversations": len(self._assignments),
                "pa_assignments": {
                    str(self._group_indices[group]): self._conversation_counts[group]
                    for group in self._groups
                },
                "pa_requests": {
                    str(self._group_indices[group]): self._request_counts[group]
                    for group in self._groups
                },
                "pa_initial_context_characters": {
                    str(self._group_indices[group]): self._initial_loads[group]
                    for group in self._groups
                },
                "pa_reserved_prefill_tokens": {
                    str(self._group_indices[group]): reserved_prefill[group]
                    for group in self._groups
                },
                "pa_reserved_kv_tokens": {
                    str(self._group_indices[group]): reserved_kv[group]
                    for group in self._groups
                },
            }


def estimate_initial_context_load(req_data: dict[str, Any]) -> int:
    """Estimate initial context cheaply from request text characters."""

    def text_characters(value: Any) -> int:
        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            if value and all(isinstance(item, int) for item in value):
                return len(value) * 4
            return sum(text_characters(item) for item in value)
        if isinstance(value, dict):
            text = value.get("text")
            return text_characters(text) if text is not None else 0
        return 0

    messages = req_data.get("messages")
    if isinstance(messages, list):
        load = sum(
            text_characters(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
    else:
        load = text_characters(req_data.get("prompt"))
    return max(1, load)


def estimate_initial_context_tokens(req_data: dict[str, Any]) -> int:
    """Estimate first-turn tokens without loading a Gateway tokenizer."""
    prompt = req_data.get("prompt")
    if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
        return max(1, len(prompt))
    characters = estimate_initial_context_load(req_data)
    return max(1, (characters + _CHARACTERS_PER_TOKEN - 1) // _CHARACTERS_PER_TOKEN)


def select_instances(
    request_number: int,
    groups: list[PAPGroup],
    projections: list[ProjectionInstance],
    *,
    routing_policy: str = "round_robin",
    conversation_id: str = "",
    conversation_router: PAPConversationRouter | None = None,
    initial_context_load: int = 1,
    initial_context_tokens: int = 1,
    decode_capacity_tokens: int = 0,
    request_id: str = "",
    current_pa_loads: Mapping[PAPGroup, Mapping[str, int]] | None = None,
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
            initial_context_load=initial_context_load,
            initial_context_tokens=initial_context_tokens,
            decode_capacity_tokens=decode_capacity_tokens,
            request_id=request_id,
            current_pa_loads=current_pa_loads,
        )
        projection_index = request_number % len(projections)
    else:
        raise ValueError(f"unsupported PAP routing policy: {routing_policy}")
    return group, projections[projection_index]
