# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-to-Attention peer membership tracking."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def pap_attention_endpoint_for_rank(value: Any, *, tp_rank: int) -> str:
    """Select one Attention control endpoint for the local TP rank."""

    rank = int(tp_rank)
    if rank < 0:
        raise ValueError("PAP TP rank must be non-negative")
    if isinstance(value, str):
        endpoints = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence):
        endpoints = tuple(str(part).strip() for part in value if str(part).strip())
    else:
        endpoints = (str(value).strip(),) if str(value).strip() else ()
    if rank >= len(endpoints):
        raise ValueError(
            f"PAP endpoint list has {len(endpoints)} endpoint(s), but TP rank is {rank}"
        )
    return endpoints[rank]


def active_pap_attention_endpoints(
    *,
    request_ids: Iterable[str],
    endpoint_by_request: Mapping[str, Any],
    tp_rank: int,
) -> tuple[str, ...]:
    """Return sorted unique Attention endpoints used by the current cohort."""

    endpoints = {
        pap_attention_endpoint_for_rank(
            endpoint_by_request[str(request_id)],
            tp_rank=tp_rank,
        )
        for request_id in request_ids
        if str(request_id) in endpoint_by_request
    }
    return tuple(sorted(endpoints))


class PAPProjectionPeerActivity:
    """Publish source membership changes without entering the layer hot path."""

    def __init__(
        self,
        *,
        source_id: str,
        notify: Callable[..., Any],
    ) -> None:
        if not str(source_id):
            raise ValueError("PAP Projection activity source id must be non-empty")
        self._source_id = str(source_id)
        self._notify = notify
        self._known_endpoints: set[str] = set()
        self._active_endpoints: set[str] = set()
        self._membership_generation = 0

    @property
    def active_endpoints(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_endpoints))

    @property
    def known_endpoints(self) -> tuple[str, ...]:
        return tuple(sorted(self._known_endpoints))

    @property
    def membership_generation(self) -> int:
        return self._membership_generation

    def update(self, active_endpoints: Iterable[str]) -> bool:
        """Notify endpoints whose active membership changed."""

        target_active = {
            str(endpoint).strip()
            for endpoint in active_endpoints
            if str(endpoint).strip()
        }
        if target_active == self._active_endpoints:
            return False

        self._membership_generation += 1
        generation = self._membership_generation
        known_endpoints = self._known_endpoints | target_active
        changed_endpoints = sorted(self._active_endpoints ^ target_active)
        for endpoint in changed_endpoints:
            self._notify(
                attention_endpoint=endpoint,
                source_id=self._source_id,
                active=endpoint in target_active,
                membership_generation=generation,
            )

        self._known_endpoints = known_endpoints
        self._active_endpoints = target_active
        return True


def sync_pap_projection_peer_activity(
    *,
    tracker: PAPProjectionPeerActivity | None,
    request_ids: Iterable[str],
    endpoint_by_request: Mapping[str, Any],
    notify: Callable[..., Any] | None = None,
) -> PAPProjectionPeerActivity | None:
    """Synchronize Projection membership after one scheduler state update."""

    projection_role = os.environ.get("PAP_PROJECTION_KV_UNAWARE", "0").lower()
    try:
        projection_count = int(os.environ.get("PAP_PROJECTION_COUNT", "1"))
    except ValueError:
        projection_count = 1
    if (
        projection_role not in {"1", "true", "yes", "on"}
        or projection_count <= 1
    ):
        return tracker

    raw_rank = os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK")
    if raw_rank is None:
        from vllm.distributed.parallel_state import get_tp_group

        tp_rank = int(get_tp_group().rank_in_group)
    else:
        tp_rank = int(raw_rank)
    active_endpoints = active_pap_attention_endpoints(
        request_ids=request_ids,
        endpoint_by_request=endpoint_by_request,
        tp_rank=tp_rank,
    )
    if tracker is None:
        if not active_endpoints:
            return None
        if notify is None:
            from vllm.pap.attention.client import (
                update_offload_exec_mailbox_activity,
            )

            notify = update_offload_exec_mailbox_activity
        tracker = PAPProjectionPeerActivity(
            source_id=(
                f"{os.environ.get('PAP_NIXL_MAILBOX_ACTOR_ID', 'projection')}"
                f"-r{tp_rank}"
            ),
            notify=notify,
        )
    tracker.update(active_endpoints)
    return tracker
