# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible gateway for arbitrary PAP topologies."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.pap.config import reject_removed_pap_flags
from vllm.pap.gateway.clients import (
    PAPServiceClient,
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    register_attention_handle,
    wait_attention_prefill_ready,
)
from vllm.pap.gateway.clients import (
    request_headers as _headers,
)
from vllm.pap.gateway.payloads import (
    attach_pap_prefill_attention_params,
    build_prefill_payload,
    build_projection_kv_unaware_payload,
    enrich_prefill_kv_params,
    requested_decode_capacity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_gateway")


def _pap_prefill_ipc_profile_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_IPC_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prefill_usage_headers(prefill_response: dict[str, Any]) -> dict[str, str]:
    usage = prefill_response.get("usage")
    if not isinstance(usage, dict):
        return {}

    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        return {}

    headers = {
        "X-PAP-Prefill-Prompt-Tokens": str(prompt_tokens),
    }
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return headers

    cached_tokens = details.get("cached_tokens")
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        return headers

    headers["X-PAP-Prefill-Cached-Tokens"] = str(cached_tokens)
    if cached_tokens <= prompt_tokens:
        headers["X-PAP-Prefill-Computed-Tokens"] = str(prompt_tokens - cached_tokens)
    return headers


PortSpec = int | tuple[int, ...]


def _parse_port_spec(value: str) -> PortSpec:
    if "|" not in value:
        return int(value)
    ports = tuple(int(part) for part in value.split("|") if part)
    if not ports:
        raise argparse.ArgumentTypeError(f"invalid empty ranked port spec {value!r}")
    return ports


def _format_ranked_endpoints(
    host: str,
    ports: PortSpec,
    *,
    scheme: str,
) -> str:
    if isinstance(ports, int):
        return f"{scheme}{host}:{ports}"
    return ",".join(f"{scheme}{host}:{port}" for port in ports)


@dataclass(frozen=True)
class PAPGroup:
    prefill_host: str
    prefill_port: int
    prefill_nixl_port: int
    attention_host: str
    attention_port: PortSpec
    attention_tcp_port: PortSpec | None = None
    attention_zmq_port: PortSpec | None = None

    @property
    def prefill_base_url(self) -> str:
        return f"http://{self.prefill_host}:{self.prefill_port}"

    @property
    def attention_base_url(self) -> str:
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_port,
            scheme="http://",
        )

    @property
    def attention_tcp_endpoint(self) -> str | None:
        if self.attention_tcp_port is None:
            return None
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_tcp_port,
            scheme="tcp://",
        )

    @property
    def attention_zmq_endpoint(self) -> str | None:
        if self.attention_zmq_port is None:
            return None
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_zmq_port,
            scheme="",
        )


@dataclass(frozen=True)
class ProjectionInstance:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class PAPConversationRouter:
    """Keep a conversation on one PA while balancing new conversations."""

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


@dataclass
class _PAPAttentionLoadAdmission:
    conversation_id: str
    prefill_group: PAPGroup
    group: PAPGroup
    history_group: PAPGroup | None
    context_tokens: int
    reserved_history_tokens: int = 0
    decode_placed: bool = False
    migration_pending: bool = False
    migration_succeeded: bool = False


class PAPAttentionLoadRouter:
    """Place Decode on the PA that minimizes peak active Attention load."""

    def __init__(
        self,
        groups: list[PAPGroup],
        *,
        migration_min_peak_gain_ratio: float = 0.3,
        migration_max_inflight: int = 1,
    ) -> None:
        if not groups:
            raise ValueError("PAP attention-load routing requires a PA group")
        if (
            not math.isfinite(migration_min_peak_gain_ratio)
            or migration_min_peak_gain_ratio < 0
        ):
            raise ValueError("migration_min_peak_gain_ratio must be finite and >= 0")
        if migration_max_inflight < 0:
            raise ValueError("migration_max_inflight must be >= 0")
        self._groups = groups
        self._group_indices = {group: index for index, group in enumerate(groups)}
        self._migration_min_peak_gain_ratio = migration_min_peak_gain_ratio
        self._migration_max_inflight = migration_max_inflight
        self._next_group = count()
        self._prefill_loads = {group: 0 for group in groups}
        self._decode_loads = {group: 0 for group in groups}
        self._migration_reserved_loads = {group: 0 for group in groups}
        self._history_loads = {group: 0 for group in groups}
        self._history_owners: dict[str, PAPGroup] = {}
        self._history_context_tokens: dict[str, int] = {}
        self._history_request_ids: dict[str, str] = {}
        self._admissions: dict[str, _PAPAttentionLoadAdmission] = {}
        self._prefill_request_counts: Counter[PAPGroup] = Counter()
        self._decode_request_counts: Counter[PAPGroup] = Counter()
        self._migration_fallback_count = 0
        self._migration_count = 0
        self._migration_miss_count = 0
        self._history_admission_count = 0
        self._migration_inflight_count = 0
        self._migration_candidate_count = 0
        self._migration_selected_count = 0
        self._migration_suppressed_count = 0
        self._migration_suppressed_by_peak_gain_count = 0
        self._migration_suppressed_by_inflight_count = 0
        self._migration_suppressed_by_prefill_busy_count = 0

    def history_context_tokens(self, conversation_id: str) -> int:
        """Return the last completed turn's context length."""
        return self._history_context_tokens.get(conversation_id, 0)

    @property
    def migration_enabled(self) -> bool:
        """Return whether this router may select cross-PA migration."""
        return self._migration_max_inflight > 0

    def history(
        self,
        conversation_id: str,
    ) -> tuple[PAPGroup, str, int] | None:
        """Return the retained owner, request ID, and context length."""
        group = self._history_owners.get(conversation_id)
        request_id = self._history_request_ids.get(conversation_id)
        if group is None or request_id is None:
            return None
        return (
            group,
            request_id,
            self._history_context_tokens.get(conversation_id, 0),
        )

    def admit(
        self,
        *,
        request_id: str,
        conversation_id: str,
        estimated_context_tokens: int,
    ) -> PAPGroup:
        """Choose the Prefill PA and reserve its pending Decode load."""
        if request_id in self._admissions:
            raise ValueError(f"duplicate PAP request admission: {request_id}")
        context_tokens = max(1, int(estimated_context_tokens))
        history_group = self._history_owners.get(conversation_id)
        reserved_history_tokens = 0
        if history_group is not None:
            self._history_admission_count += 1
            reserved_history_tokens = self._history_context_tokens.get(
                conversation_id,
                0,
            )
            self._history_loads[history_group] -= reserved_history_tokens
            if self._history_loads[history_group] < 0:
                raise RuntimeError("negative PAP reserved history load")
        committed_loads = self._committed_loads()
        if history_group is not None:
            group = history_group
        else:
            minimum_load = min(committed_loads.values())
            candidates = [
                group
                for group in self._groups
                if committed_loads[group] == minimum_load
            ]
            group = candidates[next(self._next_group) % len(candidates)]
        self._prefill_loads[group] += context_tokens
        self._prefill_request_counts[group] += 1
        self._admissions[request_id] = _PAPAttentionLoadAdmission(
            conversation_id=conversation_id,
            prefill_group=group,
            group=group,
            history_group=history_group,
            context_tokens=context_tokens,
            reserved_history_tokens=reserved_history_tokens,
        )
        return group

    def _committed_loads(self) -> dict[PAPGroup, int]:
        return {
            group: (
                self._prefill_loads[group]
                + self._decode_loads[group]
                + self._migration_reserved_loads[group]
                + self._history_loads[group]
            )
            for group in self._groups
        }

    def _decode_base_loads(self) -> dict[PAPGroup, int]:
        """Return active Decode KV plus incoming migration reservations."""
        return {
            group: (self._decode_loads[group] + self._migration_reserved_loads[group])
            for group in self._groups
        }

    def history_group(self, request_id: str) -> PAPGroup | None:
        """Return the previous turn's PA for an admitted request."""
        return self._admissions[request_id].history_group

    def observe_prefill(self, request_id: str, prompt_tokens: int) -> PAPGroup:
        """Commit exact Prefill load and choose the request's Decode PA."""
        admission = self._admissions[request_id]
        if admission.decode_placed:
            raise RuntimeError(f"PAP request already placed for Decode: {request_id}")
        exact_tokens = max(1, int(prompt_tokens))
        prefill_group = admission.prefill_group
        self._prefill_loads[prefill_group] -= admission.context_tokens
        if self._prefill_loads[prefill_group] < 0:
            raise RuntimeError("negative PAP Prefill load")
        admission.context_tokens = exact_tokens
        group = prefill_group
        migration_pending = False

        if admission.history_group is not None:
            base_loads = self._decode_base_loads()
            placement_peaks = {
                target: max(
                    load + exact_tokens if group == target else load
                    for group, load in base_loads.items()
                )
                for target in self._groups
            }
            minimum_peak = min(placement_peaks.values())
            peak_candidates = [
                target
                for target in self._groups
                if placement_peaks[target] == minimum_peak
            ]
            if prefill_group in peak_candidates:
                candidate = prefill_group
            else:
                minimum_base_load = min(
                    base_loads[target] for target in peak_candidates
                )
                lightest_candidates = [
                    target
                    for target in peak_candidates
                    if base_loads[target] == minimum_base_load
                ]
                candidate = lightest_candidates[
                    next(self._next_group) % len(lightest_candidates)
                ]
            stay_peak = placement_peaks[prefill_group]
            move_peak = placement_peaks[candidate]
            peak_gain = max(0, stay_peak - move_peak)
            peak_gain_ratio = peak_gain / stay_peak if stay_peak else 0.0
            if candidate != prefill_group and peak_gain > 0:
                self._migration_candidate_count += 1
            peak_gain_ready = (
                candidate != prefill_group
                and peak_gain > 0
                and peak_gain_ratio >= self._migration_min_peak_gain_ratio
            )
            inflight_ready = (
                self._migration_inflight_count < self._migration_max_inflight
            )
            target_prefill_ready = self._prefill_loads[candidate] == 0
            should_migrate = peak_gain_ready and inflight_ready and target_prefill_ready
            if should_migrate:
                group = candidate
                migration_pending = True
                self._migration_inflight_count += 1
                self._migration_selected_count += 1
            elif candidate != prefill_group and peak_gain > 0:
                self._migration_suppressed_count += 1
                if not peak_gain_ready:
                    self._migration_suppressed_by_peak_gain_count += 1
                elif not inflight_ready:
                    self._migration_suppressed_by_inflight_count += 1
                elif not target_prefill_ready:
                    self._migration_suppressed_by_prefill_busy_count += 1
            logger.info(
                "PAP attention-load Decode placement request_id=%s "
                "prefill_pa=%d candidate_pa=%d action=%s "
                "stay_peak_tokens=%d move_peak_tokens=%d "
                "peak_gain_ratio=%.6f threshold=%.6f inflight_ready=%d "
                "target_prefill_ready=%d target_prefill_tokens=%d "
                "base_loads=%s",
                request_id,
                self._group_indices[prefill_group],
                self._group_indices[candidate],
                "migrate" if should_migrate else "stay",
                stay_peak,
                move_peak,
                peak_gain_ratio,
                self._migration_min_peak_gain_ratio,
                int(inflight_ready),
                int(target_prefill_ready),
                self._prefill_loads[candidate],
                ",".join(str(load) for load in base_loads.values()),
            )

        admission.group = group
        admission.decode_placed = True
        admission.migration_pending = migration_pending
        if migration_pending:
            self._migration_reserved_loads[group] += exact_tokens
        else:
            self._decode_loads[group] += exact_tokens
            self._decode_request_counts[group] += 1
        return group

    def decode_group(self, request_id: str) -> PAPGroup:
        """Return the current Decode placement for an admitted request."""
        admission = self._admissions[request_id]
        if not admission.decode_placed:
            raise RuntimeError(f"PAP request is not placed for Decode: {request_id}")
        return admission.group

    def mark_migration_succeeded(self, request_id: str) -> None:
        """Record a completed cross-PA KV migration."""
        admission = self._admissions[request_id]
        if (
            admission.history_group is not None
            and admission.history_group != admission.group
            and not admission.migration_succeeded
        ):
            self._migration_reserved_loads[admission.group] -= admission.context_tokens
            if self._migration_reserved_loads[admission.group] < 0:
                raise RuntimeError("negative PAP migration reserved load")
            self._decode_loads[admission.group] += admission.context_tokens
            self._decode_request_counts[admission.group] += 1
            admission.migration_succeeded = True
            self._migration_count += 1
        self._resolve_migration(admission)

    def mark_migration_missed(self, request_id: str) -> PAPGroup:
        """Atomically fall a failed migration back to its Prefill PA."""
        admission = self._admissions[request_id]
        if admission.prefill_group != admission.group:
            self._migration_miss_count += 1
            self._migration_fallback_count += 1
            self._migration_reserved_loads[admission.group] -= admission.context_tokens
            if self._migration_reserved_loads[admission.group] < 0:
                raise RuntimeError("negative PAP migration load after fallback")
            admission.group = admission.prefill_group
            self._decode_loads[admission.group] += admission.context_tokens
            self._decode_request_counts[admission.group] += 1
        self._resolve_migration(admission)
        return admission.group

    def _resolve_migration(
        self,
        admission: _PAPAttentionLoadAdmission,
    ) -> None:
        if not admission.migration_pending:
            return
        admission.migration_pending = False
        self._migration_inflight_count -= 1
        if self._migration_inflight_count < 0:
            raise RuntimeError("negative PAP migration in-flight count")

    def finish(
        self,
        request_id: str,
        *,
        completion_tokens: int = 0,
        prefill_kv_handle: str | None = None,
    ) -> None:
        """Release active load and retain the new history owner."""
        admission = self._admissions.pop(request_id)
        if not admission.decode_placed:
            raise RuntimeError(f"PAP request finished before Decode: {request_id}")
        if admission.migration_pending:
            raise RuntimeError(f"PAP request finished during migration: {request_id}")
        self._decode_loads[admission.group] -= admission.context_tokens
        if self._decode_loads[admission.group] < 0:
            raise RuntimeError("negative PAP Decode load")
        if admission.conversation_id:
            retained_context_tokens = admission.context_tokens + max(
                0,
                int(completion_tokens),
            )
            self._history_owners[admission.conversation_id] = admission.group
            self._history_request_ids[admission.conversation_id] = str(
                prefill_kv_handle or request_id
            )
            self._history_context_tokens[admission.conversation_id] = (
                retained_context_tokens
            )
            self._history_loads[admission.group] += retained_context_tokens

    def abort(self, request_id: str) -> None:
        """Roll back a request that did not complete Decode."""
        admission = self._admissions.pop(request_id, None)
        if admission is None:
            return
        if admission.decode_placed:
            if admission.migration_pending:
                self._migration_reserved_loads[admission.group] -= (
                    admission.context_tokens
                )
                if self._migration_reserved_loads[admission.group] < 0:
                    raise RuntimeError("negative PAP migration reserved load")
            else:
                self._decode_loads[admission.group] -= admission.context_tokens
                if self._decode_loads[admission.group] < 0:
                    raise RuntimeError("negative PAP Decode load")
        else:
            self._prefill_loads[admission.prefill_group] -= admission.context_tokens
            if self._prefill_loads[admission.prefill_group] < 0:
                raise RuntimeError("negative PAP Prefill load")
        if admission.history_group is not None:
            self._history_loads[admission.history_group] += (
                admission.reserved_history_tokens
            )
        self._resolve_migration(admission)

    def snapshot(self) -> dict[str, Any]:
        """Return current token load and migration counters by PA."""
        history_counts = Counter(self._history_owners.values())
        return {
            "active_requests": len(self._admissions),
            "conversations": len(self._history_owners),
            "migration_min_peak_gain_ratio": self._migration_min_peak_gain_ratio,
            "migration_max_inflight": self._migration_max_inflight,
            "migration_inflight_count": self._migration_inflight_count,
            "history_admission_count": self._history_admission_count,
            "migration_candidate_count": self._migration_candidate_count,
            "migration_selected_count": self._migration_selected_count,
            "migration_suppressed_count": self._migration_suppressed_count,
            "migration_suppressed_by_peak_gain_count": (
                self._migration_suppressed_by_peak_gain_count
            ),
            "migration_suppressed_by_inflight_count": (
                self._migration_suppressed_by_inflight_count
            ),
            "migration_suppressed_by_prefill_busy_count": (
                self._migration_suppressed_by_prefill_busy_count
            ),
            "migration_count": self._migration_count,
            "migration_miss_count": self._migration_miss_count,
            "migration_fallback_count": self._migration_fallback_count,
            "pa_prefill_tokens": {
                str(self._group_indices[group]): self._prefill_loads[group]
                for group in self._groups
            },
            "pa_attention_tokens": {
                str(self._group_indices[group]): self._decode_loads[group]
                for group in self._groups
            },
            "pa_migration_reserved_tokens": {
                str(self._group_indices[group]): (self._migration_reserved_loads[group])
                for group in self._groups
            },
            "pa_decode_base_tokens": {
                str(self._group_indices[group]): (
                    self._decode_loads[group] + self._migration_reserved_loads[group]
                )
                for group in self._groups
            },
            "pa_history_tokens": {
                str(self._group_indices[group]): self._history_loads[group]
                for group in self._groups
            },
            "pa_committed_tokens": {
                str(self._group_indices[group]): (
                    self._prefill_loads[group]
                    + self._decode_loads[group]
                    + self._migration_reserved_loads[group]
                    + self._history_loads[group]
                )
                for group in self._groups
            },
            "pa_history_owners": {
                str(self._group_indices[group]): history_counts[group]
                for group in self._groups
            },
            "pa_prefill_requests": {
                str(self._group_indices[group]): self._prefill_request_counts[group]
                for group in self._groups
            },
            "pa_decode_requests": {
                str(self._group_indices[group]): self._decode_request_counts[group]
                for group in self._groups
            },
        }


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


def _request_text_bytes(value: Any) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return sum(_request_text_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _request_text_bytes(item)
            for key, item in value.items()
            if key in {"content", "text"}
        )
    return 0


def _estimate_context_tokens(
    req_data: dict[str, Any],
    *,
    history_context_tokens: int = 0,
    explicit_context_tokens: str | int | None = None,
) -> int:
    """Estimate a prompt only until Prefill reports its exact token count."""
    if explicit_context_tokens is not None:
        try:
            value = int(explicit_context_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("PAP context token hint must be an integer") from exc
        if value <= 0:
            raise ValueError("PAP context token hint must be positive")
        return value
    prompt = req_data.get("prompt")
    if (
        isinstance(prompt, list)
        and prompt
        and all(isinstance(token_id, int) for token_id in prompt)
    ):
        return len(prompt)
    text_bytes = _request_text_bytes(
        req_data.get("messages", prompt if prompt is not None else "")
    )
    estimated = max(1, math.ceil(text_bytes / 4))
    return max(estimated, int(history_context_tokens))


def _migration_prefix_kv_params(
    export_response: dict[str, Any],
    *,
    source_group: PAPGroup,
    block_size: int,
) -> tuple[dict[str, Any], int]:
    """Limit a retained source manifest to its exact historical prefix."""
    cached_tokens = export_response.get("seq_len")
    if not isinstance(cached_tokens, int) or cached_tokens <= 0:
        raise RuntimeError("PAP migration source has no retained historical prefix")
    if block_size <= 0:
        raise ValueError("PAP migration block size must be positive")

    kv_params = enrich_prefill_kv_params(
        export_response.get("kv_transfer_params") or {},
        prefill_host=source_group.prefill_host,
        prefill_nixl_port=source_group.prefill_nixl_port,
    )
    remote_block_ids = kv_params.get("remote_block_ids")
    if not isinstance(remote_block_ids, list) or not remote_block_ids:
        raise RuntimeError("PAP migration source returned no remote KV blocks")
    prefix_blocks = math.ceil(cached_tokens / block_size)
    trimmed_block_ids: list[list[int]] = []
    for group_blocks in remote_block_ids:
        if not isinstance(group_blocks, list) or len(group_blocks) < prefix_blocks:
            raise RuntimeError("PAP migration source returned incomplete KV blocks")
        trimmed_block_ids.append(group_blocks[:prefix_blocks])
    kv_params["remote_block_ids"] = trimmed_block_ids
    kv_params["remote_num_tokens"] = cached_tokens
    return kv_params, cached_tokens


def _migration_prefix_identity(
    export_response: dict[str, Any],
    *,
    migrated_tokens: int,
) -> tuple[list[int], list[str]]:
    """Validate the source identity used to index migrated KV blocks."""
    raw_token_ids = export_response.get("prefix_token_ids")
    raw_block_hashes = export_response.get("prefix_block_hashes")
    if not isinstance(raw_token_ids, list) or not raw_token_ids:
        raise RuntimeError("PAP migration source returned no prefix token identity")
    if not isinstance(raw_block_hashes, list) or not raw_block_hashes:
        raise RuntimeError("PAP migration source returned no prefix block hashes")

    token_ids = [int(token) for token in raw_token_ids[:migrated_tokens]]
    block_hashes: list[str] = []
    for raw_hash in raw_block_hashes:
        if not isinstance(raw_hash, str):
            raise RuntimeError("PAP migration source returned an invalid block hash")
        try:
            bytes.fromhex(raw_hash)
        except ValueError as exc:
            raise RuntimeError(
                "PAP migration source returned an invalid block hash"
            ) from exc
        block_hashes.append(raw_hash)
    return token_ids, block_hashes


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


def _parse_host_port(value: str, *, expected_parts: int, kind: str) -> list[str]:
    parts = value.split(":")
    if len(parts) != expected_parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            f"invalid {kind} spec {value!r}; expected {expected_parts} "
            "colon-separated fields"
        )
    return parts


def parse_pap_groups(spec: str) -> list[PAPGroup]:
    groups: list[PAPGroup] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in {5, 6, 7} or any(part == "" for part in parts):
            raise argparse.ArgumentTypeError(
                f"invalid PAP group spec {item!r}; expected 5, 6, or 7 "
                "colon-separated fields"
            )
        groups.append(
            PAPGroup(
                prefill_host=parts[0],
                prefill_port=int(parts[1]),
                prefill_nixl_port=int(parts[2]),
                attention_host=parts[3],
                attention_port=_parse_port_spec(parts[4]),
                attention_tcp_port=None
                if len(parts) == 5
                else _parse_port_spec(parts[5]),
                attention_zmq_port=None
                if len(parts) < 7
                else _parse_port_spec(parts[6]),
            )
        )
    if not groups:
        raise argparse.ArgumentTypeError("at least one PAP group is required")
    return groups


def parse_projection_instances(spec: str) -> list[ProjectionInstance]:
    projections: list[ProjectionInstance] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = _parse_host_port(item, expected_parts=2, kind="Projection")
        projections.append(ProjectionInstance(host=parts[0], port=int(parts[1])))
    if not projections:
        raise argparse.ArgumentTypeError("at least one Projection instance is required")
    return projections


def select_instances(
    request_number: int,
    groups: list[PAPGroup],
    projections: list[ProjectionInstance],
    *,
    routing_policy: str = "round_robin",
    conversation_id: str = "",
    conversation_router: PAPConversationRouter | None = None,
    attention_load_router: PAPAttentionLoadRouter | None = None,
    request_id: str = "",
    estimated_context_tokens: int = 1,
) -> tuple[PAPGroup, ProjectionInstance]:
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
        group_index = projection_index % len(groups)
        group = groups[group_index]
    elif routing_policy == "conversation_affinity":
        if conversation_router is None:
            raise ValueError("conversation_affinity requires a PAPConversationRouter")
        group = conversation_router.select_group(
            conversation_id,
            request_number=request_number,
        )
        projection_index = request_number % len(projections)
    elif routing_policy == "attention_load":
        if attention_load_router is None:
            raise ValueError("attention_load requires a PAPAttentionLoadRouter")
        group = attention_load_router.admit(
            request_id=request_id,
            conversation_id=conversation_id,
            estimated_context_tokens=estimated_context_tokens,
        )
        projection_index = request_number % len(projections)
    else:
        raise ValueError(f"unsupported PAP routing policy: {routing_policy}")
    return group, projections[projection_index]


def build_projection_payload_for_group(
    req_data: dict[str, Any],
    kv_transfer_params: dict[str, Any],
    group: PAPGroup,
    *,
    pap_prefill_kv_handle: str | None = None,
    pap_attention_kv_installed: bool = False,
) -> dict[str, Any]:
    return build_projection_kv_unaware_payload(
        req_data,
        kv_transfer_params,
        pap_attention_endpoint=group.attention_base_url,
        pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
        pap_offload_exec_zmq_endpoint=group.attention_zmq_endpoint,
        pap_prefill_kv_handle=pap_prefill_kv_handle,
        pap_attention_kv_installed=pap_attention_kv_installed,
    )


def _make_client(host: str, port: int, role: str) -> PAPServiceClient:
    base_url = f"http://{host}:{port}"
    return PAPServiceClient(
        client=httpx.AsyncClient(
            timeout=None,
            base_url=base_url,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        ),
        host=host,
        port=port,
        base_url=base_url,
        role=role,
    )


async def _post_json(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    resp = await client.client.post(
        endpoint,
        json=payload,
        headers=_headers(request_id),
    )
    resp.raise_for_status()
    return resp.json()


async def _export_prefill_kv(
    prefill: PAPServiceClient,
    request_id: str,
) -> dict[str, Any] | None:
    """Fetch retained KV metadata without entering the model scheduler."""
    response = await prefill.client.post(
        "/v1/pap/prefill/kv-export",
        json={"request_id": request_id},
        headers=_headers(request_id),
    )
    response.raise_for_status()
    body = response.json()
    return body if body.get("exported", False) else None


async def _wait_prefill_kv_export(
    prefill: PAPServiceClient,
    request_id: str,
) -> dict[str, Any] | None:
    """Wait for scheduler finalization to publish a completed Prefill lease."""
    timeout = float(os.environ.get("PAP_KV_EXPORT_READY_TIMEOUT", "1"))
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        exported = await _export_prefill_kv(prefill, request_id)
        if exported is not None:
            return exported
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.005)


async def _release_prefill_kv(
    prefill: PAPServiceClient,
    *,
    request_id: str,
    lease_id: str,
) -> bool:
    """Release a retained historical Prefill lease after replacement."""
    response = await prefill.client.post(
        "/v1/pap/prefill/lease-release",
        json={"request_id": request_id, "lease_id": lease_id},
        headers=_headers(request_id),
    )
    response.raise_for_status()
    body = response.json()
    return bool(
        body.get("released", False) or body.get("reason") == "unknown_or_released_lease"
    )


async def _install_completed_prefill_on_group(
    *,
    req_data: dict[str, Any],
    request_id: str,
    conversation_id: str,
    source_group: PAPGroup,
    source_prefill: PAPServiceClient,
    source_prefill_response: dict[str, Any],
    target_group: PAPGroup,
    target_prefill: PAPServiceClient,
    target_attention_clients: list[PAPServiceClient],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Import a completed Prefill KV snapshot into the Decode target PA."""
    source_kv_params = enrich_prefill_kv_params(
        source_prefill_response.get("kv_transfer_params") or {},
        prefill_host=source_group.prefill_host,
        prefill_nixl_port=source_group.prefill_nixl_port,
    )
    source_prefix_len = prefill_prefix_len_from_kv_params(source_kv_params)
    if source_prefix_len is None:
        raise RuntimeError("PAP completed Prefill returned no transferable KV")
    remote_kv_params, migrated_tokens = _migration_prefix_kv_params(
        {
            "seq_len": source_prefix_len,
            "kv_transfer_params": source_kv_params,
        },
        source_group=source_group,
        block_size=int(os.environ.get("PAP_BLOCK_SIZE", "16")),
    )
    source_request_id = prefill_kv_handle_from_kv_params(source_kv_params)
    if source_request_id is None:
        raise RuntimeError("PAP migration source returned no stable KV handle")
    source_export = await _wait_prefill_kv_export(
        source_prefill,
        source_request_id,
    )
    if source_export is None:
        raise RuntimeError("PAP migration source KV lease is unavailable")
    prefix_token_ids, prefix_block_hashes = _migration_prefix_identity(
        source_export,
        migrated_tokens=migrated_tokens,
    )
    if not target_group.attention_tcp_endpoint:
        raise RuntimeError("PAP KV migration requires Attention TCP endpoints")

    target_sessions = await register_attention_handles(
        target_attention_clients,
        request_id=request_id,
        conversation_id=conversation_id,
        prefill_endpoint=target_group.prefill_base_url,
        kv_transfer_params={},
        prefix_len=None,
    )
    try:
        decode_capacity_tokens = requested_decode_capacity(req_data)
        if decode_capacity_tokens is None:
            decode_capacity_tokens = int(
                os.environ.get(
                    "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS",
                    "0",
                )
            )
        migration_request = {
            "request_id": request_id,
            "source_kv_params": remote_kv_params,
            "prefix_len": migrated_tokens,
            "prefix_token_ids": prefix_token_ids,
            "prefix_block_hashes": prefix_block_hashes,
            "decode_capacity_tokens": decode_capacity_tokens,
            "session_handle": str(target_sessions[0].get("prefill_kv_handle")),
            "attention_tcp_endpoint": target_group.attention_tcp_endpoint,
        }
        started = time.perf_counter()
        submitted = await _post_json(
            target_prefill,
            "/v1/pap/prefill/kv-import",
            migration_request,
            request_id,
        )
        job_id = str(submitted["job_id"])
        status = submitted
        transfer_started = status.get("status") == "transferring" and bool(
            status.get("kv_transfer_params")
        )
        if status.get("status") != "ready" and not transfer_started:
            timeout = float(os.environ.get("PAP_KV_MIGRATION_TIMEOUT", "30"))
            deadline = time.monotonic() + timeout
            while True:
                if status.get("status") in {"failed", "unknown"}:
                    raise RuntimeError(
                        "PAP target KV migration failed "
                        f"job_id={job_id} detail={status.get('error')}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"PAP target KV migration timed out job_id={job_id}"
                    )
                await asyncio.sleep(0.01)
                status = await _post_json(
                    target_prefill,
                    "/v1/pap/prefill/kv-import/status",
                    {"job_id": job_id},
                    request_id,
                )
                if status.get("status") == "ready":
                    break
        target_response = {
            "kv_transfer_params": status.get("kv_transfer_params") or {},
        }
        target_kv_params = enrich_prefill_kv_params(
            target_response.get("kv_transfer_params") or {},
            prefill_host=target_group.prefill_host,
            prefill_nixl_port=target_group.prefill_nixl_port,
        )
        target_prefix_len = prefill_prefix_len_from_kv_params(target_kv_params)
        if target_prefix_len != migrated_tokens:
            raise RuntimeError(
                "PAP completed Prefill migration length mismatch "
                f"source={migrated_tokens} target={target_prefix_len}"
            )
        ready = all(
            [
                await wait_attention_prefill_ready(attention, request_id)
                for attention in target_attention_clients
            ]
        )
        if not ready:
            raise RuntimeError("PAP target Attention did not install migrated KV")
        migration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "PAP KV migration Attention ready request_id=%s job_id=%s "
            "queue_ms=%s engine_transfer_and_publish_ms=%s "
            "engine_total_ms=%s gateway_ms=%d",
            request_id,
            job_id,
            status.get("queue_ms"),
            status.get("transfer_and_publish_ms"),
            status.get("total_ms"),
            migration_ms,
        )
        if source_export is not None:
            source_lease_id = source_export.get("lease_id")
            if isinstance(source_lease_id, str) and source_lease_id:
                try:
                    released = await _release_prefill_kv(
                        source_prefill,
                        request_id=source_request_id,
                        lease_id=source_lease_id,
                    )
                    if not released:
                        logger.warning(
                            "PAP migrated source lease release not acknowledged "
                            "request_id=%s lease_id=%s",
                            request_id,
                            source_lease_id,
                        )
                except Exception:
                    logger.warning(
                        "PAP migrated source lease release failed "
                        "request_id=%s lease_id=%s",
                        request_id,
                        source_lease_id,
                        exc_info=True,
                    )
        return target_sessions, target_response, migration_ms
    except Exception:
        await _cleanup_attention_sessions(
            target_attention_clients,
            request_id,
        )
        raise


async def register_attention_handles(
    attention_clients: list[PAPServiceClient],
    *,
    request_id: str,
    conversation_id: str,
    prefill_endpoint: str,
    kv_transfer_params: dict[str, Any],
    prefix_len: int | None,
) -> list[dict[str, Any]]:
    sessions = []
    registered_attentions: list[PAPServiceClient] = []
    try:
        for attention in attention_clients:
            sessions.append(
                await register_attention_handle(
                    attention,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    prefill_endpoint=prefill_endpoint,
                    kv_transfer_params=kv_transfer_params,
                    prefix_len=prefix_len,
                )
            )
            registered_attentions.append(attention)
    except Exception:
        await _cleanup_attention_sessions(registered_attentions, request_id)
        raise
    return sessions


async def _delete_attention_session(
    attention: PAPServiceClient,
    request_id: str,
    *,
    retain_lease: bool = False,
) -> None:
    resp = await attention.client.delete(
        f"/v1/pap/attention/sessions/{request_id}",
        headers=_headers(request_id),
        params={"retain_lease": "true"} if retain_lease else None,
    )
    resp.raise_for_status()


async def _cleanup_attention_sessions(
    attention_clients: list[PAPServiceClient],
    request_id: str,
    *,
    retain_lease: bool = False,
) -> None:
    for attention in attention_clients:
        try:
            await _delete_attention_session(
                attention,
                request_id,
                retain_lease=retain_lease,
            )
        except Exception as exc:
            logger.warning(
                "failed to release PAP attention session request_id=%s "
                "attention_endpoint=%s error=%s",
                request_id,
                attention.base_url,
                exc,
            )


async def _stream_projection(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
):
    profile = _pap_prefill_ipc_profile_enabled()
    start = time.perf_counter() if profile else 0.0
    first_chunk = True
    chunk_count = 0
    byte_count = 0
    async with client.client.stream(
        "POST",
        endpoint,
        json=payload,
        headers=_headers(request_id),
    ) as resp:
        resp.raise_for_status()
        if profile:
            logger.info(
                "PAP proxy projection stream profile request_id=%s open_ms=%.3f",
                request_id,
                (time.perf_counter() - start) * 1000.0,
            )
        async for chunk in resp.aiter_bytes():
            if profile:
                chunk_count += 1
                byte_count += len(chunk)
                if first_chunk:
                    first_chunk = False
                    logger.info(
                        "PAP proxy projection stream profile request_id=%s "
                        "first_chunk_ms=%.3f first_chunk_bytes=%d",
                        request_id,
                        (time.perf_counter() - start) * 1000.0,
                        len(chunk),
                    )
            yield chunk
    if profile:
        logger.info(
            "PAP proxy projection stream profile request_id=%s total_ms=%.3f "
            "chunks=%d bytes=%d",
            request_id,
            (time.perf_counter() - start) * 1000.0,
            chunk_count,
            byte_count,
        )


async def _stream_projection_with_cleanup(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    attention_clients: list[PAPServiceClient],
    admission: PAPProjectionAdmission,
    group: PAPGroup,
    projection: ProjectionInstance,
    attention_load_router: PAPAttentionLoadRouter | None = None,
    completion_tokens: int = 0,
    retain_completed_lease: bool = False,
    prefill_kv_handle: str | None = None,
):
    terminal_marker = b"data: [DONE]"
    pending = b""
    terminal_chunks: list[bytes] = []
    try:
        async for chunk in _stream_projection(client, endpoint, payload, request_id):
            if terminal_chunks:
                terminal_chunks.append(chunk)
                continue
            pending += chunk
            marker_index = pending.find(terminal_marker)
            if marker_index < 0:
                safe_length = len(pending) - len(terminal_marker) + 1
                if safe_length > 0:
                    yield pending[:safe_length]
                    pending = pending[safe_length:]
                continue
            if marker_index:
                yield pending[:marker_index]
            terminal_chunks.append(pending[marker_index:])
            pending = b""
        if pending:
            yield pending
    finally:
        completed = bool(terminal_chunks)
        if attention_load_router is not None:
            if completed:
                attention_load_router.finish(
                    request_id,
                    completion_tokens=completion_tokens,
                    prefill_kv_handle=prefill_kv_handle,
                )
            else:
                attention_load_router.abort(request_id)
        try:
            await _cleanup_attention_sessions(
                attention_clients,
                request_id,
                retain_lease=completed and retain_completed_lease,
            )
        finally:
            await admission.release(group, projection)
    for chunk in terminal_chunks:
        yield chunk


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    reject_removed_pap_flags(os.environ)
    app.state.groups = parse_pap_groups(args.pap_groups)
    app.state.projections = parse_projection_instances(args.projections)
    app.state.request_counter = count()
    app.state.pair_counts = Counter()
    app.state.prefill_clients = {
        group: _make_client(group.prefill_host, group.prefill_port, "prefill")
        for group in app.state.groups
    }
    app.state.attention_clients = {}
    for group in app.state.groups:
        if isinstance(group.attention_port, int):
            ports = (group.attention_port,)
        else:
            ports = group.attention_port
        app.state.attention_clients[group] = [
            _make_client(group.attention_host, port, "attention") for port in ports
        ]
    app.state.projection_clients = {
        projection: _make_client(projection.host, projection.port, "projection")
        for projection in app.state.projections
    }
    app.state.conversation_router = PAPConversationRouter(app.state.groups)
    app.state.attention_load_router = PAPAttentionLoadRouter(
        app.state.groups,
        migration_min_peak_gain_ratio=(
            args.attention_load_migration_min_peak_gain_ratio
        ),
        migration_max_inflight=args.attention_load_migration_max_inflight,
    )
    app.state.prefill_admission = PAPPrefillAdmission(
        app.state.groups,
        args.prefill_max_inflight_per_pa,
    )
    app.state.projection_admission = PAPProjectionAdmission(app.state.groups)
    yield
    attention_clients = [
        client for clients in app.state.attention_clients.values() for client in clients
    ]
    for client in [
        *app.state.prefill_clients.values(),
        *attention_clients,
        *app.state.projection_clients.values(),
    ]:
        await client.client.aclose()


app = FastAPI(title="PAP Gateway", lifespan=lifespan)


def _pop_conversation_id(
    req_data: dict[str, Any],
    correlation_id: str | None,
) -> str:
    """Return the body conversation id or a session-header fallback."""

    raw_conversation_id = req_data.pop("conversation_id", None)
    if raw_conversation_id is not None:
        conversation_id = str(raw_conversation_id)
        if conversation_id:
            return conversation_id
    return correlation_id or ""


async def _handle_openai_request(api_path: str, request: Request):
    profile = _pap_prefill_ipc_profile_enabled()
    request_start = time.perf_counter() if profile else 0.0
    req_data = await request.json()
    request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    conversation_id = _pop_conversation_id(
        req_data,
        request.headers.get("X-Correlation-ID"),
    )
    client_stream = bool(req_data.get("stream", False))
    request_number = next(request.app.state.request_counter)
    attention_load_router: PAPAttentionLoadRouter | None = None
    estimated_context_tokens = 1
    history_record: tuple[PAPGroup, str, int] | None = None
    history_export: dict[str, Any] | None = None
    if request.app.state.args.routing_policy == "attention_load":
        attention_load_router = request.app.state.attention_load_router
        history_record = (
            attention_load_router.history(conversation_id) if conversation_id else None
        )
        history_context_tokens = history_record[2] if history_record else 0
        if history_record is not None and attention_load_router.migration_enabled:
            history_group, history_request_id, _ = history_record
            try:
                history_export = await _export_prefill_kv(
                    request.app.state.prefill_clients[history_group],
                    history_request_id,
                )
            except Exception as exc:
                logger.warning(
                    "PAP retained KV export failed request_id=%s "
                    "history_request_id=%s source_pa=%d error=%s",
                    request_id,
                    history_request_id,
                    request.app.state.groups.index(history_group),
                    exc,
                )
            if history_export is not None:
                exported_seq_len = history_export.get("seq_len")
                if isinstance(exported_seq_len, int) and exported_seq_len > 0:
                    history_context_tokens = exported_seq_len
        explicit_context_tokens = req_data.pop("pap_context_tokens", None)
        if explicit_context_tokens is None:
            explicit_context_tokens = request.headers.get("X-PAP-Context-Tokens")
        estimated_context_tokens = _estimate_context_tokens(
            req_data,
            history_context_tokens=history_context_tokens,
            explicit_context_tokens=explicit_context_tokens,
        )
    group, projection = select_instances(
        request_number,
        request.app.state.groups,
        request.app.state.projections,
        routing_policy=request.app.state.args.routing_policy,
        conversation_id=conversation_id,
        conversation_router=request.app.state.conversation_router,
        attention_load_router=attention_load_router,
        request_id=request_id,
        estimated_context_tokens=estimated_context_tokens,
    )
    prefill_group = group
    prefill_group_index = request.app.state.groups.index(prefill_group)
    prefill = request.app.state.prefill_clients[group]
    attention_clients = request.app.state.attention_clients[group]
    projection_client = request.app.state.projection_clients[projection]
    projection_index = request.app.state.projections.index(projection)

    attention_sessions: list[dict[str, Any]] | None = None
    handed_off_stream_cleanup = False
    prefill_admitted = False
    projection_admitted = False
    attention_load_finished = False
    try:
        prefill_admission_wait_ms = await request.app.state.prefill_admission.acquire(
            prefill_group
        )
        prefill_admitted = True
        register_start = time.perf_counter() if profile else 0.0
        attention_sessions = await register_attention_handles(
            attention_clients,
            request_id=request_id,
            conversation_id=conversation_id,
            prefill_endpoint=group.prefill_base_url,
            kv_transfer_params={},
            prefix_len=None,
        )
        register_ms = (
            (time.perf_counter() - register_start) * 1000.0 if profile else 0.0
        )
        attention_session = attention_sessions[0]

        prefill_payload_start = time.perf_counter() if profile else 0.0
        prefill_payload = attach_pap_prefill_attention_params(
            build_prefill_payload(req_data),
            pap_attention_endpoint=group.attention_base_url,
            pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
            pap_offload_exec_zmq_endpoint=group.attention_zmq_endpoint,
            pap_prefill_kv_handle=str(attention_session.get("prefill_kv_handle")),
            pap_mode=request.app.state.args.pap_mode,
        )
        prefill_payload_ms = (
            (time.perf_counter() - prefill_payload_start) * 1000.0 if profile else 0.0
        )
        t0 = time.time()
        try:
            prefill_resp = await _post_json(
                prefill,
                api_path,
                prefill_payload,
                request_id,
            )
        finally:
            try:
                await request.app.state.prefill_admission.release(prefill_group)
            finally:
                prefill_admitted = False
        prefill_ms = int((time.time() - t0) * 1000)
        migration_ms = 0
        migration_attention_ready = False
        usage = prefill_resp.get("usage")
        prompt_tokens = None
        if isinstance(usage, dict):
            value = usage.get("prompt_tokens")
            if isinstance(value, int) and value > 0:
                prompt_tokens = value
        if prompt_tokens is None:
            source_kv_params = prefill_resp.get("kv_transfer_params") or {}
            prompt_tokens = prefill_prefix_len_from_kv_params(source_kv_params)
        if prompt_tokens is None:
            prompt_tokens = estimated_context_tokens

        if attention_load_router is not None:
            decode_group = attention_load_router.observe_prefill(
                request_id,
                prompt_tokens,
            )
            if decode_group != prefill_group:
                decode_group_index = request.app.state.groups.index(decode_group)
                target_attention_clients = request.app.state.attention_clients[
                    decode_group
                ]
                logger.info(
                    "PAP completed Prefill migration planned request_id=%s "
                    "source_pa=%d target_pa=%d tokens=%d",
                    request_id,
                    prefill_group_index,
                    decode_group_index,
                    prompt_tokens,
                )
                migration_started = time.perf_counter()
                try:
                    (
                        target_sessions,
                        target_response,
                        migration_ms,
                    ) = await _install_completed_prefill_on_group(
                        req_data=req_data,
                        request_id=request_id,
                        conversation_id=conversation_id,
                        source_group=prefill_group,
                        source_prefill=(
                            request.app.state.prefill_clients[prefill_group]
                        ),
                        source_prefill_response=prefill_resp,
                        target_group=decode_group,
                        target_prefill=(
                            request.app.state.prefill_clients[decode_group]
                        ),
                        target_attention_clients=target_attention_clients,
                    )
                except Exception as exc:
                    migration_ms = int((time.perf_counter() - migration_started) * 1000)
                    group = attention_load_router.mark_migration_missed(request_id)
                    logger.warning(
                        "PAP completed Prefill migration failed; using source "
                        "request_id=%s source_pa=%d target_pa=%d error=%s",
                        request_id,
                        prefill_group_index,
                        decode_group_index,
                        exc,
                    )
                else:
                    await _cleanup_attention_sessions(
                        attention_clients,
                        request_id,
                    )
                    group = decode_group
                    prefill = request.app.state.prefill_clients[group]
                    attention_clients = target_attention_clients
                    attention_sessions = target_sessions
                    attention_session = attention_sessions[0]
                    prefill_resp = target_response
                    migration_attention_ready = True
                    attention_load_router.mark_migration_succeeded(request_id)
                    logger.info(
                        "PAP completed Prefill migration installed "
                        "request_id=%s source_pa=%d target_pa=%d "
                        "tokens=%d migration_ms=%d",
                        request_id,
                        prefill_group_index,
                        decode_group_index,
                        prompt_tokens,
                        migration_ms,
                    )
        if history_record is not None and history_export is not None:
            history_source_group, history_request_id, _ = history_record
            history_lease_id = history_export.get("lease_id")
            if isinstance(history_lease_id, str) and history_lease_id:
                try:
                    released = await _release_prefill_kv(
                        request.app.state.prefill_clients[history_source_group],
                        request_id=history_request_id,
                        lease_id=history_lease_id,
                    )
                    if not released:
                        logger.warning(
                            "PAP historical KV lease release not acknowledged "
                            "request_id=%s history_request_id=%s",
                            request_id,
                            history_request_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "PAP historical KV lease release failed request_id=%s "
                        "history_request_id=%s error=%s",
                        request_id,
                        history_request_id,
                        exc,
                    )

        projection_payload_start = time.perf_counter() if profile else 0.0
        kv_params = enrich_prefill_kv_params(
            prefill_resp.get("kv_transfer_params") or {},
            prefill_host=group.prefill_host,
            prefill_nixl_port=group.prefill_nixl_port,
        )
        prefill_kv_handle = prefill_kv_handle_from_kv_params(
            kv_params,
            fallback=attention_session.get("prefill_kv_handle"),
        )
        prefix_len = prefill_prefix_len_from_kv_params(kv_params)
        attention_ready = migration_attention_ready
        if prefix_len is not None and not attention_ready:
            attention_ready = all(
                [
                    await wait_attention_prefill_ready(attention, request_id)
                    for attention in attention_clients
                ]
            )
        projection_payload = build_projection_payload_for_group(
            req_data,
            kv_params,
            group,
            pap_prefill_kv_handle=prefill_kv_handle,
            pap_attention_kv_installed=attention_ready,
        )
        projection_payload.setdefault("stream", client_stream)
        projection_kv_params = projection_payload.get("kv_transfer_params") or {}
        group_index = request.app.state.groups.index(group)
        pair_name = f"pa{group_index}:p{projection_index}"
        request.app.state.pair_counts[pair_name] += 1
        projection_payload_ms = (
            (time.perf_counter() - projection_payload_start) * 1000.0
            if profile
            else 0.0
        )
        logger.info(
            "request_id=%s pa=%s:%d attention=%s:%s projection=%s:%d "
            "pa_index=%d projection_index=%d pair=%s "
            "prefill_pa_index=%d prefill_admission_wait_ms=%.3f "
            "prefill_ms=%d migration_ms=%d "
            "prefill_prefix_len=%s attention_ready=%s "
            "projection_kv_keys=%s",
            request_id,
            group.prefill_host,
            group.prefill_port,
            group.attention_host,
            group.attention_port,
            projection.host,
            projection.port,
            group_index,
            projection_index,
            pair_name,
            prefill_group_index,
            prefill_admission_wait_ms,
            prefill_ms,
            migration_ms,
            prefix_len,
            attention_ready,
            sorted(projection_kv_params.keys()),
        )
        if profile:
            logger.info(
                "PAP proxy prefill IPC profile request_id=%s register_ms=%.3f "
                "prefill_admission_wait_ms=%.3f prefill_payload_ms=%.3f "
                "prefill_ms=%d migration_ms=%d "
                "projection_payload_ms=%.3f pre_projection_ms=%.3f",
                request_id,
                register_ms,
                prefill_admission_wait_ms,
                prefill_payload_ms,
                prefill_ms,
                migration_ms,
                projection_payload_ms,
                (time.perf_counter() - request_start) * 1000.0,
            )

        response_headers = {
            "X-PAP-Prefill-Admission-Wait-Ms": f"{prefill_admission_wait_ms:.3f}",
            "X-PAP-Prefill-Ms": str(prefill_ms),
            "X-PAP-Migration-Ms": str(migration_ms),
            "X-PAP-Prefill-Group": str(prefill_group_index),
            "X-PAP-Group": str(group_index),
            "X-PAP-Projection": str(projection.port),
            "X-PAP-Projection-Index": str(projection_index),
            "X-PAP-Pair": pair_name,
        }
        response_headers.update(_prefill_usage_headers(prefill_resp))

        admission = request.app.state.projection_admission
        await admission.acquire(group, projection)
        projection_admitted = True

        if client_stream:
            handed_off_stream_cleanup = True
            projection_admitted = False
            return StreamingResponse(
                _stream_projection_with_cleanup(
                    projection_client,
                    api_path,
                    projection_payload,
                    request_id,
                    attention_clients,
                    admission,
                    group,
                    projection,
                    attention_load_router,
                    requested_decode_capacity(req_data) or 0,
                    bool(
                        conversation_id
                        and attention_load_router is not None
                        and attention_load_router.migration_enabled
                    ),
                    prefill_kv_handle,
                ),
                media_type="text/event-stream",
                headers=response_headers,
            )

        projection_resp = await _post_json(
            projection_client,
            api_path,
            projection_payload,
            request_id,
        )
        if attention_load_router is not None:
            response_usage = projection_resp.get("usage")
            completion_tokens = 0
            if isinstance(response_usage, dict):
                value = response_usage.get("completion_tokens")
                if isinstance(value, int):
                    completion_tokens = value
            attention_load_router.finish(
                request_id,
                completion_tokens=completion_tokens,
                prefill_kv_handle=prefill_kv_handle,
            )
            attention_load_finished = True
        return JSONResponse(
            projection_resp,
            headers=response_headers,
        )
    finally:
        if prefill_admitted:
            await request.app.state.prefill_admission.release(prefill_group)
        if not handed_off_stream_cleanup:
            try:
                if attention_sessions is not None:
                    await _cleanup_attention_sessions(
                        attention_clients,
                        request_id,
                        retain_lease=bool(
                            conversation_id
                            and attention_load_router is not None
                            and attention_load_finished
                            and attention_load_router.migration_enabled
                        ),
                    )
            finally:
                if projection_admitted:
                    await request.app.state.projection_admission.release(
                        group,
                        projection,
                    )
            if attention_load_router is not None and not attention_load_finished:
                attention_load_router.abort(request_id)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request("/v1/chat/completions", request)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "role": "multi-pap-proxy",
        "groups": len(app.state.groups),
        "projections": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "pair_counts": dict(sorted(app.state.pair_counts.items())),
        "conversation_routing": app.state.conversation_router.snapshot(),
        "attention_load_routing": app.state.attention_load_router.snapshot(),
        "prefill_admission": await app.state.prefill_admission.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
    }


@app.get("/v1/pap/topology/stats")
async def topology_stats() -> dict[str, Any]:
    pair_counts = dict(sorted(app.state.pair_counts.items()))
    return {
        "pa_count": len(app.state.groups),
        "projection_count": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "total_requests": sum(pair_counts.values()),
        "pair_counts": pair_counts,
        "conversation_routing": app.state.conversation_router.snapshot(),
        "attention_load_routing": app.state.attention_load_router.snapshot(),
        "prefill_admission": await app.state.prefill_admission.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PAP request gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--pap-groups",
        required=True,
        help=(
            "Comma-separated prefill_host:prefill_port:prefill_nixl_port:"
            "attention_host:attention_port entries"
        ),
    )
    parser.add_argument(
        "--projections",
        required=True,
        help="Comma-separated projection_host:projection_port entries",
    )
    parser.add_argument("--pap-mode", default=os.environ.get("PAP_MODE", "pap"))
    parser.add_argument(
        "--routing-policy",
        default=os.environ.get("PAP_ROUTING_POLICY", "round_robin"),
        choices=(
            "round_robin",
            "crossbar_round_robin",
            "projection_affinity",
            "projection_sticky",
            "conversation_affinity",
            "attention_load",
        ),
    )
    parser.add_argument(
        "--attention-load-migration-min-peak-gain-ratio",
        type=float,
        default=float(
            os.environ.get(
                "PAP_ATTENTION_LOAD_MIGRATION_MIN_PEAK_GAIN_RATIO",
                "0.3",
            )
        ),
        help=(
            "Minimum relative reduction in peak Decode Attention load "
            "required for a migration"
        ),
    )
    parser.add_argument(
        "--attention-load-migration-max-inflight",
        type=int,
        default=int(os.environ.get("PAP_ATTENTION_LOAD_MIGRATION_MAX_INFLIGHT", "1")),
        help="Maximum unresolved historical KV migrations; zero disables them",
    )
    parser.add_argument(
        "--prefill-max-inflight-per-pa",
        type=int,
        default=int(os.environ.get("PAP_PREFILL_MAX_INFLIGHT_PER_PA", "0")),
        help="Maximum in-flight Prefill requests per PA; zero is unbounded",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PAP gateway."""
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)


if __name__ == "__main__":
    main()
