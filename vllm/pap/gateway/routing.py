# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway conversation and Attention-load routing."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from itertools import count
from typing import Any

from vllm.pap.gateway.payloads import enrich_prefill_kv_params
from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance

logger = logging.getLogger("pap_gateway")


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
