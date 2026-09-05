# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM engine control hooks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.kv import lease as pap_lease

_DECODE_ALLOCATION_GRANULARITY_TOKENS = 256


@dataclass(frozen=True, slots=True)
class _AppliedCommit:
    commit_seq: int
    new_seq_len: int
    new_token_ids: tuple[int, ...]


class PAPEngineControl:
    """Validate and apply the Prefill KV control stream in EngineCore order."""

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler
        self._commits: dict[str, _AppliedCommit] = {}

    def apply(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one serialized endpoint-plugin operation."""
        if operation == "decode_commit":
            return self._apply_decode_commit(payload)
        if operation == "lease_release":
            return self._release_lease(payload)
        if operation == "kv_load_snapshot":
            return PAPEngineAdapter.kv_load_snapshot(self._scheduler)
        if operation == "decode_allocate":
            return self._allocate_decode(payload)
        if operation in {"projection_quiesce", "request_quiesce"}:
            request_ids = tuple(str(item) for item in payload["request_ids"])
            active = [
                request_id
                for request_id in request_ids
                if request_id in getattr(self._scheduler, "requests", {})
            ]
            return {
                "quiesced": not active,
                "active_request_ids": active,
            }
        raise ValueError(f"unknown PAP control operation: {operation!r}")

    def _allocate_decode(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extend allocator-owned storage, never the computed-token watermark."""
        from vllm.pap.kv_connector import PAPPrefillConnector
        from vllm.v1.request import RequestStatus

        scheduler = self._scheduler
        connector = scheduler.connector
        lease_id = str(payload["lease_id"])
        registry = pap_lease.get_global_kv_lease_registry()
        lease = registry.active_entry(lease_id)
        if lease is None or not isinstance(connector, PAPPrefillConnector):
            raise ValueError("unknown or released PAP allocation lease")
        request = scheduler.requests.get(lease.request_id)
        metadata = connector._request_metadata.get(lease.request_id)
        if (
            request is None
            or metadata is None
            or metadata.prefill_kv_handle != str(payload["session_handle"])
            or connector._generations[lease.request_id] != int(payload["generation"])
        ):
            raise ValueError("stale PAP allocation generation or session")
        if request.status not in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            raise ValueError("Decode allocation requires completed non-aborted Prefill")
        required = int(payload["required_tokens"])
        reserve_tokens = int(
            payload.get("reserve_tokens", _DECODE_ALLOCATION_GRANULARITY_TOKENS)
        )
        if not 0 <= reserve_tokens <= 4096:
            raise ValueError("Decode allocation reserve must be in [0, 4096]")
        limit = scheduler.max_model_len
        if metadata.decode_capacity_tokens is not None:
            limit = min(
                limit, request.num_prompt_tokens + metadata.decode_capacity_tokens
            )
        if not request.num_prompt_tokens <= required <= limit:
            raise ValueError("Decode allocation exceeds request context limit")
        manager = scheduler.kv_cache_manager
        groups = manager.get_block_ids(request.request_id)
        if len(groups) != 1:
            raise ValueError("PAP Decode allocation requires one KV group")
        block_size = int(scheduler.block_size)
        capacity = len(groups[0]) * block_size
        previous_blocks = len(groups[0])
        target = min(limit, max(required, capacity + reserve_tokens))
        if target > capacity:
            blocks = manager.allocate_slots(
                request,
                target - request.num_computed_tokens,
                delay_cache_blocks=True,
            )
            if blocks is None:
                scheduler.pap_scheduler.record_decode_allocation(blocks=0, failed=True)
                return {"allocated": False, "reason": "insufficient_kv_capacity"}
            groups = manager.get_block_ids(request.request_id)
        owned = tuple(groups[0])
        registry.extend_blocks(request.request_id, owned)
        scheduler.pap_scheduler.record_decode_allocation(
            blocks=len(owned) - previous_blocks,
            failed=False,
        )
        return {
            "allocated": True,
            "lease_id": lease_id,
            "generation": connector._generations[lease.request_id],
            "block_ids": list(owned),
            "writable_end_token": min(len(owned) * block_size, limit),
            "allocation_limit_token": limit,
        }

    def _apply_decode_commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload["request_id"])
        commit = _AppliedCommit(
            commit_seq=int(payload["commit_seq"]),
            new_seq_len=int(payload["new_seq_len"]),
            new_token_ids=tuple(int(token) for token in payload["new_token_ids"]),
        )
        if commit.commit_seq <= 0:
            raise ValueError("commit_seq must be positive")

        previous = self._commits.get(request_id)
        expected_seq = 1 if previous is None else previous.commit_seq + 1
        if previous is not None and commit.commit_seq == previous.commit_seq:
            if commit != previous:
                raise ValueError(
                    "conflicting duplicate PAP decode commit: "
                    f"request_id={request_id} commit_seq={commit.commit_seq}"
                )
            return {
                "request_id": request_id,
                "commit_seq": commit.commit_seq,
                "acked_commit_seq": commit.commit_seq,
                "new_seq_len": commit.new_seq_len,
                "applied": False,
                "idempotent": True,
            }
        if commit.commit_seq != expected_seq:
            raise ValueError(
                "non-contiguous PAP decode commit: "
                f"request_id={request_id} expected={expected_seq} "
                f"got={commit.commit_seq}"
            )
        if previous is not None and commit.new_seq_len <= previous.new_seq_len:
            raise ValueError(
                "PAP decode sequence length must increase: "
                f"request_id={request_id} previous={previous.new_seq_len} "
                f"got={commit.new_seq_len}"
            )

        result = PAPEngineAdapter.apply_decode_commit(
            self._scheduler,
            request_id,
            commit.new_seq_len,
            commit.new_token_ids,
        )
        if not result.get("applied", False):
            return {**result, "commit_seq": commit.commit_seq}

        self._commits[request_id] = commit
        return {
            **result,
            "commit_seq": commit.commit_seq,
            "acked_commit_seq": commit.commit_seq,
            "idempotent": False,
        }

    def _release_lease(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload["request_id"])
        final_commit_seq = payload.get("final_commit_seq")
        previous = self._commits.get(request_id)
        if final_commit_seq is not None:
            applied_seq = 0 if previous is None else previous.commit_seq
            if int(final_commit_seq) != applied_seq:
                raise ValueError(
                    "PAP lease release raced the decode commit stream: "
                    f"request_id={request_id} final_commit_seq={final_commit_seq} "
                    f"applied_commit_seq={applied_seq}"
                )

        result = PAPEngineAdapter.release_kv_lease(
            request_id,
            str(payload["lease_id"]),
        )
        if result.get("released", False) or result.get("reason") in {
            "unknown_or_released_lease",
            "unknown_expired_or_released_lease",
        }:
            self._commits.pop(request_id, None)
        return result


class PAPEngineAdapter:
    """Translate PAP control operations into vLLM scheduler mutations."""

    @staticmethod
    def apply_decode_commit(
        scheduler: Any,
        request_id: str,
        new_seq_len: int,
        new_token_ids: Sequence[int],
    ) -> dict[str, Any]:
        """Apply one acknowledged remote decode commit."""
        normalized_id = str(request_id)
        request = getattr(scheduler, "requests", {}).get(normalized_id)
        if request is None:
            if pap_lease.pap_has_active_lease(normalized_id):
                old_seq_len = pap_lease.pap_kv_seq_len(normalized_id) or 0
                pap_lease.pap_refresh_lease(normalized_id)
                updated = pap_lease.pap_update_kv_seq_len(
                    normalized_id,
                    int(new_seq_len),
                )
                return {
                    "request_id": normalized_id,
                    "applied": updated,
                    "old_seq_len": old_seq_len,
                    "new_seq_len": int(new_seq_len),
                    "direct_lease_commit": True,
                }
            finished_req_ids: set[str] = getattr(
                scheduler,
                "finished_req_ids",
                set(),
            )
            reason = (
                "request_finished"
                if normalized_id in finished_req_ids
                else "unknown_request"
            )
            return {
                "request_id": normalized_id,
                "applied": False,
                "reason": reason,
            }

        old_seq_len = int(request.num_computed_tokens)
        PAPEngineAdapter._advance_request_kv(
            scheduler.kv_cache_manager,
            request,
            int(new_seq_len),
            new_token_ids,
        )
        pap_lease.pap_refresh_lease(normalized_id)
        pap_lease.pap_update_kv_seq_len(normalized_id, int(new_seq_len))
        return {
            "request_id": normalized_id,
            "applied": True,
            "old_seq_len": old_seq_len,
            "new_seq_len": int(request.num_computed_tokens),
        }

    @staticmethod
    def _advance_request_kv(
        kv_cache_manager: Any,
        request: Any,
        new_seq_len: int,
        new_token_ids: Sequence[int],
    ) -> None:
        """Commit remotely produced decode tokens to a vLLM request ledger."""
        old_seq_len = int(request.num_computed_tokens)
        if new_seq_len <= old_seq_len:
            return

        delta = tuple(int(token) for token in new_token_ids)
        expected_delta = new_seq_len - old_seq_len
        if len(delta) != expected_delta:
            raise ValueError(
                "new_token_ids length must match new_seq_len delta: "
                f"expected {expected_delta}, got {len(delta)}"
            )

        existing_count = min(
            max(int(request.num_tokens) - old_seq_len, 0),
            expected_delta,
        )
        existing_delta = tuple(
            int(token)
            for token in request.all_token_ids[
                old_seq_len : old_seq_len + existing_count
            ]
        )
        if existing_delta != delta[:existing_count]:
            raise ValueError(
                "decode commit does not match existing uncomputed token IDs: "
                f"request_id={request.request_id} old_seq_len={old_seq_len} "
                f"new_seq_len={new_seq_len}"
            )

        missing_delta = delta[existing_count:]
        if missing_delta:
            request.append_output_token_ids(missing_delta)
        request.num_computed_tokens = new_seq_len
        if kv_cache_manager.enable_caching:
            kv_cache_manager.coordinator.cache_blocks(request, new_seq_len)

    @staticmethod
    def release_kv_lease(
        request_id: str,
        lease_id: str,
    ) -> dict[str, Any]:
        """Release one Prefill KV lease and report the result."""
        released = pap_lease.pap_release_lease(str(lease_id))
        did_release = bool(released)
        result: dict[str, Any] = {
            "request_id": str(request_id),
            "lease_id": str(lease_id),
            "released": did_release,
            "block_count": len(released),
        }
        if not did_release:
            result["reason"] = "unknown_or_released_lease"
        return result

    @staticmethod
    def kv_load_snapshot(scheduler: Any) -> dict[str, Any]:
        """Return resident and projected Prefill-owned KV token load."""
        block_pool = scheduler.kv_cache_manager.block_pool
        total_blocks = max(0, int(block_pool.num_gpu_blocks) - 1)
        free_blocks = int(block_pool.get_num_free_blocks())
        non_evictable_blocks = max(0, total_blocks - free_blocks)
        block_size = int(scheduler.block_size)
        running_ids = {
            str(request.request_id) for request in getattr(scheduler, "running", ())
        }
        waiting_ids = {
            str(request.request_id)
            for queue in (
                getattr(scheduler, "waiting", ()),
                getattr(scheduler, "skipped_waiting", ()),
            )
            for request in queue
        }

        running_prefill_tokens = 0
        queued_prefill_tokens = 0
        running_decode_reservation_tokens = 0
        queued_decode_reservation_tokens = 0
        running_prefill_requests = 0
        queued_prefill_requests = 0
        for request_id, request in getattr(scheduler, "requests", {}).items():
            prompt_tokens = max(0, int(request.num_prompt_tokens))
            computed_tokens = min(
                prompt_tokens,
                max(0, int(request.num_computed_tokens)),
            )
            remaining_tokens = prompt_tokens - computed_tokens
            if remaining_tokens <= 0:
                continue
            metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
            decode_reservation = int(metadata.decode_capacity_tokens or 0)
            normalized_id = str(request_id)
            if normalized_id in running_ids:
                running_prefill_tokens += remaining_tokens
                running_decode_reservation_tokens += decode_reservation
                running_prefill_requests += 1
            elif normalized_id in waiting_ids:
                queued_prefill_tokens += remaining_tokens
                queued_decode_reservation_tokens += decode_reservation
                queued_prefill_requests += 1

        non_evictable_tokens = non_evictable_blocks * block_size
        outstanding_prefill_tokens = running_prefill_tokens + queued_prefill_tokens
        outstanding_decode_reservation_tokens = (
            running_decode_reservation_tokens + queued_decode_reservation_tokens
        )
        projected_kv_tokens = (
            non_evictable_tokens
            + outstanding_prefill_tokens
            + outstanding_decode_reservation_tokens
        )
        pap_scheduler = getattr(scheduler, "pap_scheduler", None)
        return {
            "non_evictable_kv_blocks": non_evictable_blocks,
            "non_evictable_kv_tokens": non_evictable_tokens,
            "running_prefill_tokens": running_prefill_tokens,
            "queued_prefill_tokens": queued_prefill_tokens,
            "outstanding_prefill_tokens": outstanding_prefill_tokens,
            "running_decode_reservation_tokens": (running_decode_reservation_tokens),
            "queued_decode_reservation_tokens": queued_decode_reservation_tokens,
            "outstanding_decode_reservation_tokens": (
                outstanding_decode_reservation_tokens
            ),
            "running_prefill_requests": running_prefill_requests,
            "queued_prefill_requests": queued_prefill_requests,
            "projected_kv_tokens": projected_kv_tokens,
            "routing_kv_tokens": projected_kv_tokens,
            "free_kv_blocks": free_blocks,
            "total_kv_blocks": total_blocks,
            "total_kv_tokens": total_blocks * block_size,
            "kv_block_size": block_size,
            "kv_load_fraction": (
                non_evictable_blocks / total_blocks if total_blocks else 0.0
            ),
            "decode_allocation_requests": int(
                getattr(pap_scheduler, "decode_allocation_requests", 0)
            ),
            "decode_allocation_blocks": int(
                getattr(pap_scheduler, "decode_allocation_blocks", 0)
            ),
            "decode_allocation_failures": int(
                getattr(pap_scheduler, "decode_allocation_failures", 0)
            ),
            "prefill_revocations": int(
                getattr(pap_scheduler, "prefill_revocations", 0)
            ),
        }


__all__ = ["PAPEngineAdapter", "PAPEngineControl"]
