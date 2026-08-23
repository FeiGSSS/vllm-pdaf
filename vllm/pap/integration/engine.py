# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM engine control hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.lifecycle import lease as pap_lease


class PAPEngineAdapter:
    """Translate PAP control operations into vLLM scheduler mutations."""

    @staticmethod
    def is_metadata_only_request(
        params: Mapping[str, Any] | None,
    ) -> bool:
        """Return whether PAP metadata replaces ordinary KV transfer."""
        return PAPRequestMetadata.from_mapping(params).projection_kv_unaware

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
