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
        scheduler.kv_cache_manager.apply_decode_commit(
            request=request,
            new_seq_len=int(new_seq_len),
            new_token_ids=tuple(int(token) for token in new_token_ids),
        )
        pap_lease.pap_refresh_lease(normalized_id)
        pap_lease.pap_update_kv_export_seq_len(normalized_id, int(new_seq_len))
        return {
            "request_id": normalized_id,
            "applied": True,
            "old_seq_len": old_seq_len,
            "new_seq_len": int(request.num_computed_tokens),
        }

    @staticmethod
    def release_kv_lease(
        request_id: str,
        lease_id: str,
        *,
        retain: bool = False,
    ) -> dict[str, Any]:
        """Release one Prefill KV lease and report the result."""
        if retain:
            retained = pap_lease.pap_mark_kv_lease_retained(
                str(request_id),
                str(lease_id),
            )
            result: dict[str, Any] = {
                "request_id": str(request_id),
                "lease_id": str(lease_id),
                "retained": retained,
            }
            if not retained:
                result["reason"] = "unknown_expired_or_released_lease"
            return result

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
    def export_kv_lease(request_id: str) -> dict[str, Any]:
        """Return reusable NIXL metadata for one retained Prefill lease."""
        normalized_id = str(request_id)
        entry = pap_lease.pap_export_kv(normalized_id)
        if entry is None:
            return {
                "request_id": normalized_id,
                "exported": False,
                "reason": "unknown_expired_or_released_lease",
            }
        kv_transfer_params = dict(entry.kv_transfer_params)
        kv_transfer_params["remote_num_tokens"] = entry.seq_len
        return {
            "request_id": normalized_id,
            "exported": True,
            "lease_id": entry.lease_id,
            "seq_len": entry.seq_len,
            "kv_transfer_params": kv_transfer_params,
        }
