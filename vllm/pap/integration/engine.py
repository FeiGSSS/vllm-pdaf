# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM engine control hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from typing import Any

from vllm.pap.integration.migration import validate_pap_migration_tp_size
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
                entry = pap_lease.pap_export_kv(normalized_id)
                old_seq_len = entry.seq_len if entry is not None else 0
                pap_lease.pap_refresh_lease(normalized_id)
                updated = pap_lease.pap_update_kv_export_seq_len(
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
    def export_kv_lease(scheduler: Any, request_id: str) -> dict[str, Any]:
        """Return prefix identity for an active request or retained lease."""
        normalized_id = str(request_id)
        entry = pap_lease.pap_export_kv(normalized_id)
        if entry is not None:
            kv_transfer_params = dict(entry.kv_transfer_params)
            kv_transfer_params["remote_num_tokens"] = entry.seq_len
            return {
                "request_id": normalized_id,
                "exported": True,
                "lease_id": entry.lease_id,
                "seq_len": entry.seq_len,
                "kv_transfer_params": kv_transfer_params,
                "prefix_token_ids": list(entry.prefix_token_ids),
                "prefix_block_hashes": [
                    block_hash.hex() for block_hash in entry.prefix_block_hashes
                ],
            }

        request = getattr(scheduler, "requests", {}).get(normalized_id)
        if request is None:
            return {
                "request_id": normalized_id,
                "exported": False,
                "reason": "unknown_expired_or_released_lease",
            }
        seq_len = int(request.num_computed_tokens)
        hash_block_size = scheduler.kv_cache_manager.block_pool.hash_block_size
        num_prefix_hashes = seq_len // hash_block_size
        return {
            "request_id": normalized_id,
            "exported": True,
            "active_request": True,
            "seq_len": seq_len,
            "kv_transfer_params": {},
            "prefix_token_ids": list(request.all_token_ids[:seq_len]),
            "prefix_block_hashes": [
                bytes(block_hash).hex()
                for block_hash in request.block_hashes[:num_prefix_hashes]
            ],
        }

    @staticmethod
    def submit_kv_migration(
        scheduler: Any,
        migration: Mapping[str, Any],
    ) -> Future[dict[str, Any]]:
        """Enqueue a migration and notify the caller at terminal state."""
        validate_pap_migration_tp_size(
            scheduler.vllm_config.parallel_config.tensor_parallel_size
        )
        submitted = scheduler.pap_scheduler.submit_migration(
            request_id=str(migration["request_id"]),
            source_kv_params=dict(migration["source_kv_params"]),
            prefix_len=int(migration["prefix_len"]),
            prefix_token_ids=tuple(
                int(token) for token in migration["prefix_token_ids"]
            ),
            prefix_block_hashes=tuple(
                bytes.fromhex(block_hash)
                for block_hash in migration["prefix_block_hashes"]
            ),
            decode_capacity_tokens=int(migration["decode_capacity_tokens"]),
            session_handle=str(migration["session_handle"]),
            attention_tcp_endpoint=str(migration["attention_tcp_endpoint"]),
        )
        return scheduler.pap_scheduler.migration_started(submitted["job_id"])

    @staticmethod
    def kv_migration_status(
        scheduler: Any,
        job_id: str,
    ) -> dict[str, Any]:
        """Return one target PA migration job state."""
        return scheduler.pap_scheduler.migration_status(str(job_id))
