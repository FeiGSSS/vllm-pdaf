# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered PAP control operations owned by an EngineCore instance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.pap.integration.engine import PAPEngineAdapter


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
        raise ValueError(f"unknown PAP control operation: {operation!r}")

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


__all__ = ["PAPEngineControl"]
