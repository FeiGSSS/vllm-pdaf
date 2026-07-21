# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM scheduler hooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.lifecycle import lease as pap_lease


class _SchedulerRequest(Protocol):
    request_id: str
    kv_transfer_params: Mapping[str, Any] | None
    num_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class PAPProjectionScheduleState:
    """Scheduler inputs for one metadata-only Projection request."""

    remote_prefix_len: int
    remote_computed_tokens: int
    local_computed_token_offset: int
    allocate_external_computed_blocks: bool = False
    allocate_local_slots: bool = False


@dataclass(frozen=True, slots=True)
class PAPSchedulerAdapter:
    """Translate PAP metadata and lease state for the vLLM scheduler."""

    settings: PAPRuntimeSettings

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PAPSchedulerAdapter:
        """Create one scheduler adapter from process environment settings."""
        return cls(PAPRuntimeSettings.from_environ(environ))

    @staticmethod
    def projection_remote_prefix_len(request: _SchedulerRequest) -> int | None:
        """Validate and return the remote prompt prefix length."""
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if not metadata.projection_kv_unaware:
            return None
        prefix_len = metadata.remote_prefix_len
        if prefix_len is None:
            raise ValueError(
                "PAP KV-unaware Projection request requires pap_remote_prefix_len"
            )
        if prefix_len <= 0:
            raise ValueError(
                "PAP KV-unaware Projection request requires a positive "
                "pap_remote_prefix_len"
            )
        if prefix_len > request.num_prompt_tokens:
            raise ValueError(
                "PAP KV-unaware Projection prefix length cannot exceed prompt length"
            )
        return prefix_len

    @classmethod
    def projection_state(
        cls,
        request: _SchedulerRequest,
    ) -> PAPProjectionScheduleState | None:
        """Build the scheduler view of one Projection request."""
        remote_prefix_len = cls.projection_remote_prefix_len(request)
        if remote_prefix_len is None:
            return None
        remote_computed_tokens = max(remote_prefix_len - 1, 0)
        return PAPProjectionScheduleState(
            remote_prefix_len=remote_prefix_len,
            remote_computed_tokens=remote_computed_tokens,
            local_computed_token_offset=remote_computed_tokens,
        )

    def decode_capacity_tokens(self, request: _SchedulerRequest) -> int:
        """Return the local decode reservation for unified Prefill KV."""
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if not metadata.import_prefill_kv_to_attention:
            return 0
        if metadata.decode_capacity_tokens is not None:
            return metadata.decode_capacity_tokens
        return self.settings.unified_kv_decode_capacity_tokens

    @staticmethod
    def sweep_expired_leases() -> None:
        """Release expired process-local PAP KV leases."""
        pap_lease.pap_sweep_expired_leases()

    @staticmethod
    def defer_leased_blocks(
        *,
        request_id: str,
        pop_blocks: Callable[[], list[Any]],
        free_blocks: Callable[[Any], None],
    ) -> bool:
        """Transfer block-free ownership to an active PAP lease."""
        if not pap_lease.pap_has_active_lease(request_id):
            return False

        lease_id = pap_lease.pap_active_lease_id(request_id)
        blocks = pop_blocks()
        if lease_id is None:
            free_blocks(reversed(blocks))
            return True

        blocks.reverse()
        pap_lease.pap_stash_deferred_blocks(
            lease_id=lease_id,
            blocks=blocks,
            free_callback=free_blocks,
        )
        return True
