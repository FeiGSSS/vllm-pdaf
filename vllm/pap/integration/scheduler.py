# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for vLLM scheduler hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.kv import lease as pap_lease
from vllm.pap.kv.decode_token_client import DecodeTokenClient


class _DecodeTokenClient(Protocol):
    def publish_batch(self, tokens: Sequence[Mapping[str, object]]) -> None: ...

    def shutdown(self) -> None: ...


class _AcceptedTokenRequest(Protocol):
    request_id: str
    kv_transfer_params: dict[str, Any] | None


@dataclass(slots=True)
class PAPAcceptedDecodeTokenPublisher:
    """Deliver only tokens accepted by the Projection scheduler."""

    client: _DecodeTokenClient | None = None

    def build_notification(
        self,
        request: _AcceptedTokenRequest,
        token_ids: Sequence[int],
        new_seq_len: int | None,
    ) -> dict[str, object] | None:
        """Build one notification after scheduler acceptance."""
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if not metadata.projection_kv_unaware or not token_ids:
            return None
        if metadata.prefill_kv_handle is None or metadata.attention_endpoint is None:
            raise RuntimeError(
                "PAP accepted decode-token delivery is missing routing metadata "
                f"for request {request.request_id}"
            )
        if len(token_ids) != 1:
            raise RuntimeError(
                "PAP accepted decode-token delivery requires one token per "
                f"request, got {len(token_ids)} for {request.request_id}"
            )
        if new_seq_len is None:
            raise RuntimeError(
                "PAP accepted decode-token delivery is missing the GPU-frame "
                f"sequence key for request {request.request_id}"
            )
        return {
            "request_id": metadata.prefill_kv_handle,
            "new_seq_len": int(new_seq_len),
            "token_id": int(token_ids[0]),
            "endpoint": metadata.attention_endpoint,
        }

    def publish_batch(
        self,
        notifications: Sequence[Mapping[str, object]],
    ) -> None:
        """Queue one scheduler step without blocking on network progress."""
        if not notifications:
            return
        if self.client is None:
            self.client = DecodeTokenClient()
        self.client.publish_batch(notifications)

    def shutdown(self) -> None:
        """Stop the delivery worker if it was created."""
        if self.client is not None:
            self.client.shutdown()


class _SchedulerRequest(Protocol):
    request_id: str
    kv_transfer_params: dict[str, Any] | None
    num_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class PAPProjectionScheduleState:
    """Scheduler inputs for one metadata-only Projection request."""

    remote_prefix_len: int
    remote_computed_tokens: int
    allocate_external_computed_blocks: bool = False
    allocate_local_slots: bool = False


@dataclass(slots=True)
class PAPSchedulerAdapter:
    """Translate PAP metadata and lease state for the vLLM scheduler."""

    settings: PAPRuntimeSettings
    accepted_token_publisher: PAPAcceptedDecodeTokenPublisher = field(
        default_factory=PAPAcceptedDecodeTokenPublisher
    )

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
        )

    @classmethod
    def validate_projection_admission(
        cls,
        request: _SchedulerRequest,
        *,
        num_speculative_tokens: int,
    ) -> PAPProjectionScheduleState | None:
        """Validate a metadata-only Projection request before it is queued."""
        state = cls.projection_state(request)
        if state is not None and num_speculative_tokens > 0:
            raise ValueError("PAP Projection does not support speculative decoding")
        return state

    def decode_capacity_tokens(self, request: _SchedulerRequest) -> int:
        """Return the local decode reservation for unified Prefill KV."""
        metadata = PAPRequestMetadata.from_mapping(request.kv_transfer_params)
        if not metadata.import_prefill_kv_to_attention:
            return 0
        if metadata.decode_capacity_tokens is not None:
            return metadata.decode_capacity_tokens
        return self.settings.unified_kv_decode_capacity_tokens

    def accepted_decode_token_notification(
        self,
        request: _SchedulerRequest,
        token_ids: Sequence[int],
        new_seq_len: int | None,
    ) -> dict[str, object] | None:
        """Build one sideband item after the scheduler accepts output."""
        return self.accepted_token_publisher.build_notification(
            request,
            token_ids,
            new_seq_len,
        )

    def publish_accepted_decode_tokens(
        self,
        notifications: Sequence[Mapping[str, object]],
    ) -> None:
        """Publish one accepted scheduler batch asynchronously."""
        self.accepted_token_publisher.publish_batch(notifications)

    def shutdown(self) -> None:
        """Stop scheduler-owned PAP background workers."""
        self.accepted_token_publisher.shutdown()

    @staticmethod
    def sweep_expired_leases() -> None:
        """Release expired process-local PAP KV leases."""
        pap_lease.pap_sweep_expired_leases()
