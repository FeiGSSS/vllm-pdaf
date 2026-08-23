# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish scheduler-accepted PAP sampled tokens asynchronously."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.lifecycle.decode_token_client import DecodeTokenClient


class _DecodeTokenClient(Protocol):
    def publish_batch(self, tokens: Sequence[Mapping[str, object]]) -> None: ...

    def shutdown(self) -> None: ...


class _SchedulerRequest(Protocol):
    request_id: str
    kv_transfer_params: dict[str, Any] | None


@dataclass(slots=True)
class PAPAcceptedDecodeTokenPublisher:
    """Deliver only tokens accepted by the Projection scheduler."""

    client: _DecodeTokenClient | None = None

    def build_notification(
        self,
        request: _SchedulerRequest,
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
