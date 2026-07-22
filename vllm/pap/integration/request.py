# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed PAP request metadata and Projection-side request state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _optional_text(value: object) -> str | None:
    return str(value) if value else None


def _optional_positive_int(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class PAPRequestMetadata:
    """PAP fields carried through vLLM ``kv_transfer_params``."""

    projection_kv_unaware: bool = False
    attention_tcp_endpoint: str | None = None
    attention_endpoint: str | None = None
    offload_exec_zmq_endpoint: str | None = None
    remote_prefix_len: int | None = None
    decode_capacity_tokens: int | None = None
    prefill_kv_handle: str | None = None
    import_prefill_kv_to_attention: bool = False
    attention_kv_installed: bool = False

    @classmethod
    def from_mapping(
        cls,
        params: Mapping[str, Any] | None,
    ) -> PAPRequestMetadata:
        """Parse canonical PAP request fields without changing legacy coercion."""
        if not params:
            return cls()
        remote_prefix_len = params.get("pap_remote_prefix_len")
        if remote_prefix_len is None:
            remote_prefix_len = params.get("remote_num_tokens")
        return cls(
            projection_kv_unaware=bool(params.get("pap_projection_kv_unaware")),
            attention_tcp_endpoint=_optional_text(
                params.get("pap_attention_tcp_endpoint")
            ),
            attention_endpoint=_optional_text(params.get("pap_attention_endpoint")),
            offload_exec_zmq_endpoint=_optional_text(
                params.get("pap_offload_exec_zmq_endpoint")
            ),
            remote_prefix_len=(
                int(remote_prefix_len) if remote_prefix_len is not None else None
            ),
            decode_capacity_tokens=_optional_positive_int(
                params.get("pap_decode_capacity_tokens"),
                name="pap_decode_capacity_tokens",
            ),
            prefill_kv_handle=_optional_text(params.get("pap_prefill_kv_handle")),
            import_prefill_kv_to_attention=bool(
                params.get("pap_import_prefill_kv_to_attention")
            ),
            attention_kv_installed=bool(params.get("pap_attention_kv_installed")),
        )


@dataclass(slots=True)
class PAPProjectionRequestStore:
    """Projection-side PAP state indexed by vLLM request ID."""

    attention_tcp_endpoint_by_request: dict[str, str] = field(default_factory=dict)
    attention_endpoint_by_request: dict[str, str] = field(default_factory=dict)
    offload_exec_zmq_endpoint_by_request: dict[str, str] = field(default_factory=dict)
    prefill_prefix_len_by_request: dict[str, int] = field(default_factory=dict)
    decode_capacity_tokens_by_request: dict[str, int] = field(default_factory=dict)
    prefill_kv_handle_by_request: dict[str, str] = field(default_factory=dict)
    import_prefill_kv_to_attention_requests: set[str] = field(default_factory=set)
    attention_kv_installed_requests: set[str] = field(default_factory=set)

    def update(
        self,
        request_id: str,
        params: Mapping[str, Any] | None,
    ) -> PAPRequestMetadata:
        """Merge one scheduler metadata update into request state."""
        metadata = PAPRequestMetadata.from_mapping(params)
        if metadata.attention_tcp_endpoint is not None:
            self.attention_tcp_endpoint_by_request[request_id] = (
                metadata.attention_tcp_endpoint
            )
        if metadata.attention_endpoint is not None:
            self.attention_endpoint_by_request[request_id] = metadata.attention_endpoint
        if metadata.offload_exec_zmq_endpoint is not None:
            self.offload_exec_zmq_endpoint_by_request[request_id] = (
                metadata.offload_exec_zmq_endpoint
            )
        if metadata.remote_prefix_len is not None:
            self.prefill_prefix_len_by_request[request_id] = metadata.remote_prefix_len
        if metadata.decode_capacity_tokens is not None:
            self.decode_capacity_tokens_by_request[request_id] = (
                metadata.decode_capacity_tokens
            )
        if metadata.prefill_kv_handle is not None:
            self.prefill_kv_handle_by_request[request_id] = metadata.prefill_kv_handle
        if metadata.import_prefill_kv_to_attention:
            self.import_prefill_kv_to_attention_requests.add(request_id)
        if metadata.attention_kv_installed:
            self.attention_kv_installed_requests.add(request_id)
        return metadata

    def remove(self, request_id: str) -> None:
        """Remove every PAP-owned view of one vLLM request."""
        self.attention_tcp_endpoint_by_request.pop(request_id, None)
        self.attention_endpoint_by_request.pop(request_id, None)
        self.offload_exec_zmq_endpoint_by_request.pop(request_id, None)
        self.prefill_prefix_len_by_request.pop(request_id, None)
        self.decode_capacity_tokens_by_request.pop(request_id, None)
        self.prefill_kv_handle_by_request.pop(request_id, None)
        self.import_prefill_kv_to_attention_requests.discard(request_id)
        self.attention_kv_installed_requests.discard(request_id)
