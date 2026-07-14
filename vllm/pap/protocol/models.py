# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP service request and registration contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PAPAttentionRegistration(BaseModel):
    """Request metadata registered after Prefill completes."""

    request_id: str
    conversation_id: str = ""
    prefill_endpoint: str
    kv_transfer_params: dict[str, Any] = Field(default_factory=dict)
    prefix_len: int | None = None
    block_size: int = 16
    max_seq_len: int = 32768
    q_size: int | None = None
    kv_size: int | None = None


class PAPOffloadExecMailboxBindRequest(BaseModel):
    """Projection NIXL mailbox metadata used for one-time OFFLOAD_EXEC bind."""

    agent_metadata_b64: str
    source_id: str | None = None


class PAPOffloadExecMailboxActivityRequest(BaseModel):
    """Projection membership update for Attention coalescing."""

    source_id: str
    active: bool
    membership_generation: int = Field(ge=1)


class PAPDecodeTokenRequest(BaseModel):
    """One sampled token copied asynchronously by the Projection worker."""

    request_id: str
    new_seq_len: int = Field(ge=1)
    token_id: int = Field(ge=0)


class PAPDecodeTokenBatchRequest(BaseModel):
    """Sampled tokens produced by one Projection forward."""

    tokens: list[PAPDecodeTokenRequest] = Field(min_length=1)
