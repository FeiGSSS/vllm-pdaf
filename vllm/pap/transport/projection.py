# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side OFFLOAD_EXEC transport binding and capability cache."""

from __future__ import annotations

import hashlib
import os
from functools import cache
from typing import Any

from vllm.pap.config import (
    PAP_DEFAULT_OFFLOAD_EXEC_TRANSPORT,
    PAPOffloadExecTransport,
    parse_offload_exec_transport,
)
from vllm.pap.model.context import pap_tensor_parallel_rank
from vllm.pap.protocol import PAPStepPlannedOffloadExecTransport


def _pap_offload_exec_transport_kind() -> PAPOffloadExecTransport:
    return parse_offload_exec_transport(
        os.environ.get(
            "PAP_OFFLOAD_EXEC_TRANSPORT",
            PAP_DEFAULT_OFFLOAD_EXEC_TRANSPORT.value,
        )
    )


@cache
def _pap_cached_offload_exec_transport(attention_endpoint: str):
    from vllm.pap.transport.factory import build_offload_exec_transport

    local_rank = pap_tensor_parallel_rank()
    actor_base = os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection")
    endpoint_hash = hashlib.sha1(attention_endpoint.encode("utf-8")).hexdigest()[:12]
    actor_id = f"{actor_base}-r{local_rank}-{endpoint_hash}"
    return build_offload_exec_transport(
        transport=_pap_offload_exec_transport_kind(),
        actor_id=actor_id,
        local_rank=local_rank,
    )


@cache
def _pap_cached_step_planned_transport(
    attention_endpoint: str,
) -> PAPStepPlannedOffloadExecTransport | None:
    """Resolve the optional local capability once per bound endpoint."""
    transport = _pap_cached_offload_exec_transport(attention_endpoint)
    if isinstance(transport, PAPStepPlannedOffloadExecTransport):
        return transport
    return None


def _pap_offload_exec_transport_for_attention_endpoint(
    attention_endpoint: str | None,
    offload_exec_zmq_endpoint: str | None = None,
):
    del offload_exec_zmq_endpoint
    return _pap_cached_offload_exec_transport(str(attention_endpoint or ""))


def _pap_bind_offload_exec_mailbox_peer(
    transport: Any,
    attention_endpoint: str | None,
) -> None:
    if not attention_endpoint:
        raise RuntimeError(
            "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
        )
    if getattr(transport, "_pap_mailbox_bound", False):
        return
    from vllm.pap.attention.client import bind_offload_exec_mailbox

    peer_metadata = bind_offload_exec_mailbox(
        attention_endpoint=attention_endpoint,
        local_agent_metadata=transport.local_agent_metadata,
        source_id=(
            f"{os.environ.get('PAP_NIXL_MAILBOX_ACTOR_ID', 'projection')}"
            f"-r{pap_tensor_parallel_rank()}"
        ),
    )
    transport.bind_peer(peer_metadata)
    transport._pap_mailbox_bound = True
    transport._pap_mailbox_bound_attention_endpoint = attention_endpoint
