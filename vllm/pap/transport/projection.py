# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side OFFLOAD_EXEC transport binding and capability cache."""

from __future__ import annotations

import hashlib
from functools import cache
from typing import Any

from vllm.pap.model.context import pap_tensor_parallel_rank


@cache
def _pap_cached_offload_exec_transport(attention_endpoint: str):
    from vllm.pap.transport.factory import build_offload_exec_transport

    local_rank = pap_tensor_parallel_rank()
    endpoint_hash = hashlib.sha1(attention_endpoint.encode("utf-8")).hexdigest()[:12]
    actor_id = f"projection-r{local_rank}-{endpoint_hash}"
    return build_offload_exec_transport(
        actor_id=actor_id,
        local_rank=local_rank,
    )


def _pap_offload_exec_transport_for_attention_endpoint(
    attention_endpoint: str | None,
):
    return _pap_cached_offload_exec_transport(str(attention_endpoint or ""))


def _pap_bind_offload_exec_nvshmem_peer(
    transport: Any,
    attention_endpoint: str | None,
) -> None:
    if not attention_endpoint:
        raise RuntimeError("PAP NVSHMEM OFFLOAD_EXEC requires pap_attention_endpoint")
    if getattr(transport, "_pap_nvshmem_bound", False):
        return
    from vllm.pap.attention.client import bind_offload_exec_nvshmem

    peer_metadata = bind_offload_exec_nvshmem(
        attention_endpoint=attention_endpoint,
        local_agent_metadata=transport.local_agent_metadata,
        source_id=f"projection-r{pap_tensor_parallel_rank()}",
    )
    transport.bind_peer(peer_metadata)
    transport._pap_nvshmem_bound = True
    transport._pap_nvshmem_bound_attention_endpoint = attention_endpoint
