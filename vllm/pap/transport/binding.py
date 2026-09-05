# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVSHMEM transport construction, Projection caching, and peer binding."""

from __future__ import annotations

import hashlib
from functools import cache
from typing import Any

from vllm.pap.transport.nvshmem import (
    PAPNVSHMEMTransport,
    build_nvshmem_offload_exec_transport,
)


def build_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
) -> PAPNVSHMEMTransport:
    """Build the Projection-to-Attention NVSHMEM Graph backend.

    Args:
        actor_id: Stable NVSHMEM actor identifier.
        local_rank: CUDA device rank used by the transport.
        buffer_bytes: Optional transport buffer size override.

    Returns:
        NVSHMEM execution transport backend.
    """
    return build_nvshmem_offload_exec_transport(
        actor_id=actor_id,
        local_rank=local_rank,
        buffer_bytes=buffer_bytes,
    )


@cache
def _pap_cached_offload_exec_transport(attention_endpoint: str):
    from vllm.pap.model.context import pap_tensor_parallel_rank

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
    from vllm.pap.model.context import pap_tensor_parallel_rank

    peer_metadata = bind_offload_exec_nvshmem(
        attention_endpoint=attention_endpoint,
        local_agent_metadata=transport.local_agent_metadata,
        source_id=f"projection-r{pap_tensor_parallel_rank()}",
    )
    transport.bind_peer(peer_metadata)
    transport._pap_nvshmem_bound = True
