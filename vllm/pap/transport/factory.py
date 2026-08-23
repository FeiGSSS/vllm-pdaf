# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composition root for the PAP NVSHMEM Graph transport."""

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
