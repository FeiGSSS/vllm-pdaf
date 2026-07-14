# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composition factory for PAP execution transports."""

from vllm.pap.config import (
    PAPOffloadExecTransport as PAPOffloadExecTransportKind,
)
from vllm.pap.config import parse_offload_exec_transport
from vllm.pap.protocol import (
    PAPOffloadExecTransport as PAPOffloadExecTransportBackend,
)
from vllm.pap.transport.local_fast import (
    build_local_fast_offload_exec_transport,
)
from vllm.pap.transport.nixl_offload import (
    build_nixl_mailbox_offload_exec_transport,
)


def build_offload_exec_transport(
    *,
    transport: PAPOffloadExecTransportKind | str,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
) -> PAPOffloadExecTransportBackend:
    """Build the configured Projection-to-Attention transport backend.

    Args:
        transport: Canonical transport kind or supported configuration alias.
        actor_id: Stable mailbox actor identifier.
        local_rank: CUDA device rank used by the transport.
        buffer_bytes: Optional transport buffer size override.

    Returns:
        Configured execution transport backend.
    """
    kind = parse_offload_exec_transport(transport)
    if kind is PAPOffloadExecTransportKind.NIXL_MAILBOX:
        return build_nixl_mailbox_offload_exec_transport(
            actor_id=actor_id,
            local_rank=local_rank,
            buffer_bytes=buffer_bytes,
        )
    if kind is PAPOffloadExecTransportKind.LOCAL_FAST:
        return build_local_fast_offload_exec_transport(
            actor_id=actor_id,
            local_rank=local_rank,
            buffer_bytes=buffer_bytes,
        )
    raise AssertionError(f"unsupported PAP OFFLOAD_EXEC transport: {kind}")
