# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP tensor transport backends."""

from vllm.pap.protocol import PAPOffloadExecTransport, PAPTensorTransport
from vllm.pap.transport.factory import build_offload_exec_transport
from vllm.pap.transport.local_fast import (
    PAPLocalFastTransport,
    build_local_fast_offload_exec_transport,
)
from vllm.pap.transport.mailbox import PAPMailboxMessage
from vllm.pap.transport.nixl import PAPNixlMailboxEndpoint
from vllm.pap.transport.nixl_offload import (
    PAPNixlMailboxOffloadExecTransport,
    build_nixl_mailbox_offload_exec_transport,
)

__all__ = [
    "PAPLocalFastTransport",
    "PAPNixlMailboxOffloadExecTransport",
    "PAPMailboxMessage",
    "PAPNixlMailboxEndpoint",
    "PAPOffloadExecTransport",
    "PAPTensorTransport",
    "build_offload_exec_transport",
    "build_local_fast_offload_exec_transport",
    "build_nixl_mailbox_offload_exec_transport",
]
