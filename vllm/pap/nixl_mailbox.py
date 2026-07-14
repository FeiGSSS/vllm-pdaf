# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the PAP NIXL mailbox transport."""

from vllm.pap.transport.nixl import (
    InProcessPAPMailboxBackend,
    PAPMailboxActor,
    PAPMailboxBackend,
    PAPMailboxDirectSendPayload,
    PAPMailboxMessage,
    PAPNixlMailboxAgentMetadata,
    PAPNixlMailboxEndpoint,
)

__all__ = [
    "InProcessPAPMailboxBackend",
    "PAPMailboxActor",
    "PAPMailboxBackend",
    "PAPMailboxDirectSendPayload",
    "PAPMailboxMessage",
    "PAPNixlMailboxAgentMetadata",
    "PAPNixlMailboxEndpoint",
]
