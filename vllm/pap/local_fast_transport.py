# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the PAP local-fast transport."""

from vllm.pap.transport.local_fast import (
    PAPLocalFastTransport,
    build_local_fast_offload_exec_transport,
)

__all__ = [
    "PAPLocalFastTransport",
    "build_local_fast_offload_exec_transport",
]
