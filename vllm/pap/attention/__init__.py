# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention execution primitives."""

from vllm.pap.attention.dispatcher import (
    PAPAttentionDispatcher,
    PAPAttentionWorkItem,
)

__all__ = ["PAPAttentionDispatcher", "PAPAttentionWorkItem"]
