# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-facing PAP adapters."""

from vllm.pap.model.prefill import PAPPrefillKVPublisher
from vllm.pap.model.projection import (
    PAPProjectionAttentionAdapter,
    PAPProjectionAttentionExecution,
    bind_projection_attention_execution,
)

__all__ = [
    "PAPPrefillKVPublisher",
    "PAPProjectionAttentionExecution",
    "PAPProjectionAttentionAdapter",
    "bind_projection_attention_execution",
]
