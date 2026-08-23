# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP request gateway and role orchestration."""

from vllm.pap.gateway.clients import PAPServiceClient
from vllm.pap.gateway.payloads import (
    attach_pap_prefill_attention_params,
    build_prefill_payload,
    build_projection_kv_unaware_payload,
)

__all__ = [
    "PAPServiceClient",
    "attach_pap_prefill_attention_params",
    "build_prefill_payload",
    "build_projection_kv_unaware_payload",
]
