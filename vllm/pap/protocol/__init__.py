# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP protocol contracts."""

from vllm.pap.protocol.descriptors import (
    PAPCudaIPCTensorHandle,
    PAPDataPlaneChannel,
    PAPDataPlaneRole,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPOffloadExecTransport,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
    PAPTensorTransport,
    pap_offload_exec_trace_id,
)
from vllm.pap.protocol.models import (
    PAPAttentionRegistration,
    PAPDecodeTokenBatchRequest,
    PAPDecodeTokenRequest,
    PAPOffloadExecMailboxActivityRequest,
    PAPOffloadExecMailboxBindRequest,
)

__all__ = [
    "PAPAttentionRegistration",
    "PAPDecodeTokenBatchRequest",
    "PAPDecodeTokenRequest",
    "PAPOffloadExecMailboxActivityRequest",
    "PAPOffloadExecMailboxBindRequest",
    "PAPCudaIPCTensorHandle",
    "PAPDataPlaneChannel",
    "PAPDataPlaneRole",
    "PAPOffloadExecBatchDescriptor",
    "PAPOffloadExecDescriptor",
    "PAPOffloadExecTransport",
    "PAPPrefillKVCacheCatalogDescriptor",
    "PAPPrefillKVSessionManifest",
    "PAPTensorTransport",
    "pap_offload_exec_trace_id",
]
