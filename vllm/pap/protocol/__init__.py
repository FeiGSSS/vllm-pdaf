# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP protocol contracts."""

from vllm.pap.protocol.descriptors import (
    PAPCudaIPCTensorHandle,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPOffloadExecTransport,
    PAPOffloadExecTransportClosed,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
    PAPTensorTransport,
    pap_offload_exec_trace_id,
)
from vllm.pap.protocol.models import (
    PAPAttentionRegistration,
    PAPDecodeTokenBatchRequest,
    PAPDecodeTokenRequest,
    PAPOffloadExecNVSHMEMBindRequest,
)
from vllm.pap.protocol.wire import (
    deserialize_tensor_bundle,
    serialize_tensor_bundle,
)

__all__ = [
    "PAPAttentionRegistration",
    "PAPDecodeTokenBatchRequest",
    "PAPDecodeTokenRequest",
    "PAPOffloadExecNVSHMEMBindRequest",
    "PAPCudaIPCTensorHandle",
    "PAPOffloadExecBatchDescriptor",
    "PAPOffloadExecDescriptor",
    "PAPOffloadExecTransport",
    "PAPOffloadExecTransportClosed",
    "PAPPrefillKVCacheCatalogDescriptor",
    "PAPPrefillKVSessionManifest",
    "PAPTensorTransport",
    "deserialize_tensor_bundle",
    "pap_offload_exec_trace_id",
    "serialize_tensor_bundle",
]
