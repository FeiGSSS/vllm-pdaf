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

__all__ = [
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
