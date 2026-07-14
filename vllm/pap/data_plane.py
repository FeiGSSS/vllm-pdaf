# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility facade for PAP protocol, routing, and transport modules."""

from vllm.pap.protocol import (
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
from vllm.pap.protocol.offload_exec import (
    _offload_exec_batch_descriptor_from_metadata,
    _offload_exec_batch_descriptor_from_plan_payload,
    _offload_exec_batch_descriptor_to_metadata,
    _offload_exec_batch_descriptor_to_plan_metadata,
    _offload_exec_batch_plan_id,
    _offload_exec_batch_plan_payload,
    _offload_exec_descriptor_to_metadata,
)
from vllm.pap.topology.routing import (
    build_offload_exec_route_groups,
    filter_offload_exec_route_groups_for_request_slice,
)
from vllm.pap.transport.local_fast import build_local_fast_offload_exec_transport
from vllm.pap.transport.nixl_offload import (
    PAPNixlMailboxOffloadExecTransport,
    build_nixl_mailbox_offload_exec_transport,
)

__all__ = [
    "PAPCudaIPCTensorHandle",
    "PAPDataPlaneChannel",
    "PAPDataPlaneRole",
    "PAPNixlMailboxOffloadExecTransport",
    "PAPOffloadExecBatchDescriptor",
    "PAPOffloadExecDescriptor",
    "PAPOffloadExecTransport",
    "PAPPrefillKVCacheCatalogDescriptor",
    "PAPPrefillKVSessionManifest",
    "PAPTensorTransport",
    "build_local_fast_offload_exec_transport",
    "build_nixl_mailbox_offload_exec_transport",
    "build_offload_exec_route_groups",
    "filter_offload_exec_route_groups_for_request_slice",
    "pap_offload_exec_trace_id",
]
