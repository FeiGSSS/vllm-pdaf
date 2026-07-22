# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP topology, routing, and peer membership."""

from vllm.pap.topology.peer_activity import (
    PAPProjectionPeerActivity,
    active_pap_attention_endpoints,
    pap_attention_endpoint_for_rank,
    sync_pap_projection_peer_activity,
)
from vllm.pap.topology.routing import build_offload_exec_route_groups

__all__ = [
    "PAPProjectionPeerActivity",
    "active_pap_attention_endpoints",
    "build_offload_exec_route_groups",
    "pap_attention_endpoint_for_rank",
    "sync_pap_projection_peer_activity",
]
