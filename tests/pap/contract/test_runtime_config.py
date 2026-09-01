# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import FrozenInstanceError

import pytest

from vllm.pap.config import (
    PAP_REMOVED_FLAGS,
    PAPAttentionKernelPolicy,
    PAPConfigError,
    PAPMPSMode,
    PAPOffloadKVTransport,
    PAPRoutingPolicy,
    PAPRuntimeConfig,
)


def test_runtime_config_defaults_to_the_single_projection_runtime() -> None:
    config = PAPRuntimeConfig.from_env({})

    assert config.topology.name == "1pa1p"
    assert config.placement.prefill_devices == (0,)
    assert config.placement.attention_devices == (0,)
    assert config.placement.projection_devices == (1,)
    assert config.offload_kv_transport is PAPOffloadKVTransport.CUDA_IPC
    assert config.mps.mode is PAPMPSMode.STATIC
    assert config.attention.actor_id == "attention"
    assert config.attention.kernel_policy is PAPAttentionKernelPolicy.AUTO


def test_runtime_config_accepts_triton_attention_control() -> None:
    config = PAPRuntimeConfig.from_env({"PAP_ATTENTION_KERNEL_POLICY": "triton"})

    assert config.attention.kernel_policy is PAPAttentionKernelPolicy.TRITON


def test_runtime_config_accepts_7pa1p_nvshmem_graph_topology() -> None:
    config = PAPRuntimeConfig.from_env(
        {
            "PAP_TOPOLOGY": "7pa1p",
            "PAP_PREFILL_GPUS": "0,1,2,3,4,5,6",
            "PAP_ATTENTION_GPUS": "0,1,2,3,4,5,6",
            "PAP_PROJECTION_GPUS": "7",
            "PAP_ROUTING_POLICY": "conversation_affinity",
        }
    )

    assert config.topology.pa_count == 7
    assert config.topology.projection_count == 1
    assert config.routing_policy is PAPRoutingPolicy.CONVERSATION_AFFINITY


def test_runtime_config_rejects_multiple_projections() -> None:
    with pytest.raises(PAPConfigError, match="requires one Projection"):
        PAPRuntimeConfig.from_env(
            {
                "PAP_TOPOLOGY": "1pa2p",
                "PAP_PREFILL_GPUS": "0",
                "PAP_ATTENTION_GPUS": "0",
                "PAP_PROJECTION_GPUS": "1,2",
            }
        )


def test_runtime_config_rejects_tensor_parallel_execution() -> None:
    with pytest.raises(PAPConfigError, match="requires TP=1"):
        PAPRuntimeConfig.from_env(
            {
                "PAP_TP_SIZE": "2",
                "PAP_PREFILL_GPUS": "0,1",
                "PAP_ATTENTION_GPUS": "0,1",
                "PAP_PROJECTION_GPUS": "2,3",
            }
        )


@pytest.mark.parametrize(
    "name",
    [
        "PAP_OFFLOAD_EXEC_TRANSPORT",
        "PAP_NVSHMEM_GPU_GRAPH",
        "PAP_NVSHMEM_MODE",
        "PAP_NVSHMEM_SIGNAL_WAIT",
        "PAP_DIRECT_MAILBOX_OUTPUT",
        "PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND",
    ],
)
def test_removed_execution_selectors_fail_closed(name: str) -> None:
    with pytest.raises(PAPConfigError, match=name):
        PAPRuntimeConfig.from_env({name: "1"})


def test_runtime_config_rejects_non_cuda_ipc_kv_transport() -> None:
    with pytest.raises(PAPConfigError, match="must be cuda_ipc"):
        PAPRuntimeConfig.from_env({"PAP_OFFLOAD_KV_TRANSPORT": "nixl_mailbox"})


def test_runtime_config_values_are_immutable() -> None:
    config = PAPRuntimeConfig.from_env({})
    with pytest.raises(FrozenInstanceError):
        config.topology.pa_count = 2  # type: ignore[misc]


def test_removed_flag_registry_has_unique_names() -> None:
    names = [spec.name for spec in PAP_REMOVED_FLAGS]
    assert len(names) == len(set(names))
