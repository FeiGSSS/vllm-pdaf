# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import FrozenInstanceError

import pytest

from vllm.pap.config import (
    PAP_REMOVED_FLAGS,
    PAPAttentionKernelPolicy,
    PAPConfigError,
    PAPRuntimeConfig,
)


def test_runtime_config_defaults_to_the_single_projection_runtime() -> None:
    config = PAPRuntimeConfig.from_env({})

    assert config.topology.name == "1pa1p"
    assert config.placement.prefill_devices == (0,)
    assert config.placement.attention_devices == (0,)
    assert config.placement.projection_devices == (1,)
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
            "PAP_ROUTING_POLICY": "dynamo",
        }
    )

    assert config.topology.pa_count == 7
    assert config.topology.projection_count == 1


def test_runtime_config_accepts_multiple_projections() -> None:
    config = PAPRuntimeConfig.from_env(
        {
            "PAP_TOPOLOGY": "6pa2p",
            "PAP_PREFILL_GPUS": "0,1,2,3,4,5",
            "PAP_ATTENTION_GPUS": "0,1,2,3,4,5",
            "PAP_PROJECTION_GPUS": "6,7",
        }
    )

    assert config.topology.pa_count == 6
    assert config.topology.projection_count == 2


@pytest.mark.parametrize(
    "policy",
    [
        "round_robin",
        "crossbar_round_robin",
        "projection_affinity",
        "projection_sticky",
        "conversation_affinity",
        "session_round_robin",
    ],
)
def test_runtime_config_rejects_retired_routing(policy) -> None:
    with pytest.raises(PAPConfigError, match="PAP_ROUTING_POLICY"):
        PAPRuntimeConfig.from_env({"PAP_ROUTING_POLICY": policy})


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


@pytest.mark.parametrize("name", [spec.name for spec in PAP_REMOVED_FLAGS])
def test_removed_execution_selectors_fail_closed(name: str) -> None:
    with pytest.raises(PAPConfigError, match=name):
        PAPRuntimeConfig.from_env({name: "1"})


def test_delivery_clients_apply_runtime_config_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clients parse their settings through the same validated config sections."""
    from vllm.pap.kv.control_client import DecodeCommitClient, LeaseReleaseClient
    from vllm.pap.kv.decode_token_client import DecodeTokenClient

    for prefix, client in (
        ("PAP_DECODE_COMMIT", DecodeCommitClient),
        ("PAP_LEASE_RELEASE", LeaseReleaseClient),
        ("PAP_DECODE_TOKEN", DecodeTokenClient),
    ):
        monkeypatch.setenv(f"{prefix}_RETRY_INITIAL_SECONDS", "1.0")
        monkeypatch.setenv(f"{prefix}_RETRY_MAX_SECONDS", "0.5")
        with pytest.raises(PAPConfigError, match="retry maximum"):
            client()


def test_runtime_config_values_are_immutable() -> None:
    config = PAPRuntimeConfig.from_env({})
    with pytest.raises(FrozenInstanceError):
        config.topology.pa_count = 2  # type: ignore[misc]


def test_removed_flag_registry_has_unique_names() -> None:
    names = [spec.name for spec in PAP_REMOVED_FLAGS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "extra",
    [
        {"TOPOLOGY": "6pa2p"},
        {"PA_COUNT": "6"},
        {"PROJECTION_COUNT": "2"},
        {"PAP_PA_COUNT": "6"},
        {"PAP_PROJECTION_COUNT": "2"},
    ],
)
def test_replay_rejects_conflicting_topology_aliases(extra) -> None:
    with pytest.raises(PAPConfigError, match="disagrees"):
        PAPRuntimeConfig.from_env({"PAP_TOPOLOGY": "7pa1p", **extra})


def test_replay_accepts_consistent_legacy_topology() -> None:
    config = PAPRuntimeConfig.from_env(
        {"TOPOLOGY": "7PA1P", "PA_COUNT": "7", "PROJECTION_COUNT": "1"}
    )
    assert config.topology.name == "7pa1p"


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_nonfinite_timeout_cannot_disable_deadlines(timeout) -> None:
    with pytest.raises(PAPConfigError, match="must be finite"):
        PAPRuntimeConfig.from_env({"PAP_DECODE_COMMIT_TIMEOUT": timeout})
