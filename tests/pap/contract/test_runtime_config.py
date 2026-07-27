from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm.pap.config import (
    PAP_REMOVED_FLAGS,
    PAPAttentionDispatchMode,
    PAPConfigError,
    PAPMPSMode,
    PAPOffloadExecTransport,
    PAPOffloadKVTransport,
    PAPRoutingPolicy,
    PAPRuntimeConfig,
)
from vllm.pap.service import create_app

# Representative same-host configuration used by composition tests.
CURRENT_RUNTIME_ENV = {
    "TOPOLOGY": "1pa1p",
    "PAP_PREFILL_GPUS": "1",
    "PAP_PROJECTION_GPUS": "2",
    "PAP_ROUTING_POLICY": "round_robin",
    "PAP_OFFLOAD_EXEC_TRANSPORT": "local_fast",
    "PAP_OFFLOAD_KV_TRANSPORT": "cuda_ipc",
    "PAP_DIRECT_MAILBOX_OUTPUT": "1",
    "PAP_DECODE_SLOT_PLAN_CACHE_LIMIT": "256",
    "PAP_ATTENTION_DISPATCH_QUEUE_SIZE": "0",
    "PAP_PREFILL_IPC_PROFILE": "0",
    "PAP_PREFILL_TORCH_PROFILE": "0",
}


def test_runtime_config_preserves_python_defaults() -> None:
    config = PAPRuntimeConfig.from_env({})

    assert config.topology.name == "1pa1p"
    assert config.topology.tensor_parallel_size == 1
    assert config.placement.prefill_devices == (0,)
    assert config.placement.attention_devices == (0,)
    assert config.placement.projection_devices == (1,)
    assert config.offload_exec_transport is PAPOffloadExecTransport.NIXL_MAILBOX
    assert config.offload_kv_transport is PAPOffloadKVTransport.CUDA_IPC
    assert config.same_host is False
    assert config.mps.mode is PAPMPSMode.STATIC
    assert config.mps.profile_id == "static_72_20"
    assert config.features.direct_mailbox_output is False
    assert config.attention.dispatch_mode is PAPAttentionDispatchMode.DIRECT
    assert config.decode_commit.timeout_s == 5.0
    assert config.decode_commit.flush_timeout_s == 15.0
    assert config.decode_token.queue_size == 1024
    assert config.lease_release.max_attempts == 5


def test_runtime_config_supports_arbitrary_xpayp_and_tp() -> None:
    config = PAPRuntimeConfig.from_env(
        {
            "PAP_TOPOLOGY": "3pa2p",
            "PAP_TP_SIZE": "2",
            "PAP_PREFILL_GPUS": "0,1,2,3,4,5",
            "PAP_ATTENTION_GPUS": "0,1,2,3,4,5",
            "PAP_PROJECTION_GPUS": "6,7,8,9",
            "PAP_OFFLOAD_EXEC_TRANSPORT": "nixl_mailbox",
            "PAP_OFFLOAD_KV_TRANSPORT": "nixl",
            "PAP_ROUTING_POLICY": "crossbar_round_robin",
        }
    )

    assert config.topology.name == "3pa2p"
    assert config.topology.prefill_device_count == 6
    assert config.topology.projection_device_count == 4
    assert config.offload_exec_transport is PAPOffloadExecTransport.NIXL_MAILBOX
    assert config.offload_kv_transport is PAPOffloadKVTransport.NIXL_MAILBOX
    assert config.same_host is False
    assert config.attention.dispatch_mode is PAPAttentionDispatchMode.CENTRAL_COMBINE
    assert config.attention.combine_wait_us == 1000.0
    assert config.attention.active_peer_tracking is True


def test_runtime_config_accepts_conversation_affinity() -> None:
    config = PAPRuntimeConfig.from_env({"PAP_ROUTING_POLICY": "conversation_affinity"})

    assert config.routing_policy is PAPRoutingPolicy.CONVERSATION_AFFINITY


def test_runtime_config_accepts_attention_load() -> None:
    config = PAPRuntimeConfig.from_env({"PAP_ROUTING_POLICY": "attention_load"})

    assert config.routing_policy is PAPRoutingPolicy.ATTENTION_LOAD


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"PAP_TOPOLOGY": "0pa1p"}, "PAP_TOPOLOGY"),
        ({"PAP_TOPOLOGY": "2pa1p", "PAP_PA_COUNT": "1"}, "disagrees"),
        ({"PAP_OFFLOAD_EXEC_TRANSPORT": "nccl"}, "nixl_mailbox"),
        (
            {
                "PAP_TOPOLOGY": "2pa1p",
                "PAP_PREFILL_GPUS": "0",
            },
            "fewer devices",
        ),
        (
            {
                "PAP_DECODE_TOKEN_RETRY_INITIAL_SECONDS": "1",
                "PAP_DECODE_TOKEN_RETRY_MAX_SECONDS": "0.5",
            },
            "retry maximum",
        ),
    ],
)
def test_runtime_config_rejects_invalid_values(
    environment: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(PAPConfigError, match=message):
        PAPRuntimeConfig.from_env(environment)


def test_runtime_config_is_deeply_immutable_for_config_values() -> None:
    config = PAPRuntimeConfig.from_env(CURRENT_RUNTIME_ENV)

    with pytest.raises(FrozenInstanceError):
        config.topology.pa_count = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.placement.prefill_devices[0] = 0  # type: ignore[index]


def test_removed_flag_registry_has_unique_names() -> None:
    names = [spec.name for spec in PAP_REMOVED_FLAGS]

    assert len(set(names)) == len(names)


@pytest.mark.parametrize(
    ("name", "experiment_id"),
    [
        (
            "PAP_ASYNC_DECODE_TOKEN",
            "PAP-20260713-ASYNC-DECODE-TOKEN-D2H",
        ),
        (
            "PAP_PREFILL_KV_ASYNC",
            "PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC",
        ),
        ("PAP_KV_HANDOFF_MODE", "PAP-20260714-SEAL-HANDOFF-KV"),
        ("PAP_UNIFIED_KV", "PAP-20260703-UNIFIED-KV"),
        ("PAP_BATCHED_ROUTE_COPY", "PAP-20260711-ROUTE-COPY"),
        ("PAP_UNIFIED_MD_FAST_KEY", "PAP-20260712-METADATA-FAST-KEY"),
        ("PAP_ATTENTION_DISPATCH_MODE", "PAP-20260711-ATTENTION-COMBINE"),
        ("PAP_ATTENTION_COMBINE_WAIT_US", "PAP-20260711-ATTENTION-COMBINE"),
        (
            "PAP_ATTENTION_ACTIVE_PEER_TRACKING",
            "PAP-20260711-ACTIVE-PEER",
        ),
        ("PAP_MPS_MODE", "PAP-20260714-ASYNC-STATIC-BASELINE"),
        (
            "PAP_ASYNC_DECODE_TOKEN_SYNC_ONLY_BARRIER",
            "PAP-20260714-ASYNC-TTFT-ROOTCAUSE",
        ),
        (
            "PAP_PROJECTION_SYNC_ONLY_BARRIER",
            "PAP-20260714-ASYNC-TTFT-ROOTCAUSE",
        ),
        (
            "PAP_PREFILL_SYNC_ONLY_BARRIER",
            "PAP-20260714-ASYNC-TTFT-ROOTCAUSE",
        ),
        (
            "PAP_DIAG_R1_PROJECTION_GATE_COUNT",
            "PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        ),
        (
            "PAP_DIAG_R1_COMMIT_GATE_COUNT",
            "PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        ),
        (
            "PAP_DIAG_DECODE_COMMIT_GATE_FILE",
            "PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        ),
        (
            "PAP_DIAG_DECODE_COMMIT_GATE_TIMEOUT",
            "PAP-20260714-ASYNC-TTFT-STRICT-ISOLATION",
        ),
        (
            "PAP_RUNNER_MICROBATCH_COUNT",
            "PAP-20260724-SINGLE-PROJECTION-BATCH",
        ),
        (
            "PAP_ATTENTION_MAILBOX_PREFETCH",
            "PAP-20260701-ATTENTION-MAILBOX-PREFETCH",
        ),
    ],
)
def test_removed_runtime_flags_fail_closed(
    name: str,
    experiment_id: str,
) -> None:
    with pytest.raises(PAPConfigError) as error:
        PAPRuntimeConfig.from_env({name: "0"})

    message = str(error.value)
    assert name in message
    assert "was removed" in message
    assert experiment_id in message


def test_attention_composition_uses_injected_config_not_later_env(
    monkeypatch,
) -> None:
    config = PAPRuntimeConfig.from_env(CURRENT_RUNTIME_ENV)
    monkeypatch.setenv("PAP_TOPOLOGY", "2pa2p")

    app = create_app(config=config)

    assert app.state.pap_config is config
    assert app.state.registry.runtime_config is config
    assert app.state.pap_peer_manager.dispatch_mode == "direct"
    assert app.state.pap_peer_manager.dispatcher is None
