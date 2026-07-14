from __future__ import annotations

import tomllib
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vllm.pap.attention_executor import create_app
from vllm.pap.config import (
    PAPAttentionDispatchMode,
    PAPConfigError,
    PAPKVHandoffMode,
    PAPMPSMode,
    PAPOffloadExecTransport,
    PAPOffloadKVTransport,
    PAPRuntimeConfig,
    PAP_RETIRED_FLAGS,
)

ROOT = Path(__file__).resolve().parents[3]

# Config-owned fields copied from the Phase 0 formal run's effective_config.env.
P17_PHASE0_ENV = {
    "TOPOLOGY": "1pa1p",
    "PAP_PREFILL_GPUS": "1",
    "PAP_PROJECTION_GPUS": "2",
    "PAP_ROUTING_POLICY": "round_robin",
    "PAP_OFFLOAD_EXEC_TRANSPORT": "local_fast",
    "PAP_OFFLOAD_KV_TRANSPORT": "cuda_ipc",
    "PAP_BENCH_MPS_PROFILE": "baseline_static_64_28",
    "PAP_MPS_MODE": "static",
    "PAP_PREFILL_MPS_PERCENT": "70",
    "PAP_ATTENTION_MPS_PERCENT": "30",
    "PAP_STATIC_PREFILL_CHUNKS": "16",
    "PAP_STATIC_ATTENTION_CHUNKS": "7",
    "PAP_STATIC_PREFILL_EXPECTED_SMS": "64",
    "PAP_STATIC_ATTENTION_EXPECTED_SMS": "28",
    "PAP_ENABLE_MPS": "1",
    "PAP_ASYNC_DECODE_TOKEN": "1",
    "PAP_PREFILL_KV_ASYNC": "1",
    "PAP_KV_HANDOFF_MODE": "sealed_manifest",
    "PAP_UNIFIED_KV": "1",
    "PAP_BATCHED_ROUTE_COPY": "1",
    "PAP_UNIFIED_MD_FAST_KEY": "1",
    "PAP_DIRECT_MAILBOX_OUTPUT": "1",
    "PAP_LOCAL_FAST_STREAM_ORDERED": "1",
    "PAP_LOCAL_FAST_SLOT_COUNT": "2",
    "PAP_DECODE_SLOT_PLAN_CACHE_LIMIT": "256",
    "PAP_ATTENTION_DISPATCH_MODE": "legacy",
    "PAP_ATTENTION_DISPATCH_QUEUE_SIZE": "0",
    "PAP_ATTENTION_COMBINE_WAIT_US": "0",
    "PAP_ATTENTION_ACTIVE_PEER_TRACKING": "0",
    "PAP_PROJECTION_SYNC_ONLY_BARRIER": "0",
    "PAP_PREFILL_IPC_PROFILE": "0",
    "PAP_PREFILL_TORCH_PROFILE": "0",
    "PAP_DIAG_R1_PROJECTION_GATE_COUNT": "0",
    "PAP_DIAG_R1_COMMIT_GATE_COUNT": "0",
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
    assert config.mps.mode is PAPMPSMode.DYNAMIC
    assert config.features.async_decode_token is True
    assert config.features.async_prefill_kv is False
    assert config.features.kv_handoff_mode is PAPKVHandoffMode.LAYER_DESCRIPTOR
    assert config.features.unified_kv is False
    assert config.features.batched_route_copy is True
    assert config.features.unified_md_fast_key is True
    assert config.features.direct_mailbox_output is False
    assert config.attention.dispatch_mode is PAPAttentionDispatchMode.LEGACY
    assert config.decode_commit.timeout_s == 0.2
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
            "PAP_OFFLOAD_EXEC_TRANSPORT": "nixl",
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


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"PAP_TOPOLOGY": "0pa1p"}, "PAP_TOPOLOGY"),
        ({"PAP_TOPOLOGY": "2pa1p", "PAP_PA_COUNT": "1"}, "disagrees"),
        ({"PAP_ASYNC_DECODE_TOKEN": "sometimes"}, "must be a boolean"),
        ({"PAP_LOCAL_FAST_SLOT_COUNT": "0"}, "must be at least 1"),
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
    config = PAPRuntimeConfig.from_env(P17_PHASE0_ENV)

    with pytest.raises(FrozenInstanceError):
        config.protocol_version = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.topology.pa_count = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.placement.prefill_devices[0] = 0  # type: ignore[index]


def test_retired_flag_registry_is_informational_in_phase_one() -> None:
    config = PAPRuntimeConfig.from_env({})
    environment = {
        "PAP_ASYNC_DECODE_TOKEN": "0",
        "PAP_KV_HANDOFF_MODE": "sealed-manifest",
        "UNRELATED": "1",
    }

    settings = config.configured_retired_flags(environment)

    assert len({spec.name for spec in PAP_RETIRED_FLAGS}) == len(PAP_RETIRED_FLAGS)
    assert [setting.spec.name for setting in settings] == [
        "PAP_ASYNC_DECODE_TOKEN",
        "PAP_KV_HANDOFF_MODE",
    ]
    assert settings[0].matches_p17 is False
    assert settings[1].matches_p17 is True


def test_p17_effective_config_matches_frozen_profile_field_by_field() -> None:
    profile_path = ROOT / "benchmarks" / "pap" / "profiles" / "p17_1pa1p.toml"
    with profile_path.open("rb") as profile_file:
        profile = tomllib.load(profile_file)
    expected = {
        section: profile[section]
        for section in ("topology", "placement", "transport", "mps", "runtime")
    }

    config = PAPRuntimeConfig.from_env(P17_PHASE0_ENV)

    assert config.p17_profile_contract() == expected


def test_attention_composition_uses_injected_config_not_later_env(
    monkeypatch,
) -> None:
    config = PAPRuntimeConfig.from_env(P17_PHASE0_ENV)
    monkeypatch.setenv("PAP_ATTENTION_DISPATCH_MODE", "central_combine")
    monkeypatch.setenv("PAP_ATTENTION_ACTIVE_PEER_TRACKING", "1")

    app = create_app(config=config)

    assert app.state.pap_config is config
    assert app.state.registry.runtime_config is config
    assert app.state.offload_exec_dispatch_mode == "legacy"
    assert app.state.offload_exec_dispatcher is None
