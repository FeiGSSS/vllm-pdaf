# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.layers.attention.execution import (
    register_attention_execution_factory,
    resolve_attention_execution,
)
from vllm.pap.integration import (
    PAPAcceptedDecodeTokenPublisher,
    PAPEngineAdapter,
    PAPModelRunnerAdapter,
    PAPProjectionRequestStore,
    PAPRuntimeSettings,
    PAPSchedulerAdapter,
    PAPWorkerAdapter,
    build_projection_forward_context,
    install_pap_control_routes,
    prepare_pap_projection_chat_input,
    prepare_pap_tokenized_chat_input,
    select_projection_request_ids,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1


class _DecodeTokenClient:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def publish_batch(self, tokens) -> None:
        self._events.append(f"publish:{len(tokens)}")

    def shutdown(self) -> None:
        self._events.append("shutdown")


def _runner_adapter(
    *,
    supports_async_sampled_tokens: bool,
    globally_enabled: bool = False,
) -> PAPModelRunnerAdapter:
    return PAPModelRunnerAdapter(
        globally_enabled=globally_enabled,
        attention_tcp_endpoint=None,
        block_size=16,
        supports_async_sampled_tokens=supports_async_sampled_tokens,
        projection_kv_unaware=True,
        debug_decision=False,
    )


def _scheduler_request(
    params=None,
    *,
    prompt_tokens: int = 10,
):
    return SimpleNamespace(
        request_id="req-a",
        kv_transfer_params=params,
        num_prompt_tokens=prompt_tokens,
    )


def test_scheduler_adapter_owns_projection_metadata_validation() -> None:
    request = _scheduler_request(
        {
            "pap_projection_kv_unaware": True,
            "pap_remote_prefix_len": 10,
            "pap_attention_kv_installed": True,
        }
    )

    state = PAPSchedulerAdapter.projection_state(request)

    assert state is not None
    assert state.remote_prefix_len == 10
    assert state.remote_computed_tokens == 9
    assert state.local_computed_token_offset == 9
    assert not state.allocate_external_computed_blocks
    assert not state.allocate_local_slots

    request.kv_transfer_params = {"pap_projection_kv_unaware": True}
    with pytest.raises(ValueError, match="pap_remote_prefix_len"):
        PAPSchedulerAdapter.projection_state(request)

    request.kv_transfer_params = {
        "pap_projection_kv_unaware": True,
        "pap_remote_prefix_len": 11,
    }
    with pytest.raises(ValueError, match="cannot exceed"):
        PAPSchedulerAdapter.projection_state(request)

    request.kv_transfer_params = None
    assert PAPSchedulerAdapter.projection_state(request) is None


def test_runtime_settings_are_parsed_once_per_owner() -> None:
    settings = PAPRuntimeSettings.from_environ(
        {
            "PAP_PROJECTION_KV_UNAWARE": "true",
            "PAP_PROJECTION_CRITICAL_TRACE": "on",
            "PAP_DEBUG_DECISION": "yes",
            "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "64",
            "PAP_RUNTIME_CUDA_CONTEXT_ROLE": "projection",
        }
    )
    worker = PAPWorkerAdapter(settings)

    assert settings.critical_trace
    assert settings.debug_decision
    assert settings.unified_kv_decode_capacity_tokens == 64
    assert worker.projection_kv_unaware
    assert worker.skip_local_attention_kernel_warmup


def test_projection_allows_vllm_async_scheduling(monkeypatch) -> None:
    monkeypatch.setenv("PAP_PROJECTION_KV_UNAWARE", "1")
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        max_concurrent_batches=2,
    )

    adapter = PAPModelRunnerAdapter.from_vllm_config(
        config,
        supports_async_sampled_tokens=True,
    )

    assert adapter.projection_kv_unaware
    assert adapter.supports_async_sampled_tokens


def test_projection_skips_only_local_attention_kernel_warmup() -> None:
    worker = PAPWorkerAdapter.from_environ(
        {
            "PAP_PROJECTION_KV_UNAWARE": "1",
        }
    )

    assert worker.skip_local_attention_kernel_warmup


def test_generic_attention_execution_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    layer = SimpleNamespace(
        execution_override=lambda *args, **kwargs: calls.append((*args, kwargs)),
        impl=SimpleNamespace(
            forward=lambda *_args, **_kwargs: pytest.fail(
                "the local Attention backend must be bypassed"
            )
        ),
    )
    kv_cache = object()
    metadata = object()
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda _name: (metadata, layer, kv_cache, None),
    )
    query = object()
    key = object()
    value = object()
    output = object()

    attention_module.unified_attention_with_output(
        query,  # type: ignore[arg-type]
        key,  # type: ignore[arg-type]
        value,  # type: ignore[arg-type]
        output,  # type: ignore[arg-type]
        "layer",
    )

    assert calls == [
        (
            layer,
            query,
            key,
            value,
            output,
            kv_cache,
            metadata,
            {"output_scale": None, "output_block_scale": None},
        )
    ]


def test_attention_execution_factory_is_model_independent() -> None:
    selected_attention = object()
    execution = object()

    def factory(attention, _vllm_config):
        return execution if attention is selected_attention else None

    factory_name = "pap-test-vllm-integration-model-independent"
    register_attention_execution_factory(factory_name, factory)

    assert resolve_attention_execution(selected_attention, object()) is execution
    assert resolve_attention_execution(object(), object()) is None


def test_request_decode_capacity_overrides_environment_fallback() -> None:
    scheduler = PAPSchedulerAdapter(
        PAPRuntimeSettings.from_environ({"PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "64"})
    )
    request = _scheduler_request(
        {
            "pap_import_prefill_kv_to_attention": True,
            "pap_decode_capacity_tokens": 24,
        }
    )

    assert scheduler.decode_capacity_tokens(request) == 24

    request.kv_transfer_params.pop("pap_decode_capacity_tokens")
    assert scheduler.decode_capacity_tokens(request) == 64


def test_engine_adapter_recognizes_metadata_only_request() -> None:
    assert PAPEngineAdapter.is_metadata_only_request(
        {"pap_projection_kv_unaware": True}
    )
    assert not PAPEngineAdapter.is_metadata_only_request(None)


def test_projection_chat_admission_reuses_prefill_token_ids() -> None:
    request = SimpleNamespace(
        messages=[{"role": "user", "content": "large prompt"}],
        cache_salt="salt",
        kv_transfer_params={
            "pap_projection_kv_unaware": True,
            "pap_remote_prefix_len": 3,
            "pap_prompt_token_ids": [11, 12, 13],
        },
    )

    conversation, engine_inputs = prepare_pap_projection_chat_input(request)

    assert conversation == request.messages
    assert engine_inputs == [
        {
            "type": "token",
            "prompt_token_ids": [11, 12, 13],
            "cache_salt": "salt",
        }
    ]
    assert "pap_prompt_token_ids" not in request.kv_transfer_params


def test_prefill_chat_admission_reuses_gateway_token_ids() -> None:
    request = SimpleNamespace(
        messages=[{"role": "user", "content": "large prompt"}],
        cache_salt=None,
        kv_transfer_params={
            "pap_tokenized_input": True,
            "pap_prompt_token_ids": [21, 22, 23],
        },
    )

    conversation, engine_inputs = prepare_pap_tokenized_chat_input(request)

    assert conversation == request.messages
    assert engine_inputs == [
        {
            "type": "token",
            "prompt_token_ids": [21, 22, 23],
        }
    ]
    assert request.kv_transfer_params == {}


def test_api_adapter_installs_control_routes_only_for_unified_kv() -> None:
    installed: list[object] = []
    app = SimpleNamespace(include_router=lambda router: installed.append(router))

    assert not install_pap_control_routes(app, {})
    assert install_pap_control_routes(
        app,
        {"PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "64"},
    )
    assert len(installed) == 1


@pytest.mark.parametrize("runner_type", [GPUModelRunnerV1, GPUModelRunnerV2])
def test_model_runners_share_one_pap_adapter_boundary(runner_type) -> None:
    runner = object.__new__(runner_type)
    runner.pap_runner = _runner_adapter(
        supports_async_sampled_tokens=runner_type is GPUModelRunnerV2
    )

    runner.pap_runner.update_request(
        "req-a",
        {
            "pap_attention_endpoint": "http://attention",
            "remote_num_tokens": "16",
        },
    )
    runner.pap_runner.update_request(
        "req-a",
        {"pap_prefill_kv_handle": "session-a"},
    )

    assert runner.pap_runner.store.attention_endpoint_by_request == {
        "req-a": "http://attention"
    }
    assert runner.pap_runner.store.prefill_prefix_len_by_request == {"req-a": 16}
    assert runner.pap_runner.store.prefill_kv_handle_by_request == {
        "req-a": "session-a"
    }


def test_projection_batch_adapter_builds_filtered_forward_context() -> None:
    store = PAPProjectionRequestStore()
    store.update(
        "req-a",
        {
            "pap_attention_tcp_endpoint": "tcp://attention",
            "pap_attention_endpoint": "http://attention",
            "pap_remote_prefix_len": 16,
            "pap_decode_capacity_tokens": 24,
            "pap_prefill_kv_handle": "session-a",
            "pap_import_prefill_kv_to_attention": True,
            "pap_attention_kv_installed": True,
        },
    )
    positions = object()
    assert select_projection_request_ids(
        store,
        ("req-a", "req-b"),
        globally_enabled=False,
    ) == {"req-a"}
    assert select_projection_request_ids(
        store,
        ("req-a", "req-b"),
        globally_enabled=True,
    ) == {"req-a", "req-b"}

    context = build_projection_forward_context(
        store,
        request_ids=("req-a", "req-b"),
        num_scheduled_tokens=(1, 2),
        num_actual_tokens=3,
        positions=positions,
        seq_lens_cpu_upper_bound=(17, 9),
        pap_enabled=True,
        attention_tcp_endpoint=None,
        block_size=16,
        finished_request_ids=("req-z",),
    )

    assert context["pap_positions"] is positions
    assert context["pap_prefill_kv_handle_by_request"] == {"req-a": "session-a"}
    assert context["pap_decode_capacity_tokens_by_request"] == {"req-a": 24}
    assert context["pap_attention_kv_installed_by_request"] == {"req-a"}
    route_group = context["pap_offload_exec_route_groups"][0]
    assert route_group["steps"] == (17,)
    assert route_group["session_request_ids"] == ("session-a",)
    assert route_group["batch_id_suffix"] == "session-a@17"
    assert route_group["metadata_template"] == {
        "r": ("session-a",),
        "s": (17,),
    }
    assert context["pap_finished_request_ids"] == ("req-z",)


def test_v2_runner_groups_decode_requests_by_attention_peer() -> None:
    adapter = _runner_adapter(supports_async_sampled_tokens=True)
    for request_id, attention_endpoint in (
        ("req-a", "http://attention-0"),
        ("req-b", "http://attention-1"),
        ("req-c", "http://attention-0"),
        ("req-d", "http://attention-1"),
    ):
        adapter.store.update(
            request_id,
            {
                "pap_attention_endpoint": attention_endpoint,
                "pap_attention_kv_installed": True,
            },
        )

    grouped = adapter.group_decode_request_ids(
        ("req-a", "req-b", "req-c", "req-d"),
        {
            "req-a": 1,
            "req-b": 1,
            "req-c": 1,
            "req-d": 1,
        },
    )

    assert grouped == ("req-a", "req-c", "req-b", "req-d")
    context = adapter.build_forward_context(
        request_ids=grouped,
        num_scheduled_tokens=(1, 1, 1, 1),
        num_actual_tokens=4,
        positions=object(),
        seq_lens_cpu_upper_bound=(11, 13, 12, 14),
    )
    route_groups = context["pap_offload_exec_route_groups"]
    assert tuple(group["req_indices"] for group in route_groups) == (
        (0, 1),
        (2, 3),
    )
    assert tuple(group["request_ids"] for group in route_groups) == (
        ("req-a", "req-c"),
        ("req-b", "req-d"),
    )


def test_runner_builds_complete_graph_capture_context() -> None:
    adapter = _runner_adapter(supports_async_sampled_tokens=True)
    positions = SimpleNamespace(numel=lambda: 8)

    context = adapter.build_capture_forward_context({"positions": positions})

    assert context["pap_positions"] is positions
    assert context["pap_request_ids"] == ()
    assert context["pap_num_actual_tokens"] == 8
    assert not context["pap_enabled"]


def test_projection_warmup_has_no_network_side_effects() -> None:
    adapter = _runner_adapter(supports_async_sampled_tokens=True)
    positions = SimpleNamespace(numel=lambda: 8)

    prepared = adapter.prepare_model_forward(
        request_ids=("_warmup_0_",),
        num_scheduled_tokens=(8,),
        num_actual_tokens=8,
        positions=positions,  # type: ignore[arg-type]
        seq_lens_cpu_upper_bound=(8,),
        finished_request_ids=(),
        dtype=torch.float16,
        native_cudagraph_mode=CUDAGraphMode.NONE,
    )

    assert prepared.step_preparation is None
    assert prepared.additional_kwargs["pap_enabled"] is False


@pytest.mark.parametrize(
    ("request_ids", "num_scheduled_tokens"),
    [
        (("req-a", "req-b"), {"req-a": 1, "req-b": 2}),
        (("req-a", "req-missing"), {"req-a": 1, "req-missing": 1}),
    ],
)
def test_v2_runner_preserves_order_when_pap_grouping_is_not_safe(
    request_ids,
    num_scheduled_tokens,
) -> None:
    adapter = _runner_adapter(supports_async_sampled_tokens=True)
    adapter.store.update(
        "req-a",
        {
            "pap_attention_endpoint": "http://attention-0",
        },
    )

    assert (
        adapter.group_decode_request_ids(request_ids, num_scheduled_tokens)
        == request_ids
    )


def test_v2_runner_captures_frame_local_pap_decode_sequence_keys() -> None:
    adapter = _runner_adapter(
        supports_async_sampled_tokens=True,
        globally_enabled=False,
    )
    adapter.store.update(
        "req-a",
        {
            "pap_attention_endpoint": "http://attention-0",
            "pap_prefill_kv_handle": "session-a",
        },
    )

    seq_lens = adapter.decode_token_seq_lens(
        ("req-a", "req-b"),
        (16, 32),
    )

    assert seq_lens == {"req-a": 17}


def test_scheduler_publishes_only_accepted_decode_tokens() -> None:
    events: list[str] = []
    adapter = PAPSchedulerAdapter(PAPRuntimeSettings.from_environ({}))
    adapter.accepted_token_publisher = PAPAcceptedDecodeTokenPublisher(
        client=_DecodeTokenClient(events)
    )
    request = _scheduler_request(
        {
            "pap_projection_kv_unaware": True,
            "pap_attention_endpoint": "http://attention",
            "pap_prefill_kv_handle": "session-a",
        },
    )

    notification = adapter.accepted_decode_token_notification(
        request,
        (42,),
        18,
    )
    assert notification == {
        "request_id": "session-a",
        "new_seq_len": 18,
        "token_id": 42,
        "endpoint": "http://attention",
    }
    adapter.publish_accepted_decode_tokens((notification,))

    assert events == ["publish:1"]


def test_v1_runner_rejects_async_pap_decode_token_delivery() -> None:
    adapter = _runner_adapter(
        supports_async_sampled_tokens=False,
        globally_enabled=True,
    )

    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        adapter.build_forward_context(
            request_ids=("req-a",),
            num_scheduled_tokens=(1,),
            num_actual_tokens=1,
            positions=object(),
            seq_lens_cpu_upper_bound=(1,),
        )
