from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.pap.integration import (
    PAPDecodeTokenBridge,
    PAPEngineAdapter,
    PAPModelRunnerAdapter,
    PAPRuntimeSettings,
    PAPSchedulerAdapter,
    PAPWorkerAdapter,
    bind_projection_request_store,
    build_projection_forward_context,
    install_pap_control_routes,
    select_projection_request_ids,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1


class _RequestStates:
    def remove_request(self, _request_id: str) -> None:
        return None


class _ModelState:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def remove_request(self, request_id: str) -> None:
        self._events.append(f"remove:{request_id}")


class _DecodeTokenClient:
    def __init__(self, events: list[str], *, flush_succeeds: bool) -> None:
        self._events = events
        self._flush_succeeds = flush_succeeds

    def publish_batch(self, _tokens) -> None:
        raise AssertionError("not used by request-removal tests")

    def flush_request(self, request_id: str) -> bool:
        self._events.append(f"flush:{request_id}")
        return self._flush_succeeds

    def forget_request(self, request_id: str) -> None:
        self._events.append(f"forget:{request_id}")

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


def _scheduler_request(params=None, *, prompt_tokens: int = 10):
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
            "PAP_RUNNER_MICROBATCH_COUNT": "4",
            "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "64",
            "PAP_RUNTIME_CUDA_CONTEXT_ROLE": "projection",
        }
    )
    worker = PAPWorkerAdapter(settings)

    assert settings.critical_trace
    assert settings.debug_decision
    assert settings.unified_kv_decode_capacity_tokens == 64
    assert worker.projection_kv_unaware
    assert worker.runner_microbatch_count == 4


def test_request_decode_capacity_overrides_environment_fallback() -> None:
    scheduler = PAPSchedulerAdapter(
        PAPRuntimeSettings.from_environ(
            {"PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "64"}
        )
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


def test_api_adapter_installs_control_routes_only_for_unified_kv() -> None:
    app = SimpleNamespace(include_router=lambda router: installed.append(router))
    installed = []

    assert not install_pap_control_routes(app, {})
    assert install_pap_control_routes(
        app,
        {"PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS": "64"},
    )
    assert len(installed) == 1


def _v2_runner_for_removal(
    *,
    flush_succeeds: bool,
) -> tuple[GPUModelRunnerV2, list[str]]:
    events: list[str] = []
    runner = object.__new__(GPUModelRunnerV2)
    runner.pap_runner = _runner_adapter(supports_async_sampled_tokens=True)
    runner.pap_runner.store.update(
        "req-a",
        {
            "pap_attention_tcp_endpoint": "tcp",
            "pap_attention_endpoint": "http",
            "pap_offload_exec_zmq_endpoint": "zmq",
            "pap_remote_prefix_len": 16,
            "pap_decode_capacity_tokens": 24,
            "pap_prefill_kv_handle": "session-a",
            "pap_import_prefill_kv_to_attention": True,
            "pap_attention_kv_installed": True,
        },
    )
    runner.pap_runner.decode_token_bridge = PAPDecodeTokenBridge(
        client=_DecodeTokenClient(
            events,
            flush_succeeds=flush_succeeds,
        )
    )
    runner.model_state = _ModelState(events)
    runner.req_states = _RequestStates()
    return runner, events


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
    owner = SimpleNamespace()
    store = bind_projection_request_store(owner)
    store.update(
        "req-a",
        {
            "pap_attention_tcp_endpoint": "tcp://attention",
            "pap_attention_endpoint": "http://attention",
            "pap_offload_exec_zmq_endpoint": "tcp://mailbox",
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
    assert context["pap_prefill_kv_handle_by_request"] == {
        "req-a": "session-a"
    }
    assert context["pap_decode_capacity_tokens_by_request"] == {"req-a": 24}
    assert context["pap_attention_kv_installed_by_request"] == {"req-a"}
    assert context["pap_offload_exec_route_groups"][0]["steps"] == (17,)
    assert context["pap_finished_request_ids"] == ("req-z",)


def test_v2_runner_flushes_decode_tokens_before_removing_request() -> None:
    runner, events = _v2_runner_for_removal(flush_succeeds=True)

    assert runner._remove_request("req-a") is False

    assert events == ["flush:session-a", "forget:session-a", "remove:req-a"]
    assert "req-a" not in runner.pap_runner.store.prefill_kv_handle_by_request


def test_v2_runner_fails_closed_when_decode_token_flush_fails() -> None:
    runner, events = _v2_runner_for_removal(flush_succeeds=False)

    with pytest.raises(RuntimeError, match="delivery failed before request removal"):
        runner._remove_request("req-a")

    assert events == ["flush:session-a"]
    assert runner.pap_runner.store.prefill_kv_handle_by_request == {
        "req-a": "session-a"
    }


def test_v1_runner_rejects_pap_without_async_sampled_token_callback() -> None:
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
