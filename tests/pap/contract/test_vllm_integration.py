from __future__ import annotations

from types import MethodType

import pytest

from vllm.pap.integration import bind_projection_request_store
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

    def flush_request(self, request_id: str) -> bool:
        self._events.append(f"flush:{request_id}")
        return self._flush_succeeds

    def forget_request(self, request_id: str) -> None:
        self._events.append(f"forget:{request_id}")


def _v2_runner_for_removal(
    *,
    flush_succeeds: bool,
) -> tuple[GPUModelRunnerV2, list[str]]:
    events: list[str] = []
    runner = object.__new__(GPUModelRunnerV2)
    store = bind_projection_request_store(runner)
    store.update(
        "req-a",
        {
            "pap_attention_tcp_endpoint": "tcp",
            "pap_attention_endpoint": "http",
            "pap_offload_exec_zmq_endpoint": "zmq",
            "pap_remote_prefix_len": 16,
            "pap_prefill_kv_handle": "session-a",
            "pap_import_prefill_kv_to_attention": True,
            "pap_attention_kv_installed": True,
        },
    )
    runner.pap_decode_token_client = _DecodeTokenClient(
        events,
        flush_succeeds=flush_succeeds,
    )
    runner.model_state = _ModelState(events)
    runner.req_states = _RequestStates()
    return runner, events


@pytest.mark.parametrize("runner_type", [GPUModelRunnerV1, GPUModelRunnerV2])
def test_model_runners_share_typed_projection_request_state(runner_type) -> None:
    runner = object.__new__(runner_type)
    store = bind_projection_request_store(runner)

    runner._add_pap_attention_endpoint(
        "req-a",
        {
            "pap_attention_endpoint": "http://attention",
            "remote_num_tokens": "16",
        },
    )
    runner._add_pap_attention_endpoint(
        "req-a",
        {"pap_prefill_kv_handle": "session-a"},
    )

    assert store.attention_endpoint_by_request == {
        "req-a": "http://attention"
    }
    assert runner.pap_prefill_prefix_len_by_req_id == {"req-a": 16}
    assert runner.pap_prefill_kv_handle_by_req_id == {"req-a": "session-a"}


def test_v2_runner_flushes_decode_tokens_before_removing_request() -> None:
    runner, events = _v2_runner_for_removal(flush_succeeds=True)

    assert runner._remove_request("req-a") is False

    assert events == ["flush:session-a", "forget:session-a", "remove:req-a"]
    assert "req-a" not in runner.pap_prefill_kv_handle_by_req_id


def test_v2_runner_fails_closed_when_decode_token_flush_fails() -> None:
    runner, events = _v2_runner_for_removal(flush_succeeds=False)

    with pytest.raises(RuntimeError, match="delivery failed before request removal"):
        runner._remove_request("req-a")

    assert events == ["flush:session-a"]
    assert runner.pap_prefill_kv_handle_by_req_id == {"req-a": "session-a"}


def test_v1_runner_rejects_pap_without_async_sampled_token_callback() -> None:
    runner = object.__new__(GPUModelRunnerV1)
    runner._pap_enabled_for_request_ids = MethodType(
        lambda _self, _request_ids: True,
        runner,
    )

    with pytest.raises(RuntimeError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
        runner._pap_forward_context_kwargs(
            request_ids=("req-a",),
            num_scheduled_tokens=(1,),
            num_actual_tokens=1,
            positions=object(),
            seq_lens_cpu_upper_bound=(1,),
        )
