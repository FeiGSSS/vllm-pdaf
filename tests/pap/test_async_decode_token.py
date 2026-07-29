from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from vllm.pap.integration import (
    PAPDecodeTokenBridge,
    PAPProjectionRequestStore,
)
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)
from vllm.pap.lifecycle.decode_token_client import DecodeTokenClient
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput


def test_deferred_decode_token_dispatches_after_token_and_kv_are_ready() -> None:
    dispatched: list[DeferredDecodeCommit] = []
    committer = DeferredDecodeTokenCommitter(dispatched.append)

    assert (
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
        == "pending"
    )
    assert dispatched == []

    assert (
        committer.record_kv_ready(
            request_id="req-a",
            new_seq_len=17,
            endpoint="http://127.0.0.1:8100/commit",
        )
        == "matched"
    )
    assert dispatched == [
        DeferredDecodeCommit(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
            endpoint="http://127.0.0.1:8100/commit",
        )
    ]


def test_deferred_decode_token_dispatches_when_token_arrives_last() -> None:
    dispatched: list[DeferredDecodeCommit] = []
    committer = DeferredDecodeTokenCommitter(dispatched.append)

    assert (
        committer.record_kv_ready(
            request_id="req-a",
            new_seq_len=17,
            endpoint="http://127.0.0.1:8100/commit",
        )
        == "pending"
    )
    assert (
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
        == "matched"
    )
    assert len(dispatched) == 1


def test_deferred_decode_token_retries_are_idempotent_and_mismatch_fails() -> None:
    dispatched: list[DeferredDecodeCommit] = []
    committer = DeferredDecodeTokenCommitter(dispatched.append)

    assert (
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
        == "pending"
    )
    assert (
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
        == "duplicate"
    )
    with pytest.raises(ValueError, match="changed token IDs"):
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(43,),
        )

    assert (
        committer.record_kv_ready(
            request_id="req-a",
            new_seq_len=17,
            endpoint="http://127.0.0.1:8100/commit",
        )
        == "matched"
    )
    assert (
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
        == "duplicate"
    )
    assert len(dispatched) == 1


def test_deferred_decode_token_flush_waits_for_token_and_dispatch() -> None:
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()

    def dispatch(_commit: DeferredDecodeCommit) -> None:
        dispatch_started.set()
        assert release_dispatch.wait(timeout=1.0)

    committer = DeferredDecodeTokenCommitter(dispatch)
    committer.record_kv_ready(
        request_id="req-a",
        new_seq_len=17,
        endpoint="http://127.0.0.1:8100/commit",
    )

    token_thread = threading.Thread(
        target=lambda: committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
    )
    token_thread.start()
    assert dispatch_started.wait(timeout=1.0)

    flush_results: list[bool] = []
    flush_thread = threading.Thread(
        target=lambda: flush_results.append(
            committer.flush_request("req-a", timeout_s=1.0)
        )
    )
    flush_thread.start()
    time.sleep(0.02)
    assert flush_thread.is_alive()

    release_dispatch.set()
    token_thread.join(timeout=1.0)
    flush_thread.join(timeout=1.0)
    assert flush_results == [True]


def test_deferred_decode_token_forget_drops_only_unmatched_final_token() -> None:
    committer = DeferredDecodeTokenCommitter(lambda _commit: None)
    committer.record_token(
        request_id="req-a",
        new_seq_len=18,
        token_ids=(99,),
    )

    assert committer.flush_request("req-a", timeout_s=0.0)
    committer.forget_request("req-a")

    assert committer.stats() == {
        "decode_token_received": 1,
        "decode_kv_ready": 0,
        "decode_token_matched": 0,
        "decode_token_duplicates": 0,
        "decode_token_mismatches": 0,
        "decode_token_dispatch_failures": 0,
        "decode_token_pending_tokens": 0,
        "decode_token_pending_kv": 0,
        "decode_token_dispatching": 0,
        "decode_token_only_dropped": 1,
    }


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"status": "accepted"}


def _patch_decode_token_http_client(
    monkeypatch: pytest.MonkeyPatch,
    post,
) -> None:
    class FakeHTTPClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, url, *, json, timeout):
            return post(url, json=json, timeout=timeout)

    monkeypatch.setattr(
        "vllm.pap.lifecycle.decode_token_client.httpx.Client",
        FakeHTTPClient,
    )


def test_decode_token_client_reuses_one_http_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []
    clients_created = 0
    clients_closed = 0

    class FakeHTTPClient:
        def __enter__(self):
            nonlocal clients_created
            clients_created += 1
            return self

        def __exit__(self, *_args) -> None:
            nonlocal clients_closed
            clients_closed += 1

        def post(self, url, *, json, timeout):
            payloads.append({"url": url, "json": dict(json), "timeout": timeout})
            return _FakeResponse()

    def reject_one_shot_post(*_args, **_kwargs):
        raise AssertionError("one-shot httpx.post must not be used")

    monkeypatch.setattr(
        "vllm.pap.lifecycle.decode_token_client.httpx.Client",
        FakeHTTPClient,
    )
    monkeypatch.setattr(
        "vllm.pap.lifecycle.decode_token_client.httpx.post",
        reject_one_shot_post,
    )
    client = DecodeTokenClient(queue_size=8, max_attempts=1)

    for new_seq_len in (17, 18):
        client.publish(
            request_id="req-a",
            new_seq_len=new_seq_len,
            token_id=40 + new_seq_len,
            endpoint="http://127.0.0.1:8300",
        )

    assert client.flush_request("req-a", timeout_s=1.0)
    client.shutdown()

    assert clients_created == 1
    assert clients_closed == 1
    assert len(payloads) == 2


def test_decode_token_client_batches_one_forward_into_one_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, object]] = []

    def fake_post(url, *, json, timeout):
        payloads.append({"url": url, "json": dict(json), "timeout": timeout})
        return _FakeResponse()

    _patch_decode_token_http_client(monkeypatch, fake_post)
    client = DecodeTokenClient(queue_size=8, max_attempts=1)
    client.publish_batch(
        (
            {
                "request_id": "req-a",
                "new_seq_len": 17,
                "token_id": 42,
                "endpoint": "http://127.0.0.1:8300",
            },
            {
                "request_id": "req-b",
                "new_seq_len": 23,
                "token_id": 43,
                "endpoint": "http://127.0.0.1:8300",
            },
        )
    )

    assert client.flush_request("req-a", timeout_s=1.0)
    assert client.flush_request("req-b", timeout_s=1.0)
    client.shutdown()

    assert payloads == [
        {
            "url": "http://127.0.0.1:8300/v1/pap/attention/decode-tokens",
            "json": {
                "tokens": [
                    {"request_id": "req-a", "new_seq_len": 17, "token_id": 42},
                    {"request_id": "req-b", "new_seq_len": 23, "token_id": 43},
                ]
            },
            "timeout": 0.2,
        }
    ]


def test_decode_token_client_publish_is_nonblocking_and_flush_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_started = threading.Event()
    release_post = threading.Event()
    payloads: list[dict[str, object]] = []
    attempts = 0

    def fake_post(url, *, json, timeout):
        nonlocal attempts
        attempts += 1
        payloads.append({"url": url, "json": dict(json), "timeout": timeout})
        post_started.set()
        assert release_post.wait(timeout=1.0)
        if attempts == 1:
            raise RuntimeError("retry me")
        return _FakeResponse()

    _patch_decode_token_http_client(monkeypatch, fake_post)
    client = DecodeTokenClient(
        timeout_s=0.2,
        queue_size=8,
        max_attempts=2,
        retry_initial_s=0.0,
        retry_max_s=0.0,
    )

    client.publish(
        request_id="req-a",
        new_seq_len=17,
        token_id=42,
        endpoint="http://127.0.0.1:8300",
    )
    assert post_started.wait(timeout=1.0)

    flush_results: list[bool] = []
    flush_thread = threading.Thread(
        target=lambda: flush_results.append(
            client.flush_request("req-a", timeout_s=1.0)
        )
    )
    flush_thread.start()
    time.sleep(0.02)
    assert flush_thread.is_alive()

    release_post.set()
    flush_thread.join(timeout=1.0)
    assert flush_results == [True]
    assert [payload["json"] for payload in payloads] == [
        {
            "tokens": [
                {"request_id": "req-a", "new_seq_len": 17, "token_id": 42}
            ]
        },
        {
            "tokens": [
                {"request_id": "req-a", "new_seq_len": 17, "token_id": 42}
            ]
        },
    ]
    assert all(
        payload["url"] == "http://127.0.0.1:8300/v1/pap/attention/decode-tokens"
        for payload in payloads
    )
    client.shutdown()


def test_decode_token_client_queue_full_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_started = threading.Event()
    release_post = threading.Event()

    def fake_post(_url, *, json, timeout):
        del json, timeout
        post_started.set()
        assert release_post.wait(timeout=1.0)
        return _FakeResponse()

    _patch_decode_token_http_client(monkeypatch, fake_post)
    client = DecodeTokenClient(queue_size=1, max_attempts=1)
    client.publish(
        request_id="req-a",
        new_seq_len=17,
        token_id=42,
        endpoint="http://127.0.0.1:8300",
    )
    assert post_started.wait(timeout=1.0)
    client.publish(
        request_id="req-a",
        new_seq_len=18,
        token_id=43,
        endpoint="http://127.0.0.1:8300",
    )

    with pytest.raises(RuntimeError, match="queue is full"):
        client.publish(
            request_id="req-a",
            new_seq_len=19,
            token_id=44,
            endpoint="http://127.0.0.1:8300",
        )

    release_post.set()
    assert client.flush_request("req-a", timeout_s=1.0)
    client.shutdown()


def test_async_output_callback_receives_trimmed_sampled_tokens() -> None:
    order: list[str] = []

    class FakeEvent:
        def synchronize(self) -> None:
            order.append("synchronized")

    model_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
    )
    async_output = AsyncOutput.__new__(AsyncOutput)
    async_output.copy_event = FakeEvent()
    async_output.sampled_token_ids = np.array([[42, -1], [43, 44]])
    async_output.num_sampled_tokens_np = np.array([1, 2])
    async_output.model_runner_output = model_output
    async_output.num_nans = None
    async_output.logprobs_tensors = None
    async_output.prompt_logprobs_dict = {}
    async_output.output_ready_callback = lambda output: order.append(
        f"callback:{output.sampled_token_ids}"
    )

    result = async_output.get_output()

    assert result is model_output
    assert result.sampled_token_ids == [[42], [43, 44]]
    assert order == ["synchronized", "callback:[[42], [43, 44]]"]


def test_decode_token_bridge_uses_captured_route_and_next_seq_len() -> None:
    published: list[tuple[dict[str, object], ...]] = []

    class FakeClient:
        def publish_batch(self, tokens) -> None:
            published.append(tuple(tokens))

        def flush_request(self, _request_id: str) -> bool:
            return True

        def forget_request(self, _request_id: str) -> None:
            return None

        def shutdown(self) -> None:
            return None

    output = ModelRunnerOutput(
        req_ids=["projection-a", "projection-b"],
        req_id_to_index={"projection-a": 0, "projection-b": 1},
        sampled_token_ids=[[42], []],
    )
    store = PAPProjectionRequestStore()
    store.update(
        "projection-a",
        {
            "pap_prefill_kv_handle": "prefill-a",
            "pap_attention_endpoint": "http://127.0.0.1:8300",
        },
    )
    bridge = PAPDecodeTokenBridge(client=FakeClient())
    callback = bridge.build_callback(
        store,
        pap_request_ids=frozenset({"projection-a"}),
        request_ids=("projection-a", "projection-b"),
        seq_lens_cpu_upper_bound=(16, 8),
    )

    assert callback is not None
    output.req_ids[:] = ["reused-slot", "projection-b"]
    callback(output)

    assert published == [
        (
            {
                "request_id": "prefill-a",
                "new_seq_len": 17,
                "token_id": 42,
                "endpoint": "http://127.0.0.1:8300",
            },
        )
    ]

    incomplete_store = PAPProjectionRequestStore()
    incomplete_store.update(
        "projection-a",
        {"pap_attention_endpoint": "http://127.0.0.1:8300"},
    )
    incomplete_callback = bridge.build_callback(
        incomplete_store,
        pap_request_ids=frozenset({"projection-a"}),
        request_ids=("projection-a",),
        seq_lens_cpu_upper_bound=(16,),
    )
    assert incomplete_callback is not None
    incomplete_output = ModelRunnerOutput(
        req_ids=["projection-a"],
        req_id_to_index={"projection-a": 0},
        sampled_token_ids=[[42]],
    )
    with pytest.raises(RuntimeError, match="missing routing metadata"):
        incomplete_callback(incomplete_output)
