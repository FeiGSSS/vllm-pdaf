# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from vllm.pap.integration import (
    PAPAcceptedDecodeTokenPublisher,
)
from vllm.pap.kv.registry import PAPAttentionRegistry
from vllm.pap.lifecycle.decode_token import (
    DeferredDecodeCommit,
    DeferredDecodeTokenCommitter,
)
from vllm.pap.lifecycle.decode_token_client import DecodeTokenClient
from vllm.pap.protocol import PAPAttentionRegistration


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
    assert dispatched == []
    assert committer.flush_request("req-a")
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
    assert dispatched == []
    assert committer.flush_request("req-a")
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
    assert committer.flush_request("req-a")
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

    assert (
        committer.record_token(
            request_id="req-a",
            new_seq_len=17,
            token_ids=(42,),
        )
        == "matched"
    )

    flush_results: list[bool] = []
    flush_thread = threading.Thread(
        target=lambda: flush_results.append(
            committer.flush_request("req-a", timeout_s=1.0)
        )
    )
    flush_thread.start()
    assert dispatch_started.wait(timeout=1.0)
    time.sleep(0.02)
    assert flush_thread.is_alive()

    release_dispatch.set()
    flush_thread.join(timeout=1.0)
    assert flush_results == [True]


def test_deferred_decode_token_flush_combines_contiguous_positions() -> None:
    dispatched: list[DeferredDecodeCommit] = []
    committer = DeferredDecodeTokenCommitter(dispatched.append)

    for new_seq_len, token_id in ((17, 42), (18, 43), (19, 44)):
        assert (
            committer.record_token(
                request_id="req-a",
                new_seq_len=new_seq_len,
                token_ids=(token_id,),
            )
            == "pending"
        )
        assert (
            committer.record_kv_ready(
                request_id="req-a",
                new_seq_len=new_seq_len,
                endpoint="http://127.0.0.1:8100/commit",
            )
            == "matched"
        )

    assert dispatched == []
    assert committer.flush_request("req-a")
    assert dispatched == [
        DeferredDecodeCommit(
            request_id="req-a",
            new_seq_len=19,
            token_ids=(42, 43, 44),
            endpoint="http://127.0.0.1:8100/commit",
        )
    ]


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


def test_attention_kv_ready_ignores_released_session(monkeypatch) -> None:
    class CommitClient:
        @staticmethod
        def flush_submitted_request(_request_id: str) -> bool:
            return True

        flush_request = flush_submitted_request

        @staticmethod
        def forget_request(_request_id: str) -> None:
            pass

    monkeypatch.setattr(
        "vllm.pap.kv.session_registry._get_commit_client",
        lambda: CommitClient(),
    )
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            prefill_endpoint="http://127.0.0.1:8100",
            q_size=4096,
            kv_size=1024,
        )
    )
    entry = registry.offload_exec_batch_session_entries(
        ("req-a",),
        default_q_size=4096,
        default_kv_size=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )[0]

    assert registry.release_session("req-a")
    assert (
        registry.record_decode_kv_ready(
            request_id="req-a",
            session_epoch=entry.session_epoch,
            new_seq_len=17,
            endpoint="http://127.0.0.1:8100/commit",
        )
        == "released"
    )
    stats = registry.decode_token_stats()
    assert stats["decode_kv_ready"] == 0
    assert stats["decode_token_pending_kv"] == 0


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
        {"tokens": [{"request_id": "req-a", "new_seq_len": 17, "token_id": 42}]},
        {"tokens": [{"request_id": "req-a", "new_seq_len": 17, "token_id": 42}]},
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


def test_accepted_decode_token_publisher_uses_gpu_frame_sequence_key() -> None:
    published: list[tuple[dict[str, object], ...]] = []

    class FakeClient:
        def publish_batch(self, tokens) -> None:
            published.append(tuple(tokens))

        def shutdown(self) -> None:
            return None

    request = SimpleNamespace(
        request_id="projection-a",
        kv_transfer_params={
            "pap_projection_kv_unaware": True,
            "pap_prefill_kv_handle": "prefill-a",
            "pap_attention_endpoint": "http://127.0.0.1:8300",
        },
    )
    publisher = PAPAcceptedDecodeTokenPublisher(client=FakeClient())
    notification = publisher.build_notification(request, (42,), 17)

    assert notification == {
        "request_id": "prefill-a",
        "new_seq_len": 17,
        "token_id": 42,
        "endpoint": "http://127.0.0.1:8300",
    }
    publisher.publish_batch((notification,))
    publisher.shutdown()

    assert published == [(notification,)]

    request.kv_transfer_params = {
        "pap_projection_kv_unaware": True,
        "pap_attention_endpoint": "http://127.0.0.1:8300",
    }
    with pytest.raises(RuntimeError, match="missing routing metadata"):
        publisher.build_notification(request, (42,), 17)

    request.kv_transfer_params = None
    assert publisher.build_notification(request, (42,), 17) is None

    request.kv_transfer_params = {
        "pap_projection_kv_unaware": True,
        "pap_prefill_kv_handle": "prefill-a",
        "pap_attention_endpoint": "http://127.0.0.1:8300",
    }
    with pytest.raises(RuntimeError, match="GPU-frame sequence key"):
        publisher.build_notification(request, (42,), None)
