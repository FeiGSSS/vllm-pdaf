# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PAP gateway client and handoff tests."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from vllm.pap.gateway.clients import (
    PAPServiceClient,
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    register_attention_handle,
    wait_attention_prefill_ready,
)
from vllm.pap.gateway.payloads import (
    attach_pap_prefill_attention_params,
    build_prefill_payload,
)
from vllm.pap.gateway.payloads import (
    build_projection_kv_unaware_payload as build_projection_payload,
)


def test_prefill_payload_can_attach_attention_import_params() -> None:
    payload = attach_pap_prefill_attention_params(
        build_prefill_payload({"model": "qwen", "prompt": "hello"}),
        pap_attention_endpoint="http://127.0.0.1:8300",
        pap_attention_tcp_endpoint="tcp://127.0.0.1:9300",
        pap_prefill_kv_handle="req-7",
        pap_mode="pap",
    )

    assert payload["kv_transfer_params"]["pap_attention_endpoint"] == (
        "http://127.0.0.1:8300"
    )
    assert payload["kv_transfer_params"]["pap_attention_tcp_endpoint"] == (
        "tcp://127.0.0.1:9300"
    )
    assert payload["kv_transfer_params"]["pap_prefill_kv_handle"] == "req-7"
    assert payload["kv_transfer_params"]["pap_mode"] == "pap"


def test_build_projection_payload_strips_prefill_kv_transport() -> None:
    payload = build_projection_payload(
        {"model": "qwen", "prompt": "hello"},
        {
            "remote_engine_id": "prefill-0",
            "remote_block_ids": [[4, 5]],
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
            "remote_num_tokens": 11,
        },
        prompt_token_ids=list(range(11)),
    )

    assert payload["kv_transfer_params"]["pap_projection_kv_unaware"] is True
    assert payload["kv_transfer_params"]["pap_remote_prefix_len"] == 11
    assert "remote_engine_id" not in payload["kv_transfer_params"]
    assert "remote_block_ids" not in payload["kv_transfer_params"]


def test_build_projection_payload_for_pap_is_kv_unaware() -> None:
    payload = build_projection_payload(
        {"model": "qwen", "prompt": "hello"},
        {
            "remote_engine_id": "prefill-0",
            "remote_request_id": "prefill-req",
            "remote_block_ids": [[4, 5]],
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
            "remote_num_tokens": 17,
        },
        prompt_token_ids=list(range(17)),
        pap_prefill_kv_handle="req-7",
        pap_attention_kv_installed=True,
    )

    kv_params = payload["kv_transfer_params"]
    assert kv_params["pap_projection_kv_unaware"] is True
    assert kv_params["pap_remote_prefix_len"] == 17
    assert kv_params["pap_prefill_kv_handle"] == "req-7"
    assert kv_params["pap_attention_kv_installed"] is True
    for key in (
        "remote_engine_id",
        "remote_request_id",
        "remote_block_ids",
        "remote_host",
        "remote_port",
    ):
        assert key not in kv_params


def test_prefill_kv_handle_prefers_remote_request_id() -> None:
    assert (
        prefill_kv_handle_from_kv_params(
            {
                "remote_request_id": "cmpl-prefill-0-cafebabe",
                "pap_prefill_kv_handle": "external",
            },
            fallback="fallback",
        )
        == "cmpl-prefill-0-cafebabe"
    )


def test_prefill_kv_handle_falls_back_to_attention_session() -> None:
    assert prefill_kv_handle_from_kv_params({}, fallback="external") == "external"


@pytest.mark.parametrize("timeout", [None, float("nan"), float("inf"), -1.0])
def test_readiness_rejects_invalid_timeout_before_http(monkeypatch, timeout):
    monkeypatch.setenv("PAP_ATTENTION_PREFILL_READY_TIMEOUT", "nan")

    def unexpected_http(*args, **kwargs):
        raise AssertionError("invalid timeout must not reach HTTP")

    attention = SimpleNamespace(client=SimpleNamespace(get=unexpected_http))
    with pytest.raises(ValueError, match="finite|non-negative"):
        asyncio.run(wait_attention_prefill_ready(attention, "req", timeout_s=timeout))


def test_wait_attention_prefill_ready_uses_one_long_poll() -> None:
    class FakeClient:
        def __init__(self):
            self.paths = []
            self.params = []

        async def get(self, path, headers=None, params=None):
            self.paths.append(path)
            self.params.append(params)
            return httpx.Response(
                200,
                json={
                    "request_id": "req-1",
                    "session_handle": "session-1",
                    "ready_prefix_len": 17,
                    "ready": True,
                    "failed": False,
                    "timed_out": False,
                },
                request=httpx.Request("GET", f"http://testserver{path}"),
            )

    fake = FakeClient()
    attention = PAPServiceClient(
        client=fake,
        host="127.0.0.1",
        port=8300,
        base_url="http://127.0.0.1:8300",
        role="attention",
    )

    ready = asyncio.run(
        wait_attention_prefill_ready(
            attention,
            "req-1",
            expected_prefix_len=17,
            expected_session_handle="session-1",
            timeout_s=0.5,
        )
    )

    assert ready is True
    assert fake.paths == ["/v1/pap/attention/sessions/req-1/prefill-readiness"]
    assert fake.params == [
        {
            "expected_prefix_len": 17,
            "expected_session_handle": "session-1",
            "timeout_s": 0.5,
        }
    ]


def test_projection_payload_does_not_claim_attention_kv_by_default() -> None:
    kv_params = {"remote_engine_id": "prefill-0"}
    payload = build_projection_payload(
        {"model": "qwen", "prompt": "hello"},
        kv_params,
        prompt_token_ids=[1],
        pap_prefill_kv_handle="req-7",
    )

    assert payload["kv_transfer_params"]["pap_prefill_kv_handle"] == "req-7"
    assert "pap_attention_kv_installed" not in payload["kv_transfer_params"]
    assert "pap_prefill_kv_handle" not in kv_params


def test_prefill_prefix_len_from_kv_params_uses_remote_num_tokens() -> None:
    assert prefill_prefix_len_from_kv_params({"remote_num_tokens": 17}) == 17
    assert prefill_prefix_len_from_kv_params({"remote_num_tokens": "19"}) == 19
    assert prefill_prefix_len_from_kv_params({}) is None
    assert prefill_prefix_len_from_kv_params({"remote_num_tokens": 0}) is None


def test_prefill_prefix_len_from_kv_params_rejects_invalid_values() -> None:
    try:
        prefill_prefix_len_from_kv_params({"remote_num_tokens": "not-an-int"})
    except ValueError as exc:
        assert "remote_num_tokens" in str(exc)
    else:
        raise AssertionError("invalid remote_num_tokens should raise ValueError")


class FakeAsyncClient:
    def __init__(self) -> None:
        self.posts: list[tuple[object, object, object]] = []
        self.gets: list[tuple[object, object]] = []

    async def post(self, endpoint, json, headers):
        self.posts.append((endpoint, json, headers))
        return FakeResponse({"ok": True})

    async def get(self, endpoint, headers):
        self.gets.append((endpoint, headers))
        return FakeResponse(
            {
                "session_id": "req-3",
                "seq_len": 9,
                "ready": True,
                "layers": {},
            }
        )


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._body


def test_register_attention_handle_posts_internal_registration() -> None:
    fake_http = FakeAsyncClient()
    client = PAPServiceClient(
        client=fake_http,
        host="localhost",
        port=8300,
        base_url="http://localhost:8300",
        role="attention",
    )

    asyncio.run(
        register_attention_handle(
            client,
            request_id="req-3",
            conversation_id="conv-3",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={"remote_engine_id": "prefill-0"},
            prefix_len=9,
        )
    )

    assert fake_http.posts == [
        (
            "/v1/pap/attention/register",
            {
                "request_id": "req-3",
                "conversation_id": "conv-3",
                "prefill_endpoint": "http://localhost:8100",
                "kv_transfer_params": {"remote_engine_id": "prefill-0"},
                "prefix_len": 9,
            },
            {},
        )
    ]


def test_register_attention_handle_retries_transport_error() -> None:
    class FlakyAsyncClient(FakeAsyncClient):
        async def post(self, endpoint, json, headers):
            self.posts.append((endpoint, json, headers))
            if len(self.posts) == 1:
                request = httpx.Request("POST", "http://localhost:8300")
                raise httpx.ReadError("stale keep-alive", request=request)
            return FakeResponse({"ok": True})

    fake_http = FlakyAsyncClient()
    client = PAPServiceClient(
        client=fake_http,
        host="localhost",
        port=8300,
        base_url="http://localhost:8300",
        role="attention",
    )

    result = asyncio.run(
        register_attention_handle(
            client,
            request_id="req-retry",
            conversation_id="conv-retry",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
            prefix_len=None,
        )
    )

    assert result == {"ok": True}
    assert len(fake_http.posts) == 2
