import asyncio
from pathlib import Path

import httpx

from examples.pap.pap_proxy_server import (
    PAPServiceClient,
    attach_pap_prefill_attention_params,
    build_prefill_payload,
    build_projection_payload,
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    register_attention_handle,
    wait_attention_prefill_ready,
)

ROOT = Path(__file__).resolve().parents[2]


def test_build_prefill_payload_forces_single_token_non_streaming() -> None:
    payload = build_prefill_payload(
        {
            "model": "qwen",
            "prompt": "hello",
            "stream": True,
            "max_tokens": 32,
            "max_completion_tokens": 32,
            "stream_options": {"include_usage": True},
        }
    )

    assert payload["stream"] is False
    assert payload["max_tokens"] == 1
    assert payload["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    assert "max_completion_tokens" not in payload
    assert "stream_options" not in payload


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
    assert (
        prefill_kv_handle_from_kv_params({}, fallback="external")
        == "external"
    )


def test_wait_attention_prefill_ready_polls_until_layers_ready() -> None:
    class FakeClient:
        def __init__(self):
            self.paths = []
            self.responses = [
                {"request_id": "req-1", "layers": []},
                {
                    "request_id": "req-1",
                    "layers": [{"layer_name": "layer0", "ready": True}],
                },
            ]

        async def get(self, path, headers=None):
            self.paths.append(path)
            return httpx.Response(
                200,
                json=self.responses.pop(0),
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
            timeout_s=0.5,
            poll_interval_s=0.0,
        )
    )

    assert ready is True
    assert fake.paths == [
        "/v1/pap/attention/sessions/req-1/prefill-readiness",
        "/v1/pap/attention/sessions/req-1/prefill-readiness",
    ]


def test_single_proxy_marks_attention_kv_installed_only_after_readiness() -> None:
    text = (ROOT / "examples/pap/pap_proxy_server.py").read_text()

    prefill = text.index("prefill_resp = await _post_json")
    prefix_len = text.index("prefix_len = prefill_prefix_len_from_kv_params")
    readiness = text.index("attention_ready =")
    installed = text.index("pap_attention_kv_installed=attention_ready")
    assert prefill < prefix_len < readiness < installed


def test_single_proxy_has_no_decode_barrier() -> None:
    text = (ROOT / "examples/pap/pap_proxy_server.py").read_text()

    assert "decode_barrier" not in text
    assert "DecodeBarrier" not in text
    assert not (ROOT / "examples/pap/decode_barrier.py").exists()


def test_build_projection_payload_does_not_claim_attention_kv_installed_by_default(
) -> None:
    kv_params = {"remote_engine_id": "prefill-0"}
    payload = build_projection_payload(
        {"model": "qwen", "prompt": "hello"},
        kv_params,
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
        self.posts = []
        self.gets = []

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
