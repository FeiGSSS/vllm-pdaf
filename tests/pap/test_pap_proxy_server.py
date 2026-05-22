import asyncio

from examples.pap.pap_proxy_server import (
    PAPServiceClient,
    build_prefill_payload,
    build_projection_payload,
    register_attention_handle,
)


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


def test_build_projection_payload_attaches_prefill_kv_params() -> None:
    payload = build_projection_payload(
        {"model": "qwen", "prompt": "hello"},
        {
            "remote_engine_id": "prefill-0",
            "remote_block_ids": [[4, 5]],
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
        },
    )

    assert payload["kv_transfer_params"]["remote_engine_id"] == "prefill-0"
    assert payload["kv_transfer_params"]["remote_block_ids"] == [[4, 5]]


class FakeAsyncClient:
    def __init__(self) -> None:
        self.posts = []

    async def post(self, endpoint, json, headers):
        self.posts.append((endpoint, json, headers))
        return FakeResponse({"ok": True})


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
