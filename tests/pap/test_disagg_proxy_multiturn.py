# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial, wraps
from typing import Any

import anyio
import httpx
import pytest

from examples.disaggregated.disaggregated_serving import (
    disagg_proxy_multiturn as proxy,
)

pytestmark = pytest.mark.cpu_test


def _run_async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return anyio.run(partial(function, *args, **kwargs))

    return run


class _Upstream:
    def __init__(
        self,
        responses: list[
            httpx.Response | Callable[[httpx.Request], httpx.Response]
        ],
    ) -> None:
        self._responses = responses
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        response_or_factory = self._responses.pop(0)
        if callable(response_or_factory):
            return response_or_factory(request)
        response_or_factory.request = request
        return response_or_factory


@asynccontextmanager
async def _proxy_client(
    *,
    prefill_responses: list[
        httpx.Response | Callable[[httpx.Request], httpx.Response]
    ],
    decode_responses: list[
        httpx.Response | Callable[[httpx.Request], httpx.Response]
    ],
    reuse_mode: str = "bidirectional",
    require_conversation_id: bool = False,
    cache: proxy.ConversationKVCache | None = None,
) -> AsyncIterator[
    tuple[
        httpx.AsyncClient,
        _Upstream,
        _Upstream,
        proxy.ConversationKVCache,
    ]
]:
    prefill = _Upstream(prefill_responses)
    decode = _Upstream(decode_responses)
    prefill_http = httpx.AsyncClient(
        transport=httpx.MockTransport(prefill),
        base_url="http://prefill.test/v1",
    )
    decode_http = httpx.AsyncClient(
        transport=httpx.MockTransport(decode),
        base_url="http://decode.test/v1",
    )
    cache = cache or proxy.ConversationKVCache(ttl_seconds=450.0)
    state = proxy.app.state
    state.prefill_clients = [
        proxy.ServiceClient(prefill_http, "prefill.test", 8100, 0)
    ]
    state.decode_clients = [proxy.ServiceClient(decode_http, "decode.test", 8200, 0)]
    state.prefill_iter = itertools.cycle([0])
    state.decode_iter = itertools.cycle([0])
    state.reuse_mode = reuse_mode
    state.require_conversation_id = require_conversation_id
    state.kv_cache = cache
    state.stats = proxy.ProxyStats()
    if hasattr(proxy, "kv_cache"):
        proxy.kv_cache = cache

    transport = httpx.ASGITransport(app=proxy.app, raise_app_exceptions=False)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy.test",
        ) as client:
            yield client, prefill, decode, cache
    finally:
        await prefill_http.aclose()
        await decode_http.aclose()


def _prefill_response(
    *,
    prompt: int = 128,
    local: int = 32,
    external: int = 64,
) -> dict:
    return {
        "usage": {
            "prompt_tokens": prompt,
            "prompt_tokens_details": {
                "cached_tokens": local + external,
                "local_cached_tokens": local,
                "external_cached_tokens": external,
            },
        },
        "kv_transfer_params": {
            "remote_engine_id": "prefill-0",
            "remote_block_ids": [[10, 11]],
            "remote_num_tokens": prompt,
        },
    }


def _decode_response() -> dict[str, Any]:
    return {
        "id": "decode-response",
        "object": "chat.completion",
        "created": 123456,
        "model": "served-model",
        "system_fingerprint": "fp-test",
        "prompt_token_ids": [101, 102, 103],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reason",
                },
                "finish_reason": "stop",
                "token_ids": [201, 202],
                "logprobs": {"content": []},
            }
        ],
        "usage": {
            "prompt_tokens": 128,
            "completion_tokens": 2,
            "total_tokens": 130,
            "prompt_tokens_details": {
                "cached_tokens": 96,
                "local_cached_tokens": 32,
                "external_cached_tokens": 64,
            },
        },
        "kv_transfer_params": {
            "remote_engine_id": "decode-0",
            "remote_request_id": "decode-request",
            "remote_block_ids": [[20, 21]],
            "remote_num_tokens": 130,
            "lifecycle": {"lease": "held"},
        },
    }


def _chat_request(*, stream: bool = False, conversation_id: str | None = "conv"):
    request: dict[str, Any] = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "question"}],
        "stream": stream,
        "max_tokens": 8,
    }
    if conversation_id is not None:
        request["conversation_id"] = conversation_id
    return request


def test_cache_peek_is_retry_safe_and_consume_is_single_use() -> None:
    now = 100.0
    cache = proxy.ConversationKVCache(
        ttl_seconds=10.0,
        clock=lambda: now,
    )
    cache.put("conversation", {"remote_block_ids": [[1, 2]]})

    first = cache.peek("conversation")
    retry = cache.peek("conversation")

    assert first.status == "hit"
    assert retry.status == "hit"
    assert first.cache_token == retry.cache_token
    assert cache.size == 1
    assert cache.consume("conversation", first.cache_token) is True
    assert cache.consume("conversation", first.cache_token) is False
    assert cache.size == 0


def test_cache_peek_returns_a_nested_defensive_copy() -> None:
    cache = proxy.ConversationKVCache(ttl_seconds=10.0)
    cache.put(
        "conversation",
        {"remote_block_ids": [[1, 2]], "nested": {"values": [3]}},
    )

    first = cache.peek("conversation")
    assert first.kv_transfer_params is not None
    first.kv_transfer_params["remote_block_ids"][0].append(99)
    first.kv_transfer_params["nested"]["values"].append(99)

    second = cache.peek("conversation")
    assert second.kv_transfer_params == {
        "remote_block_ids": [[1, 2]],
        "nested": {"values": [3]},
    }


def test_cache_peek_evicts_and_reports_expired_entry() -> None:
    now = 100.0
    cache = proxy.ConversationKVCache(
        ttl_seconds=10.0,
        clock=lambda: now,
    )
    cache.put("conversation", {"remote_block_ids": [[1, 2]]})
    now = 111.0

    result = cache.peek("conversation")

    assert result.status == "expired"
    assert result.kv_transfer_params is None
    assert result.cache_token is None
    assert cache.size == 0


def test_prefill_accounting_validates_exclusive_token_buckets() -> None:
    accounting = proxy.validate_prefill_accounting(_prefill_response())

    assert accounting.prompt_tokens == 128
    assert accounting.local_cached_tokens == 32
    assert accounting.external_cached_tokens == 64
    assert accounting.computed_tokens == 32


@pytest.mark.parametrize(
    "prefill_response",
    [
        {},
        {"usage": {"prompt_tokens": 8}},
        {
            "usage": {
                "prompt_tokens": 8,
                "prompt_tokens_details": {
                    "cached_tokens": 4,
                    "local_cached_tokens": 2,
                },
            }
        },
        {
            "usage": {
                "prompt_tokens": 8,
                "prompt_tokens_details": {
                    "cached_tokens": 7,
                    "local_cached_tokens": 3,
                    "external_cached_tokens": 3,
                },
            }
        },
        {
            "usage": {
                "prompt_tokens": 4,
                "prompt_tokens_details": {
                    "cached_tokens": 6,
                    "local_cached_tokens": 3,
                    "external_cached_tokens": 3,
                },
            }
        },
    ],
)
def test_prefill_accounting_rejects_missing_or_inconsistent_usage(
    prefill_response: dict,
) -> None:
    with pytest.raises(ValueError, match="prefill usage"):
        proxy.validate_prefill_accounting(prefill_response)


def test_prefill_accounting_rejects_external_tokens_in_oneway_mode() -> None:
    with pytest.raises(ValueError, match="oneway"):
        proxy.validate_prefill_accounting(
            _prefill_response(external=64),
            reuse_mode="oneway",
        )


def test_prefill_accounting_headers_are_generic_and_mode_explicit() -> None:
    accounting = proxy.validate_prefill_accounting(_prefill_response())

    assert proxy.prefill_accounting_headers(
        accounting,
        reuse_mode="bidirectional",
    ) == {
        "X-VLLM-Prefill-Prompt-Tokens": "128",
        "X-VLLM-Prefill-Local-Cached-Tokens": "32",
        "X-VLLM-Prefill-External-Cached-Tokens": "64",
        "X-VLLM-Prefill-Computed-Tokens": "32",
        "X-VLLM-PD-Reuse-Mode": "bidirectional",
        "X-VLLM-D2P-Transfer-Selected": "true",
    }


def test_parse_args_exposes_compatible_and_strict_reuse_controls() -> None:
    defaults = proxy.parse_args([])
    strict_oneway = proxy.parse_args(
        ["--reuse-mode", "oneway", "--require-conversation-id"]
    )

    assert defaults.reuse_mode == "bidirectional"
    assert defaults.require_conversation_id is False
    assert strict_oneway.reuse_mode == "oneway"
    assert strict_oneway.require_conversation_id is True


def test_stats_snapshot_has_bounded_counters_and_unknown_timing_reasons() -> None:
    stats = proxy.ProxyStats()
    stats.requests_total = 3
    stats.streaming_requests = 1
    stats.non_streaming_requests = 2
    stats.cache_hits = 1
    stats.cache_misses = 1
    stats.cache_expired = 1
    stats.d2p_selected_requests = 1
    stats.d2p_selected_tokens = 64
    stats.p2d_offered_requests = 2
    stats.p2d_offered_tokens = 256
    stats.failures_total = 1
    stats.prefill_failures = 1

    snapshot = stats.snapshot(reuse_mode="bidirectional", cache_size=2)

    assert snapshot == {
        "reuse_mode": "bidirectional",
        "requests": {
            "total": 3,
            "streaming": 1,
            "non_streaming": 2,
            "missing_conversation_id": 0,
        },
        "cache": {
            "entries": 2,
            "handles_offered": 1,
            "misses": 1,
            "expired": 1,
        },
        "d2p": {"selected_requests": 1, "selected_tokens": 64},
        "p2d": {
            "offered_requests": 2,
            "offered_tokens": 256,
            "unknown_token_requests": 0,
        },
        "failures": {
            "total": 1,
            "prefill_request": 1,
            "prefill_accounting": 0,
            "missing_prefill_kv": 0,
            "decode_request": 0,
        },
        "timings_ms": {
            "last_prefill": {
                "value": None,
                "reason": "no prefill request has completed",
            },
            "last_decode": {
                "value": None,
                "reason": "no decode response has completed",
            },
        },
    }


def test_conversation_log_tag_is_a_privacy_safe_sha256_prefix() -> None:
    conversation_id = "private-customer-conversation"

    tag = proxy.conversation_log_tag(conversation_id)

    assert tag == hashlib.sha256(conversation_id.encode()).hexdigest()[:12]
    assert conversation_id not in tag


@_run_async_test
async def test_oneway_never_reads_or_writes_decoder_cache() -> None:
    cache = proxy.ConversationKVCache(ttl_seconds=450.0)
    old_params = {
        "remote_engine_id": "decode-old",
        "remote_block_ids": [[1, 2]],
    }
    cache.put("conv", old_params)
    old_token = cache.peek("conv").cache_token
    prefill_response = _prefill_response(prompt=64, local=32, external=0)

    async with _proxy_client(
        prefill_responses=[httpx.Response(200, json=prefill_response)],
        decode_responses=[httpx.Response(200, json=_decode_response())],
        reuse_mode="oneway",
        cache=cache,
    ) as (client, prefill, _, cache):
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_request(),
        )

    assert response.status_code == 200
    assert prefill.requests[0]["kv_transfer_params"]["remote_engine_id"] is None
    cached = cache.peek("conv")
    assert cached.cache_token == old_token
    assert cached.kv_transfer_params == old_params
    assert response.headers["X-VLLM-Prefill-External-Cached-Tokens"] == "0"
    assert response.headers["X-VLLM-PD-Reuse-Mode"] == "oneway"
    assert response.headers["X-VLLM-D2P-Transfer-Selected"] == "false"


@_run_async_test
async def test_bidirectional_consumes_old_handle_after_prefill_and_stores_new(
    caplog: pytest.LogCaptureFixture,
) -> None:
    conversation_id = "customer-secret-conversation"
    cache = proxy.ConversationKVCache(ttl_seconds=450.0)
    cache.put(
        conversation_id,
        {
            "remote_engine_id": "decode-old",
            "remote_request_id": "old-request",
            "remote_block_ids": [[1, 2]],
            "nested": {"blocks": [1, 2]},
        },
    )
    old_token = cache.peek(conversation_id).cache_token
    decode_response = _decode_response()

    with caplog.at_level("INFO", logger="disagg_proxy"):
        async with _proxy_client(
            prefill_responses=[httpx.Response(200, json=_prefill_response())],
            decode_responses=[httpx.Response(200, json=decode_response)],
            cache=cache,
        ) as (client, prefill, decode, cache):
            response = await client.post(
                "/v1/chat/completions",
                json=_chat_request(conversation_id=conversation_id),
            )
            stats_response = await client.get("/stats")

    assert response.status_code == 200
    offered = prefill.requests[0]["kv_transfer_params"]
    assert offered["remote_engine_id"] == "decode-old"
    assert offered["do_remote_decode"] is True
    assert offered["do_remote_prefill"] is False
    assert decode.requests[0]["kv_transfer_params"]["remote_engine_id"] == "prefill-0"
    assert decode.requests[0]["kv_transfer_params"]["remote_host"] == "prefill.test"

    cached = cache.peek(conversation_id)
    assert cached.status == "hit"
    assert cached.cache_token != old_token
    assert cached.kv_transfer_params["remote_engine_id"] == "decode-0"
    assert cached.kv_transfer_params["remote_host"] == "decode.test"
    assert response.headers["X-VLLM-Prefill-External-Cached-Tokens"] == "64"
    assert response.headers["X-VLLM-D2P-Transfer-Selected"] == "true"

    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["cache"]["handles_offered"] == 1
    assert stats["d2p"] == {"selected_requests": 1, "selected_tokens": 64}
    assert stats["p2d"]["offered_requests"] == 1
    assert stats["p2d"]["offered_tokens"] == 128
    assert conversation_id not in caplog.text
    assert proxy.conversation_log_tag(conversation_id) in caplog.text


@_run_async_test
async def test_failed_prefill_keeps_handle_for_retry() -> None:
    cache = proxy.ConversationKVCache(ttl_seconds=450.0)
    cache.put(
        "conv",
        {"remote_engine_id": "decode-old", "remote_block_ids": [[1, 2]]},
    )
    old_token = cache.peek("conv").cache_token

    async with _proxy_client(
        prefill_responses=[
            httpx.Response(503, json={"error": "temporarily unavailable"}),
            httpx.Response(200, json=_prefill_response()),
        ],
        decode_responses=[httpx.Response(200, json=_decode_response())],
        cache=cache,
    ) as (client, prefill, decode, cache):
        failed = await client.post("/v1/chat/completions", json=_chat_request())
        assert cache.peek("conv").cache_token == old_token

        retried = await client.post("/v1/chat/completions", json=_chat_request())

    assert failed.status_code == 502
    assert retried.status_code == 200
    assert prefill.requests[0]["kv_transfer_params"]["remote_engine_id"] == "decode-old"
    assert prefill.requests[1]["kv_transfer_params"]["remote_engine_id"] == "decode-old"
    assert len(decode.requests) == 1
    assert cache.peek("conv").kv_transfer_params["remote_engine_id"] == "decode-0"


@_run_async_test
async def test_expired_bidirectional_entry_is_not_offered() -> None:
    now = 100.0
    cache = proxy.ConversationKVCache(ttl_seconds=10.0, clock=lambda: now)
    cache.put("conv", {"remote_engine_id": "decode-old"})
    now = 111.0

    async with _proxy_client(
        prefill_responses=[
            httpx.Response(
                200,
                json=_prefill_response(prompt=64, local=32, external=0),
            )
        ],
        decode_responses=[httpx.Response(200, json=_decode_response())],
        cache=cache,
    ) as (client, prefill, _, _):
        response = await client.post("/v1/chat/completions", json=_chat_request())
        stats_response = await client.get("/stats")

    assert response.status_code == 200
    assert prefill.requests[0]["kv_transfer_params"]["remote_engine_id"] is None
    assert stats_response.json()["cache"]["expired"] == 1


@_run_async_test
async def test_strict_conversation_id_rejects_before_upstream_requests() -> None:
    async with _proxy_client(
        prefill_responses=[],
        decode_responses=[],
        require_conversation_id=True,
    ) as (client, prefill, decode, _):
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_request(conversation_id=None),
        )

    assert response.status_code == 400
    assert "conversation_id" in response.json()["error"]["message"]
    assert prefill.requests == []
    assert decode.requests == []


@_run_async_test
async def test_compatible_missing_conversation_id_disables_cache_reuse() -> None:
    async with _proxy_client(
        prefill_responses=[
            httpx.Response(
                200,
                json=_prefill_response(prompt=64, local=32, external=0),
            )
        ],
        decode_responses=[httpx.Response(200, json=_decode_response())],
    ) as (client, prefill, _, cache):
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_request(conversation_id=None),
        )
        stats_response = await client.get("/stats")

    assert response.status_code == 200
    assert "conversation_id" not in prefill.requests[0]
    assert prefill.requests[0]["kv_transfer_params"]["remote_engine_id"] is None
    assert cache.size == 0
    assert stats_response.json()["requests"]["missing_conversation_id"] == 1


@_run_async_test
async def test_malformed_prefill_usage_fails_closed_without_consuming_handle() -> None:
    cache = proxy.ConversationKVCache(ttl_seconds=450.0)
    cache.put("conv", {"remote_engine_id": "decode-old"})
    old_token = cache.peek("conv").cache_token
    malformed = _prefill_response()
    malformed["usage"]["prompt_tokens_details"]["external_cached_tokens"] = 63

    async with _proxy_client(
        prefill_responses=[httpx.Response(200, json=malformed)],
        decode_responses=[],
        cache=cache,
    ) as (client, _, decode, cache):
        response = await client.post("/v1/chat/completions", json=_chat_request())

    assert response.status_code == 502
    assert "prefill usage" in response.json()["error"]["message"]
    assert decode.requests == []
    assert cache.peek("conv").cache_token == old_token


@_run_async_test
async def test_missing_new_prefill_kv_never_forwards_stale_decoder_payload() -> None:
    cache = proxy.ConversationKVCache(ttl_seconds=450.0)
    cache.put(
        "conv",
        {"remote_engine_id": "decode-old", "remote_block_ids": [[1, 2]]},
    )
    old_token = cache.peek("conv").cache_token
    prefill_response = _prefill_response()
    del prefill_response["kv_transfer_params"]

    async with _proxy_client(
        prefill_responses=[httpx.Response(200, json=prefill_response)],
        decode_responses=[],
        cache=cache,
    ) as (client, _, decode, cache):
        response = await client.post("/v1/chat/completions", json=_chat_request())

    assert response.status_code == 502
    assert "kv_transfer_params" in response.json()["error"]["message"]
    assert decode.requests == []
    assert cache.peek("conv").cache_token == old_token


@_run_async_test
async def test_nonstream_preserves_complete_decoder_payload_and_headers() -> None:
    decode_response = _decode_response()
    async with _proxy_client(
        prefill_responses=[httpx.Response(200, json=_prefill_response())],
        decode_responses=[httpx.Response(200, json=decode_response)],
    ) as (client, _, _, _):
        response = await client.post("/v1/chat/completions", json=_chat_request())

    assert response.status_code == 200
    assert response.json() == decode_response
    assert response.headers["X-VLLM-Prefill-Prompt-Tokens"] == "128"
    assert response.headers["X-VLLM-Prefill-Local-Cached-Tokens"] == "32"
    assert response.headers["X-VLLM-Prefill-External-Cached-Tokens"] == "64"
    assert response.headers["X-VLLM-Prefill-Computed-Tokens"] == "32"
    assert response.headers["X-VLLM-PD-Reuse-Mode"] == "bidirectional"
    assert response.headers["X-VLLM-D2P-Transfer-Selected"] == "true"


@_run_async_test
async def test_stream_response_preserves_sse_and_has_accounting_headers() -> None:
    chunks = [
        {
            "id": "decode-stream",
            "object": "chat.completion.chunk",
            "created": 123456,
            "model": "served-model",
            "prompt_token_ids": [101, 102],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "answer"},
                    "token_ids": [201],
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "decode-stream",
            "object": "chat.completion.chunk",
            "created": 123456,
            "model": "served-model",
            "choices": [
                {"index": 0, "delta": {}, "token_ids": [202], "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 128, "completion_tokens": 2},
            "kv_transfer_params": _decode_response()["kv_transfer_params"],
        },
    ]
    sse = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    sse += "data: [DONE]\n\n"

    async with _proxy_client(
        prefill_responses=[httpx.Response(200, json=_prefill_response())],
        decode_responses=[
            httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        ],
    ) as (client, _, _, cache):
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_request(stream=True),
        )

    assert response.status_code == 200
    assert response.text == sse
    assert response.headers["X-VLLM-Prefill-Prompt-Tokens"] == "128"
    assert response.headers["X-VLLM-Prefill-External-Cached-Tokens"] == "64"
    assert response.headers["X-VLLM-PD-Reuse-Mode"] == "bidirectional"
    cached = cache.peek("conv").kv_transfer_params
    assert cached["remote_engine_id"] == "decode-0"
    assert cached["remote_host"] == "decode.test"
