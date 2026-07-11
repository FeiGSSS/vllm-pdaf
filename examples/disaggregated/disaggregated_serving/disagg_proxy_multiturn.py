# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Prefill/Decode Proxy with Explicit Multi-turn Reuse Modes

This proxy sits between clients and a vLLM Prefill/Decode (P/D) deployment,
routing requests in either one-way P→D mode or bidirectional mode.
Bidirectional mode can reuse KV blocks from the previous Decode turn.

Architecture:
    Client  ──►  Proxy  ──►  Prefill (P)  ──►  Decode (D)
                   │              │                 │
                   │   kv_transfer_params flow:     │
                   │   D finish ──► proxy caches    │
                   │   next turn ──► proxy sends    │
                   │   cached D blocks to P ──►     │
                   │   P reads D blocks (bidir)     │
                   │   P sends its blocks to D      │

Per-request flow:
    1. Client sends chat/completions request to proxy.
    2. In bidirectional mode, proxy peeks at cached D block info from the
       previous turn (keyed by conversation_id). One-way skips the cache.
    3. On a bidirectional cache hit, proxy offers D's block info to P.
    4. Proxy sends request to P (max_tokens=1, non-streaming).
    5. P returns kv_transfer_params with its own block info.
    6. Proxy forwards request + P's block info to D.
    7. In bidirectional mode, proxy caches D's kv_transfer_params for the
       next turn.
    8. Proxy returns D's response to the client.

Conversation isolation:
    Each request must include a ``conversation_id`` field (top-level in
    the JSON body) to scope the KV cache across turns. Without it, the
    proxy cannot link turns and falls back to no-cache behavior unless
    ``--require-conversation-id`` is set, which rejects the request.

    ``conversation_id`` is a non-standard extension to the OpenAI Chat
    Completions schema, consumed by this proxy and not forwarded to the
    vLLM engine. Strict OpenAI-compatible frontends reject unknown
    fields, so clients must opt in only when targeting this proxy.

Usage:
    python disagg_proxy_multiturn.py \\
        --host 0.0.0.0 --port 8000 \\
        --prefiller-host 10.0.0.1 --prefiller-port 8100 \\
        --decoder-host 10.0.0.2 --decoder-port 8200 \\
        --reuse-mode bidirectional

Benchmarking:
    Use ``benchmarks/multi_turn/benchmark_serving_multi_turn.py`` with
    the ``--send-conversation-id`` flag to inject a per-conversation
    ``conversation_id`` into every request so this proxy can key cross-turn KV
    reuse. For controlled runs, start the proxy with an explicit
    ``--reuse-mode`` and ``--require-conversation-id``.

    Example:
        python benchmarks/multi_turn/benchmark_serving_multi_turn.py \\
            --model <MODEL> --served-model-name <NAME> \\
            --url http://<proxy_host>:8000 \\
            --input-file generate_multi_turn.json \\
            --num-clients 2 --max-active-conversations 6 \\
            --send-conversation-id

    See ``docs/features/nixl_connector_usage.md`` for the broader
    one-way and bidirectional setup these benchmarks exercise.

Dependencies:
    pip install fastapi uvicorn httpx
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("disagg_proxy")


def conversation_log_tag(conversation_id: str) -> str:
    """Return a stable privacy-safe identifier for logs."""
    return hashlib.sha256(conversation_id.encode()).hexdigest()[:12]


# Data structures
@dataclass
class CachedKVEntry:
    """KV transfer parameters cached from D's response for one turn."""

    kv_transfer_params: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    cache_token: int = 0


@dataclass(frozen=True)
class CachePeekResult:
    """Result of a non-consuming conversation cache lookup."""

    status: Literal["hit", "miss", "expired"]
    kv_transfer_params: dict[str, Any] | None = None
    cache_token: int | None = None


class ConversationKVCache:
    """Per-conversation KV block cache.

    Each conversation is identified by a ``conversation_id`` supplied by
    the client. After D finishes a turn, its ``kv_transfer_params`` are
    stored here. On the next turn, the proxy retrieves them so P can
    read D's blocks via bidirectional KV transfer.
    """

    def __init__(
        self,
        ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store: dict[str, CachedKVEntry] = {}
        self._ttl = ttl_seconds
        self._clock = clock
        self._next_cache_token = 1

    def peek(self, conversation_id: str) -> CachePeekResult:
        """Return a defensive snapshot without consuming a live entry."""
        entry = self._store.get(conversation_id)
        if entry is None:
            return CachePeekResult(status="miss")
        if self._clock() - entry.timestamp > self._ttl:
            del self._store[conversation_id]
            return CachePeekResult(status="expired")
        return CachePeekResult(
            status="hit",
            kv_transfer_params=copy.deepcopy(entry.kv_transfer_params),
            cache_token=entry.cache_token,
        )

    def consume(self, conversation_id: str, cache_token: int | None) -> bool:
        """Consume the entry only if it still matches a prior snapshot."""
        entry = self._store.get(conversation_id)
        if entry is None or entry.cache_token != cache_token:
            return False
        del self._store[conversation_id]
        return True

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve and consume cached KV params for a conversation.

        Returns a *copy* of the kv_transfer_params dict, or None.
        The entry is removed after retrieval (single-use).
        """
        result = self.peek(conversation_id)
        if result.status != "hit":
            return None
        self.consume(conversation_id, result.cache_token)
        return result.kv_transfer_params

    def put(self, conversation_id: str, kv_params: dict[str, Any]) -> None:
        """Store D's kv_transfer_params for a conversation."""
        cache_token = self._next_cache_token
        self._next_cache_token += 1
        self._store[conversation_id] = CachedKVEntry(
            kv_transfer_params=copy.deepcopy(kv_params),
            timestamp=self._clock(),
            cache_token=cache_token,
        )
        logger.info(
            "conv=%s: cached D blocks (remote_request_id=%s, blocks=%d)",
            conversation_log_tag(conversation_id),
            kv_params.get("remote_request_id", "?"),
            len(kv_params.get("remote_block_ids", [[]])[0])
            if kv_params.get("remote_block_ids")
            else 0,
        )

    def evict_stale(self) -> int:
        """Remove entries older than TTL. Returns count of evicted entries."""
        now = self._clock()
        stale = [
            cid
            for cid, entry in self._store.items()
            if now - entry.timestamp > self._ttl
        ]
        for cid in stale:
            del self._store[cid]
        return len(stale)

    @property
    def size(self) -> int:
        return len(self._store)


@dataclass(frozen=True)
class PrefillAccounting:
    """Validated exclusive prompt-token accounting from the prefill node."""

    prompt_tokens: int
    local_cached_tokens: int
    external_cached_tokens: int
    computed_tokens: int


@dataclass
class ProxyStats:
    """Fixed-cardinality counters and last-observed proxy timings."""

    requests_total: int = 0
    streaming_requests: int = 0
    non_streaming_requests: int = 0
    missing_conversation_id: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_expired: int = 0
    d2p_selected_requests: int = 0
    d2p_selected_tokens: int = 0
    p2d_offered_requests: int = 0
    p2d_offered_tokens: int = 0
    p2d_unknown_token_requests: int = 0
    failures_total: int = 0
    prefill_failures: int = 0
    prefill_accounting_failures: int = 0
    missing_prefill_kv_failures: int = 0
    decode_failures: int = 0
    last_prefill_ms: float | None = None
    last_decode_ms: float | None = None

    def snapshot(
        self,
        *,
        reuse_mode: Literal["oneway", "bidirectional"],
        cache_size: int,
    ) -> dict[str, Any]:
        """Return a bounded, JSON-serializable observability snapshot."""
        return {
            "reuse_mode": reuse_mode,
            "requests": {
                "total": self.requests_total,
                "streaming": self.streaming_requests,
                "non_streaming": self.non_streaming_requests,
                "missing_conversation_id": self.missing_conversation_id,
            },
            "cache": {
                "entries": cache_size,
                "handles_offered": self.cache_hits,
                "misses": self.cache_misses,
                "expired": self.cache_expired,
            },
            "d2p": {
                "selected_requests": self.d2p_selected_requests,
                "selected_tokens": self.d2p_selected_tokens,
            },
            "p2d": {
                "offered_requests": self.p2d_offered_requests,
                "offered_tokens": self.p2d_offered_tokens,
                "unknown_token_requests": self.p2d_unknown_token_requests,
            },
            "failures": {
                "total": self.failures_total,
                "prefill_request": self.prefill_failures,
                "prefill_accounting": self.prefill_accounting_failures,
                "missing_prefill_kv": self.missing_prefill_kv_failures,
                "decode_request": self.decode_failures,
            },
            "timings_ms": {
                "last_prefill": {
                    "value": self.last_prefill_ms,
                    "reason": (
                        None
                        if self.last_prefill_ms is not None
                        else "no prefill request has completed"
                    ),
                },
                "last_decode": {
                    "value": self.last_decode_ms,
                    "reason": (
                        None
                        if self.last_decode_ms is not None
                        else "no decode response has completed"
                    ),
                },
            },
        }


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"invalid prefill usage: {field_name} must be a non-negative integer"
        )
    return value


def validate_prefill_accounting(
    prefill_response: dict[str, Any],
    *,
    reuse_mode: Literal["oneway", "bidirectional"] = "bidirectional",
) -> PrefillAccounting:
    """Validate and adapt exclusive prefill prompt-token accounting."""
    if reuse_mode not in {"oneway", "bidirectional"}:
        raise ValueError(f"invalid prefill usage reuse mode: {reuse_mode}")
    usage = prefill_response.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("invalid prefill usage: missing usage object")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise ValueError("invalid prefill usage: missing prompt_tokens_details")

    prompt = _require_non_negative_int(usage.get("prompt_tokens"), "prompt_tokens")
    cached = _require_non_negative_int(details.get("cached_tokens"), "cached_tokens")
    local = _require_non_negative_int(
        details.get("local_cached_tokens"),
        "local_cached_tokens",
    )
    external = _require_non_negative_int(
        details.get("external_cached_tokens"),
        "external_cached_tokens",
    )
    if cached != local + external:
        raise ValueError(
            "invalid prefill usage: cached_tokens must equal local_cached_tokens "
            "+ external_cached_tokens"
        )
    computed = prompt - local - external
    if computed < 0:
        raise ValueError(
            "invalid prefill usage: prompt_tokens must equal local cached + "
            "external cached + non-negative computed tokens"
        )
    if reuse_mode == "oneway" and external != 0:
        raise ValueError(
            "invalid prefill usage: oneway mode requires zero external cached tokens"
        )
    return PrefillAccounting(
        prompt_tokens=prompt,
        local_cached_tokens=local,
        external_cached_tokens=external,
        computed_tokens=computed,
    )


def prefill_accounting_headers(
    accounting: PrefillAccounting,
    *,
    reuse_mode: Literal["oneway", "bidirectional"],
) -> dict[str, str]:
    """Build mode-explicit response headers from validated accounting."""
    return {
        "X-VLLM-Prefill-Prompt-Tokens": str(accounting.prompt_tokens),
        "X-VLLM-Prefill-Local-Cached-Tokens": str(
            accounting.local_cached_tokens
        ),
        "X-VLLM-Prefill-External-Cached-Tokens": str(
            accounting.external_cached_tokens
        ),
        "X-VLLM-Prefill-Computed-Tokens": str(accounting.computed_tokens),
        "X-VLLM-PD-Reuse-Mode": reuse_mode,
        "X-VLLM-D2P-Transfer-Selected": str(
            accounting.external_cached_tokens > 0
        ).lower(),
    }


# Service client helpers
@dataclass
class ServiceClient:
    """Wrapper around an httpx.AsyncClient for a P or D instance."""

    client: httpx.AsyncClient
    host: str
    port: int
    id: int


def _make_headers(request_id: str) -> dict[str, str]:
    """Build HTTP headers for upstream requests."""
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _send_to_prefill(
    client: ServiceClient,
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Send a non-streaming prefill request (max_tokens=1).

    Returns the JSON response from P, which includes kv_transfer_params.
    """
    payload = req_data.copy()
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload.pop("max_completion_tokens", None)
    payload.pop("min_tokens", None)
    payload.pop("stream_options", None)

    resp = await client.client.post(
        endpoint,
        json=payload,
        headers=_make_headers(request_id),
    )
    resp.raise_for_status()
    return resp.json()


async def _send_to_decode(
    client: ServiceClient,
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """Send a non-streaming decode request and return its complete JSON."""
    payload = req_data.copy()
    payload["stream"] = False
    resp = await client.client.post(
        endpoint,
        json=payload,
        headers=_make_headers(request_id),
    )
    resp.raise_for_status()
    return resp.json()


def _store_decoder_kv(
    response: dict[str, Any],
    *,
    client: ServiceClient,
    conversation_id: str,
    reuse_mode: Literal["oneway", "bidirectional"],
    cache: ConversationKVCache,
) -> None:
    if reuse_mode != "bidirectional" or not conversation_id:
        return
    kv_params = response.get("kv_transfer_params")
    if not isinstance(kv_params, dict) or not kv_params:
        return
    cached_params = copy.deepcopy(kv_params)
    cached_params["remote_host"] = client.host
    cache.put(conversation_id, cached_params)


async def _stream_from_decode_sse(
    client: ServiceClient,
    endpoint: str,
    req_data: dict[str, Any],
    request_id: str,
    conversation_id: str,
    reuse_mode: Literal["oneway", "bidirectional"],
    cache: ConversationKVCache,
    stats: ProxyStats,
):
    """Yield SSE chunks from D to the client, capturing kv_transfer_params."""
    payload = req_data.copy()
    payload["stream"] = True
    started = time.perf_counter()
    try:
        async with client.client.stream(
            "POST",
            endpoint,
            json=payload,
            headers=_make_headers(request_id),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    yield "\n"
                    continue

                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        _store_decoder_kv(
                            chunk,
                            client=client,
                            conversation_id=conversation_id,
                            reuse_mode=reuse_mode,
                            cache=cache,
                        )
                    except json.JSONDecodeError:
                        pass

                yield line + "\n"
    except Exception:
        stats.failures_total += 1
        stats.decode_failures += 1
        raise
    finally:
        stats.last_decode_ms = (time.perf_counter() - started) * 1000.0


# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize HTTP clients for P and D instances."""
    app.state.reuse_mode = global_args.reuse_mode
    app.state.require_conversation_id = global_args.require_conversation_id
    app.state.kv_cache = ConversationKVCache(
        ttl_seconds=450.0
    )  # Must be below VLLM_NIXL_ABORT_REQUEST_TIMEOUT (480s).
    app.state.stats = ProxyStats()
    app.state.prefill_clients: list[ServiceClient] = []
    app.state.decode_clients: list[ServiceClient] = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append(
            ServiceClient(
                client=httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                ),
                host=host,
                port=port,
                id=i,
            )
        )

    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append(
            ServiceClient(
                client=httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                ),
                host=host,
                port=port,
                id=i,
            )
        )

    app.state.prefill_iter = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iter = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Ready: %d prefill, %d decode instances, reuse_mode=%s",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
        app.state.reuse_mode,
    )
    yield

    for sc in app.state.prefill_clients + app.state.decode_clients:
        await sc.client.aclose()


app = FastAPI(title="Disaggregated P/D Proxy (Multi-turn)", lifespan=lifespan)


def _next_client(app_state, role: str) -> ServiceClient:
    if role == "prefill":
        return app_state.prefill_clients[next(app_state.prefill_iter)]
    return app_state.decode_clients[next(app_state.decode_iter)]


def _empty_remote_kv_params() -> dict[str, Any]:
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def _error_response(message: str, code: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "proxy_error",
                "code": code,
            }
        },
    )


# Request handler
async def _handle_request(api_path: str, request: Request):
    """Core request handler for both /v1/chat/completions and /v1/completions."""
    req_data = await request.json()
    request_id = str(uuid.uuid4())
    raw_conversation_id = req_data.pop("conversation_id", "")
    conversation_id = (
        raw_conversation_id if isinstance(raw_conversation_id, str) else ""
    )
    client_wants_stream = bool(req_data.get("stream", False))
    state = request.app.state
    reuse_mode: Literal["oneway", "bidirectional"] = state.reuse_mode
    cache: ConversationKVCache = state.kv_cache
    stats: ProxyStats = state.stats

    stats.requests_total += 1
    if client_wants_stream:
        stats.streaming_requests += 1
    else:
        stats.non_streaming_requests += 1

    if not conversation_id:
        stats.missing_conversation_id += 1
        if state.require_conversation_id:
            stats.failures_total += 1
            return _error_response(
                "conversation_id is required by this proxy",
                "missing_conversation_id",
                400,
            )
        logger.warning(
            "[%s] No conversation_id provided — KV cache reuse disabled "
            "for this request. Add a 'conversation_id' field to enable "
            "cross-turn KV sharing. When using "
            "benchmarks/multi_turn/benchmark_serving_multi_turn.py, pass "
            "--send-conversation-id (off by default).",
            request_id,
        )

    cache_peek = CachePeekResult(status="miss")
    if reuse_mode == "bidirectional" and conversation_id:
        cache_peek = cache.peek(conversation_id)
        if cache_peek.status == "hit":
            stats.cache_hits += 1
        elif cache_peek.status == "expired":
            stats.cache_expired += 1
        else:
            stats.cache_misses += 1
    elif reuse_mode == "bidirectional":
        stats.cache_misses += 1

    cached_kv = cache_peek.kv_transfer_params
    if reuse_mode == "bidirectional" and cached_kv is not None:
        cached_kv["do_remote_decode"] = True
        cached_kv["do_remote_prefill"] = False
        req_data["kv_transfer_params"] = cached_kv
        logger.info(
            "[%s] conv=%s: sending D's cached blocks to P (remote_request_id=%s)",
            request_id,
            conversation_log_tag(conversation_id),
            cached_kv.get("remote_request_id"),
        )
    else:
        req_data["kv_transfer_params"] = _empty_remote_kv_params()
        logger.info(
            "[%s] conv=%s: no D-to-P handle offered (reuse_mode=%s)",
            request_id,
            conversation_log_tag(conversation_id) if conversation_id else "none",
            reuse_mode,
        )

    prefill_client = _next_client(request.app.state, "prefill")
    prefill_started = time.perf_counter()
    try:
        prefill_resp = await _send_to_prefill(
            prefill_client,
            api_path,
            req_data,
            request_id,
        )
    except Exception as exc:
        stats.last_prefill_ms = (
            time.perf_counter() - prefill_started
        ) * 1000.0
        stats.failures_total += 1
        stats.prefill_failures += 1
        logger.warning(
            "[%s] Prefill request failed: %s",
            request_id,
            type(exc).__name__,
        )
        return _error_response(
            "prefill request failed",
            "prefill_request_failed",
            502,
        )
    stats.last_prefill_ms = (time.perf_counter() - prefill_started) * 1000.0
    logger.info(
        "[%s] Prefill done in %.0fms",
        request_id,
        stats.last_prefill_ms,
    )

    try:
        accounting = validate_prefill_accounting(
            prefill_resp,
            reuse_mode=reuse_mode,
        )
    except ValueError as exc:
        stats.failures_total += 1
        stats.prefill_accounting_failures += 1
        return _error_response(str(exc), "invalid_prefill_usage", 502)

    p_kv_params = prefill_resp.get("kv_transfer_params")
    if not isinstance(p_kv_params, dict) or not p_kv_params:
        stats.failures_total += 1
        stats.missing_prefill_kv_failures += 1
        return _error_response(
            "prefill response is missing new kv_transfer_params",
            "missing_prefill_kv",
            502,
        )

    if accounting.external_cached_tokens > 0:
        stats.d2p_selected_requests += 1
        stats.d2p_selected_tokens += accounting.external_cached_tokens

    p_kv_params = copy.deepcopy(p_kv_params)
    p_kv_params["remote_host"] = prefill_client.host
    stats.p2d_offered_requests += 1
    p2d_tokens = p_kv_params.get("remote_num_tokens")
    if isinstance(p2d_tokens, int) and not isinstance(p2d_tokens, bool):
        stats.p2d_offered_tokens += max(p2d_tokens, 0)
    else:
        stats.p2d_unknown_token_requests += 1

    if cache_peek.status == "hit":
        cache.consume(conversation_id, cache_peek.cache_token)

    decode_req_data = req_data.copy()
    decode_req_data["kv_transfer_params"] = p_kv_params
    decode_client = _next_client(request.app.state, "decode")
    response_headers = prefill_accounting_headers(
        accounting,
        reuse_mode=reuse_mode,
    )

    if client_wants_stream:
        return StreamingResponse(
            _stream_from_decode_sse(
                decode_client,
                api_path,
                decode_req_data,
                request_id,
                conversation_id,
                reuse_mode,
                cache,
                stats,
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )

    decode_started = time.perf_counter()
    try:
        decode_resp = await _send_to_decode(
            decode_client,
            api_path,
            decode_req_data,
            request_id,
        )
    except Exception as exc:
        stats.last_decode_ms = (time.perf_counter() - decode_started) * 1000.0
        stats.failures_total += 1
        stats.decode_failures += 1
        logger.warning(
            "[%s] Decode request failed: %s",
            request_id,
            type(exc).__name__,
        )
        return _error_response(
            "decode request failed",
            "decode_request_failed",
            502,
        )
    stats.last_decode_ms = (time.perf_counter() - decode_started) * 1000.0
    _store_decoder_kv(
        decode_resp,
        client=decode_client,
        conversation_id=conversation_id,
        reuse_mode=reuse_mode,
        cache=cache,
    )
    return JSONResponse(content=decode_resp, headers=response_headers)


# Routes
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_request("/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_request("/completions", request)


@app.get("/health")
async def health(request: Request):
    cache: ConversationKVCache = request.app.state.kv_cache
    stats: ProxyStats = request.app.state.stats
    evicted = cache.evict_stale()
    stats.cache_expired += evicted
    return {
        "status": "ok",
        "cached_conversations": cache.size,
        "evicted_stale": evicted,
    }


@app.get("/stats")
async def proxy_stats(request: Request):
    return request.app.state.stats.snapshot(
        reuse_mode=request.app.state.reuse_mode,
        cache_size=request.app.state.kv_cache.size,
    )


# CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Disaggregated P/D proxy with bidirectional KV transfer",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--reuse-mode",
        choices=("oneway", "bidirectional"),
        default="bidirectional",
        help="D-to-P reuse mode (default: bidirectional)",
    )
    p.add_argument(
        "--require-conversation-id",
        action="store_true",
        help="Reject requests without conversation_id",
    )
    p.add_argument(
        "--prefiller-host",
        "--prefiller-hosts",
        dest="prefiller_hosts",
        nargs="+",
        default=["localhost"],
    )
    p.add_argument(
        "--prefiller-port",
        "--prefiller-ports",
        dest="prefiller_ports",
        type=int,
        nargs="+",
        default=[8100],
    )
    p.add_argument(
        "--decoder-host",
        "--decoder-hosts",
        dest="decoder_hosts",
        nargs="+",
        default=["localhost"],
    )
    p.add_argument(
        "--decoder-port",
        "--decoder-ports",
        dest="decoder_ports",
        type=int,
        nargs="+",
        default=[8200],
    )
    args = p.parse_args(argv)

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        p.error("Number of prefiller hosts must match ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        p.error("Number of decoder hosts must match ports")

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
