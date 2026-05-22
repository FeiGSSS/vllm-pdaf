# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native vLLM PD proxy for Qwen3-8B NIXL consistency experiments."""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from examples.pap.pd_payloads import (
        build_decode_payload,
        build_prefill_payload,
        enrich_prefill_kv_params,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from pd_payloads import (  # type: ignore[no-redef]
        build_decode_payload,
        build_prefill_payload,
        enrich_prefill_kv_params,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("native_pd_proxy")


@dataclass
class PDServiceClient:
    client: httpx.AsyncClient
    host: str
    port: int
    base_url: str
    role: str


def _headers(request_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


async def _post_json(
    client: PDServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    resp = await client.client.post(
        endpoint,
        json=payload,
        headers=_headers(request_id),
    )
    resp.raise_for_status()
    return resp.json()


async def _stream_decode(
    client: PDServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
):
    async with client.client.stream(
        "POST",
        endpoint,
        json=payload,
        headers=_headers(request_id),
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk


def _make_client(host: str, port: int, role: str) -> PDServiceClient:
    base_url = f"http://{host}:{port}"
    return PDServiceClient(
        client=httpx.AsyncClient(
            timeout=None,
            base_url=base_url,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        ),
        host=host,
        port=port,
        base_url=base_url,
        role=role,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    app.state.prefill = _make_client(args.prefill_host, args.prefill_port, "prefill")
    app.state.decode = _make_client(args.decode_host, args.decode_port, "decode")
    yield
    await app.state.prefill.client.aclose()
    await app.state.decode.client.aclose()


app = FastAPI(title="Native PD Proxy", lifespan=lifespan)


async def _handle_openai_request(api_path: str, request: Request):
    req_data = await request.json()
    request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    client_stream = bool(req_data.get("stream", False))

    prefill_payload = build_prefill_payload(req_data)
    t0 = time.time()
    prefill_resp = await _post_json(
        request.app.state.prefill,
        api_path,
        prefill_payload,
        request_id,
    )
    prefill_ms = int((time.time() - t0) * 1000)

    kv_params = enrich_prefill_kv_params(
        prefill_resp.get("kv_transfer_params") or {},
        prefill_host=request.app.state.prefill.host,
        prefill_nixl_port=request.app.state.args.prefill_nixl_port,
    )
    logger.info(
        "request_id=%s prefill_ms=%d prefill_kv_keys=%s",
        request_id,
        prefill_ms,
        sorted(kv_params.keys()),
    )

    decode_payload = build_decode_payload(req_data, kv_params)
    decode_payload.setdefault("stream", client_stream)
    logger.info(
        "request_id=%s decode_kv_keys=%s",
        request_id,
        sorted(decode_payload["kv_transfer_params"].keys()),
    )

    if client_stream:
        return StreamingResponse(
            _stream_decode(
                request.app.state.decode,
                api_path,
                decode_payload,
                request_id,
            ),
            media_type="text/event-stream",
            headers={"X-PD-Prefill-Ms": str(prefill_ms)},
        )

    decode_resp = await _post_json(
        request.app.state.decode,
        api_path,
        decode_payload,
        request_id,
    )
    return JSONResponse(
        decode_resp,
        headers={"X-PD-Prefill-Ms": str(prefill_ms)},
    )


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request("/v1/chat/completions", request)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "role": "native-pd-proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native PD proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--prefill-host", default="127.0.0.1")
    parser.add_argument("--prefill-port", type=int, default=8110)
    parser.add_argument("--prefill-nixl-port", type=int, default=None)
    parser.add_argument("--decode-host", default="127.0.0.1")
    parser.add_argument("--decode-port", type=int, default=8210)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)
