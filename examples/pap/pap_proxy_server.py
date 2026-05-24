# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP proxy for the first Qwen3-8B NIXL experiment.

Externally this is an OpenAI-compatible proxy. Internally it exposes PAP roles:
Prefill is a vLLM NIXL producer, Projection is a vLLM NIXL consumer that owns
decode/lm_head/sampling for this first slice, and Attention is an internal
executor that records the prefill KV handle.
"""

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
        attach_pap_prefill_attention_params,
        build_prefill_payload,
        build_projection_kv_unaware_payload,
        enrich_prefill_kv_params,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from pd_payloads import (  # type: ignore[no-redef]
        attach_pap_prefill_attention_params,
        build_prefill_payload,
        build_projection_kv_unaware_payload,
        enrich_prefill_kv_params,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_proxy")


@dataclass
class PAPServiceClient:
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


def prefill_prefix_len_from_kv_params(
    kv_transfer_params: dict[str, Any],
) -> int | None:
    """Return the number of prompt KV tokens exported by Prefill."""
    value = kv_transfer_params.get("remote_num_tokens")
    if value is None:
        return None
    try:
        prefix_len = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid remote_num_tokens in Prefill kv_transfer_params: {value!r}"
        ) from exc
    if prefix_len < 0:
        raise ValueError(
            f"invalid remote_num_tokens in Prefill kv_transfer_params: {value!r}"
        )
    if prefix_len == 0:
        return None
    return prefix_len


async def register_attention_handle(
    attention: PAPServiceClient,
    request_id: str,
    conversation_id: str,
    prefill_endpoint: str,
    kv_transfer_params: dict[str, Any],
    prefix_len: int | None,
) -> dict[str, Any]:
    payload = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "prefill_endpoint": prefill_endpoint,
        "kv_transfer_params": dict(kv_transfer_params),
        "prefix_len": prefix_len,
    }
    resp = await attention.client.post(
        "/v1/pap/attention/register",
        json=payload,
        headers={},
    )
    resp.raise_for_status()
    return resp.json()


async def get_attention_resident_prefix(
    attention: PAPServiceClient,
    request_id: str,
) -> dict[str, Any]:
    resp = await attention.client.get(
        f"/v1/pap/attention/sessions/{request_id}/resident-prefix",
        headers={},
    )
    resp.raise_for_status()
    return resp.json()


def build_projection_payload(
    req_data: dict[str, Any],
    kv_transfer_params: dict[str, Any],
    *,
    pap_prefill_kv_handle: str | None = None,
    pap_attention_kv_installed: bool = False,
) -> dict[str, Any]:
    return build_projection_kv_unaware_payload(
        req_data,
        kv_transfer_params,
        pap_prefill_kv_handle=pap_prefill_kv_handle,
        pap_attention_kv_installed=pap_attention_kv_installed,
    )


async def _post_json(
    client: PAPServiceClient,
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


async def _stream_projection(
    client: PAPServiceClient,
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


def _make_client(host: str, port: int, role: str) -> PAPServiceClient:
    base_url = f"http://{host}:{port}"
    return PAPServiceClient(
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
    app.state.projection = _make_client(
        args.projection_host, args.projection_port, "projection"
    )
    app.state.attention = _make_client(
        args.attention_host, args.attention_port, "attention"
    )
    yield
    await app.state.prefill.client.aclose()
    await app.state.projection.client.aclose()
    await app.state.attention.client.aclose()


app = FastAPI(title="PAP Proxy", lifespan=lifespan)


async def _handle_openai_request(api_path: str, request: Request):
    req_data = await request.json()
    request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    conversation_id = str(req_data.pop("conversation_id", ""))
    client_stream = bool(req_data.get("stream", False))
    attention_session = await register_attention_handle(
        request.app.state.attention,
        request_id=request_id,
        conversation_id=conversation_id,
        prefill_endpoint=request.app.state.prefill.base_url,
        kv_transfer_params={},
        prefix_len=None,
    )
    prefill_payload = attach_pap_prefill_attention_params(
        build_prefill_payload(req_data),
        pap_attention_endpoint=request.app.state.attention.base_url,
        pap_prefill_kv_handle=str(attention_session.get("prefill_kv_handle")),
        pap_mode=request.app.state.args.pap_mode,
    )
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
    prefix_len = prefill_prefix_len_from_kv_params(kv_params)
    logger.info(
        "request_id=%s prefill_ms=%d prefill_prefix_len=%s prefill_kv_keys=%s",
        request_id,
        prefill_ms,
        prefix_len,
        sorted(kv_params.keys()),
    )

    projection_payload = build_projection_payload(
        req_data,
        kv_params,
        pap_prefill_kv_handle=attention_session.get("prefill_kv_handle"),
        pap_attention_kv_installed=prefix_len is not None,
    )
    logger.info(
        "request_id=%s projection_kv_keys=%s attention_endpoint=%s",
        request_id,
        sorted(projection_payload["kv_transfer_params"].keys()),
        request.app.state.attention.base_url,
    )
    projection_payload.setdefault("stream", client_stream)

    if client_stream:
        return StreamingResponse(
            _stream_projection(
                request.app.state.projection,
                api_path,
                projection_payload,
                request_id,
            ),
            media_type="text/event-stream",
            headers={"X-PAP-Prefill-Ms": str(prefill_ms)},
        )

    projection_resp = await _post_json(
        request.app.state.projection,
        api_path,
        projection_payload,
        request_id,
    )
    return JSONResponse(
        projection_resp,
        headers={"X-PAP-Prefill-Ms": str(prefill_ms)},
    )


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request("/v1/chat/completions", request)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "role": "pap-proxy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAP proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefill-host", default="127.0.0.1")
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--prefill-nixl-port", type=int, default=None)
    parser.add_argument("--attention-host", default="127.0.0.1")
    parser.add_argument("--attention-port", type=int, default=8300)
    parser.add_argument("--projection-host", default="127.0.0.1")
    parser.add_argument("--projection-port", type=int, default=8200)
    parser.add_argument("--pap-mode", default=os.environ.get("PAP_MODE", "true_split"))
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)
