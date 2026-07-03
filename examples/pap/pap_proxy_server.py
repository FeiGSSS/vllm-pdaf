# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP proxy for the first Qwen3-8B NIXL experiment.

Externally this is an OpenAI-compatible proxy. Internally it exposes PAP roles:
Prefill computes prompt KV, Projection runs the decode model path without
prompt KV bytes, and Attention is an internal executor that reads Prefill KV and
computes attention.
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


async def _delete_attention_session(
    attention: PAPServiceClient,
    request_id: str,
) -> None:
    resp = await attention.client.delete(
        f"/v1/pap/attention/sessions/{request_id}",
        headers=_headers(request_id),
    )
    resp.raise_for_status()


async def _cleanup_attention_sessions(
    attentions: list[PAPServiceClient],
    request_id: str,
) -> None:
    for attention in attentions:
        try:
            await _delete_attention_session(attention, request_id)
        except Exception as exc:
            logger.warning(
                "failed to release PAP attention session request_id=%s "
                "attention_endpoint=%s error=%s",
                request_id,
                attention.base_url,
                exc,
            )


async def _stream_projection_with_cleanup(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    attentions: list[PAPServiceClient],
):
    try:
        async for chunk in _stream_projection(client, endpoint, payload, request_id):
            yield chunk
    finally:
        await _cleanup_attention_sessions(attentions, request_id)


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
    attention_client = request.app.state.attention
    attention_session: dict[str, Any] | None = None
    handed_off_stream_cleanup = False
    try:
        attention_session = await register_attention_handle(
            attention_client,
            request_id=request_id,
            conversation_id=conversation_id,
            prefill_endpoint=request.app.state.prefill.base_url,
            kv_transfer_params={},
            prefix_len=None,
        )
        prefill_payload = attach_pap_prefill_attention_params(
            build_prefill_payload(req_data),
            pap_attention_endpoint=attention_client.base_url,
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
            attention_client.base_url,
        )
        projection_payload.setdefault("stream", client_stream)

        if client_stream:
            handed_off_stream_cleanup = True
            return StreamingResponse(
                _stream_projection_with_cleanup(
                    request.app.state.projection,
                    api_path,
                    projection_payload,
                    request_id,
                    [attention_client],
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
    finally:
        if attention_session is not None and not handed_off_stream_cleanup:
            await _cleanup_attention_sessions([attention_client], request_id)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request("/v1/chat/completions", request)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "role": "pap-proxy",
    }


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
    parser.add_argument("--pap-mode", default=os.environ.get("PAP_MODE", "pap"))
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)
