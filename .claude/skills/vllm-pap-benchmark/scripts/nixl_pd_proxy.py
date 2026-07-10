# SPDX-License-Identifier: Apache-2.0
"""Minimal 1P1D NIXL proxy used by the bundled PD benchmark runner."""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    app.state.prefill_clients = _build_clients(args.prefiller_instances)
    app.state.decode_clients = _build_clients(args.decoder_instances)
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients))
    )
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients))
    )
    logger.info(
        "PD proxy ready: prefill=%d decode=%d",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )
    yield
    for client_info in (
        *app.state.prefill_clients,
        *app.state.decode_clients,
    ):
        await client_info["client"].aclose()


def _build_clients(instances: list[tuple[str, int]]) -> list[dict[str, Any]]:
    clients = []
    for index, (host, port) in enumerate(instances):
        clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=f"http://{host}:{port}/v1",
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": index,
            }
        )
    return clients


app = FastAPI(lifespan=lifespan)


def _next_client(app_state: Any, kind: str) -> dict[str, Any]:
    if kind == "prefill":
        index = next(app_state.prefill_iterator)
        return app_state.prefill_clients[index]
    index = next(app_state.decode_iterator)
    return app_state.decode_clients[index]


def _request_headers(request_id: str) -> dict[str, str]:
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def _send_prefill(
    client_info: dict[str, Any],
    endpoint: str,
    request_data: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    body = request_data.copy()
    body["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    body["stream"] = False
    body["max_tokens"] = 1
    if "max_completion_tokens" in body:
        body["max_completion_tokens"] = 1
    body.pop("stream_options", None)
    min_tokens = body.pop("min_tokens", None)
    min_completion_tokens = body.pop("min_completion_tokens", None)

    response = await client_info["client"].post(
        endpoint,
        json=body,
        headers=_request_headers(request_id),
    )
    response.raise_for_status()
    await response.aread()
    kv_params = response.json().get("kv_transfer_params", {})
    body["min_tokens"] = min_tokens
    body["min_completion_tokens"] = min_completion_tokens
    return kv_params


async def _stream_decode(
    client_info: dict[str, Any],
    endpoint: str,
    request_data: dict[str, Any],
    request_id: str,
):
    async with client_info["client"].stream(
        "POST",
        endpoint,
        json=request_data,
        headers=_request_headers(request_id),
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_request(endpoint: str, request: Request) -> StreamingResponse:
    request_data = await request.json()
    request_id = str(uuid.uuid4())
    prefill_client = _next_client(request.app.state, "prefill")
    kv_params = await _send_prefill(
        prefill_client,
        endpoint,
        request_data,
        request_id,
    )
    if kv_params:
        request_data["kv_transfer_params"] = kv_params
    decode_client = _next_client(request.app.state, "decode")

    async def generate():
        async for chunk in _stream_decode(
            decode_client,
            endpoint,
            request_data,
            request_id,
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="application/json")


@app.post("/v1/completions")
async def completions(request: Request) -> StreamingResponse:
    return await _handle_request("/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    return await _handle_request("/chat/completions", request)


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": "nixl-pd-proxy", "object": "model"}],
    }


@app.get("/healthcheck")
async def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bundled NIXL PD proxy")
    parser.add_argument("--port", type=int, default=19410)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefiller-hosts", nargs="+", default=["127.0.0.1"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[18100])
    parser.add_argument("--decoder-hosts", nargs="+", default=["127.0.0.1"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[19100])
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("prefiller hosts/ports length mismatch")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("decoder hosts/ports length mismatch")
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports, strict=True)
    )
    args.decoder_instances = list(
        zip(args.decoder_hosts, args.decoder_ports, strict=True)
    )
    return args


def main() -> None:
    args = parse_args()
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
