# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-instance PAP proxy for 6 Prefill+Attention and 2 Projection runs."""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import count
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from examples.pap.pap_proxy_server import (
        prefill_prefix_len_from_kv_params,
        register_attention_handle,
    )
    from examples.pap.pd_payloads import (
        attach_pap_prefill_attention_params,
        build_decode_payload,
        build_prefill_payload,
        enrich_prefill_kv_params,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from pap_proxy_server import (  # type: ignore[no-redef]
        prefill_prefix_len_from_kv_params,
        register_attention_handle,
    )
    from pd_payloads import (  # type: ignore[no-redef]
        attach_pap_prefill_attention_params,
        build_decode_payload,
        build_prefill_payload,
        enrich_prefill_kv_params,
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multi_pap_proxy")


@dataclass(frozen=True)
class PAPGroup:
    prefill_host: str
    prefill_port: int
    prefill_nixl_port: int
    attention_host: str
    attention_port: int
    attention_tcp_port: int | None = None
    attention_zmq_port: int | None = None

    @property
    def prefill_base_url(self) -> str:
        return f"http://{self.prefill_host}:{self.prefill_port}"

    @property
    def attention_base_url(self) -> str:
        return f"http://{self.attention_host}:{self.attention_port}"

    @property
    def attention_tcp_endpoint(self) -> str | None:
        if self.attention_tcp_port is None:
            return None
        return f"tcp://{self.attention_host}:{self.attention_tcp_port}"

    @property
    def attention_zmq_endpoint(self) -> str | None:
        if self.attention_zmq_port is None:
            return None
        return f"{self.attention_host}:{self.attention_zmq_port}"


@dataclass(frozen=True)
class ProjectionInstance:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class MultiPAPServiceClient:
    client: httpx.AsyncClient
    host: str
    port: int
    base_url: str
    role: str


def _parse_host_port(value: str, *, expected_parts: int, kind: str) -> list[str]:
    parts = value.split(":")
    if len(parts) != expected_parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            f"invalid {kind} spec {value!r}; expected {expected_parts} "
            "colon-separated fields"
        )
    return parts


def parse_pap_groups(spec: str) -> list[PAPGroup]:
    groups: list[PAPGroup] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in {5, 6, 7} or any(part == "" for part in parts):
            raise argparse.ArgumentTypeError(
                f"invalid PAP group spec {item!r}; expected 5, 6, or 7 "
                "colon-separated fields"
            )
        groups.append(
            PAPGroup(
                prefill_host=parts[0],
                prefill_port=int(parts[1]),
                prefill_nixl_port=int(parts[2]),
                attention_host=parts[3],
                attention_port=int(parts[4]),
                attention_tcp_port=None if len(parts) == 5 else int(parts[5]),
                attention_zmq_port=None if len(parts) < 7 else int(parts[6]),
            )
        )
    if not groups:
        raise argparse.ArgumentTypeError("at least one PAP group is required")
    return groups


def parse_projection_instances(spec: str) -> list[ProjectionInstance]:
    projections: list[ProjectionInstance] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = _parse_host_port(item, expected_parts=2, kind="Projection")
        projections.append(ProjectionInstance(host=parts[0], port=int(parts[1])))
    if not projections:
        raise argparse.ArgumentTypeError("at least one Projection instance is required")
    return projections


def select_instances(
    request_number: int,
    groups: list[PAPGroup],
    projections: list[ProjectionInstance],
    *,
    routing_policy: str = "round_robin",
) -> tuple[PAPGroup, ProjectionInstance]:
    group_index = request_number % len(groups)
    group = groups[group_index]
    if routing_policy == "round_robin":
        projection_index = request_number % len(projections)
    elif routing_policy == "projection_affinity":
        groups_per_projection = (len(groups) + len(projections) - 1) // len(
            projections
        )
        projection_index = min(
            group_index // groups_per_projection,
            len(projections) - 1,
        )
    elif routing_policy == "projection_sticky":
        projection_index = request_number % len(projections)
        group_index = projection_index % len(groups)
        group = groups[group_index]
    else:
        raise ValueError(f"unsupported PAP routing policy: {routing_policy}")
    return group, projections[projection_index]


def build_projection_payload_for_group(
    req_data: dict[str, Any],
    kv_transfer_params: dict[str, Any],
    group: PAPGroup,
    *,
    pap_prefill_kv_handle: str | None = None,
    pap_attention_kv_installed: bool = False,
) -> dict[str, Any]:
    kv_params = dict(kv_transfer_params)
    kv_params["pap_attention_endpoint"] = group.attention_base_url
    if group.attention_tcp_endpoint is not None:
        kv_params["pap_attention_tcp_endpoint"] = group.attention_tcp_endpoint
    if group.attention_zmq_endpoint is not None:
        kv_params["pap_offload_exec_zmq_endpoint"] = group.attention_zmq_endpoint
    return build_decode_payload(
        req_data,
        kv_params,
        pap_prefill_kv_handle=pap_prefill_kv_handle,
        pap_attention_kv_installed=pap_attention_kv_installed,
    )


def _headers(request_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def _make_client(host: str, port: int, role: str) -> MultiPAPServiceClient:
    base_url = f"http://{host}:{port}"
    return MultiPAPServiceClient(
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


async def _post_json(
    client: MultiPAPServiceClient,
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
    client: MultiPAPServiceClient,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    app.state.groups = parse_pap_groups(args.pap_groups)
    app.state.projections = parse_projection_instances(args.projections)
    app.state.request_counter = count()
    app.state.prefill_clients = {
        group: _make_client(group.prefill_host, group.prefill_port, "prefill")
        for group in app.state.groups
    }
    app.state.attention_clients = {
        group: _make_client(group.attention_host, group.attention_port, "attention")
        for group in app.state.groups
    }
    app.state.projection_clients = {
        projection: _make_client(projection.host, projection.port, "projection")
        for projection in app.state.projections
    }
    yield
    for client in [
        *app.state.prefill_clients.values(),
        *app.state.attention_clients.values(),
        *app.state.projection_clients.values(),
    ]:
        await client.client.aclose()


app = FastAPI(title="Multi PAP Proxy", lifespan=lifespan)


async def _handle_openai_request(api_path: str, request: Request):
    req_data = await request.json()
    request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    conversation_id = str(req_data.pop("conversation_id", ""))
    client_stream = bool(req_data.get("stream", False))
    request_number = next(request.app.state.request_counter)
    group, projection = select_instances(
        request_number,
        request.app.state.groups,
        request.app.state.projections,
        routing_policy=request.app.state.args.routing_policy,
    )
    prefill = request.app.state.prefill_clients[group]
    attention = request.app.state.attention_clients[group]
    projection_client = request.app.state.projection_clients[projection]

    attention_session = await register_attention_handle(
        attention,
        request_id=request_id,
        conversation_id=conversation_id,
        prefill_endpoint=group.prefill_base_url,
        kv_transfer_params={},
        prefix_len=None,
    )

    prefill_payload = attach_pap_prefill_attention_params(
        build_prefill_payload(req_data),
        pap_attention_endpoint=group.attention_base_url,
        pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
        pap_prefill_kv_handle=str(attention_session.get("prefill_kv_handle")),
        pap_mode=request.app.state.args.pap_mode,
    )
    t0 = time.time()
    prefill_resp = await _post_json(prefill, api_path, prefill_payload, request_id)
    prefill_ms = int((time.time() - t0) * 1000)

    kv_params = enrich_prefill_kv_params(
        prefill_resp.get("kv_transfer_params") or {},
        prefill_host=group.prefill_host,
        prefill_nixl_port=group.prefill_nixl_port,
    )
    prefix_len = prefill_prefix_len_from_kv_params(kv_params)
    projection_payload = build_projection_payload_for_group(
        req_data,
        kv_params,
        group,
        pap_prefill_kv_handle=attention_session.get("prefill_kv_handle"),
        pap_attention_kv_installed=True,
    )
    projection_payload.setdefault("stream", client_stream)
    logger.info(
        "request_id=%s pa=%s:%d attention=%s:%d projection=%s:%d "
        "prefill_ms=%d prefill_prefix_len=%s",
        request_id,
        group.prefill_host,
        group.prefill_port,
        group.attention_host,
        group.attention_port,
        projection.host,
        projection.port,
        prefill_ms,
        prefix_len,
    )

    if client_stream:
        return StreamingResponse(
            _stream_projection(
                projection_client,
                api_path,
                projection_payload,
                request_id,
            ),
            media_type="text/event-stream",
            headers={
                "X-PAP-Prefill-Ms": str(prefill_ms),
                "X-PAP-Group": str(request_number % len(request.app.state.groups)),
                "X-PAP-Projection": str(projection.port),
            },
        )

    projection_resp = await _post_json(
        projection_client,
        api_path,
        projection_payload,
        request_id,
    )
    return JSONResponse(
        projection_resp,
        headers={
            "X-PAP-Prefill-Ms": str(prefill_ms),
            "X-PAP-Group": str(request_number % len(request.app.state.groups)),
            "X-PAP-Projection": str(projection.port),
        },
    )


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
        "role": "multi-pap-proxy",
        "groups": len(app.state.groups),
        "projections": len(app.state.projections),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-instance PAP proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--pap-groups",
        required=True,
        help=(
            "Comma-separated prefill_host:prefill_port:prefill_nixl_port:"
            "attention_host:attention_port entries"
        ),
    )
    parser.add_argument(
        "--projections",
        required=True,
        help="Comma-separated projection_host:projection_port entries",
    )
    parser.add_argument("--pap-mode", default=os.environ.get("PAP_MODE", "true_split"))
    parser.add_argument(
        "--routing-policy",
        default=os.environ.get("PAP_ROUTING_POLICY", "round_robin"),
        choices=("round_robin", "projection_affinity", "projection_sticky"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)
