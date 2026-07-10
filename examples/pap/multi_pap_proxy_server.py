# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-instance proxy for arbitrary PAP Prefill+Attention/Projection ratios."""

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
        prefill_kv_handle_from_kv_params,
        prefill_prefix_len_from_kv_params,
        register_attention_handle,
        wait_attention_prefill_ready,
    )
    from examples.pap.pd_payloads import (
        attach_pap_prefill_attention_params,
        build_prefill_payload,
        build_projection_kv_unaware_payload,
        enrich_prefill_kv_params,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from pap_proxy_server import (  # type: ignore[no-redef]
        prefill_kv_handle_from_kv_params,
        prefill_prefix_len_from_kv_params,
        register_attention_handle,
        wait_attention_prefill_ready,
    )
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
logger = logging.getLogger("multi_pap_proxy")


def _pap_prefill_ipc_profile_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_IPC_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


PortSpec = int | tuple[int, ...]


def _parse_port_spec(value: str) -> PortSpec:
    if "|" not in value:
        return int(value)
    ports = tuple(int(part) for part in value.split("|") if part)
    if not ports:
        raise argparse.ArgumentTypeError(f"invalid empty ranked port spec {value!r}")
    return ports


def _format_ranked_endpoints(
    host: str,
    ports: PortSpec,
    *,
    scheme: str,
) -> str:
    if isinstance(ports, int):
        return f"{scheme}{host}:{ports}"
    return ",".join(f"{scheme}{host}:{port}" for port in ports)


@dataclass(frozen=True)
class PAPGroup:
    prefill_host: str
    prefill_port: int
    prefill_nixl_port: int
    attention_host: str
    attention_port: PortSpec
    attention_tcp_port: PortSpec | None = None
    attention_zmq_port: PortSpec | None = None

    @property
    def prefill_base_url(self) -> str:
        return f"http://{self.prefill_host}:{self.prefill_port}"

    @property
    def attention_base_url(self) -> str:
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_port,
            scheme="http://",
        )

    @property
    def attention_base_urls(self) -> tuple[str, ...]:
        return tuple(self.attention_base_url.split(","))

    @property
    def attention_tcp_endpoint(self) -> str | None:
        if self.attention_tcp_port is None:
            return None
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_tcp_port,
            scheme="tcp://",
        )

    @property
    def attention_zmq_endpoint(self) -> str | None:
        if self.attention_zmq_port is None:
            return None
        return _format_ranked_endpoints(
            self.attention_host,
            self.attention_zmq_port,
            scheme="",
        )


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
                attention_port=_parse_port_spec(parts[4]),
                attention_tcp_port=None
                if len(parts) == 5
                else _parse_port_spec(parts[5]),
                attention_zmq_port=None
                if len(parts) < 7
                else _parse_port_spec(parts[6]),
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
        groups_per_projection = (len(groups) + len(projections) - 1) // len(projections)
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
    return build_projection_kv_unaware_payload(
        req_data,
        kv_transfer_params,
        pap_attention_endpoint=group.attention_base_url,
        pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
        pap_offload_exec_zmq_endpoint=group.attention_zmq_endpoint,
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


async def register_attention_handles(
    attention_clients: list[MultiPAPServiceClient],
    *,
    request_id: str,
    conversation_id: str,
    prefill_endpoint: str,
    kv_transfer_params: dict[str, Any],
    prefix_len: int | None,
) -> list[dict[str, Any]]:
    sessions = []
    registered_attentions: list[MultiPAPServiceClient] = []
    try:
        for attention in attention_clients:
            sessions.append(
                await register_attention_handle(
                    attention,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    prefill_endpoint=prefill_endpoint,
                    kv_transfer_params=kv_transfer_params,
                    prefix_len=prefix_len,
                )
            )
            registered_attentions.append(attention)
    except Exception:
        await _cleanup_attention_sessions(registered_attentions, request_id)
        raise
    return sessions


async def _delete_attention_session(
    attention: MultiPAPServiceClient,
    request_id: str,
) -> None:
    resp = await attention.client.delete(
        f"/v1/pap/attention/sessions/{request_id}",
        headers=_headers(request_id),
    )
    resp.raise_for_status()


async def _cleanup_attention_sessions(
    attention_clients: list[MultiPAPServiceClient],
    request_id: str,
) -> None:
    for attention in attention_clients:
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


async def _stream_projection(
    client: MultiPAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
):
    profile = _pap_prefill_ipc_profile_enabled()
    start = time.perf_counter() if profile else 0.0
    first_chunk = True
    chunk_count = 0
    byte_count = 0
    async with client.client.stream(
        "POST",
        endpoint,
        json=payload,
        headers=_headers(request_id),
    ) as resp:
        resp.raise_for_status()
        if profile:
            logger.info(
                "PAP proxy projection stream profile request_id=%s open_ms=%.3f",
                request_id,
                (time.perf_counter() - start) * 1000.0,
            )
        async for chunk in resp.aiter_bytes():
            if profile:
                chunk_count += 1
                byte_count += len(chunk)
                if first_chunk:
                    first_chunk = False
                    logger.info(
                        "PAP proxy projection stream profile request_id=%s "
                        "first_chunk_ms=%.3f first_chunk_bytes=%d",
                        request_id,
                        (time.perf_counter() - start) * 1000.0,
                        len(chunk),
                    )
            yield chunk
    if profile:
        logger.info(
            "PAP proxy projection stream profile request_id=%s total_ms=%.3f "
            "chunks=%d bytes=%d",
            request_id,
            (time.perf_counter() - start) * 1000.0,
            chunk_count,
            byte_count,
        )


async def _stream_projection_with_cleanup(
    client: MultiPAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    attention_clients: list[MultiPAPServiceClient],
):
    try:
        async for chunk in _stream_projection(client, endpoint, payload, request_id):
            yield chunk
    finally:
        await _cleanup_attention_sessions(attention_clients, request_id)


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
    app.state.attention_clients = {}
    for group in app.state.groups:
        if isinstance(group.attention_port, int):
            ports = (group.attention_port,)
        else:
            ports = group.attention_port
        app.state.attention_clients[group] = [
            _make_client(group.attention_host, port, "attention") for port in ports
        ]
    app.state.projection_clients = {
        projection: _make_client(projection.host, projection.port, "projection")
        for projection in app.state.projections
    }
    yield
    attention_clients = [
        client for clients in app.state.attention_clients.values() for client in clients
    ]
    for client in [
        *app.state.prefill_clients.values(),
        *attention_clients,
        *app.state.projection_clients.values(),
    ]:
        await client.client.aclose()


app = FastAPI(title="Multi PAP Proxy", lifespan=lifespan)


async def _handle_openai_request(api_path: str, request: Request):
    profile = _pap_prefill_ipc_profile_enabled()
    request_start = time.perf_counter() if profile else 0.0
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
    attention_clients = request.app.state.attention_clients[group]
    projection_client = request.app.state.projection_clients[projection]
    group_index = request.app.state.groups.index(group)

    attention_sessions: list[dict[str, Any]] | None = None
    handed_off_stream_cleanup = False
    try:
        register_start = time.perf_counter() if profile else 0.0
        attention_sessions = await register_attention_handles(
            attention_clients,
            request_id=request_id,
            conversation_id=conversation_id,
            prefill_endpoint=group.prefill_base_url,
            kv_transfer_params={},
            prefix_len=None,
        )
        register_ms = (
            (time.perf_counter() - register_start) * 1000.0 if profile else 0.0
        )
        attention_session = attention_sessions[0]

        prefill_payload_start = time.perf_counter() if profile else 0.0
        prefill_payload = attach_pap_prefill_attention_params(
            build_prefill_payload(req_data),
            pap_attention_endpoint=group.attention_base_url,
            pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
            pap_offload_exec_zmq_endpoint=group.attention_zmq_endpoint,
            pap_prefill_kv_handle=str(attention_session.get("prefill_kv_handle")),
            pap_mode=request.app.state.args.pap_mode,
        )
        prefill_payload_ms = (
            (time.perf_counter() - prefill_payload_start) * 1000.0 if profile else 0.0
        )
        t0 = time.time()
        prefill_resp = await _post_json(prefill, api_path, prefill_payload, request_id)
        prefill_ms = int((time.time() - t0) * 1000)

        projection_payload_start = time.perf_counter() if profile else 0.0
        kv_params = enrich_prefill_kv_params(
            prefill_resp.get("kv_transfer_params") or {},
            prefill_host=group.prefill_host,
            prefill_nixl_port=group.prefill_nixl_port,
        )
        prefill_kv_handle = prefill_kv_handle_from_kv_params(
            kv_params,
            fallback=attention_session.get("prefill_kv_handle"),
        )
        prefix_len = prefill_prefix_len_from_kv_params(kv_params)
        attention_ready = False
        if prefix_len is not None:
            attention_ready = all(
                [
                    await wait_attention_prefill_ready(attention, request_id)
                    for attention in attention_clients
                ]
            )
        projection_payload = build_projection_payload_for_group(
            req_data,
            kv_params,
            group,
            pap_prefill_kv_handle=prefill_kv_handle,
            pap_attention_kv_installed=attention_ready,
        )
        projection_payload.setdefault("stream", client_stream)
        projection_kv_params = projection_payload.get("kv_transfer_params") or {}
        projection_payload_ms = (
            (time.perf_counter() - projection_payload_start) * 1000.0
            if profile
            else 0.0
        )
        logger.info(
            "request_id=%s pa=%s:%d attention=%s:%s projection=%s:%d "
            "prefill_ms=%d prefill_prefix_len=%s attention_ready=%s "
            "projection_kv_keys=%s",
            request_id,
            group.prefill_host,
            group.prefill_port,
            group.attention_host,
            group.attention_port,
            projection.host,
            projection.port,
            prefill_ms,
            prefix_len,
            attention_ready,
            sorted(projection_kv_params.keys()),
        )
        if profile:
            logger.info(
                "PAP proxy prefill IPC profile request_id=%s register_ms=%.3f "
                "prefill_payload_ms=%.3f prefill_ms=%d projection_payload_ms=%.3f "
                "pre_projection_ms=%.3f",
                request_id,
                register_ms,
                prefill_payload_ms,
                prefill_ms,
                projection_payload_ms,
                (time.perf_counter() - request_start) * 1000.0,
            )

        if client_stream:
            handed_off_stream_cleanup = True
            return StreamingResponse(
                _stream_projection_with_cleanup(
                    projection_client,
                    api_path,
                    projection_payload,
                    request_id,
                    attention_clients,
                ),
                media_type="text/event-stream",
                headers={
                    "X-PAP-Prefill-Ms": str(prefill_ms),
                    "X-PAP-Group": str(group_index),
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
                "X-PAP-Group": str(group_index),
                "X-PAP-Projection": str(projection.port),
            },
        )
    finally:
        if attention_sessions is not None and not handed_off_stream_cleanup:
            await _cleanup_attention_sessions(attention_clients, request_id)


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
    parser.add_argument("--pap-mode", default=os.environ.get("PAP_MODE", "pap"))
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
