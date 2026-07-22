# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible gateway for arbitrary PAP topologies."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.pap.config import reject_removed_pap_flags
from vllm.pap.gateway.clients import (
    PAPServiceClient,
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    register_attention_handle,
    wait_attention_prefill_ready,
)
from vllm.pap.gateway.clients import (
    request_headers as _headers,
)
from vllm.pap.gateway.payloads import (
    attach_pap_prefill_attention_params,
    build_prefill_payload,
    build_projection_kv_unaware_payload,
    enrich_prefill_kv_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_gateway")


def _pap_prefill_ipc_profile_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_IPC_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _prefill_usage_headers(prefill_response: dict[str, Any]) -> dict[str, str]:
    usage = prefill_response.get("usage")
    if not isinstance(usage, dict):
        return {}

    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        return {}

    headers = {
        "X-PAP-Prefill-Prompt-Tokens": str(prompt_tokens),
    }
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return headers

    cached_tokens = details.get("cached_tokens")
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        return headers

    headers["X-PAP-Prefill-Cached-Tokens"] = str(cached_tokens)
    if cached_tokens <= prompt_tokens:
        headers["X-PAP-Prefill-Computed-Tokens"] = str(prompt_tokens - cached_tokens)
    return headers


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


class PAPConversationRouter:
    """Keep a conversation on one PA while balancing new conversations."""

    def __init__(self, groups: list[PAPGroup]) -> None:
        if not groups:
            raise ValueError("PAP conversation routing requires a PA group")
        self._groups = groups
        self._group_indices = {group: index for index, group in enumerate(groups)}
        self._next_group = count()
        self._assignments: dict[str, PAPGroup] = {}
        self._request_counts: Counter[PAPGroup] = Counter()

    def select_group(
        self,
        conversation_id: str,
        *,
        request_number: int,
    ) -> PAPGroup:
        """Return the resident PA or round-robin a new conversation."""
        if conversation_id:
            group = self._assignments.get(conversation_id)
            if group is None:
                group = self._groups[next(self._next_group) % len(self._groups)]
                self._assignments[conversation_id] = group
        else:
            group = self._groups[request_number % len(self._groups)]
        self._request_counts[group] += 1
        return group

    def snapshot(self) -> dict[str, Any]:
        """Return token-free assignment and request counts by PA."""
        assignment_counts = Counter(self._assignments.values())
        return {
            "conversations": len(self._assignments),
            "pa_assignments": {
                str(self._group_indices[group]): assignment_counts[group]
                for group in self._groups
            },
            "pa_requests": {
                str(self._group_indices[group]): self._request_counts[group]
                for group in self._groups
            },
        }


@dataclass
class _PAPProjectionAdmissionState:
    owner: ProjectionInstance | None = None
    active_requests: int = 0
    waiters: list[tuple[object, ProjectionInstance]] = field(default_factory=list)


class PAPProjectionAdmission:
    """Keep each PA on one Projection source for a complete request wave."""

    def __init__(self, groups: list[PAPGroup]) -> None:
        self._condition = asyncio.Condition()
        self._states = {group: _PAPProjectionAdmissionState() for group in groups}
        self._group_indices = {group: index for index, group in enumerate(groups)}

    async def acquire(
        self,
        group: PAPGroup,
        projection: ProjectionInstance,
    ) -> None:
        """Admit a request without changing the PA owner mid-wave."""
        ticket = object()
        async with self._condition:
            state = self._states[group]
            state.waiters.append((ticket, projection))
            try:
                while True:
                    if state.owner is None:
                        state.owner = state.waiters[0][1]
                        self._condition.notify_all()
                    if state.owner == projection and self._is_next_owner_ticket(
                        state,
                        ticket,
                    ):
                        state.waiters = [
                            item for item in state.waiters if item[0] is not ticket
                        ]
                        state.active_requests += 1
                        self._condition.notify_all()
                        return
                    await self._condition.wait()
            except BaseException:
                state.waiters = [
                    item for item in state.waiters if item[0] is not ticket
                ]
                if state.active_requests == 0 and not any(
                    waiting_projection == state.owner
                    for _, waiting_projection in state.waiters
                ):
                    state.owner = None
                self._condition.notify_all()
                raise

    @staticmethod
    def _is_next_owner_ticket(
        state: _PAPProjectionAdmissionState,
        ticket: object,
    ) -> bool:
        for waiting_ticket, waiting_projection in state.waiters:
            if waiting_ticket is ticket:
                return True
            if waiting_projection != state.owner:
                return False
        return False

    async def release(
        self,
        group: PAPGroup,
        projection: ProjectionInstance,
    ) -> None:
        """Release one request and hand the idle PA to the next source."""
        async with self._condition:
            state = self._states[group]
            if state.owner != projection or state.active_requests <= 0:
                raise RuntimeError("invalid PAP Projection admission release")
            state.active_requests -= 1
            if state.active_requests == 0:
                state.owner = None
            self._condition.notify_all()

    async def snapshot(self) -> list[dict[str, int | None]]:
        """Return the current PA admission state for audits."""
        async with self._condition:
            return [
                {
                    "pa_index": self._group_indices[group],
                    "projection_port": (
                        None if state.owner is None else state.owner.port
                    ),
                    "active_requests": state.active_requests,
                    "waiting_requests": len(state.waiters),
                }
                for group, state in self._states.items()
            ]


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
    conversation_id: str = "",
    conversation_router: PAPConversationRouter | None = None,
) -> tuple[PAPGroup, ProjectionInstance]:
    group_index = request_number % len(groups)
    group = groups[group_index]
    if routing_policy == "round_robin":
        projection_index = request_number % len(projections)
    elif routing_policy == "crossbar_round_robin":
        projection_index = (request_number // len(groups) + group_index) % len(
            projections
        )
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
    elif routing_policy == "conversation_affinity":
        if conversation_router is None:
            raise ValueError("conversation_affinity requires a PAPConversationRouter")
        group = conversation_router.select_group(
            conversation_id,
            request_number=request_number,
        )
        projection_index = request_number % len(projections)
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


async def register_attention_handles(
    attention_clients: list[PAPServiceClient],
    *,
    request_id: str,
    conversation_id: str,
    prefill_endpoint: str,
    kv_transfer_params: dict[str, Any],
    prefix_len: int | None,
) -> list[dict[str, Any]]:
    sessions = []
    registered_attentions: list[PAPServiceClient] = []
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
    attention: PAPServiceClient,
    request_id: str,
) -> None:
    resp = await attention.client.delete(
        f"/v1/pap/attention/sessions/{request_id}",
        headers=_headers(request_id),
    )
    resp.raise_for_status()


async def _cleanup_attention_sessions(
    attention_clients: list[PAPServiceClient],
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
    client: PAPServiceClient,
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
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    attention_clients: list[PAPServiceClient],
    admission: PAPProjectionAdmission,
    group: PAPGroup,
    projection: ProjectionInstance,
):
    terminal_marker = b"data: [DONE]"
    pending = b""
    terminal_chunks: list[bytes] = []
    try:
        async for chunk in _stream_projection(client, endpoint, payload, request_id):
            if terminal_chunks:
                terminal_chunks.append(chunk)
                continue
            pending += chunk
            marker_index = pending.find(terminal_marker)
            if marker_index < 0:
                safe_length = len(pending) - len(terminal_marker) + 1
                if safe_length > 0:
                    yield pending[:safe_length]
                    pending = pending[safe_length:]
                continue
            if marker_index:
                yield pending[:marker_index]
            terminal_chunks.append(pending[marker_index:])
            pending = b""
        if pending:
            yield pending
    finally:
        try:
            await _cleanup_attention_sessions(attention_clients, request_id)
        finally:
            await admission.release(group, projection)
    for chunk in terminal_chunks:
        yield chunk


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    reject_removed_pap_flags(os.environ)
    app.state.groups = parse_pap_groups(args.pap_groups)
    app.state.projections = parse_projection_instances(args.projections)
    app.state.request_counter = count()
    app.state.pair_counts = Counter()
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
    app.state.conversation_router = PAPConversationRouter(app.state.groups)
    app.state.projection_admission = PAPProjectionAdmission(app.state.groups)
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


app = FastAPI(title="PAP Gateway", lifespan=lifespan)


def _pop_conversation_id(
    req_data: dict[str, Any],
    correlation_id: str | None,
) -> str:
    """Return the body conversation id or a session-header fallback."""

    raw_conversation_id = req_data.pop("conversation_id", None)
    if raw_conversation_id is not None:
        conversation_id = str(raw_conversation_id)
        if conversation_id:
            return conversation_id
    return correlation_id or ""


async def _handle_openai_request(api_path: str, request: Request):
    profile = _pap_prefill_ipc_profile_enabled()
    request_start = time.perf_counter() if profile else 0.0
    req_data = await request.json()
    request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    conversation_id = _pop_conversation_id(
        req_data,
        request.headers.get("X-Correlation-ID"),
    )
    client_stream = bool(req_data.get("stream", False))
    request_number = next(request.app.state.request_counter)
    group, projection = select_instances(
        request_number,
        request.app.state.groups,
        request.app.state.projections,
        routing_policy=request.app.state.args.routing_policy,
        conversation_id=conversation_id,
        conversation_router=request.app.state.conversation_router,
    )
    prefill = request.app.state.prefill_clients[group]
    attention_clients = request.app.state.attention_clients[group]
    projection_client = request.app.state.projection_clients[projection]
    group_index = request.app.state.groups.index(group)
    projection_index = request.app.state.projections.index(projection)
    pair_name = f"pa{group_index}:p{projection_index}"
    request.app.state.pair_counts[pair_name] += 1

    attention_sessions: list[dict[str, Any]] | None = None
    handed_off_stream_cleanup = False
    projection_admitted = False
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
            "pa_index=%d projection_index=%d pair=%s "
            "prefill_ms=%d prefill_prefix_len=%s attention_ready=%s "
            "projection_kv_keys=%s",
            request_id,
            group.prefill_host,
            group.prefill_port,
            group.attention_host,
            group.attention_port,
            projection.host,
            projection.port,
            group_index,
            projection_index,
            pair_name,
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

        response_headers = {
            "X-PAP-Prefill-Ms": str(prefill_ms),
            "X-PAP-Group": str(group_index),
            "X-PAP-Projection": str(projection.port),
            "X-PAP-Projection-Index": str(projection_index),
            "X-PAP-Pair": pair_name,
        }
        response_headers.update(_prefill_usage_headers(prefill_resp))

        admission = request.app.state.projection_admission
        await admission.acquire(group, projection)
        projection_admitted = True

        if client_stream:
            handed_off_stream_cleanup = True
            projection_admitted = False
            return StreamingResponse(
                _stream_projection_with_cleanup(
                    projection_client,
                    api_path,
                    projection_payload,
                    request_id,
                    attention_clients,
                    admission,
                    group,
                    projection,
                ),
                media_type="text/event-stream",
                headers=response_headers,
            )

        projection_resp = await _post_json(
            projection_client,
            api_path,
            projection_payload,
            request_id,
        )
        return JSONResponse(
            projection_resp,
            headers=response_headers,
        )
    finally:
        if not handed_off_stream_cleanup:
            try:
                if attention_sessions is not None:
                    await _cleanup_attention_sessions(
                        attention_clients,
                        request_id,
                    )
            finally:
                if projection_admitted:
                    await request.app.state.projection_admission.release(
                        group,
                        projection,
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
        "routing_policy": app.state.args.routing_policy,
        "pair_counts": dict(sorted(app.state.pair_counts.items())),
        "conversation_routing": app.state.conversation_router.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
    }


@app.get("/v1/pap/topology/stats")
async def topology_stats() -> dict[str, Any]:
    pair_counts = dict(sorted(app.state.pair_counts.items()))
    return {
        "pa_count": len(app.state.groups),
        "projection_count": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "total_requests": sum(pair_counts.values()),
        "pair_counts": pair_counts,
        "conversation_routing": app.state.conversation_router.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PAP request gateway")
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
        choices=(
            "round_robin",
            "crossbar_round_robin",
            "projection_affinity",
            "projection_sticky",
            "conversation_affinity",
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the PAP gateway."""
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)


if __name__ == "__main__":
    main()
