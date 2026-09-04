# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway KV handoff and Projection streaming lifecycle."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from vllm.pap.gateway.clients import PAPServiceClient, register_attention_handle
from vllm.pap.gateway.clients import request_headers as _headers
from vllm.pap.gateway.lifecycle import (
    PAPRequestLifecycle,
    release_attention_sessions,
)
from vllm.pap.gateway.observability import (
    _merge_prefill_cache_usage_sse_event,
    _pap_prefill_ipc_profile_enabled,
    _PrefillCacheUsage,
)

logger = logging.getLogger("pap_gateway")


@dataclass
class OpenProjectionStream:
    response: httpx.Response
    start: float
    profile: bool
    profile_origin: float = 0.0


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
    """Register the request with every Attention instance in its PA group."""
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
    except BaseException:
        cleanup_task = asyncio.create_task(
            release_attention_sessions(registered_attentions, request_id)
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cleanup_task.add_done_callback(_log_registration_cleanup_failure)
        raise
    return sessions


def _log_registration_cleanup_failure(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("PAP partial Attention registration cleanup failed")


_cleanup_attention_sessions = release_attention_sessions


async def _stream_projection(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
):
    profile = _pap_prefill_ipc_profile_enabled()
    start = time.perf_counter() if profile else 0.0
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
        async for chunk in _stream_projection_response(
            resp,
            request_id,
            start=start,
            profile=profile,
        ):
            yield chunk


async def open_projection_stream(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    *,
    profile_origin: float = 0.0,
) -> OpenProjectionStream:
    """Receive Projection headers before committing the downstream response."""
    profile = _pap_prefill_ipc_profile_enabled()
    start = time.perf_counter() if profile else 0.0
    request = client.client.build_request(
        "POST",
        endpoint,
        json=payload,
        headers=_headers(request_id),
    )
    response = await client.client.send(request, stream=True)
    try:
        response.raise_for_status()
    except Exception:
        await response.aclose()
        raise
    if profile:
        logger.info(
            "PAP proxy projection stream profile request_id=%s open_ms=%.3f",
            request_id,
            (time.perf_counter() - start) * 1000.0,
        )
    return OpenProjectionStream(response, start, profile, profile_origin)


async def _stream_projection_response(
    response: httpx.Response,
    request_id: str,
    *,
    start: float,
    profile: bool,
    profile_origin: float = 0.0,
):
    first_chunk = True
    chunk_count = 0
    byte_count = 0
    async for chunk in response.aiter_bytes():
        if profile:
            chunk_count += 1
            byte_count += len(chunk)
            if first_chunk:
                first_chunk = False
                logger.info(
                    "PAP proxy projection stream profile request_id=%s "
                    "first_chunk_ms=%.3f request_to_first_chunk_ms=%.3f "
                    "first_chunk_bytes=%d",
                    request_id,
                    (time.perf_counter() - start) * 1000.0,
                    (
                        (time.perf_counter() - profile_origin) * 1000.0
                        if profile_origin
                        else 0.0
                    ),
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


async def _stream_with_prefill_cache_usage(
    chunks: AsyncIterator[bytes],
    prefill_usage: _PrefillCacheUsage | None,
) -> AsyncIterator[bytes]:
    if prefill_usage is None:
        async for chunk in chunks:
            yield chunk
        return

    pending = b""
    async for chunk in chunks:
        pending += chunk
        while True:
            boundary = pending.find(b"\n\n")
            boundary_size = 2
            crlf_boundary = pending.find(b"\r\n\r\n")
            if crlf_boundary >= 0 and (boundary < 0 or crlf_boundary < boundary):
                boundary = crlf_boundary
                boundary_size = 4
            if boundary < 0:
                break
            event = pending[:boundary]
            pending = pending[boundary + boundary_size :]
            delimiter = b"\r\n\r\n" if boundary_size == 4 else b"\n\n"
            merged_event = _merge_prefill_cache_usage_sse_event(
                event,
                prefill_usage,
            )
            yield merged_event + delimiter
    if pending:
        yield pending


async def _stream_projection_with_cleanup(
    client: PAPServiceClient,
    endpoint: str,
    payload: dict[str, Any],
    request_id: str,
    lifecycle: PAPRequestLifecycle,
    opened_stream: OpenProjectionStream | None = None,
    prefill_usage: _PrefillCacheUsage | None = None,
):
    """Stream Projection output and release the fixed PA lifecycle."""
    terminal_marker = b"data: [DONE]"
    pending = b""
    terminal_chunks: list[bytes] = []
    try:
        if opened_stream is None:
            projection_chunks = _stream_projection(
                client,
                endpoint,
                payload,
                request_id,
            )
        else:
            projection_chunks = _stream_projection_response(
                opened_stream.response,
                request_id,
                start=opened_stream.start,
                profile=opened_stream.profile,
                profile_origin=opened_stream.profile_origin,
            )
        projection_chunks = _stream_with_prefill_cache_usage(
            projection_chunks,
            prefill_usage,
        )
        async for chunk in projection_chunks:
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
        if terminal_chunks:
            lifecycle.mark_projection_completed()
    finally:
        await lifecycle.terminate("projection_stream_closed")
    for chunk in terminal_chunks:
        yield chunk
