# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway KV handoff and Projection streaming lifecycle."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.clients import PAPServiceClient, register_attention_handle
from vllm.pap.gateway.clients import request_headers as _headers
from vllm.pap.gateway.observability import _pap_prefill_ipc_profile_enabled
from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance

logger = logging.getLogger("pap_gateway")


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
    """Release all Attention sessions associated with one request."""
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
    on_cleanup_complete: Callable[[], None] | None = None,
):
    """Stream Projection output and release the fixed PA lifecycle."""
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
            try:
                await admission.release(group, projection)
            finally:
                if on_cleanup_complete is not None:
                    on_cleanup_complete()
    for chunk in terminal_chunks:
        yield chunk
