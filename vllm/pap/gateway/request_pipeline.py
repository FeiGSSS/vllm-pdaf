# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI request pipeline for the PAP gateway."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.pap.gateway.clients import (
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    wait_attention_prefill_ready,
)
from vllm.pap.gateway.handoff import (
    _cleanup_attention_sessions,
    _post_json,
    _stream_projection_with_cleanup,
    register_attention_handles,
)
from vllm.pap.gateway.observability import (
    _pap_prefill_ipc_profile_enabled,
    _prefill_usage_headers,
)
from vllm.pap.gateway.payloads import (
    attach_pap_prefill_attention_params,
    build_prefill_payload,
)
from vllm.pap.gateway.routing import select_instances
from vllm.pap.gateway.topology import build_projection_payload_for_group

logger = logging.getLogger("pap_gateway")


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
    projection_client = request.app.state.projection_clients[projection]
    projection_index = request.app.state.projections.index(projection)
    group_index = request.app.state.groups.index(group)
    prefill = request.app.state.prefill_clients[group]
    attention_clients = request.app.state.attention_clients[group]

    attention_sessions: list[dict[str, Any]] | None = None
    handed_off_stream_cleanup = False
    projection_admitted = False
    active_request_ids: set[str] = request.app.state.pap_active_request_ids
    active_request_ids.add(request_id)
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
            pap_prefill_kv_handle=str(attention_session.get("prefill_kv_handle")),
            pap_mode=request.app.state.args.pap_mode,
        )
        prefill_payload_ms = (
            (time.perf_counter() - prefill_payload_start) * 1000.0 if profile else 0.0
        )
        prefill_start = time.time()
        prefill_resp = await _post_json(
            prefill,
            api_path,
            prefill_payload,
            request_id,
        )
        prefill_ms = int((time.time() - prefill_start) * 1000)

        prompt_token_ids = prefill_resp.get("prompt_token_ids")
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(not isinstance(token_id, int) for token_id in prompt_token_ids)
        ):
            raise RuntimeError("PAP Prefill returned no valid prompt token IDs")
        prompt_text = prefill_resp.get("prompt_text")
        if not isinstance(prompt_text, str):
            prompt_text = None

        projection_payload_start = time.perf_counter() if profile else 0.0
        kv_params = dict(prefill_resp.get("kv_transfer_params") or {})
        prefill_kv_handle = prefill_kv_handle_from_kv_params(
            kv_params,
            fallback=attention_session.get("prefill_kv_handle"),
        )
        prefix_len = prefill_prefix_len_from_kv_params(kv_params)
        if prefix_len is None or len(prompt_token_ids) != prefix_len:
            raise RuntimeError(
                "PAP Prefill prompt token length mismatch "
                f"prompt_token_ids={len(prompt_token_ids)} prefix_len={prefix_len}"
            )
        attention_ready = all(
            [
                await wait_attention_prefill_ready(
                    attention,
                    request_id,
                    expected_prefix_len=prefix_len,
                    expected_session_handle=str(session["prefill_kv_handle"]),
                )
                for attention, session in zip(
                    attention_clients,
                    attention_sessions,
                )
            ]
        )
        projection_payload = build_projection_payload_for_group(
            req_data,
            kv_params,
            group,
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
            pap_prefill_kv_handle=prefill_kv_handle,
            pap_attention_kv_installed=attention_ready,
        )
        projection_payload.setdefault("stream", client_stream)
        projection_kv_params = projection_payload.get("kv_transfer_params") or {}
        pair_name = f"pa{group_index}:p{projection_index}"
        request.app.state.pair_counts[pair_name] += 1
        projection_payload_ms = (
            (time.perf_counter() - projection_payload_start) * 1000.0
            if profile
            else 0.0
        )
        logger.info(
            "request_id=%s pa=%s:%d attention=%s:%s projection=%s:%d "
            "pa_index=%d projection_index=%d pair=%s prefill_ms=%d "
            "prefill_prefix_len=%s attention_ready=%s projection_kv_keys=%s",
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
                "prefill_payload_ms=%.3f prefill_ms=%d "
                "projection_payload_ms=%.3f pre_projection_ms=%.3f",
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
                    on_cleanup_complete=lambda: active_request_ids.discard(request_id),
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
        return JSONResponse(projection_resp, headers=response_headers)
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
            active_request_ids.discard(request_id)
