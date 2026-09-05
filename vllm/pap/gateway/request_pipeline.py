# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI request pipeline for the PAP gateway."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from vllm.pap.gateway.clients import (
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    wait_attention_prefill_ready,
)
from vllm.pap.gateway.handoff import (
    _post_json,
    _stream_projection_with_cleanup,
    open_projection_stream,
    register_attention_handles,
)
from vllm.pap.gateway.observability import (
    _extract_prefill_cache_usage,
    _merge_prefill_cache_usage,
    _pap_prefill_ipc_profile_enabled,
    _prefill_usage_headers,
)
from vllm.pap.gateway.payloads import (
    attach_pap_prefill_attention_params,
    build_prefill_payload,
    requested_decode_capacity,
)
from vllm.pap.gateway.topology import (
    build_projection_payload_for_group,
    select_projection_for_group,
)

logger = logging.getLogger("pap_gateway")


async def _cancel_on_client_disconnect(
    request: Request,
    request_task: asyncio.Task[Any],
    disconnected: asyncio.Event | None = None,
) -> None:
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            if disconnected is not None:
                disconnected.set()
            request_task.cancel("PAP client disconnected")
            return


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
    if req_data.get("cache_salt") is not None:
        return JSONResponse(
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "PAP Dynamo routing does not support cache_salt isolation; "
                        "the current selector cannot index salted KV events safely."
                    ),
                }
            },
            status_code=400,
        )
    request_id = request.headers.get("X-Request-Id", uuid.uuid4().hex)
    pending_route_request_ids: set[str] = (
        request.app.state.pap_pending_route_request_ids
    )
    if (
        request_id in pending_route_request_ids
        or request_id in request.app.state.pap_active_request_ids
    ):
        return JSONResponse({"error": "duplicate active X-Request-Id"}, status_code=409)
    pending_route_request_ids.add(request_id)
    request_task = asyncio.current_task()
    if request_task is None:
        raise RuntimeError("PAP request pipeline has no asyncio task")
    client_disconnected = asyncio.Event()
    disconnect_watcher = asyncio.create_task(
        _cancel_on_client_disconnect(
            request,
            request_task,
            client_disconnected,
        ),
        name=f"pap-disconnect-{request_id}",
    )
    try:
        return await _handle_parsed_openai_request(
            api_path,
            request,
            req_data=req_data,
            request_id=request_id,
            profile=profile,
            request_start=request_start,
        )
    except asyncio.CancelledError:
        if client_disconnected.is_set():
            return JSONResponse(
                {"error": "client disconnected"},
                status_code=499,
            )
        raise
    finally:
        disconnect_watcher.cancel()
        if request_id in pending_route_request_ids:
            # No lifecycle coordinator took ownership (routing/setup failed).
            request.app.state.pap_dynamo_router.finish_request(request_id)
            request.app.state.pap_load_tracker.finish_request(request_id)
            request.app.state.pap_active_request_ids.discard(request_id)
        pending_route_request_ids.discard(request_id)


async def _handle_parsed_openai_request(
    api_path: str,
    request: Request,
    *,
    req_data: dict[str, Any],
    request_id: str,
    profile: bool,
    request_start: float,
):
    conversation_id = _pop_conversation_id(
        req_data,
        request.headers.get("X-Correlation-ID"),
    )
    client_stream = bool(req_data.get("stream", False))
    tokenizer = request.app.state.pap_prompt_tokenizer
    tokenization_start = time.perf_counter() if profile else 0.0
    prefix_input = await tokenizer.prompt_hashes(api_path, req_data)
    if prefix_input is None:
        raise RuntimeError("Dynamo routing requires Gateway prompt token IDs")
    routing_token_ids, prompt_hashes, routing_prompt_text = prefix_input
    initial_context_tokens = len(routing_token_ids)
    tokenization_ms = (
        (time.perf_counter() - tokenization_start) * 1000.0 if profile else 0.0
    )
    decode_capacity_tokens = requested_decode_capacity(req_data) or 0
    forwarded_prompt_text = (
        routing_prompt_text if req_data.get("return_prompt_text") else None
    )
    dynamo_router = request.app.state.pap_dynamo_router
    routing_start = time.perf_counter() if profile else 0.0
    group = await dynamo_router.select_group(
        routing_token_ids,
        request_id=request_id,
        expected_output_tokens=decode_capacity_tokens,
    )
    projection = select_projection_for_group(
        group, request.app.state.groups, request.app.state.projections
    )
    routing_ms = (time.perf_counter() - routing_start) * 1000.0 if profile else 0.0
    projection_client = request.app.state.projection_clients[projection]
    projection_index = request.app.state.projections.index(projection)
    group_index = request.app.state.groups.index(group)
    prefill = request.app.state.prefill_clients[group]
    attention_clients = request.app.state.attention_clients[group]
    load_tracker = request.app.state.pap_load_tracker
    load_tracker.begin_request(
        request_id,
        group,
        prefill_tokens=max(1, initial_context_tokens),
        decode_capacity_tokens=decode_capacity_tokens,
        prompt_hashes=prompt_hashes,
    )

    handed_off_stream_cleanup = False
    active_request_ids: set[str] = request.app.state.pap_active_request_ids
    active_request_ids.add(request_id)
    admission = request.app.state.projection_admission

    def finish_request() -> None:
        dynamo_router.finish_request(request_id)
        load_tracker.finish_request(request_id)
        active_request_ids.discard(request_id)

    lifecycle = request.app.state.pap_lifecycle_manager.create(
        request_id=request_id,
        attention_clients=attention_clients,
        prefill_client=prefill,
        projection_client=projection_client,
        admission=admission,
        group=group,
        projection=projection,
        on_finished=finish_request,
    )
    request.app.state.pap_pending_route_request_ids.discard(request_id)
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
        lifecycle.mark_attention_registered()
        register_ms = (
            (time.perf_counter() - register_start) * 1000.0 if profile else 0.0
        )
        attention_session = attention_sessions[0]

        prefill_payload_start = time.perf_counter() if profile else 0.0
        prefill_payload = attach_pap_prefill_attention_params(
            build_prefill_payload(
                req_data,
                prompt_token_ids=routing_token_ids,
                prompt_text=forwarded_prompt_text,
            ),
            pap_attention_endpoint=group.attention_base_url,
            pap_attention_tcp_endpoint=group.attention_tcp_endpoint,
            pap_prefill_kv_handle=str(attention_session.get("prefill_kv_handle")),
            pap_mode=request.app.state.args.pap_mode,
        )
        prefill_payload_ms = (
            (time.perf_counter() - prefill_payload_start) * 1000.0 if profile else 0.0
        )
        prefill_start = time.time()
        lifecycle.mark_prefill_started()
        prefill_resp = await _post_json(
            prefill,
            api_path,
            prefill_payload,
            request_id,
        )
        lifecycle.mark_prefill_completed()
        prefill_ms = int((time.time() - prefill_start) * 1000)

        prompt_token_ids = prefill_resp.get("prompt_token_ids")
        if prompt_token_ids is None:
            choices = prefill_resp.get("choices")
            if (
                isinstance(choices, list)
                and len(choices) == 1
                and isinstance(choices[0], dict)
            ):
                prompt_token_ids = choices[0].get("prompt_token_ids")
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(not isinstance(token_id, int) for token_id in prompt_token_ids)
        ):
            raise RuntimeError("PAP Prefill returned no valid prompt token IDs")
        if routing_token_ids is not None and prompt_token_ids != routing_token_ids:
            raise RuntimeError("PAP Prefill changed the Gateway prompt token IDs")
        prompt_text = prefill_resp.get("prompt_text")
        if not isinstance(prompt_text, str):
            prompt_text = forwarded_prompt_text

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
        prefill_cache_usage = _extract_prefill_cache_usage(prefill_resp)
        exact_prompt_hashes = tokenizer.hash_token_ids(
            prompt_token_ids,
            cache_salt=req_data.get("cache_salt"),
        )
        load_tracker.mark_prefill_completed(
            request_id,
            prefix_len,
            prompt_hashes=exact_prompt_hashes,
            cached_tokens=(
                prefill_cache_usage.cached_tokens
                if prefill_cache_usage is not None
                else 0
            ),
        )
        attention_ready_start = time.perf_counter() if profile else 0.0
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
        attention_ready_ms = (
            (time.perf_counter() - attention_ready_start) * 1000.0 if profile else 0.0
        )
        await dynamo_router.mark_prefill_completed(request_id)
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
                "tokenization_ms=%.3f routing_ms=%.3f "
                "prefill_payload_ms=%.3f prefill_ms=%d readiness_ms=%.3f "
                "projection_payload_ms=%.3f pre_projection_ms=%.3f",
                request_id,
                register_ms,
                tokenization_ms,
                routing_ms,
                prefill_payload_ms,
                prefill_ms,
                attention_ready_ms,
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

        admission_start = time.perf_counter() if profile else 0.0
        await admission.acquire(group, projection)
        admission_ms = (
            (time.perf_counter() - admission_start) * 1000.0 if profile else 0.0
        )
        lifecycle.mark_projection_admitted()
        if profile:
            logger.info(
                "PAP proxy projection admission profile request_id=%s "
                "admission_ms=%.3f request_to_admission_ms=%.3f",
                request_id,
                admission_ms,
                (time.perf_counter() - request_start) * 1000.0,
            )

        if client_stream:
            lifecycle.mark_projection_started()
            opened_stream = await open_projection_stream(
                projection_client,
                api_path,
                projection_payload,
                request_id,
                profile_origin=request_start,
            )
            lifecycle.attach_projection_response(opened_stream.response)

            handed_off_stream_cleanup = True
            return StreamingResponse(
                _stream_projection_with_cleanup(
                    projection_client,
                    api_path,
                    projection_payload,
                    request_id,
                    lifecycle,
                    opened_stream=opened_stream,
                    prefill_usage=prefill_cache_usage,
                ),
                media_type="text/event-stream",
                headers=response_headers,
                background=BackgroundTask(
                    lifecycle.terminate,
                    "response_background",
                ),
            )

        lifecycle.mark_projection_started()
        projection_resp = await _post_json(
            projection_client,
            api_path,
            projection_payload,
            request_id,
        )
        lifecycle.mark_projection_completed()
        projection_resp = _merge_prefill_cache_usage(
            projection_resp,
            prefill_cache_usage,
        )
        return JSONResponse(projection_resp, headers=response_headers)
    finally:
        if not handed_off_stream_cleanup:
            await lifecycle.terminate("request_pipeline_exit")
