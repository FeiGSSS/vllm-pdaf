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
    _export_prefill_kv,
    _install_completed_prefill_on_group,
    _post_json,
    _release_prefill_kv,
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
    enrich_prefill_kv_params,
    requested_decode_capacity,
)
from vllm.pap.gateway.routing import (
    PAPAttentionLoadRouter,
    _estimate_context_tokens,
    select_instances,
)
from vllm.pap.gateway.topology import (
    PAPGroup,
    build_projection_payload_for_group,
)

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
    attention_load_router: PAPAttentionLoadRouter | None = None
    estimated_context_tokens = 1
    history_record: tuple[PAPGroup, str, int] | None = None
    history_export: dict[str, Any] | None = None
    if request.app.state.args.routing_policy == "attention_load":
        attention_load_router = request.app.state.attention_load_router
        history_record = (
            attention_load_router.history(conversation_id) if conversation_id else None
        )
        history_context_tokens = history_record[2] if history_record else 0
        if history_record is not None and attention_load_router.migration_enabled:
            history_group, history_request_id, _ = history_record
            try:
                history_export = await _export_prefill_kv(
                    request.app.state.prefill_clients[history_group],
                    history_request_id,
                )
            except Exception as exc:
                logger.warning(
                    "PAP retained KV export failed request_id=%s "
                    "history_request_id=%s source_pa=%d error=%s",
                    request_id,
                    history_request_id,
                    request.app.state.groups.index(history_group),
                    exc,
                )
            if history_export is not None:
                exported_seq_len = history_export.get("seq_len")
                if isinstance(exported_seq_len, int) and exported_seq_len > 0:
                    history_context_tokens = exported_seq_len
        explicit_context_tokens = req_data.pop("pap_context_tokens", None)
        if explicit_context_tokens is None:
            explicit_context_tokens = request.headers.get("X-PAP-Context-Tokens")
        estimated_context_tokens = _estimate_context_tokens(
            req_data,
            history_context_tokens=history_context_tokens,
            explicit_context_tokens=explicit_context_tokens,
        )
    group, projection = select_instances(
        request_number,
        request.app.state.groups,
        request.app.state.projections,
        routing_policy=request.app.state.args.routing_policy,
        conversation_id=conversation_id,
        conversation_router=request.app.state.conversation_router,
        attention_load_router=attention_load_router,
        request_id=request_id,
        estimated_context_tokens=estimated_context_tokens,
    )
    prefill_group = group
    prefill_group_index = request.app.state.groups.index(prefill_group)
    prefill = request.app.state.prefill_clients[group]
    attention_clients = request.app.state.attention_clients[group]
    projection_client = request.app.state.projection_clients[projection]
    projection_index = request.app.state.projections.index(projection)

    attention_sessions: list[dict[str, Any]] | None = None
    handed_off_stream_cleanup = False
    prefill_admitted = False
    projection_admitted = False
    attention_load_finished = False
    try:
        prefill_admission_wait_ms = await request.app.state.prefill_admission.acquire(
            prefill_group
        )
        prefill_admitted = True
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
        try:
            prefill_resp = await _post_json(
                prefill,
                api_path,
                prefill_payload,
                request_id,
            )
        finally:
            try:
                await request.app.state.prefill_admission.release(prefill_group)
            finally:
                prefill_admitted = False
        prefill_ms = int((time.time() - t0) * 1000)
        migration_ms = 0
        migration_attention_ready = False
        usage = prefill_resp.get("usage")
        prompt_tokens = None
        if isinstance(usage, dict):
            value = usage.get("prompt_tokens")
            if isinstance(value, int) and value > 0:
                prompt_tokens = value
        if prompt_tokens is None:
            source_kv_params = prefill_resp.get("kv_transfer_params") or {}
            prompt_tokens = prefill_prefix_len_from_kv_params(source_kv_params)
        if prompt_tokens is None:
            prompt_tokens = estimated_context_tokens

        if attention_load_router is not None:
            decode_group = attention_load_router.observe_prefill(
                request_id,
                prompt_tokens,
            )
            if decode_group != prefill_group:
                decode_group_index = request.app.state.groups.index(decode_group)
                target_attention_clients = request.app.state.attention_clients[
                    decode_group
                ]
                logger.info(
                    "PAP completed Prefill migration planned request_id=%s "
                    "source_pa=%d target_pa=%d tokens=%d",
                    request_id,
                    prefill_group_index,
                    decode_group_index,
                    prompt_tokens,
                )
                migration_started = time.perf_counter()
                try:
                    (
                        target_sessions,
                        target_response,
                        migration_ms,
                    ) = await _install_completed_prefill_on_group(
                        req_data=req_data,
                        request_id=request_id,
                        conversation_id=conversation_id,
                        source_group=prefill_group,
                        source_prefill=(
                            request.app.state.prefill_clients[prefill_group]
                        ),
                        source_prefill_response=prefill_resp,
                        target_group=decode_group,
                        target_prefill=(
                            request.app.state.prefill_clients[decode_group]
                        ),
                        target_attention_clients=target_attention_clients,
                    )
                except Exception as exc:
                    migration_ms = int((time.perf_counter() - migration_started) * 1000)
                    group = attention_load_router.mark_migration_missed(request_id)
                    logger.warning(
                        "PAP completed Prefill migration failed; using source "
                        "request_id=%s source_pa=%d target_pa=%d error=%s",
                        request_id,
                        prefill_group_index,
                        decode_group_index,
                        exc,
                    )
                else:
                    await _cleanup_attention_sessions(
                        attention_clients,
                        request_id,
                    )
                    group = decode_group
                    prefill = request.app.state.prefill_clients[group]
                    attention_clients = target_attention_clients
                    attention_sessions = target_sessions
                    attention_session = attention_sessions[0]
                    prefill_resp = target_response
                    migration_attention_ready = True
                    attention_load_router.mark_migration_succeeded(request_id)
                    logger.info(
                        "PAP completed Prefill migration installed "
                        "request_id=%s source_pa=%d target_pa=%d "
                        "tokens=%d migration_ms=%d",
                        request_id,
                        prefill_group_index,
                        decode_group_index,
                        prompt_tokens,
                        migration_ms,
                    )
        if history_record is not None and history_export is not None:
            history_source_group, history_request_id, _ = history_record
            history_lease_id = history_export.get("lease_id")
            if isinstance(history_lease_id, str) and history_lease_id:
                try:
                    released = await _release_prefill_kv(
                        request.app.state.prefill_clients[history_source_group],
                        request_id=history_request_id,
                        lease_id=history_lease_id,
                    )
                    if not released:
                        logger.warning(
                            "PAP historical KV lease release not acknowledged "
                            "request_id=%s history_request_id=%s",
                            request_id,
                            history_request_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "PAP historical KV lease release failed request_id=%s "
                        "history_request_id=%s error=%s",
                        request_id,
                        history_request_id,
                        exc,
                    )

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
        attention_ready = migration_attention_ready
        if prefix_len is not None and not attention_ready:
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
        group_index = request.app.state.groups.index(group)
        pair_name = f"pa{group_index}:p{projection_index}"
        request.app.state.pair_counts[pair_name] += 1
        projection_payload_ms = (
            (time.perf_counter() - projection_payload_start) * 1000.0
            if profile
            else 0.0
        )
        logger.info(
            "request_id=%s pa=%s:%d attention=%s:%s projection=%s:%d "
            "pa_index=%d projection_index=%d pair=%s "
            "prefill_pa_index=%d prefill_admission_wait_ms=%.3f "
            "prefill_ms=%d migration_ms=%d "
            "prefill_prefix_len=%s attention_ready=%s "
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
            prefill_group_index,
            prefill_admission_wait_ms,
            prefill_ms,
            migration_ms,
            prefix_len,
            attention_ready,
            sorted(projection_kv_params.keys()),
        )
        if profile:
            logger.info(
                "PAP proxy prefill IPC profile request_id=%s register_ms=%.3f "
                "prefill_admission_wait_ms=%.3f prefill_payload_ms=%.3f "
                "prefill_ms=%d migration_ms=%d "
                "projection_payload_ms=%.3f pre_projection_ms=%.3f",
                request_id,
                register_ms,
                prefill_admission_wait_ms,
                prefill_payload_ms,
                prefill_ms,
                migration_ms,
                projection_payload_ms,
                (time.perf_counter() - request_start) * 1000.0,
            )

        response_headers = {
            "X-PAP-Prefill-Admission-Wait-Ms": f"{prefill_admission_wait_ms:.3f}",
            "X-PAP-Prefill-Ms": str(prefill_ms),
            "X-PAP-Migration-Ms": str(migration_ms),
            "X-PAP-Prefill-Group": str(prefill_group_index),
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
                    attention_load_router,
                    requested_decode_capacity(req_data) or 0,
                    bool(
                        conversation_id
                        and attention_load_router is not None
                        and attention_load_router.migration_enabled
                    ),
                    prefill_kv_handle,
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
        if attention_load_router is not None:
            response_usage = projection_resp.get("usage")
            completion_tokens = 0
            if isinstance(response_usage, dict):
                value = response_usage.get("completion_tokens")
                if isinstance(value, int):
                    completion_tokens = value
            attention_load_router.finish(
                request_id,
                completion_tokens=completion_tokens,
                prefill_kv_handle=prefill_kv_handle,
            )
            attention_load_finished = True
        return JSONResponse(
            projection_resp,
            headers=response_headers,
        )
    finally:
        if prefill_admitted:
            await request.app.state.prefill_admission.release(prefill_group)
        if not handed_off_stream_cleanup:
            try:
                if attention_sessions is not None:
                    await _cleanup_attention_sessions(
                        attention_clients,
                        request_id,
                        retain_lease=bool(
                            conversation_id
                            and attention_load_router is not None
                            and attention_load_finished
                            and attention_load_router.migration_enabled
                        ),
                    )
            finally:
                if projection_admitted:
                    await request.app.state.projection_admission.release(
                        group,
                        projection,
                    )
            if attention_load_router is not None and not attention_load_finished:
                attention_load_router.abort(request_id)
