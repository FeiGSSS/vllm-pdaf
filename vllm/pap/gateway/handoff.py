# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway KV handoff and Projection streaming lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.clients import (
    PAPServiceClient,
    prefill_kv_handle_from_kv_params,
    prefill_prefix_len_from_kv_params,
    register_attention_handle,
    wait_attention_prefill_ready,
)
from vllm.pap.gateway.clients import request_headers as _headers
from vllm.pap.gateway.observability import _pap_prefill_ipc_profile_enabled
from vllm.pap.gateway.payloads import (
    enrich_prefill_kv_params,
    requested_decode_capacity,
)
from vllm.pap.gateway.routing import (
    PAPAttentionLoadRouter,
    _migration_prefix_identity,
    _migration_prefix_kv_params,
)
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


async def _export_prefill_kv(
    prefill: PAPServiceClient,
    request_id: str,
) -> dict[str, Any] | None:
    """Fetch retained KV metadata without entering the model scheduler."""
    response = await prefill.client.post(
        "/v1/pap/prefill/kv-export",
        json={"request_id": request_id},
        headers=_headers(request_id),
    )
    response.raise_for_status()
    body = response.json()
    return body if body.get("exported", False) else None


async def _wait_prefill_kv_export(
    prefill: PAPServiceClient,
    request_id: str,
) -> dict[str, Any] | None:
    """Wait for scheduler finalization to publish a completed Prefill lease."""
    timeout = float(os.environ.get("PAP_KV_EXPORT_READY_TIMEOUT", "1"))
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        exported = await _export_prefill_kv(prefill, request_id)
        if exported is not None:
            return exported
        if time.monotonic() >= deadline:
            return None
        await asyncio.sleep(0.005)


async def _release_prefill_kv(
    prefill: PAPServiceClient,
    *,
    request_id: str,
    lease_id: str,
) -> bool:
    """Release a retained historical Prefill lease after replacement."""
    response = await prefill.client.post(
        "/v1/pap/prefill/lease-release",
        json={"request_id": request_id, "lease_id": lease_id},
        headers=_headers(request_id),
    )
    response.raise_for_status()
    body = response.json()
    return bool(
        body.get("released", False) or body.get("reason") == "unknown_or_released_lease"
    )


async def _install_completed_prefill_on_group(
    *,
    req_data: dict[str, Any],
    request_id: str,
    conversation_id: str,
    source_group: PAPGroup,
    source_prefill: PAPServiceClient,
    source_prefill_response: dict[str, Any],
    target_group: PAPGroup,
    target_prefill: PAPServiceClient,
    target_attention_clients: list[PAPServiceClient],
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Import a completed Prefill KV snapshot into the Decode target PA."""
    source_kv_params = enrich_prefill_kv_params(
        source_prefill_response.get("kv_transfer_params") or {},
        prefill_host=source_group.prefill_host,
        prefill_nixl_port=source_group.prefill_nixl_port,
    )
    source_prefix_len = prefill_prefix_len_from_kv_params(source_kv_params)
    if source_prefix_len is None:
        raise RuntimeError("PAP completed Prefill returned no transferable KV")
    remote_kv_params, migrated_tokens = _migration_prefix_kv_params(
        {
            "seq_len": source_prefix_len,
            "kv_transfer_params": source_kv_params,
        },
        source_group=source_group,
        block_size=int(os.environ.get("PAP_BLOCK_SIZE", "16")),
    )
    source_request_id = prefill_kv_handle_from_kv_params(source_kv_params)
    if source_request_id is None:
        raise RuntimeError("PAP migration source returned no stable KV handle")
    source_export = await _wait_prefill_kv_export(
        source_prefill,
        source_request_id,
    )
    if source_export is None:
        raise RuntimeError("PAP migration source KV lease is unavailable")
    prefix_token_ids, prefix_block_hashes = _migration_prefix_identity(
        source_export,
        migrated_tokens=migrated_tokens,
    )
    if not target_group.attention_tcp_endpoint:
        raise RuntimeError("PAP KV migration requires Attention TCP endpoints")

    target_sessions = await register_attention_handles(
        target_attention_clients,
        request_id=request_id,
        conversation_id=conversation_id,
        prefill_endpoint=target_group.prefill_base_url,
        kv_transfer_params={},
        prefix_len=None,
    )
    try:
        decode_capacity_tokens = requested_decode_capacity(req_data)
        if decode_capacity_tokens is None:
            decode_capacity_tokens = int(
                os.environ.get(
                    "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS",
                    "0",
                )
            )
        migration_request = {
            "request_id": request_id,
            "source_kv_params": remote_kv_params,
            "prefix_len": migrated_tokens,
            "prefix_token_ids": prefix_token_ids,
            "prefix_block_hashes": prefix_block_hashes,
            "decode_capacity_tokens": decode_capacity_tokens,
            "session_handle": str(target_sessions[0].get("prefill_kv_handle")),
            "attention_tcp_endpoint": target_group.attention_tcp_endpoint,
        }
        started = time.perf_counter()
        submitted = await _post_json(
            target_prefill,
            "/v1/pap/prefill/kv-import",
            migration_request,
            request_id,
        )
        job_id = str(submitted["job_id"])
        status = submitted
        transfer_started = status.get("status") == "transferring" and bool(
            status.get("kv_transfer_params")
        )
        if status.get("status") != "ready" and not transfer_started:
            timeout = float(os.environ.get("PAP_KV_MIGRATION_TIMEOUT", "30"))
            deadline = time.monotonic() + timeout
            while True:
                if status.get("status") in {"failed", "unknown"}:
                    raise RuntimeError(
                        "PAP target KV migration failed "
                        f"job_id={job_id} detail={status.get('error')}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"PAP target KV migration timed out job_id={job_id}"
                    )
                await asyncio.sleep(0.01)
                status = await _post_json(
                    target_prefill,
                    "/v1/pap/prefill/kv-import/status",
                    {"job_id": job_id},
                    request_id,
                )
                if status.get("status") == "ready":
                    break
        target_response = {
            "kv_transfer_params": status.get("kv_transfer_params") or {},
        }
        target_kv_params = enrich_prefill_kv_params(
            target_response.get("kv_transfer_params") or {},
            prefill_host=target_group.prefill_host,
            prefill_nixl_port=target_group.prefill_nixl_port,
        )
        target_prefix_len = prefill_prefix_len_from_kv_params(target_kv_params)
        if target_prefix_len != migrated_tokens:
            raise RuntimeError(
                "PAP completed Prefill migration length mismatch "
                f"source={migrated_tokens} target={target_prefix_len}"
            )
        ready = all(
            [
                await wait_attention_prefill_ready(attention, request_id)
                for attention in target_attention_clients
            ]
        )
        if not ready:
            raise RuntimeError("PAP target Attention did not install migrated KV")
        migration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "PAP KV migration Attention ready request_id=%s job_id=%s "
            "queue_ms=%s engine_transfer_and_publish_ms=%s "
            "engine_total_ms=%s gateway_ms=%d",
            request_id,
            job_id,
            status.get("queue_ms"),
            status.get("transfer_and_publish_ms"),
            status.get("total_ms"),
            migration_ms,
        )
        if source_export is not None:
            source_lease_id = source_export.get("lease_id")
            if isinstance(source_lease_id, str) and source_lease_id:
                try:
                    released = await _release_prefill_kv(
                        source_prefill,
                        request_id=source_request_id,
                        lease_id=source_lease_id,
                    )
                    if not released:
                        logger.warning(
                            "PAP migrated source lease release not acknowledged "
                            "request_id=%s lease_id=%s",
                            request_id,
                            source_lease_id,
                        )
                except Exception:
                    logger.warning(
                        "PAP migrated source lease release failed "
                        "request_id=%s lease_id=%s",
                        request_id,
                        source_lease_id,
                        exc_info=True,
                    )
        return target_sessions, target_response, migration_ms
    except Exception:
        await _cleanup_attention_sessions(
            target_attention_clients,
            request_id,
        )
        raise


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
    *,
    retain_lease: bool = False,
) -> None:
    resp = await attention.client.delete(
        f"/v1/pap/attention/sessions/{request_id}",
        headers=_headers(request_id),
        params={"retain_lease": "true"} if retain_lease else None,
    )
    resp.raise_for_status()


async def _cleanup_attention_sessions(
    attention_clients: list[PAPServiceClient],
    request_id: str,
    *,
    retain_lease: bool = False,
) -> None:
    for attention in attention_clients:
        try:
            await _delete_attention_session(
                attention,
                request_id,
                retain_lease=retain_lease,
            )
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
    attention_load_router: PAPAttentionLoadRouter | None = None,
    completion_tokens: int = 0,
    retain_completed_lease: bool = False,
    prefill_kv_handle: str | None = None,
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
        completed = bool(terminal_chunks)
        if attention_load_router is not None:
            if completed:
                attention_load_router.finish(
                    request_id,
                    completion_tokens=completion_tokens,
                    prefill_kv_handle=prefill_kv_handle,
                )
            else:
                attention_load_router.abort(request_id)
        try:
            await _cleanup_attention_sessions(
                attention_clients,
                request_id,
                retain_lease=completed and retain_completed_lease,
            )
        finally:
            await admission.release(group, projection)
    for chunk in terminal_chunks:
        yield chunk
