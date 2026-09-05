# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP clients and Prefill-to-Attention handoff helpers."""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx

from vllm.pap.config import read_env_float


@dataclass
class PAPServiceClient:
    """One HTTP client bound to a PAP role endpoint."""

    client: httpx.AsyncClient
    host: str
    port: int
    base_url: str
    role: str


def request_headers(request_id: str | None = None) -> dict[str, str]:
    """Build headers shared by gateway role requests."""
    headers: dict[str, str] = {}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request_id:
        headers["X-Request-Id"] = request_id
    return headers


def prefill_prefix_len_from_kv_params(
    kv_transfer_params: dict[str, Any],
) -> int | None:
    """Return the number of prompt KV tokens exported by Prefill."""
    value = kv_transfer_params.get("remote_num_tokens")
    if value is None:
        return None
    try:
        prefix_len = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid remote_num_tokens in Prefill kv_transfer_params: {value!r}"
        ) from exc
    if prefix_len < 0:
        raise ValueError(
            f"invalid remote_num_tokens in Prefill kv_transfer_params: {value!r}"
        )
    return prefix_len or None


def prefill_kv_handle_from_kv_params(
    kv_transfer_params: dict[str, Any],
    *,
    fallback: Any | None = None,
) -> str | None:
    """Resolve the stable Prefill KV handle returned by vLLM."""
    for key in ("remote_request_id", "pap_prefill_kv_handle"):
        value = kv_transfer_params.get(key)
        if value:
            return str(value)
    if fallback:
        return str(fallback)
    return None


async def register_attention_handle(
    attention: PAPServiceClient,
    request_id: str,
    conversation_id: str,
    prefill_endpoint: str,
    kv_transfer_params: dict[str, Any],
    prefix_len: int | None,
) -> dict[str, Any]:
    """Create the Attention session before Prefill publishes its KV manifest."""
    payload = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "prefill_endpoint": prefill_endpoint,
        "kv_transfer_params": dict(kv_transfer_params),
        "prefix_len": prefix_len,
    }
    for attempt in range(2):
        try:
            response = await attention.client.post(
                "/v1/pap/attention/register",
                json=payload,
                headers={},
            )
            break
        except httpx.TransportError:
            if attempt:
                raise
            await asyncio.sleep(0)
    response.raise_for_status()
    return response.json()


async def wait_attention_prefill_ready(
    attention: PAPServiceClient,
    request_id: str,
    *,
    expected_prefix_len: int | None = None,
    expected_session_handle: str | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Wait until final Prefill KV is GPU-ready before Decode admission."""
    timeout = (
        float(timeout_s)
        if timeout_s is not None
        else read_env_float(
            os.environ, "PAP_ATTENTION_PREFILL_READY_TIMEOUT", 5.0, minimum=0.0
        )
    )
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Prefill readiness timeout must be finite and non-negative")
    path = f"/v1/pap/attention/sessions/{request_id}/prefill-readiness"
    response = await attention.client.get(
        path,
        headers=request_headers(request_id),
        params={
            "expected_prefix_len": expected_prefix_len,
            "expected_session_handle": expected_session_handle,
            "timeout_s": timeout,
        },
    )
    response.raise_for_status()
    readiness = response.json()
    session_handle = readiness.get("session_handle")
    if (
        expected_session_handle is not None
        and session_handle is not None
        and str(session_handle) != expected_session_handle
    ):
        raise RuntimeError(
            "PAP Attention readiness session changed "
            f"request_id={request_id} expected={expected_session_handle} "
            f"actual={session_handle}"
        )
    if readiness.get("failed"):
        raise RuntimeError(
            "PAP Attention prefill readiness failed "
            f"request_id={request_id} error={readiness.get('error')}"
        )
    if readiness.get("timed_out") or not readiness.get("ready"):
        raise TimeoutError(
            "timed out waiting for PAP Attention prefill readiness "
            f"request_id={request_id}"
        )
    return True


async def get_prefill_kv_load(prefill: PAPServiceClient) -> dict[str, int]:
    """Read Prefill compute backlog and projected KV capacity pressure."""
    response = await prefill.client.get(
        "/v1/pap/prefill/kv-load",
        headers=request_headers(),
    )
    response.raise_for_status()
    snapshot = response.json()
    return {
        key: max(0, int(snapshot[key]))
        for key in (
            "outstanding_prefill_tokens",
            "projected_kv_tokens",
            "non_evictable_kv_tokens",
            "total_kv_tokens",
            "kv_block_size",
        )
    }
