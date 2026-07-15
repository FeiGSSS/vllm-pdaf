# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP clients and Prefill-to-Attention handoff helpers."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


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
            "invalid remote_num_tokens in Prefill kv_transfer_params: "
            f"{value!r}"
        ) from exc
    if prefix_len < 0:
        raise ValueError(
            "invalid remote_num_tokens in Prefill kv_transfer_params: "
            f"{value!r}"
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
    response = await attention.client.post(
        "/v1/pap/attention/register",
        json=payload,
        headers={},
    )
    response.raise_for_status()
    return response.json()


async def wait_attention_prefill_ready(
    attention: PAPServiceClient,
    request_id: str,
    *,
    timeout_s: float | None = None,
    poll_interval_s: float | None = None,
) -> bool:
    """Wait until every registered Prefill layer is ready for Attention."""
    timeout = (
        float(timeout_s)
        if timeout_s is not None
        else float(os.environ.get("PAP_ATTENTION_PREFILL_READY_TIMEOUT", "5.0"))
    )
    poll_interval = (
        float(poll_interval_s)
        if poll_interval_s is not None
        else float(os.environ.get("PAP_ATTENTION_PREFILL_READY_POLL", "0.01"))
    )
    deadline = time.monotonic() + timeout
    path = f"/v1/pap/attention/sessions/{request_id}/prefill-readiness"
    while True:
        response = await attention.client.get(
            path,
            headers=request_headers(request_id),
        )
        response.raise_for_status()
        layers = list(response.json().get("layers") or ())
        if layers and all(bool(layer.get("ready")) for layer in layers):
            return True
        failed = [layer for layer in layers if layer.get("failed")]
        if failed:
            raise RuntimeError(
                "PAP Attention prefill readiness failed "
                f"request_id={request_id} layers={failed}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for PAP Attention prefill readiness "
                f"request_id={request_id}"
            )
        await asyncio.sleep(max(0.0, poll_interval))
