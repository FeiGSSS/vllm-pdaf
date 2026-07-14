# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side client helpers for PAP Attention control endpoints."""

from __future__ import annotations

import base64
import json
import os
import socket
from typing import Any
from urllib.parse import urlsplit


def select_attention_endpoint_for_request(
    request_id: str | None,
    *,
    default_endpoint: str | None = None,
    endpoint_by_request: dict[str, str] | None = None,
) -> str | None:
    """Select the request-specific Attention endpoint when one is present."""

    if request_id is not None and endpoint_by_request:
        endpoint = endpoint_by_request.get(str(request_id))
        if endpoint:
            return str(endpoint)
    return default_endpoint


def bind_offload_exec_mailbox(
    *,
    attention_endpoint: str,
    local_agent_metadata: bytes,
    source_id: str | None = None,
    timeout: float | None = None,
) -> bytes:
    """Bind Projection's NIXL mailbox endpoint to one Attention endpoint."""

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    parsed = urlsplit(attention_endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"unsupported PAP attention endpoint: {attention_endpoint}")
    port = "" if parsed.port is None else f":{parsed.port}"
    path = "/v1/pap/attention/offload-exec-mailbox/bind"
    request_payload = {
        "agent_metadata_b64": base64.b64encode(local_agent_metadata).decode("ascii")
    }
    if source_id is not None:
        request_payload["source_id"] = str(source_id)
    body = json.dumps(request_payload, separators=(",", ":")).encode("utf-8")
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body
    with socket.create_connection(
        (parsed.hostname, int(parsed.port or 80)),
        timeout=request_timeout,
    ) as sock:
        sock.settimeout(request_timeout)
        sock.sendall(request)
        response = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
    header, _, payload = response.partition(b"\r\n\r\n")
    status_line = header.splitlines()[0].decode("ascii", errors="replace")
    if " 200 " not in status_line:
        raise RuntimeError(f"PAP mailbox bind failed: {status_line} {payload[:256]!r}")
    data = json.loads(payload.decode("utf-8"))
    return base64.b64decode(str(data["agent_metadata_b64"]).encode("ascii"))


def update_offload_exec_mailbox_activity(
    *,
    attention_endpoint: str,
    source_id: str,
    active: bool,
    membership_generation: int,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Publish one Projection source's active membership to Attention."""

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    generation = int(membership_generation)
    if generation <= 0:
        raise ValueError("PAP mailbox membership generation must be positive")
    parsed = urlsplit(attention_endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"unsupported PAP attention endpoint: {attention_endpoint}")
    port = "" if parsed.port is None else f":{parsed.port}"
    path = "/v1/pap/attention/offload-exec-mailbox/activity"
    body = json.dumps(
        {
            "source_id": str(source_id),
            "active": bool(active),
            "membership_generation": generation,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body
    with socket.create_connection(
        (parsed.hostname, int(parsed.port or 80)),
        timeout=request_timeout,
    ) as sock:
        sock.settimeout(request_timeout)
        sock.sendall(request)
        response = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
    header, _, payload = response.partition(b"\r\n\r\n")
    status_line = header.splitlines()[0].decode("ascii", errors="replace")
    if " 200 " not in status_line:
        raise RuntimeError(
            f"PAP mailbox activity update failed: {status_line} {payload[:256]!r}"
        )
    result = json.loads(payload.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("PAP mailbox activity response must be an object")
    return result


__all__ = [
    "bind_offload_exec_mailbox",
    "select_attention_endpoint_for_request",
    "update_offload_exec_mailbox_activity",
]
