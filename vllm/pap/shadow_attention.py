# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP control-plane helpers.

TCP control messages trigger remote Attention execution and publish sealed
Prefill KV catalog/manifest state. OFFLOAD_EXEC tensors use the PAP data plane.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import time
from collections.abc import Sequence
from threading import local
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import torch
from torch.multiprocessing.reductions import reduce_tensor

if TYPE_CHECKING:
    from vllm.pap.protocol import PAPCudaIPCTensorHandle

logger = logging.getLogger(__name__)

_TCP_CONNECTIONS = local()


def _pap_unified_kv_decode_capacity_tokens() -> int:
    raw = os.environ.get("PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS", "")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    parsed = urlsplit(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    if parsed.scheme != "tcp" or parsed.hostname is None or parsed.port is None:
        raise ValueError(f"unsupported PAP attention TCP endpoint: {endpoint}")
    return parsed.hostname, int(parsed.port)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("PAP attention TCP peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _post_bytes_tcp(
    *,
    endpoint: str,
    payload: bytes,
    timeout: float,
) -> bytes:
    host, port = _parse_tcp_endpoint(endpoint)
    cache: dict[tuple[str, int], socket.socket] = getattr(
        _TCP_CONNECTIONS, "connections", {}
    )
    if not cache:
        _TCP_CONNECTIONS.connections = cache
    key = (host, port)

    last_error: Exception | None = None
    for _ in range(2):
        sock = cache.get(key)
        if sock is None:
            sock = socket.create_connection((host, port), timeout=float(timeout))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            cache[key] = sock
        sock.settimeout(float(timeout))
        try:
            sock.sendall(len(payload).to_bytes(8, byteorder="little") + payload)
            header = _recv_exact(sock, 8)
            response_len = int.from_bytes(header, byteorder="little")
            if response_len <= 0:
                raise ValueError("PAP attention TCP response length <= 0")
            return _recv_exact(sock, response_len)
        except (BrokenPipeError, ConnectionError, EOFError, OSError) as exc:
            last_error = exc
            cache.pop(key, None)
            sock.close()
    assert last_error is not None
    raise last_error


def select_attention_endpoint_for_request(
    request_id: str | None,
    *,
    default_endpoint: str | None = None,
    endpoint_by_request: dict[str, str] | None = None,
) -> str | None:
    if request_id is not None and endpoint_by_request:
        endpoint = endpoint_by_request.get(str(request_id))
        if endpoint:
            return str(endpoint)
    return default_endpoint


def trigger_offload_exec_attention(
    *,
    tcp_endpoint: str | None = None,
    request_id: str,
    layer_name: str,
    step: int,
    scale: float,
    remote_address: str,
    timeout: float | None = None,
) -> None:
    """Trigger Attention to receive QKV and send O over OFFLOAD_EXEC."""

    if not tcp_endpoint:
        raise RuntimeError("PAP OFFLOAD_EXEC trigger requires a TCP control endpoint")

    from vllm.pap.remote_attention import (
        deserialize_compact_offload_exec_ack,
        serialize_compact_offload_exec_command,
    )

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )

    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=serialize_compact_offload_exec_command(
            request_id=request_id,
            layer_name=layer_name,
            step=int(step),
            scale=float(scale),
            remote_address=str(remote_address),
        ),
        timeout=request_timeout,
    )
    deserialize_compact_offload_exec_ack(response_body)


def trigger_offload_exec_attention_batch(
    *,
    tcp_endpoint: str | None = None,
    layer_name: str,
    items: Sequence[dict[str, Any]],
    remote_address: str,
    timeout: float | None = None,
) -> None:
    """Trigger Attention to receive one batched QKV tensor and send batched O."""

    if not tcp_endpoint:
        raise RuntimeError(
            "PAP OFFLOAD_EXEC batch trigger requires a TCP control endpoint"
        )

    from vllm.pap.remote_attention import (
        deserialize_compact_offload_exec_ack,
        serialize_compact_offload_exec_batch_command,
    )

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )

    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=serialize_compact_offload_exec_batch_command(
            layer_name=layer_name,
            remote_address=str(remote_address),
            items=list(items),
        ),
        timeout=request_timeout,
    )
    deserialize_compact_offload_exec_ack(response_body)


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


def _gpu_uuid_for_tensor(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cuda":
        return "cpu"
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return str(torch.cuda.get_device_properties(device_index).uuid)


def _make_cuda_ipc_tensor_handle(
    tensor: torch.Tensor,
) -> PAPCudaIPCTensorHandle:
    from vllm.pap.protocol import PAPCudaIPCTensorHandle

    _, ipc_args = reduce_tensor(tensor)
    return PAPCudaIPCTensorHandle(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(dim) for dim in tensor.shape),
        ipc_handle={_gpu_uuid_for_tensor(tensor): tuple(ipc_args)},
    )


def register_prefill_kv_catalog(
    *,
    catalog_id: str,
    layer_name: str,
    kv_cache: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    layout: str,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
) -> str:
    """Register one process-lifetime Prefill KV-cache tensor in Attention."""

    from vllm.pap.protocol import PAPPrefillKVCacheCatalogDescriptor
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    if not tcp_endpoint:
        raise RuntimeError("PAP Prefill KV catalog requires a TCP endpoint")
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    descriptor = PAPPrefillKVCacheCatalogDescriptor(
        catalog_id=catalog_id,
        layer_name=layer_name,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        layout=layout,
        kv_cache=_make_cuda_ipc_tensor_handle(kv_cache),
    )
    request_body = serialize_tensor_bundle(
        {
            "command": "register_prefill_kv_catalog",
            "descriptor": descriptor.to_dict(),
        },
        {},
    )
    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=request_body,
        timeout=request_timeout,
    )
    response_metadata, _ = deserialize_tensor_bundle(response_body)
    status = str(response_metadata.get("status", ""))
    if status not in {"registered", "existing"}:
        raise RuntimeError(
            "PAP Prefill KV catalog registration failed "
            f"catalog_id={catalog_id} layer={layer_name} status={status!r}"
        )
    return status


def publish_prefill_kv_session_manifest(
    *,
    request_id: str,
    catalog_id: str,
    block_ids: Sequence[int],
    prefix_len: int,
    block_size: int,
    expected_layer_count: int,
    ready_event_handle: bytes | None,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
) -> int:
    """Atomically publish one request's sealed Prefill KV layout."""

    from vllm.pap.protocol import PAPPrefillKVSessionManifest
    from vllm.pap.lifecycle.lease import (
        pap_active_lease_id,
        pap_has_active_lease,
        pap_leased_block_ids,
        pap_pin_blocks,
    )
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    if not tcp_endpoint:
        raise RuntimeError("PAP Prefill KV manifest requires a TCP endpoint")
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    normalized_block_ids = tuple(int(block_id) for block_id in block_ids)
    try:
        if pap_has_active_lease(request_id):
            lease_id = pap_active_lease_id(request_id)
            leased_block_ids = pap_leased_block_ids(request_id)
        else:
            lease_id = pap_pin_blocks(
                request_id=request_id,
                block_ids=normalized_block_ids,
            )
            leased_block_ids = normalized_block_ids
    except Exception as exc:
        logger.exception(
            "PAP sealed KV lease pin failed request_id=%s blocks=%d",
            request_id,
            len(normalized_block_ids),
        )
        raise RuntimeError(
            f"PAP sealed KV lease pin failed for request_id={request_id}"
        ) from exc
    if lease_id is None:
        raise RuntimeError(f"PAP sealed KV lease missing for request_id={request_id}")

    block_capacity = len(normalized_block_ids) * int(block_size)
    planned_capacity = min(
        int(prefix_len) + _pap_unified_kv_decode_capacity_tokens(),
        block_capacity,
    )
    manifest = PAPPrefillKVSessionManifest(
        request_id=request_id,
        catalog_id=catalog_id,
        prefix_len=prefix_len,
        block_ids=normalized_block_ids,
        block_size=block_size,
        expected_layer_count=expected_layer_count,
        lease_id=lease_id,
        leased_block_ids=leased_block_ids,
        lease_capacity_tokens=planned_capacity,
        writable_start_token=prefix_len,
        writable_end_token=planned_capacity,
        ready_event_handle=ready_event_handle,
    )
    request_body = serialize_tensor_bundle(
        {
            "command": "publish_prefill_kv_manifest",
            "manifest": manifest.to_dict(),
        },
        {},
    )
    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=request_body,
        timeout=request_timeout,
    )
    response_metadata, _ = deserialize_tensor_bundle(response_body)
    if response_metadata.get("status") != "ready":
        raise RuntimeError(
            "PAP Prefill KV manifest publication failed "
            f"request_id={request_id} status={response_metadata.get('status')!r}"
        )
    return int(response_metadata["prefix_len"])
