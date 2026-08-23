# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side publisher for sealed PAP Prefill KV handoff."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Sequence
from threading import local
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import torch
from torch.multiprocessing.reductions import reduce_tensor

if TYPE_CHECKING:
    from vllm.pap.kv.registry import PAPAttentionRegistry
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


def _gpu_uuid_for_tensor(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cuda":
        return "cpu"
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
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


def accept_prefill_kv_handoff(
    registry: PAPAttentionRegistry,
    payload: bytes,
) -> bytes:
    """Install one sealed Prefill KV catalog or request manifest."""
    from vllm.pap.kv.ipc import (
        open_ipc_tensor_handle,
        open_prefill_manifest_event,
    )
    from vllm.pap.protocol import (
        PAPPrefillKVCacheCatalogDescriptor,
        PAPPrefillKVSessionManifest,
    )
    from vllm.pap.protocol.wire import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    metadata, _tensors = deserialize_tensor_bundle(payload)
    command = str(metadata.get("command", ""))
    if command == "register_prefill_kv_catalog":
        descriptor = PAPPrefillKVCacheCatalogDescriptor.from_dict(
            metadata["descriptor"]
        )
        kv_cache = open_ipc_tensor_handle(descriptor.kv_cache)
        installed = registry.register_prefill_kv_catalog(
            descriptor=descriptor,
            kv_cache=kv_cache,
        )
        return serialize_tensor_bundle(
            {
                "status": "registered" if installed else "existing",
                "catalog_id": descriptor.catalog_id,
                "layer_name": descriptor.layer_name,
            },
            {},
        )
    if command == "publish_prefill_kv_manifest":
        manifest = PAPPrefillKVSessionManifest.from_dict(metadata["manifest"])
        prefix_len = registry.install_prefill_kv_session_manifest(
            manifest=manifest,
            ready_event=open_prefill_manifest_event(manifest),
        )
        return serialize_tensor_bundle(
            {
                "status": "ready",
                "request_id": manifest.request_id,
                "catalog_id": manifest.catalog_id,
                "prefix_len": prefix_len,
            },
            {},
        )
    raise ValueError(f"unsupported PAP wire command {command!r}; use sealed KV handoff")


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
    from vllm.pap.protocol.wire import (
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
    session_handle: str,
    catalog_id: str,
    block_ids: Sequence[int],
    prefix_len: int,
    block_size: int,
    expected_layer_count: int,
    ready_event_handle: bytes | None,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
    decode_capacity_tokens: int | None = None,
) -> int:
    """Atomically publish one request's sealed Prefill KV layout."""

    from vllm.pap.lifecycle.lease import (
        pap_active_lease_id,
        pap_has_active_lease,
        pap_leased_block_ids,
        pap_pin_blocks,
    )
    from vllm.pap.protocol import PAPPrefillKVSessionManifest
    from vllm.pap.protocol.wire import (
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
    if decode_capacity_tokens is None:
        decode_capacity_tokens = _pap_unified_kv_decode_capacity_tokens()
    decode_capacity_tokens = max(0, int(decode_capacity_tokens))
    planned_capacity = min(
        int(prefix_len) + decode_capacity_tokens,
        block_capacity,
    )
    manifest = PAPPrefillKVSessionManifest(
        request_id=request_id,
        session_handle=session_handle,
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


__all__ = [
    "accept_prefill_kv_handoff",
    "publish_prefill_kv_session_manifest",
    "register_prefill_kv_catalog",
]
