# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side publisher for sealed PAP Prefill KV handoff."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Sequence
from threading import local
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.pap.protocol import (
    PAPCudaIPCTensorHandle,
    PAPPrefillKVSessionManifest,
)


def _normalize_gpu_uuid(value: object) -> str:
    return str(value).removeprefix("GPU-").lower()


def _resolve_ipc_gpu_uuid(
    physical_gpu_id: object,
    available_gpu_ids: list[str],
) -> str | None:
    normalized = _normalize_gpu_uuid(physical_gpu_id)
    return next(
        (
            gpu_id
            for gpu_id in available_gpu_ids
            if _normalize_gpu_uuid(gpu_id) == normalized
        ),
        None,
    )


def open_ipc_tensor_handle(handle: PAPCudaIPCTensorHandle) -> torch.Tensor:
    """Open one CUDA IPC tensor handle on the current physical GPU."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    device_index = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device_index)
    physical_gpu_id = str(props.uuid)
    ipc_handle = handle.ipc_handle
    ipc_gpu_id = _resolve_ipc_gpu_uuid(physical_gpu_id, list(ipc_handle))
    if ipc_gpu_id is None:
        raise ValueError(
            f"IPC handle not found for GPU UUID {physical_gpu_id}. "
            f"Available UUIDs: {list(ipc_handle.keys())}"
        )
    args = list(ipc_handle[ipc_gpu_id])
    args[6] = device_index
    return rebuild_cuda_tensor(*args)


def open_prefill_manifest_event(
    manifest: PAPPrefillKVSessionManifest,
) -> Any | None:
    """Open an interprocess CUDA event carried by a Prefill manifest."""
    if manifest.ready_event_handle is None:
        return None
    device_index = torch.accelerator.current_device_index()
    return torch.cuda.Event.from_ipc_handle(
        device_index,
        manifest.ready_event_handle,
    )


if TYPE_CHECKING:
    from vllm.pap.kv.registry import PAPAttentionRegistry
    from vllm.pap.protocol import PAPCudaIPCTensorHandle


logger = logging.getLogger(__name__)


_TCP_CONNECTIONS = local()


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
    if command == "revoke_prefill_kv":
        result = registry.revoke_prefill_kv(
            session_handle=str(metadata["session_handle"]),
            generation=int(metadata["generation"]),
        )
        return serialize_tensor_bundle(result, {})
    raise ValueError(f"unsupported PAP wire command {command!r}; use sealed KV handoff")


def revoke_prefill_kv(*, endpoint: str, session_handle: str, generation: int) -> None:
    """Fence old Prefill publications before allocator ownership is recycled."""
    from vllm.pap.protocol.wire import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    result, _ = deserialize_tensor_bundle(
        _post_bytes_tcp(
            endpoint=endpoint,
            payload=serialize_tensor_bundle(
                {
                    "command": "revoke_prefill_kv",
                    "session_handle": session_handle,
                    "generation": generation,
                },
                {},
            ),
            timeout=float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0")),
        )
    )
    if not result.get("revoked"):
        raise RuntimeError(f"PAP Prefill revocation not acknowledged: {result}")


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
    writable_tail_tokens: int = 0,
    lease_id: str,
    generation: int = 0,
) -> int:
    """Atomically publish one request's sealed Prefill KV layout."""

    from vllm.pap.kv.lease import get_global_kv_lease_registry
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
    if len(set(normalized_block_ids)) != len(normalized_block_ids):
        raise ValueError("PAP manifest aliases physical blocks within one request")
    try:
        leased_block_ids = get_global_kv_lease_registry().extend_blocks_if_active(
            request_id=request_id,
            lease_id=lease_id,
            block_ids=normalized_block_ids,
        )
    except Exception as exc:
        logger.exception(
            "PAP sealed KV lease validation failed request_id=%s blocks=%d",
            request_id,
            len(normalized_block_ids),
        )
        raise RuntimeError(
            f"PAP sealed KV lease validation failed for request_id={request_id}"
        ) from exc
    if leased_block_ids is None:
        # Revocation won the race. Never recreate ownership for an old generation.
        return 0

    block_capacity = len(normalized_block_ids) * int(block_size)
    writable_tail_tokens = max(0, int(writable_tail_tokens))
    planned_capacity = min(
        int(prefix_len) + writable_tail_tokens,
        block_capacity,
    )
    if int(prefix_len) + writable_tail_tokens > block_capacity:
        raise ValueError("PAP manifest exceeds allocated KV capacity")
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
        generation=generation,
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
