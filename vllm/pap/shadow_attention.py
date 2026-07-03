# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP control-plane helpers.

TCP control messages (compact binary format) for triggering the remote
Attention executor and importing prefill KV. Tensor data for OFFLOAD_EXEC
uses the PAP data plane; prefill KV import uses TCP binary bundles.
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
    from vllm.pap.data_plane import PAPCudaIPCTensorHandle, PAPTensorTransport

logger = logging.getLogger(__name__)

_TCP_CONNECTIONS = local()


def _pap_prefill_ipc_profile_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_IPC_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pap_prefill_kv_async_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_KV_ASYNC", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _pap_unified_kv_enabled() -> bool:
    return os.environ.get("PAP_UNIFIED_KV", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
        raise RuntimeError(
            "PAP OFFLOAD_EXEC trigger requires a TCP control endpoint"
        )

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
    body = json.dumps(
        {
            "agent_metadata_b64": base64.b64encode(local_agent_metadata).decode(
                "ascii"
            )
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
            f"PAP mailbox bind failed: {status_line} {payload[:256]!r}"
        )
    data = json.loads(payload.decode("utf-8"))
    return base64.b64decode(str(data["agent_metadata_b64"]).encode("ascii"))


def _block_ids_from_block_table(
    *,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> list[int]:
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("PAP KV import supports one request per call")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    blocks = block_table[0, :num_blocks].to(device="cpu", dtype=torch.long).tolist()
    return [int(block_id) for block_id in blocks]


def _normalize_offload_kv_transport(
    transport: Any | None,
) -> PAPTensorTransport | None:
    if transport is None:
        return None
    from vllm.pap.data_plane import PAPTensorTransport

    return PAPTensorTransport(transport)


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
    from vllm.pap.data_plane import PAPCudaIPCTensorHandle

    _, ipc_args = reduce_tensor(tensor)
    return PAPCudaIPCTensorHandle(
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(int(dim) for dim in tensor.shape),
        ipc_handle={_gpu_uuid_for_tensor(tensor): tuple(ipc_args)},
    )


def _maybe_synchronize_cuda_ipc_tensors(*tensors: torch.Tensor) -> None:
    if not torch.cuda.is_available():
        return
    synced_devices: set[int] = set()
    for tensor in tensors:
        if tensor.device.type != "cuda":
            continue
        device_index = tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if device_index in synced_devices:
            continue
        torch.cuda.current_stream(device_index).synchronize()
        synced_devices.add(device_index)


def _post_prefill_kv_ipc(
    *,
    request_id: str,
    layer_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_len: int,
    block_ids: Sequence[int] | None,
    tcp_endpoint: str,
    timeout: float,
) -> int:
    from vllm.pap.data_plane import PAPOffloadKVIPCDescriptor
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    _maybe_synchronize_cuda_ipc_tensors(key, value)
    descriptor = PAPOffloadKVIPCDescriptor(
        request_id=request_id,
        layer_name=layer_name,
        seq_len=int(seq_len),
        block_ids=tuple([] if block_ids is None else [int(b) for b in block_ids]),
        key=_make_cuda_ipc_tensor_handle(key),
        value=_make_cuda_ipc_tensor_handle(value),
    )
    request_body = serialize_tensor_bundle(
        {
            "command": "import_prefill_kv_ipc",
            "descriptor": descriptor.to_dict(),
        },
        {},
    )
    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=request_body,
        timeout=timeout,
    )
    response_metadata, _ = deserialize_tensor_bundle(response_body)
    return int(response_metadata["seq_len"])


def import_prefill_paged_kv(
    *,
    request_id: str,
    layer_name: str,
    kv_cache: torch.Tensor,
    block_ids: Sequence[int],
    seq_len: int,
    block_size: int,
    num_kv_heads: int,
    layout: str,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
) -> int:
    """Install Prefill-owned paged KV backing storage in Attention."""

    from vllm.pap.data_plane import PAPOffloadKVPagedIPCDescriptor
    from vllm.pap.kv_lease import (
        pap_active_lease_id,
        pap_has_active_lease,
        pap_pin_blocks,
    )
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    if not tcp_endpoint:
        raise RuntimeError("PAP paged OFFLOAD_KV requires a TCP endpoint")
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    profile = _pap_prefill_ipc_profile_enabled()
    total_start = time.perf_counter() if profile else 0.0
    sync_start = time.perf_counter() if profile else 0.0
    _maybe_synchronize_cuda_ipc_tensors(kv_cache)
    sync_ms = (
        (time.perf_counter() - sync_start) * 1000.0 if profile else 0.0
    )

    lease_id: str | None = None
    leased_block_ids: tuple[int, ...] | None = None
    lease_capacity_tokens: int | None = None
    unified_kv_mode = _pap_unified_kv_enabled()
    prefix_len_value: int | None = None
    writable_start_token: int | None = None
    writable_end_token: int | None = None
    try:
        if unified_kv_mode and not pap_has_active_lease(request_id):
            lease_id = pap_pin_blocks(
                request_id=request_id,
                block_ids=tuple(int(b) for b in block_ids),
            )
            leased_block_ids = tuple(int(b) for b in block_ids)
            lease_capacity_tokens = int(seq_len)
        elif unified_kv_mode and pap_has_active_lease(request_id):
            lease_id = pap_active_lease_id(request_id)
    except Exception:  # noqa: BLE001
        # Lease registry must not break Prefill export; fail to no-lease mode.
        lease_id = None
        leased_block_ids = None
        lease_capacity_tokens = None
        unified_kv_mode = False
    if unified_kv_mode:
        prefix_len_value = int(seq_len)
        planned_capacity = (
            int(seq_len) + _pap_unified_kv_decode_capacity_tokens()
        )
        writable_start_token = int(seq_len)
        writable_end_token = planned_capacity
        lease_capacity_tokens = planned_capacity

    descriptor_start = time.perf_counter() if profile else 0.0
    descriptor = PAPOffloadKVPagedIPCDescriptor(
        request_id=request_id,
        layer_name=layer_name,
        seq_len=int(seq_len),
        block_ids=tuple(int(block_id) for block_id in block_ids),
        block_size=int(block_size),
        num_kv_heads=int(num_kv_heads),
        layout=str(layout),
        kv_cache=_make_cuda_ipc_tensor_handle(kv_cache),
        lease_id=lease_id,
        leased_block_ids=leased_block_ids,
        lease_seq_len=int(seq_len) if lease_id is not None else None,
        lease_capacity_tokens=lease_capacity_tokens,
        unified_kv_mode=unified_kv_mode,
        prefix_len=prefix_len_value,
        writable_start_token=writable_start_token,
        writable_end_token=writable_end_token,
    )
    descriptor_ms = (
        (time.perf_counter() - descriptor_start) * 1000.0
        if profile
        else 0.0
    )
    serialize_start = time.perf_counter() if profile else 0.0
    async_import = _pap_prefill_kv_async_enabled()
    request_body = serialize_tensor_bundle(
        {
            "command": "import_prefill_paged_kv_ipc",
            "descriptor": descriptor.to_dict(),
            "async": async_import,
        },
        {},
    )
    serialize_ms = (
        (time.perf_counter() - serialize_start) * 1000.0
        if profile
        else 0.0
    )
    post_start = time.perf_counter() if profile else 0.0
    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=request_body,
        timeout=request_timeout,
    )
    post_ms = (time.perf_counter() - post_start) * 1000.0 if profile else 0.0
    deserialize_start = time.perf_counter() if profile else 0.0
    response_metadata, _ = deserialize_tensor_bundle(response_body)
    deserialize_ms = (
        (time.perf_counter() - deserialize_start) * 1000.0
        if profile
        else 0.0
    )
    if profile:
        logger.info(
            "PAP prefill IPC transport profile request_id=%s layer=%s "
            "seq_len=%d blocks=%d async=%s status=%s sync_ms=%.3f "
            "descriptor_ms=%.3f serialize_ms=%.3f response_wait_ms=%.3f "
            "deserialize_ms=%.3f total_ms=%.3f request_bytes=%d "
            "response_bytes=%d endpoint=%s lease_id=%s",
            request_id,
            layer_name,
            int(seq_len),
            len(block_ids),
            async_import,
            str(response_metadata.get("status", "ready")),
            sync_ms,
            descriptor_ms,
            serialize_ms,
            post_ms,
            deserialize_ms,
            (time.perf_counter() - total_start) * 1000.0,
            len(request_body),
            len(response_body),
            tcp_endpoint,
            lease_id,
        )
    return int(response_metadata["seq_len"])


def import_prefill_kv(
    *,
    request_id: str,
    layer_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_len: int,
    block_ids: Sequence[int] | None = None,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
    transport: Any | None = None,
) -> int:
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    if not tcp_endpoint:
        raise RuntimeError("PAP KV import requires a TCP control endpoint")

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    from vllm.pap.data_plane import PAPTensorTransport

    if _normalize_offload_kv_transport(transport) is PAPTensorTransport.CUDA_IPC:
        return _post_prefill_kv_ipc(
            request_id=request_id,
            layer_name=layer_name,
            key=key,
            value=value,
            seq_len=int(seq_len),
            block_ids=block_ids,
            tcp_endpoint=tcp_endpoint,
            timeout=request_timeout,
        )

    metadata = {
        "command": "import_prefill_kv",
        "request_id": request_id,
        "layer_name": layer_name,
        "seq_len": int(seq_len),
        "block_ids": [] if block_ids is None else [int(b) for b in block_ids],
    }
    request_body = serialize_tensor_bundle(metadata, {"key": key, "value": value})
    response_body = _post_bytes_tcp(
        endpoint=tcp_endpoint,
        payload=request_body,
        timeout=request_timeout,
    )
    response_metadata, _ = deserialize_tensor_bundle(response_body)
    return int(response_metadata["seq_len"])


def import_prefill_kv_from_paged_cache(
    *,
    request_id: str,
    layer_name: str,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
    num_kv_heads: int,
    layout: str,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
    transport: Any | None = None,
) -> int:
    from vllm.pap.remote_attention import gather_paged_kv

    if layout not in {"NHD", "HND"}:
        raise ValueError(f"unsupported KV cache layout: {layout}")
    key, value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=block_table,
        seq_len=int(seq_len),
        num_kv_heads=int(num_kv_heads),
        layout=layout,
    )
    return import_prefill_kv(
        request_id=request_id,
        layer_name=layer_name,
        key=key,
        value=value,
        seq_len=int(seq_len),
        block_ids=_block_ids_from_block_table(
            block_table=block_table, seq_len=int(seq_len), block_size=int(block_size)
        ),
        tcp_endpoint=tcp_endpoint,
        timeout=timeout,
        transport=transport,
    )
