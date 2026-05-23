# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP control-plane helpers.

TCP control messages (compact binary format) for triggering the remote
Attention executor and importing prefill KV. Tensor data for OFFLOAD_EXEC
travels over NCCL; prefill KV import uses TCP binary bundles.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from collections.abc import Sequence
from threading import local
from typing import Any
from urllib.parse import urlsplit

import torch

logger = logging.getLogger(__name__)

_OPENAI_REQUEST_ID_PREFIXES = ("cmpl-", "chatcmpl-")
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
    """Trigger Attention to receive QKV and send O over OFFLOAD_EXEC NCCL."""

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
    )
