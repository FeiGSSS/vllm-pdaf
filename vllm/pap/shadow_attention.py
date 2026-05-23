# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side PAP shadow attention hook.

This module reports metadata at the true Qwen3 q/k/v -> attention boundary
and provides the first remote-output path for the PAP prototype. The current
path is deliberately conservative: projection can still run local attention for
KV update/fallback, then ask the internal attention executor for an output to
feed into ``o_proj``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections.abc import Sequence
from http.client import HTTPConnection, HTTPSConnection, RemoteDisconnected
from threading import local
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import torch

logger = logging.getLogger(__name__)

_OPENAI_REQUEST_ID_PREFIXES = ("cmpl-", "chatcmpl-")
_HTTP_CONNECTIONS = local()
_TCP_CONNECTIONS = local()


def _pap_config() -> dict[str, Any]:
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except Exception:
        return {}
    if vllm_config is None or vllm_config.kv_transfer_config is None:
        return {}
    return vllm_config.kv_transfer_config.kv_connector_extra_config


def _enabled(enabled: bool | str | None = None) -> bool:
    if enabled is not None:
        return str(enabled).lower() in {"1", "true", "yes"}
    config = _pap_config()
    configured = config.get(
        "pap_shadow_attention", os.environ.get("PAP_SHADOW_ATTENTION", "")
    )
    return str(configured).lower() in {"1", "true", "yes"}


def _attention_endpoint(endpoint: str | None = None) -> str:
    if endpoint is not None:
        return str(endpoint).rstrip("/")
    config = _pap_config()
    configured = config.get(
        "pap_attention_endpoint",
        os.environ.get("PAP_ATTENTION_ENDPOINT", "http://127.0.0.1:8300"),
    )
    return str(configured).rstrip("/")


def _post_json(
    *,
    endpoint: str | None,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    base_url = _attention_endpoint(endpoint)
    body = json.dumps(payload).encode("utf-8")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"unsupported PAP attention endpoint: {base_url}")

    cache: dict[tuple[str, str], HTTPConnection] = getattr(
        _HTTP_CONNECTIONS, "connections", {}
    )
    if not cache:
        _HTTP_CONNECTIONS.connections = cache
    key = (parsed.scheme, parsed.netloc)
    url_path = path if parsed.path in {"", "/"} else f"{parsed.path.rstrip('/')}{path}"

    last_error: Exception | None = None
    for _ in range(2):
        conn = cache.get(key)
        if conn is None:
            conn_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
            conn = conn_cls(parsed.netloc, timeout=float(timeout))
            cache[key] = conn
        try:
            conn.request(
                "POST",
                url_path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            response_body = response.read()
            if response.status >= 400:
                response_message = response_body.decode("utf-8", errors="replace")
                raise HTTPError(
                    f"{base_url}{path}",
                    response.status,
                    f"{response.reason}: {response_message}",
                    response.headers,
                    None,
                )
            return json.loads(response_body.decode("utf-8"))
        except (BrokenPipeError, ConnectionError, RemoteDisconnected, OSError) as exc:
            last_error = exc
            cache.pop(key, None)
            conn.close()
    assert last_error is not None
    raise last_error


def _post_bytes(
    *,
    endpoint: str | None,
    path: str,
    payload: bytes,
    timeout: float,
) -> bytes:
    base_url = _attention_endpoint(endpoint)
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"unsupported PAP attention endpoint: {base_url}")

    cache: dict[tuple[str, str], HTTPConnection] = getattr(
        _HTTP_CONNECTIONS, "connections", {}
    )
    if not cache:
        _HTTP_CONNECTIONS.connections = cache
    key = (parsed.scheme, parsed.netloc)
    url_path = path if parsed.path in {"", "/"} else f"{parsed.path.rstrip('/')}{path}"

    last_error: Exception | None = None
    for _ in range(2):
        conn = cache.get(key)
        if conn is None:
            conn_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
            conn = conn_cls(parsed.netloc, timeout=float(timeout))
            cache[key] = conn
        try:
            conn.request(
                "POST",
                url_path,
                body=payload,
                headers={"Content-Type": "application/octet-stream"},
            )
            response = conn.getresponse()
            response_body = response.read()
            if response.status >= 400:
                response_message = response_body.decode("utf-8", errors="replace")
                raise HTTPError(
                    f"{base_url}{path}",
                    response.status,
                    f"{response.reason}: {response_message}",
                    response.headers,
                    None,
                )
            return response_body
        except (BrokenPipeError, ConnectionError, RemoteDisconnected, OSError) as exc:
            last_error = exc
            cache.pop(key, None)
            conn.close()
    assert last_error is not None
    raise last_error


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


def _remote_attention_transport() -> str:
    return os.environ.get("PAP_REMOTE_ATTENTION_TRANSPORT", "http").lower()


def _compact_tcp_enabled() -> bool:
    return os.environ.get("PAP_REMOTE_ATTENTION_COMPACT_TCP", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _select_request_id(request_ids: Sequence[str] | None) -> str | None:
    if not request_ids:
        return None
    for request_id in request_ids:
        request_id_str = str(request_id)
        if request_id_str.startswith(_OPENAI_REQUEST_ID_PREFIXES):
            return request_id_str
    return None


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


def _is_decode(num_scheduled_tokens: Sequence[int] | None, query: torch.Tensor) -> bool:
    if num_scheduled_tokens is not None and len(num_scheduled_tokens) > 0:
        return all(int(num_tokens) == 1 for num_tokens in num_scheduled_tokens)
    return query.shape[0] == 1


def build_layer_event_payload(
    *,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    request_ids: Sequence[str] | None,
    num_scheduled_tokens: Sequence[int] | None,
    num_reqs: int | None,
    num_actual_tokens: int | None,
    max_seq_len: int | None,
) -> dict[str, Any] | None:
    request_id = _select_request_id(request_ids)
    if request_id is None:
        return None

    return {
        "request_id": request_id,
        "layer_name": layer_name,
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "dtype": str(query.dtype),
        "device": str(query.device),
        "is_decode": _is_decode(num_scheduled_tokens, query),
        "num_reqs": None if num_reqs is None else int(num_reqs),
        "num_actual_tokens": None
        if num_actual_tokens is None
        else int(num_actual_tokens),
        "max_seq_len": None if max_seq_len is None else int(max_seq_len),
    }


def maybe_report_qkv_boundary(
    *,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    request_ids: Sequence[str] | None,
    num_scheduled_tokens: Sequence[int] | None,
    num_reqs: int | None,
    num_actual_tokens: int | None,
    max_seq_len: int | None,
    enabled: bool | str | None = None,
    endpoint: str | None = None,
) -> None:
    if not _enabled(enabled):
        return

    payload = build_layer_event_payload(
        layer_name=layer_name,
        query=query,
        key=key,
        value=value,
        request_ids=request_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_seq_len=max_seq_len,
    )
    if payload is None:
        return

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{_attention_endpoint(endpoint)}/v1/pap/attention/layer-event",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=0.5):
            return
    except (TimeoutError, URLError, OSError):
        logger.debug(
            "failed to report PAP shadow attention event request_id=%s endpoint=%s",
            payload.get("request_id"),
            _attention_endpoint(endpoint),
            exc_info=True,
        )


def build_remote_attention_request(
    *,
    request_id: str,
    layer_name: str,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    scale: float,
    layout: str,
) -> dict[str, Any]:
    from vllm.pap.remote_attention import gather_paged_kv, serialize_tensor

    if layout not in {"NHD", "HND"}:
        raise ValueError(f"unsupported KV cache layout: {layout}")
    key, value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=block_table,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        layout=layout,
    )
    return {
        "request_id": request_id,
        "layer_name": layer_name,
        "query": serialize_tensor(query),
        "key": serialize_tensor(key),
        "value": serialize_tensor(value),
        "scale": float(scale),
    }


def import_prefill_kv(
    *,
    request_id: str,
    layer_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_len: int,
    block_ids: Sequence[int] | None = None,
    endpoint: str | None = None,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
) -> int:
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor,
        serialize_tensor_bundle,
    )

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    use_binary = os.environ.get("PAP_REMOTE_ATTENTION_BINARY", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_binary:
        metadata = {
            "command": "import_prefill_kv",
            "request_id": request_id,
            "layer_name": layer_name,
            "seq_len": int(seq_len),
            "block_ids": [] if block_ids is None else [
                int(block_id) for block_id in block_ids
            ],
        }
        request_body = serialize_tensor_bundle(
            metadata,
            {"key": key, "value": value},
        )
        if _remote_attention_transport() == "tcp" and tcp_endpoint:
            response_body = _post_bytes_tcp(
                endpoint=tcp_endpoint,
                payload=request_body,
                timeout=request_timeout,
            )
        else:
            response_body = _post_bytes(
                endpoint=endpoint,
                path="/v1/pap/attention/import-prefill-kv-binary",
                payload=request_body,
                timeout=request_timeout,
            )
        response_metadata, _ = deserialize_tensor_bundle(response_body)
        return int(response_metadata["seq_len"])

    payload = {
        "request_id": request_id,
        "layer_name": layer_name,
        "key": serialize_tensor(key),
        "value": serialize_tensor(value),
        "seq_len": int(seq_len),
    }
    if block_ids is not None:
        payload["block_ids"] = [int(block_id) for block_id in block_ids]
    result = _post_json(
        endpoint=endpoint,
        path="/v1/pap/attention/import-prefill-kv",
        payload=payload,
        timeout=request_timeout,
    )
    return int(result["seq_len"])


def _block_ids_from_block_table(
    *,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> list[int]:
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("PAP prototype supports one decode request per call")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    blocks = block_table[0, :num_blocks].to(device="cpu", dtype=torch.long).tolist()
    return [int(block_id) for block_id in blocks]


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
    endpoint: str | None = None,
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
        endpoint=endpoint,
        tcp_endpoint=tcp_endpoint,
        timeout=timeout,
    )


def compute_stateful_remote_attention_output(
    *,
    request_id: str,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    block_id: int | None = None,
    slot: int | None = None,
    seq_len: int | None = None,
    endpoint: str | None = None,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
) -> torch.Tensor:
    from vllm.pap.remote_attention import (
        deserialize_attention_result,
        deserialize_tensor_bundle,
        serialize_tensor,
        serialize_tensor_bundle,
    )

    payload = {
        "request_id": request_id,
        "layer_name": layer_name,
        "query": serialize_tensor(query),
        "key": serialize_tensor(key),
        "value": serialize_tensor(value),
        "scale": float(scale),
    }
    if block_id is not None or slot is not None or seq_len is not None:
        if block_id is None or slot is None or seq_len is None:
            raise ValueError("block_id, slot, and seq_len must be provided together")
        payload.update(
            {
                "block_id": int(block_id),
                "slot": int(slot),
                "seq_len": int(seq_len),
            }
        )
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    use_binary = os.environ.get("PAP_REMOTE_ATTENTION_BINARY", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_binary:
        request_body = serialize_tensor_bundle(
            {
                "request_id": request_id,
                "layer_name": layer_name,
                "scale": float(scale),
                "block_id": None if block_id is None else int(block_id),
                "slot": None if slot is None else int(slot),
                "seq_len": None if seq_len is None else int(seq_len),
            },
            {"query": query, "key": key, "value": value},
        )
        if _remote_attention_transport() == "tcp" and tcp_endpoint:
            response_body = _post_bytes_tcp(
                endpoint=tcp_endpoint,
                payload=request_body,
                timeout=request_timeout,
            )
        else:
            response_body = _post_bytes(
                endpoint=endpoint,
                path="/v1/pap/attention/append-and-compute-binary",
                payload=request_body,
                timeout=request_timeout,
            )
        _, tensors = deserialize_tensor_bundle(response_body)
        return tensors["output"]

    result = _post_json(
        endpoint=endpoint,
        path="/v1/pap/attention/append-and-compute",
        payload=payload,
        timeout=request_timeout,
    )
    return deserialize_attention_result(result["output"])


def trigger_offload_exec_attention(
    *,
    endpoint: str | None,
    tcp_endpoint: str | None = None,
    request_id: str,
    layer_name: str,
    step: int,
    scale: float,
    remote_address: str,
    timeout: float | None = None,
) -> None:
    """Trigger Attention to receive QKV and send O over OFFLOAD_EXEC."""

    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    payload = {
        "request_id": request_id,
        "layer_name": layer_name,
        "step": int(step),
        "scale": float(scale),
        "remote_address": str(remote_address),
    }
    if _remote_attention_transport() == "tcp" and tcp_endpoint:
        from vllm.pap.remote_attention import (
            deserialize_tensor_bundle,
            serialize_tensor_bundle,
        )

        response_body = _post_bytes_tcp(
            endpoint=tcp_endpoint,
            payload=serialize_tensor_bundle(
                {
                    "command": "offload_exec",
                    **payload,
                },
                {},
            ),
            timeout=request_timeout,
        )
        deserialize_tensor_bundle(response_body)
        return

    _post_json(
        endpoint=endpoint,
        path="/v1/pap/attention/offload-exec",
        payload=payload,
        timeout=request_timeout,
    )


def compute_stateful_remote_attention_outputs_batch(
    *,
    calls: Sequence[dict[str, Any]],
    endpoint: str | None = None,
    tcp_endpoint: str | None = None,
    timeout: float | None = None,
) -> list[torch.Tensor]:
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    if not calls:
        return []
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    use_binary = os.environ.get("PAP_REMOTE_ATTENTION_BINARY", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    use_compact_tcp = (
        use_binary
        and _remote_attention_transport() == "tcp"
        and bool(tcp_endpoint)
        and _compact_tcp_enabled()
    )
    if not use_binary:
        return [
            compute_stateful_remote_attention_output(
                **call,
                endpoint=endpoint if call.get("endpoint") is None else call["endpoint"],
                tcp_endpoint=tcp_endpoint
                if call.get("tcp_endpoint") is None
                else call["tcp_endpoint"],
                timeout=request_timeout,
            )
            for call in calls
        ]

    items: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    compact_qkv_tensors: list[torch.Tensor] = []
    trace_remote_attention = os.environ.get(
        "PAP_OFFLOAD_EXEC_TRACE", ""
    ).lower() in ("1", "true", "yes", "on")
    trace_total_start = time.perf_counter() if trace_remote_attention else 0.0
    trace_serialize_start = time.perf_counter() if trace_remote_attention else 0.0
    for index, call in enumerate(calls):
        block_id = call.get("block_id")
        slot = call.get("slot")
        seq_len = call.get("seq_len")
        items.append(
            {
                "request_id": str(call["request_id"]),
                "layer_name": str(call["layer_name"]),
                "scale": float(call["scale"]),
                "block_id": None if block_id is None else int(block_id),
                "slot": None if slot is None else int(slot),
                "seq_len": None if seq_len is None else int(seq_len),
            }
        )
        qkv = torch.cat(
            [
                call["query"].reshape(1, -1),
                call["key"].reshape(1, -1),
                call["value"].reshape(1, -1),
            ],
            dim=-1,
        )
        tensors[f"qkv_{index}"] = qkv
        if use_compact_tcp:
            query = call["query"]
            key = call["key"]
            value = call["value"]
            if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
                raise ValueError(
                    "compact PAP attention requires 3D query/key/value tensors"
                )
            item = items[-1]
            item["q_size"] = int(query.numel())
            item["kv_size"] = int(key.numel())
            item["num_heads"] = int(query.shape[1])
            item["num_kv_heads"] = int(key.shape[1])
            item["head_dim"] = int(query.shape[2])
            compact_qkv_tensors.append(qkv)

    if use_compact_tcp:
        from vllm.pap.remote_attention import serialize_compact_attention_batch

        request_body = serialize_compact_attention_batch(items, compact_qkv_tensors)
    else:
        request_body = serialize_tensor_bundle({"items": items}, tensors)
    trace_serialize_ms = (
        (time.perf_counter() - trace_serialize_start) * 1000.0
        if trace_remote_attention
        else 0.0
    )
    trace_rpc_start = time.perf_counter() if trace_remote_attention else 0.0
    if _remote_attention_transport() == "tcp" and tcp_endpoint:
        response_body = _post_bytes_tcp(
            endpoint=tcp_endpoint,
            payload=request_body,
            timeout=request_timeout,
        )
    else:
        response_body = _post_bytes(
            endpoint=endpoint,
            path="/v1/pap/attention/append-and-compute-batch-binary",
            payload=request_body,
            timeout=request_timeout,
        )
    trace_rpc_ms = (
        (time.perf_counter() - trace_rpc_start) * 1000.0
        if trace_remote_attention
        else 0.0
    )
    trace_deserialize_start = time.perf_counter() if trace_remote_attention else 0.0
    if use_compact_tcp:
        from vllm.pap.remote_attention import deserialize_compact_attention_response

        outputs = deserialize_compact_attention_response(response_body)
    else:
        metadata, response_tensors = deserialize_tensor_bundle(response_body)
        outputs = []
        for index in range(len(metadata["items"])):
            outputs.append(response_tensors[f"output_{index}"])
    if trace_remote_attention:
        trace_deserialize_ms = (
            time.perf_counter() - trace_deserialize_start
        ) * 1000.0
        trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
        layer_name = str(calls[0]["layer_name"]) if calls else ""
        logger.info(
            "PAP remote attention batch trace layer=%s calls=%d "
            "serialize_ms=%.3f rpc_ms=%.3f deserialize_ms=%.3f "
            "total_ms=%.3f request_bytes=%d response_bytes=%d",
            layer_name,
            len(calls),
            trace_serialize_ms,
            trace_rpc_ms,
            trace_deserialize_ms,
            trace_total_ms,
            len(request_body),
            len(response_body),
        )
    return outputs


def compute_remote_attention_output(
    *,
    request_id: str,
    layer_name: str,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    scale: float,
    layout: str,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> torch.Tensor:
    from vllm.pap.remote_attention import deserialize_attention_result

    payload = build_remote_attention_request(
        request_id=request_id,
        layer_name=layer_name,
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        scale=scale,
        layout=layout,
    )
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    result = _post_json(
        endpoint=endpoint,
        path="/v1/pap/attention/compute",
        payload=payload,
        timeout=request_timeout,
    )
    return deserialize_attention_result(result["output"])
