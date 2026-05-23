# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small PAP remote-attention data-plane helpers."""

from __future__ import annotations

import base64
import json
import os
import struct
from typing import Any, Literal

import torch

COMPACT_ATTENTION_REQUEST_MAGIC = b"PAPATN1\0"
COMPACT_ATTENTION_RESPONSE_MAGIC = b"PAPOUT1\0"
COMPACT_OFFLOAD_EXEC_MAGIC = b"PAPEXE1\0"
COMPACT_OFFLOAD_EXEC_OK_MAGIC = b"PAPOKAY\0"

_COMPACT_COUNT_STRUCT = struct.Struct("<8sI")
_COMPACT_REQUEST_HEADER_STRUCT = struct.Struct("<HHHHIIIIIqqqfI")
_COMPACT_RESPONSE_HEADER_STRUCT = struct.Struct("<HHIIII")
_COMPACT_OFFLOAD_EXEC_STRUCT = struct.Struct("<8sHHHqf")

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}
_DTYPE_ID_BY_DTYPE: dict[torch.dtype, int] = {
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float32: 3,
}
_DTYPE_BY_ID = {value: key for key, value in _DTYPE_ID_BY_DTYPE.items()}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unsupported PAP tensor dtype: {name}") from exc


def _dtype_id(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_ID_BY_DTYPE[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported PAP tensor dtype: {dtype}") from exc


def _dtype_from_id(dtype_id: int) -> torch.dtype:
    try:
        return _DTYPE_BY_ID[int(dtype_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported PAP tensor dtype id: {dtype_id}") from exc


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().contiguous().cpu()
    raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": _dtype_name(cpu_tensor.dtype),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def deserialize_tensor(payload: dict[str, Any]) -> torch.Tensor:
    dtype = _dtype_from_name(str(payload["dtype"]))
    raw = base64.b64decode(str(payload["data"]).encode("ascii"))
    byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return byte_tensor.view(dtype).reshape(tuple(int(dim) for dim in payload["shape"]))


def serialize_attention_result(tensor: torch.Tensor) -> dict[str, Any]:
    return serialize_tensor(tensor)


def deserialize_attention_result(payload: dict[str, Any]) -> torch.Tensor:
    return deserialize_tensor(payload)


def serialize_tensor_bundle(
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> bytes:
    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    for name, tensor in tensors.items():
        cpu_tensor = tensor.detach().contiguous().cpu()
        raw = cpu_tensor.view(torch.uint8).numpy().tobytes()
        entries.append(
            {
                "name": name,
                "shape": list(cpu_tensor.shape),
                "dtype": _dtype_name(cpu_tensor.dtype),
                "nbytes": len(raw),
            }
        )
        chunks.append(raw)

    header = json.dumps(
        {"metadata": metadata, "tensors": entries},
        separators=(",", ":"),
    ).encode("utf-8")
    return len(header).to_bytes(8, "little") + header + b"".join(chunks)


def deserialize_tensor_bundle(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if len(payload) < 8:
        raise ValueError("tensor bundle is missing header length")
    header_len = int.from_bytes(payload[:8], "little")
    header_end = 8 + header_len
    if len(payload) < header_end:
        raise ValueError("tensor bundle header is truncated")
    header = json.loads(payload[8:header_end].decode("utf-8"))
    offset = header_end
    tensors: dict[str, torch.Tensor] = {}
    for entry in header.get("tensors", []):
        nbytes = int(entry["nbytes"])
        raw = payload[offset : offset + nbytes]
        if len(raw) != nbytes:
            raise ValueError(f"tensor bundle payload is truncated for {entry['name']}")
        dtype = _dtype_from_name(str(entry["dtype"]))
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        tensors[str(entry["name"])] = byte_tensor.view(dtype).reshape(
            tuple(int(dim) for dim in entry["shape"])
        )
        offset += nbytes
    if offset != len(payload):
        raise ValueError("tensor bundle has trailing bytes")
    return dict(header.get("metadata", {})), tensors


def serialize_compact_attention_batch(
    items: list[dict[str, Any]],
    qkv_tensors: list[torch.Tensor],
) -> bytes:
    if len(items) != len(qkv_tensors):
        raise ValueError("compact attention items and tensors length mismatch")

    chunks = [
        _COMPACT_COUNT_STRUCT.pack(COMPACT_ATTENTION_REQUEST_MAGIC, len(items))
    ]
    for item, qkv in zip(items, qkv_tensors):
        request_id = str(item["request_id"]).encode("utf-8")
        layer_name = str(item["layer_name"]).encode("utf-8")
        qkv_cpu = qkv.detach().contiguous().cpu()
        raw = qkv_cpu.view(torch.uint8).numpy().tobytes()
        block_id = -1 if item.get("block_id") is None else int(item["block_id"])
        slot = -1 if item.get("slot") is None else int(item["slot"])
        seq_len = -1 if item.get("seq_len") is None else int(item["seq_len"])
        chunks.append(
            _COMPACT_REQUEST_HEADER_STRUCT.pack(
                len(request_id),
                len(layer_name),
                _dtype_id(qkv_cpu.dtype),
                0,
                int(item["q_size"]),
                int(item["kv_size"]),
                int(item["num_heads"]),
                int(item["num_kv_heads"]),
                int(item["head_dim"]),
                block_id,
                slot,
                seq_len,
                float(item["scale"]),
                len(raw),
            )
        )
        chunks.extend([request_id, layer_name, raw])
    return b"".join(chunks)


def deserialize_compact_attention_batch(
    payload: bytes,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    if len(payload) < _COMPACT_COUNT_STRUCT.size:
        raise ValueError("compact attention request is truncated")
    magic, count = _COMPACT_COUNT_STRUCT.unpack_from(payload, 0)
    if magic != COMPACT_ATTENTION_REQUEST_MAGIC:
        raise ValueError("invalid compact attention request magic")
    offset = _COMPACT_COUNT_STRUCT.size
    items: list[dict[str, Any]] = []
    qkv_tensors: list[torch.Tensor] = []
    for _ in range(int(count)):
        if len(payload) < offset + _COMPACT_REQUEST_HEADER_STRUCT.size:
            raise ValueError("compact attention request item header is truncated")
        (
            request_id_len,
            layer_name_len,
            dtype_id,
            _reserved,
            q_size,
            kv_size,
            num_heads,
            num_kv_heads,
            head_dim,
            block_id,
            slot,
            seq_len,
            scale,
            qkv_nbytes,
        ) = _COMPACT_REQUEST_HEADER_STRUCT.unpack_from(payload, offset)
        offset += _COMPACT_REQUEST_HEADER_STRUCT.size
        strings_end = offset + int(request_id_len) + int(layer_name_len)
        raw_end = strings_end + int(qkv_nbytes)
        if len(payload) < raw_end:
            raise ValueError("compact attention request item payload is truncated")
        request_id = payload[offset : offset + int(request_id_len)].decode("utf-8")
        offset += int(request_id_len)
        layer_name = payload[offset : offset + int(layer_name_len)].decode("utf-8")
        offset += int(layer_name_len)
        raw = payload[offset : offset + int(qkv_nbytes)]
        offset += int(qkv_nbytes)
        dtype = _dtype_from_id(dtype_id)
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        qkv = byte_tensor.view(dtype).reshape(1, int(q_size) + 2 * int(kv_size))
        items.append(
            {
                "request_id": request_id,
                "layer_name": layer_name,
                "scale": float(scale),
                "block_id": None if block_id < 0 else int(block_id),
                "slot": None if slot < 0 else int(slot),
                "seq_len": None if seq_len < 0 else int(seq_len),
                "q_size": int(q_size),
                "kv_size": int(kv_size),
                "num_heads": int(num_heads),
                "num_kv_heads": int(num_kv_heads),
                "head_dim": int(head_dim),
            }
        )
        qkv_tensors.append(qkv)
    if offset != len(payload):
        raise ValueError("compact attention request has trailing bytes")
    return items, qkv_tensors


def serialize_compact_attention_response(outputs: list[torch.Tensor]) -> bytes:
    chunks = [
        _COMPACT_COUNT_STRUCT.pack(COMPACT_ATTENTION_RESPONSE_MAGIC, len(outputs))
    ]
    for output in outputs:
        if output.ndim != 3:
            raise ValueError("compact attention output must have 3 dimensions")
        output_cpu = output.detach().contiguous().cpu()
        raw = output_cpu.view(torch.uint8).numpy().tobytes()
        chunks.append(
            _COMPACT_RESPONSE_HEADER_STRUCT.pack(
                _dtype_id(output_cpu.dtype),
                0,
                len(raw),
                int(output_cpu.shape[0]),
                int(output_cpu.shape[1]),
                int(output_cpu.shape[2]),
            )
        )
        chunks.append(raw)
    return b"".join(chunks)


def deserialize_compact_attention_response(payload: bytes) -> list[torch.Tensor]:
    if len(payload) < _COMPACT_COUNT_STRUCT.size:
        raise ValueError("compact attention response is truncated")
    magic, count = _COMPACT_COUNT_STRUCT.unpack_from(payload, 0)
    if magic != COMPACT_ATTENTION_RESPONSE_MAGIC:
        raise ValueError("invalid compact attention response magic")
    offset = _COMPACT_COUNT_STRUCT.size
    outputs: list[torch.Tensor] = []
    for _ in range(int(count)):
        if len(payload) < offset + _COMPACT_RESPONSE_HEADER_STRUCT.size:
            raise ValueError("compact attention response item header is truncated")
        dtype_id, _reserved, nbytes, dim0, dim1, dim2 = (
            _COMPACT_RESPONSE_HEADER_STRUCT.unpack_from(payload, offset)
        )
        offset += _COMPACT_RESPONSE_HEADER_STRUCT.size
        raw_end = offset + int(nbytes)
        if len(payload) < raw_end:
            raise ValueError("compact attention response item payload is truncated")
        raw = payload[offset:raw_end]
        offset = raw_end
        dtype = _dtype_from_id(dtype_id)
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        outputs.append(
            byte_tensor.view(dtype).reshape(int(dim0), int(dim1), int(dim2))
        )
    if offset != len(payload):
        raise ValueError("compact attention response has trailing bytes")
    return outputs


def serialize_compact_offload_exec_command(
    *,
    request_id: str,
    layer_name: str,
    step: int,
    scale: float,
    remote_address: str,
) -> bytes:
    request_id_bytes = str(request_id).encode("utf-8")
    layer_name_bytes = str(layer_name).encode("utf-8")
    remote_address_bytes = str(remote_address).encode("utf-8")
    return b"".join(
        [
            _COMPACT_OFFLOAD_EXEC_STRUCT.pack(
                COMPACT_OFFLOAD_EXEC_MAGIC,
                len(request_id_bytes),
                len(layer_name_bytes),
                len(remote_address_bytes),
                int(step),
                float(scale),
            ),
            request_id_bytes,
            layer_name_bytes,
            remote_address_bytes,
        ]
    )


def deserialize_compact_offload_exec_command(payload: bytes) -> dict[str, Any]:
    if len(payload) < _COMPACT_OFFLOAD_EXEC_STRUCT.size:
        raise ValueError("compact offload-exec command is truncated")
    (
        magic,
        request_id_len,
        layer_name_len,
        remote_address_len,
        step,
        scale,
    ) = _COMPACT_OFFLOAD_EXEC_STRUCT.unpack_from(payload, 0)
    if magic != COMPACT_OFFLOAD_EXEC_MAGIC:
        raise ValueError("invalid compact offload-exec command magic")
    offset = _COMPACT_OFFLOAD_EXEC_STRUCT.size
    request_id_end = offset + int(request_id_len)
    layer_name_end = request_id_end + int(layer_name_len)
    remote_address_end = layer_name_end + int(remote_address_len)
    if len(payload) < remote_address_end:
        raise ValueError("compact offload-exec command payload is truncated")
    if remote_address_end != len(payload):
        raise ValueError("compact offload-exec command has trailing bytes")
    return {
        "request_id": payload[offset:request_id_end].decode("utf-8"),
        "layer_name": payload[request_id_end:layer_name_end].decode("utf-8"),
        "remote_address": payload[layer_name_end:remote_address_end].decode("utf-8"),
        "step": int(step),
        "scale": float(scale),
    }


def serialize_compact_offload_exec_ack() -> bytes:
    return COMPACT_OFFLOAD_EXEC_OK_MAGIC


def deserialize_compact_offload_exec_ack(payload: bytes) -> None:
    if payload != COMPACT_OFFLOAD_EXEC_OK_MAGIC:
        raise ValueError("invalid compact offload-exec ack")


def gather_paged_kv(
    *,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    layout: Literal["NHD", "HND"],
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("PAP prototype supports one decode request per call")
    if kv_cache.shape[0] != 2:
        raise ValueError("expected KV cache first dimension to contain K and V")

    blocks = block_table[0].to(device="cpu", dtype=torch.long).tolist()
    if layout == "NHD":
        logical_nhd = True
    elif layout == "HND":
        # vLLM allocates HND physically, then binds a permuted logical view back
        # to Attention. In that common case shape[2] is still block_size and
        # shape[3] is num_kv_heads; direct physical-HND tensors have the reverse.
        logical_nhd = int(kv_cache.shape[3]) == int(num_kv_heads)
    else:
        raise ValueError(f"unsupported KV cache layout: {layout}")
    block_size = int(kv_cache.shape[2] if logical_nhd else kv_cache.shape[3])

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    remaining = int(seq_len)
    for block_id in blocks:
        if remaining <= 0:
            break
        take = min(block_size, remaining)
        if logical_nhd:
            keys.append(kv_cache[0, block_id, :take, :num_kv_heads, :])
            values.append(kv_cache[1, block_id, :take, :num_kv_heads, :])
        else:
            keys.append(kv_cache[0, block_id, :num_kv_heads, :take, :].transpose(0, 1))
            values.append(
                kv_cache[1, block_id, :num_kv_heads, :take, :].transpose(0, 1)
            )
        remaining -= take

    if remaining > 0:
        raise ValueError("block table does not cover requested sequence length")
    if not keys:
        empty_shape = (0, num_kv_heads, int(kv_cache.shape[-1]))
        return (
            torch.empty(empty_shape, dtype=kv_cache.dtype, device=kv_cache.device),
            torch.empty(empty_shape, dtype=kv_cache.dtype, device=kv_cache.device),
        )
    return torch.cat(keys, dim=0), torch.cat(values, dim=0)


def compute_attention_output(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "expected query/key/value tensors with shape [tokens, heads, dim]"
        )
    if query.shape[0] != 1:
        raise ValueError("PAP prototype supports decode attention with one query token")
    if key.shape[1] == 0 or value.shape[1] == 0:
        raise ValueError("key/value tensors must have at least one KV head")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")

    compute_dtype = torch.float32
    q = query.to(compute_dtype)
    k = key.to(compute_dtype)
    v = value.to(compute_dtype)
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->qhk", q, k) * float(scale)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("qhk,khd->qhd", probs, v)
    return out.to(query.dtype)


def compute_segmented_attention_output(
    *,
    query: torch.Tensor,
    segments: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float,
) -> torch.Tensor:
    if query.ndim != 3:
        raise ValueError("expected query tensor with shape [tokens, heads, dim]")
    if query.shape[0] != 1:
        raise ValueError("PAP prototype supports decode attention with one query token")

    non_empty_segments = [
        (key, value) for key, value in segments if key.numel() > 0
    ]
    if not non_empty_segments:
        raise ValueError("segmented attention requires at least one KV token")
    for key, value in non_empty_segments:
        if key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "expected key/value tensors with shape [tokens, heads, dim]"
            )
        if key.shape[1] == 0 or value.shape[1] == 0:
            raise ValueError("key/value tensors must have at least one KV head")
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("query heads must be divisible by KV heads")

    use_cuda_segmented = (
        query.is_cuda
        and len(non_empty_segments) > 1
        and os.environ.get("PAP_REMOTE_ATTENTION_CUDA_SEGMENTED", "0").lower()
        in {"1", "true", "yes"}
    )
    if query.is_cuda and not use_cuda_segmented:
        key = torch.cat([segment_key for segment_key, _ in non_empty_segments], dim=0)
        value = torch.cat(
            [segment_value for _, segment_value in non_empty_segments], dim=0
        )
        q_sdpa = query.permute(1, 0, 2).unsqueeze(0)
        k_sdpa = key.permute(1, 0, 2).unsqueeze(0)
        v_sdpa = value.permute(1, 0, 2).unsqueeze(0)
        output = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=0.0,
            scale=float(scale),
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output.squeeze(0).permute(1, 0, 2).to(query.dtype)

    score_segments: list[torch.Tensor] = []
    value_segments: list[torch.Tensor] = []
    compute_dtype = query.dtype if query.is_cuda else torch.float32
    q = query.to(compute_dtype)
    for key, value in non_empty_segments:
        repeat = query.shape[1] // key.shape[1]
        k = key.to(compute_dtype).repeat_interleave(repeat, dim=1)
        v = value.to(compute_dtype).repeat_interleave(repeat, dim=1)
        score_segments.append(torch.einsum("qhd,khd->qhk", q, k) * float(scale))
        value_segments.append(v)

    scores = torch.cat(score_segments, dim=-1)
    probs = torch.softmax(scores, dim=-1)

    output = torch.zeros_like(q)
    offset = 0
    for value in value_segments:
        length = int(value.shape[0])
        output += torch.einsum(
            "qhk,khd->qhd", probs[..., offset : offset + length], value
        )
        offset += length
    return output.to(query.dtype)
