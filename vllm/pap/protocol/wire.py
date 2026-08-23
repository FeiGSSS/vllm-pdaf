# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Binary wire codec for PAP sealed KV control messages."""

from __future__ import annotations

import json
from typing import Any

import torch

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unsupported PAP tensor dtype: {name}") from exc


def serialize_tensor_bundle(
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> bytes:
    """Serialize metadata and optional CPU tensor payloads."""

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
    """Deserialize one PAP metadata and tensor bundle."""

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


__all__ = ["deserialize_tensor_bundle", "serialize_tensor_bundle"]
