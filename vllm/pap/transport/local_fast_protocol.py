# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Doorbell wire layout for the PAP local-fast transport."""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
from dataclasses import dataclass
from typing import Any

import torch

DOORBELL_RECORD_STRUCT = struct.Struct("<QQQQQQQQiiII")
DOORBELL_HEADER_BYTES = DOORBELL_RECORD_STRUCT.size
DOORBELL_RECORD_BYTES = int(
    os.environ.get("PAP_LOCAL_FAST_DOORBELL_RECORD_BYTES", str(64 * 1024))
)
if DOORBELL_RECORD_BYTES < DOORBELL_HEADER_BYTES:
    raise RuntimeError("PAP_LOCAL_FAST_DOORBELL_RECORD_BYTES is too small")

DIR_QKV = 0
DIR_OUTPUT = 1

RECORD_FLAG_PLAN_FULL = 1 << 0
RECORD_FLAG_PLAN_REF = 1 << 1
RECORD_FLAG_OUTPUT_DESCRIPTORLESS = 1 << 2
RECORD_FLAG_FIXED_TENSOR = 1 << 3

DTYPE_CODE_NONE = 0
DTYPE_CODE_FLOAT16 = 1
DTYPE_CODE_BFLOAT16 = 2
DTYPE_CODE_FLOAT32 = 3

_DTYPE_TO_CODE = {
    torch.float16: DTYPE_CODE_FLOAT16,
    torch.bfloat16: DTYPE_CODE_BFLOAT16,
    torch.float32: DTYPE_CODE_FLOAT32,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
_LAYER_INDEX_PATTERN = re.compile(r"^(.*\.layers\.)(\d+)(\..*)$")

SIGNAL_READY_QKV = 0
SIGNAL_READY_OUTPUT = 1
SIGNAL_RELEASE_QKV = 2
SIGNAL_RELEASE_OUTPUT = 3


def _doorbell_bytes(slot_count: int) -> int:
    return 2 * int(slot_count) * DOORBELL_RECORD_BYTES


def _doorbell_record_offset(direction: int, slot_id: int, slot_count: int) -> int:
    if direction not in (DIR_QKV, DIR_OUTPUT):
        raise ValueError(f"invalid PAP local fast direction: {direction}")
    if slot_id < 0 or slot_id >= slot_count:
        raise ValueError(f"invalid PAP local fast slot: {slot_id}")
    return (direction * slot_count + slot_id) * DOORBELL_RECORD_BYTES


def _signal_index(
    direction: int,
    slot_id: int,
    slot_count: int,
    *,
    release: bool,
) -> int:
    if direction == DIR_QKV:
        kind = SIGNAL_RELEASE_QKV if release else SIGNAL_READY_QKV
    elif direction == DIR_OUTPUT:
        kind = SIGNAL_RELEASE_OUTPUT if release else SIGNAL_READY_OUTPUT
    else:
        raise ValueError(f"invalid PAP local fast direction: {direction}")
    if slot_id < 0 or slot_id >= slot_count:
        raise ValueError(f"invalid PAP local fast slot: {slot_id}")
    return kind * slot_count + slot_id


@dataclass(frozen=True)
class _DoorbellRecord:
    seq: int
    nbytes: int
    offset: int
    metadata_len: int
    ack: int
    plan_id: int
    dim0: int
    dim1: int
    layer_index: int
    dtype_code: int
    flags: int


def _layer_index_and_template(layer_name: str) -> tuple[int, tuple[str, str]] | None:
    match = _LAYER_INDEX_PATTERN.match(str(layer_name))
    if match is None:
        return None
    return int(match.group(2)), (match.group(1), match.group(3))


def _layer_name_from_template(template: tuple[str, str], layer_index: int) -> str:
    return f"{template[0]}{int(layer_index)}{template[1]}"


def _doorbell_read_record(mm: mmap.mmap, record_offset: int) -> _DoorbellRecord:
    raw = bytes(mm[record_offset : record_offset + DOORBELL_HEADER_BYTES])
    unpacked = DOORBELL_RECORD_STRUCT.unpack(raw)
    return _DoorbellRecord(
        seq=int(unpacked[0]),
        nbytes=int(unpacked[1]),
        offset=int(unpacked[2]),
        metadata_len=int(unpacked[3]),
        ack=int(unpacked[4]),
        plan_id=int(unpacked[5]),
        dim0=int(unpacked[6]),
        dim1=int(unpacked[7]),
        layer_index=int(unpacked[8]),
        dtype_code=int(unpacked[9]),
        flags=int(unpacked[10]),
    )


def _doorbell_write(
    mm: mmap.mmap,
    record_offset: int,
    *,
    seq: int,
    nbytes: int,
    offset: int,
    metadata: dict[str, Any] | None,
    plan_id: int = 0,
    shape: tuple[int, int] = (0, 0),
    layer_index: int = -1,
    dtype_code: int = DTYPE_CODE_NONE,
    flags: int = 0,
) -> None:
    meta = (
        json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        if metadata
        else b""
    )
    if len(meta) > DOORBELL_RECORD_BYTES - DOORBELL_HEADER_BYTES:
        raise RuntimeError(
            "PAP local fast metadata is too large for the doorbell record"
        )
    start = int(record_offset)
    body_start = start + DOORBELL_HEADER_BYTES
    mm[body_start : body_start + len(meta)] = meta
    header = DOORBELL_RECORD_STRUCT.pack(
        0,
        nbytes,
        offset,
        len(meta),
        0,
        int(plan_id),
        int(shape[0]),
        int(shape[1]),
        int(layer_index),
        int(dtype_code),
        int(flags),
        0,
    )
    # ACK is an independent receiver-owned watermark.  Do not overwrite it:
    # descriptorless output receive may publish the ACK before this descriptor
    # is written, concurrently with the sender.
    ack_start = 4 * 8
    ack_end = ack_start + 8
    mm[start : start + ack_start] = header[:ack_start]
    mm[start + ack_end : start + DOORBELL_HEADER_BYTES] = header[ack_end:]
    mm[start : start + 8] = struct.pack("<Q", seq)


def _doorbell_read_header(
    mm: mmap.mmap,
    record_offset: int,
) -> tuple[int, int, int, int, int]:
    record = _doorbell_read_record(mm, record_offset)
    return (
        record.seq,
        record.nbytes,
        record.offset,
        record.metadata_len,
        record.ack,
    )


def _doorbell_ack(mm: mmap.mmap, record_offset: int, seq: int) -> None:
    ack_offset = int(record_offset) + 32
    mm[ack_offset : ack_offset + 8] = struct.pack("<Q", seq)


def _doorbell_read_metadata(
    mm: mmap.mmap,
    record_offset: int,
    metadata_len: int,
) -> dict[str, Any]:
    if metadata_len < 0 or metadata_len > DOORBELL_RECORD_BYTES - DOORBELL_HEADER_BYTES:
        raise RuntimeError("PAP local fast doorbell metadata length is invalid")
    start = int(record_offset) + DOORBELL_HEADER_BYTES
    raw = bytes(mm[start : start + metadata_len])
    data = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(data, dict):
        raise RuntimeError("PAP local fast doorbell metadata must be a dict")
    return data


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"unsupported PAP local fast tensor dtype: {name}")
    return dtype


def _payload_metadata(
    descriptor_metadata: dict[str, Any],
    tensor: torch.Tensor,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
    }
    if descriptor_metadata:
        metadata["descriptor"] = descriptor_metadata
    return metadata


@dataclass
class _WireMetadata:
    metadata: dict[str, Any] | None
    plan_id: int = 0
    shape: tuple[int, int] = (0, 0)
    layer_index: int = -1
    dtype_code: int = DTYPE_CODE_NONE
    flags: int = 0
