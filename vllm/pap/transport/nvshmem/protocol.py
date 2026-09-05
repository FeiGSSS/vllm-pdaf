# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned CPU control-record codec for PAP's NVSHMEM transport."""

from __future__ import annotations

import struct

import torch

from vllm.pap.protocol import PAPOffloadExecBatchDescriptor
from vllm.pap.protocol.offload_exec import dtype_from_name, dtype_name
from vllm.pap.transport.nvshmem.runtime import PAPNVSHMEMError

METADATA_VERSION = 3
_CONTROL_MAGIC = b"PNSH"
_CONTROL_HEADER = struct.Struct("<4sBBHIHdIHHH")
_CONTROL_ROW = struct.Struct("<HI")


def encode_step_plan(
    descriptor: PAPOffloadExecBatchDescriptor,
    *,
    dtype: torch.dtype,
    qkv_width: int,
    layer_count: int,
) -> bytes:
    template = descriptor.metadata_template
    if template is None:
        request_ids = tuple(item.request_id for item in descriptor.items)
        steps = tuple(int(item.step) for item in descriptor.items)
        scales = tuple(float(item.scale) for item in descriptor.items)
    else:
        request_ids = tuple(str(value) for value in template["r"])
        steps = tuple(int(value) for value in template["s"])
        scales = tuple(float(value) for value in template["a"])
    if not request_ids or not (len(request_ids) == len(steps) == len(scales)):
        raise PAPNVSHMEMError("PAP NVSHMEM step rows are malformed")
    scale = scales[0]
    if any(value != scale for value in scales):
        raise PAPNVSHMEMError("PAP NVSHMEM step has mixed attention scales")
    if qkv_width <= 0 or qkv_width >= 1 << 32:
        raise PAPNVSHMEMError("PAP NVSHMEM QKV width is out of range")
    if layer_count <= 0 or layer_count >= 1 << 16:
        raise PAPNVSHMEMError("PAP NVSHMEM layer count is out of range")
    if len(request_ids) >= 1 << 16:
        raise PAPNVSHMEMError("PAP NVSHMEM row count is out of range")

    layer_name = descriptor.layer_name.encode("utf-8")
    encoded_dtype_name = dtype_name(dtype).encode("ascii")
    batch_suffix = (
        descriptor.batch_id_suffix
        or ",".join(
            f"{request_id}@{step}" for request_id, step in zip(request_ids, steps)
        )
    ).encode("utf-8")
    static_fields = (layer_name, encoded_dtype_name, batch_suffix)
    if any(len(value) >= 1 << 16 for value in static_fields):
        raise PAPNVSHMEMError("PAP NVSHMEM step string is too large")

    encoded_rows: list[tuple[bytes, int]] = []
    rows_bytes = 0
    for request_id, step in zip(request_ids, steps):
        encoded_id = request_id.encode("utf-8")
        if len(encoded_id) >= 1 << 16 or step < 0 or step >= 1 << 32:
            raise PAPNVSHMEMError("PAP NVSHMEM step row is out of range")
        encoded_rows.append((encoded_id, step))
        rows_bytes += _CONTROL_ROW.size + len(encoded_id)
    record_bytes = (
        _CONTROL_HEADER.size + sum(len(value) for value in static_fields) + rows_bytes
    )
    record = bytearray(record_bytes)
    _CONTROL_HEADER.pack_into(
        record,
        0,
        _CONTROL_MAGIC,
        METADATA_VERSION,
        0,
        layer_count,
        qkv_width,
        len(request_ids),
        scale,
        record_bytes,
        len(layer_name),
        len(encoded_dtype_name),
        len(batch_suffix),
    )
    cursor = _CONTROL_HEADER.size
    for value in static_fields:
        record[cursor : cursor + len(value)] = value
        cursor += len(value)
    for encoded_id, step in encoded_rows:
        _CONTROL_ROW.pack_into(record, cursor, len(encoded_id), step)
        cursor += _CONTROL_ROW.size
        record[cursor : cursor + len(encoded_id)] = encoded_id
        cursor += len(encoded_id)
    return bytes(record)


def decode_step_plan(
    control: torch.Tensor,
    *,
    capacity: int,
) -> tuple[PAPOffloadExecBatchDescriptor, torch.dtype, int, int]:
    host_view = memoryview(control.numpy())
    if capacity < _CONTROL_HEADER.size:
        raise PAPNVSHMEMError("PAP NVSHMEM control record is truncated")
    (
        magic,
        version,
        flags,
        layer_count,
        qkv_width,
        row_count,
        scale,
        record_bytes,
        layer_name_bytes,
        dtype_name_bytes,
        batch_suffix_bytes,
    ) = _CONTROL_HEADER.unpack_from(host_view, 0)
    if magic != _CONTROL_MAGIC or version != METADATA_VERSION or flags != 0:
        raise PAPNVSHMEMError("PAP NVSHMEM control header is incompatible")
    if record_bytes < _CONTROL_HEADER.size or record_bytes > capacity:
        raise PAPNVSHMEMError("PAP NVSHMEM control record length is invalid")

    cursor = _CONTROL_HEADER.size

    def read_string(size: int, encoding: str) -> str:
        nonlocal cursor
        end = cursor + size
        if end > record_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM control string is truncated")
        value = bytes(host_view[cursor:end]).decode(encoding)
        cursor = end
        return value

    layer_name = read_string(layer_name_bytes, "utf-8")
    dtype_name = read_string(dtype_name_bytes, "ascii")
    batch_suffix = read_string(batch_suffix_bytes, "utf-8")
    request_ids: list[str] = []
    steps: list[int] = []
    for _ in range(row_count):
        if cursor + _CONTROL_ROW.size > record_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM control row is truncated")
        request_id_bytes, step = _CONTROL_ROW.unpack_from(host_view, cursor)
        cursor += _CONTROL_ROW.size
        request_ids.append(read_string(request_id_bytes, "utf-8"))
        steps.append(step)
    if cursor != record_bytes or not request_ids:
        raise PAPNVSHMEMError("PAP NVSHMEM control record has trailing data")
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=(),
        batch_id_suffix=batch_suffix,
        metadata_template={
            "r": tuple(request_ids),
            "s": tuple(steps),
            "a": (float(scale),) * len(request_ids),
        },
    )
    return descriptor, dtype_from_name(dtype_name), qkv_width, layer_count
