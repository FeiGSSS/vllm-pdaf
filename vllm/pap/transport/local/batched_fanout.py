# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched same-host QKV fan-out over CUDA P2P DMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.pap.cuda_stream_memops import (
    make_stream_write_value32_batch,
    stream_write_value32_batch,
)
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    end_deferred_cuda_span,
)
from vllm.pap.transport.local.protocol import DIR_QKV


def local_qkv_batched_fanout_available() -> bool:
    """Return whether the CUDA runtime exposes batched pointer copies."""
    try:
        from cuda.bindings import runtime

        result = runtime.cudaRuntimeGetVersion()
        return (
            result[0] == runtime.cudaError_t.cudaSuccess
            and int(result[1]) >= 13000
            and hasattr(runtime, "cudaMemcpyBatchAsync")
        )
    except (ImportError, TypeError):
        return False


def _make_memcpy_attributes() -> object:
    from cuda.bindings import runtime

    attributes = runtime.cudaMemcpyAttributes()
    attributes.srcAccessOrder = (
        runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    )
    attributes.flags = (
        runtime.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
    )
    return attributes


def _submit_memcpy_batch(
    *,
    destinations: tuple[int, ...],
    sources: list[int],
    sizes: tuple[int, ...],
    attributes: object,
    stream: torch.cuda.Stream,
) -> None:
    from cuda.bindings import runtime

    result = runtime.cudaMemcpyBatchAsync(
        destinations,
        sources,
        sizes,
        len(destinations),
        [attributes],
        [0],
        1,
        runtime.cudaStream_t(stream.cuda_stream),
    )
    if result[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpyBatchAsync failed: {result[0]}")


@dataclass
class PAPLocalQKVBatchedFanoutPlan:
    """CPU-prepared routing metadata reused by every layer in one step."""

    transports: tuple[Any, ...]
    stream: torch.cuda.Stream
    destination_addresses: tuple[int, ...]
    source_byte_offsets: tuple[int, ...]
    byte_counts: tuple[int, ...]
    signal_batches: tuple[tuple[object, ...], ...]
    memcpy_attributes: object
    dtype: torch.dtype
    qkv_width: int
    num_layers: int
    num_source_rows: int
    _next_layer: int = 0

    def launch(self, qkv: torch.Tensor) -> None:
        """Fan one grouped QKV tensor out with one batched DMA submission."""
        layer_index = int(self._next_layer)
        if layer_index < 0 or layer_index >= self.num_layers:
            raise RuntimeError(
                f"PAP local batched fan-out layer {layer_index} is out of range"
            )
        if qkv.device != torch.device(self.stream.device):
            raise RuntimeError("PAP local batched fan-out source device changed")
        if qkv.dtype != self.dtype or qkv.ndim != 2 or not qkv.is_contiguous():
            raise RuntimeError(
                "PAP local batched fan-out requires the planned contiguous "
                "QKV tensor"
            )
        if int(qkv.shape[0]) < self.num_source_rows:
            raise RuntimeError("PAP local batched fan-out source has too few rows")
        if int(qkv.shape[1]) != self.qkv_width:
            raise RuntimeError("PAP local batched fan-out QKV width changed")

        current_stream = torch.cuda.current_stream(qkv.device)
        # The preceding QKV projection is the only producer dependency. The
        # next layer cannot produce QKV until Attention publishes this layer's
        # output, so the single receive buffer is safe to reuse in step order.
        self.stream.wait_stream(current_stream)
        source_base = int(qkv.data_ptr())
        sources = [
            source_base + source_offset
            for source_offset in self.source_byte_offsets
        ]
        copy_trace = begin_deferred_cuda_span(
            "qkv_batched_fanout_gpu_ms",
            self.stream,
        )
        try:
            _submit_memcpy_batch(
                destinations=self.destination_addresses,
                sources=sources,
                sizes=self.byte_counts,
                attributes=self.memcpy_attributes,
                stream=self.stream,
            )
            stream_write_value32_batch(
                self.signal_batches[layer_index],
                self.stream,
            )
        finally:
            end_deferred_cuda_span(copy_trace)
        for transport in self.transports:
            peer = transport._require_peer()
            peer.source_refs[DIR_QKV] = qkv
        self._next_layer += 1


def build_local_qkv_batched_fanout_plan(
    entries: list[tuple[Any, int, int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    qkv_width: int,
    num_layers: int,
    num_source_rows: int,
) -> PAPLocalQKVBatchedFanoutPlan:
    """Prepare peer pointers, row ranges, generations, and signal batches."""
    if not local_qkv_batched_fanout_available():
        raise RuntimeError("PAP local batched fan-out requires CUDA 13")
    if not entries:
        raise RuntimeError("PAP local batched fan-out requires at least one peer")
    if num_layers <= 0 or qkv_width <= 0 or num_source_rows <= 0:
        raise RuntimeError("PAP local batched fan-out plan shape is invalid")
    element_size = torch.empty((), dtype=dtype).element_size()
    row_bytes = int(qkv_width) * int(element_size)

    destinations: list[int] = []
    ready_addresses: list[int] = []
    source_offsets: list[int] = []
    byte_counts: list[int] = []
    base_sequences: list[int] = []
    transports: list[Any] = []

    for transport, row_start, row_count in entries:
        if row_start < 0 or row_count <= 0:
            raise RuntimeError("PAP local batched fan-out row range is invalid")
        if row_start + row_count > num_source_rows:
            raise RuntimeError(
                "PAP local batched fan-out row range exceeds batch"
            )
        peer_plan = transport.reserve_qkv_fanout(num_layers=num_layers)
        destinations.append(int(peer_plan["destination_address"]))
        ready_addresses.append(int(peer_plan["ready_address"]))
        source_offsets.append(int(row_start * row_bytes))
        byte_counts.append(int(row_count * row_bytes))
        base_sequences.append(int(peer_plan["base_sequence"]))
        transports.append(transport)

    ready_tuple = tuple(ready_addresses)
    signal_batches = tuple(
        make_stream_write_value32_batch(
            ready_tuple,
            tuple(base + layer for base in base_sequences),
        )
        for layer in range(num_layers)
    )
    return PAPLocalQKVBatchedFanoutPlan(
        transports=tuple(transports),
        stream=entries[0][0]._qkv_fanout_stream,
        destination_addresses=tuple(destinations),
        source_byte_offsets=tuple(source_offsets),
        byte_counts=tuple(byte_counts),
        signal_batches=signal_batches,
        memcpy_attributes=_make_memcpy_attributes(),
        dtype=dtype,
        qkv_width=int(qkv_width),
        num_layers=int(num_layers),
        num_source_rows=int(num_source_rows),
    )
