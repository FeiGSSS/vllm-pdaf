# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-model CUDA Graph ownership for PAP Projection."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.config import read_env_bool
from vllm.pap.model.projection_routing import _pap_offload_exec_step_groups
from vllm.pap.protocol import PAPOffloadExecBatchDescriptor
from vllm.pap.protocol.offload_exec import layer_index_and_template
from vllm.pap.transport.projection import (
    _pap_bind_offload_exec_nvshmem_peer,
    _pap_offload_exec_transport_for_attention_endpoint,
)

logger = init_logger(__name__)

_PROJECTION_ADAPTERS: dict[str, Any] = {}
_ROUTED_GRAPH_BUFFERS: dict[tuple[Any, ...], _RoutedGraphBuffers] = {}
_ROUTE_INDEX_LOCK = threading.Lock()
_TLS = threading.local()


def pap_projection_step_graph_enabled() -> bool:
    """Return whether this process owns the PAP Projection model graph."""
    return read_env_bool(os.environ, "PAP_PROJECTION_KV_UNAWARE")


def register_projection_step_graph_adapter(
    layer_name: str,
    adapter: Any,
) -> None:
    """Register one process-lifetime Projection adapter."""
    if pap_projection_step_graph_enabled():
        _PROJECTION_ADAPTERS[str(layer_name)] = adapter


@dataclass(frozen=True)
class PAPProjectionStepGraphRoute:
    endpoint: str
    req_indices: tuple[int, ...]
    transport: Any


@dataclass
class _RoutedGraphBuffers:
    host_indices: torch.Tensor
    host_counts: torch.Tensor
    copy_done_events: list[torch.Event | None]
    next_host_slot: int
    indices: torch.Tensor
    counts: torch.Tensor
    peer_ranks: torch.Tensor
    packed_qkv: torch.Tensor
    controller: Any


@dataclass
class PAPProjectionStepGraphContext:
    """Capture-time route state shared by all model layers."""

    layer_count: int
    routed: _RoutedGraphBuffers

    @staticmethod
    def layer_index(layer_name: str) -> int:
        layer_info = layer_index_and_template(layer_name)
        if layer_info is None:
            raise RuntimeError(f"PAP step graph layer name is invalid: {layer_name}")
        return int(layer_info[0])


def current_projection_step_graph_context() -> PAPProjectionStepGraphContext | None:
    """Return the context active during whole-model capture."""
    return getattr(_TLS, "projection_step_graph_context", None)


@contextmanager
def projection_step_graph_capture_context(
    context: PAPProjectionStepGraphContext,
):
    previous = current_projection_step_graph_context()
    _TLS.projection_step_graph_context = context
    try:
        yield
    finally:
        _TLS.projection_step_graph_context = previous


@dataclass(frozen=True)
class PAPProjectionStepPreparation:
    key: tuple[Any, ...]
    context: PAPProjectionStepGraphContext


def _update_routed_graph_buffers(
    *,
    routes: tuple[PAPProjectionStepGraphRoute, ...],
    batch_rows: int,
    qkv_width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _RoutedGraphBuffers:
    if not routes:
        raise RuntimeError("PAP Projection graph has no active routes")
    controller = routes[0].transport
    world = controller.world
    peer_rank_values = tuple(
        rank for rank in range(world.world_size) if rank != world.rank
    )
    key = (
        str(torch.device(device)),
        peer_rank_values,
        batch_rows,
        qkv_width,
        str(dtype),
    )
    with _ROUTE_INDEX_LOCK:
        buffers = _ROUTED_GRAPH_BUFFERS.get(key)
        if buffers is None:
            peer_count = len(peer_rank_values)
            host_indices = torch.empty(
                (2, peer_count, batch_rows),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            host_counts = torch.empty(
                (2, peer_count),
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            indices = torch.empty(
                (peer_count, batch_rows),
                dtype=torch.int64,
                device=device,
            )
            counts = torch.empty(
                peer_count,
                dtype=torch.int32,
                device=device,
            )
            peer_ranks = torch.tensor(
                peer_rank_values,
                dtype=torch.int32,
                device=device,
            )
            packed_qkv = torch.empty(
                (peer_count, batch_rows, qkv_width),
                dtype=dtype,
                device=device,
            )
            buffers = _RoutedGraphBuffers(
                host_indices=host_indices,
                host_counts=host_counts,
                copy_done_events=[None, None],
                next_host_slot=0,
                indices=indices,
                counts=counts,
                peer_ranks=peer_ranks,
                packed_qkv=packed_qkv,
                controller=controller,
            )
            _ROUTED_GRAPH_BUFFERS[key] = buffers
        host_slot = buffers.next_host_slot
        buffers.next_host_slot = (host_slot + 1) % len(buffers.copy_done_events)
        copy_done = buffers.copy_done_events[host_slot]
        if copy_done is not None:
            copy_done.synchronize()
        host_indices = buffers.host_indices[host_slot]
        host_counts = buffers.host_counts[host_slot]
        host_indices.zero_()
        host_counts.zero_()
        peer_slots = {rank: slot for slot, rank in enumerate(peer_rank_values)}
        routed_rows = 0
        for route in routes:
            peer_rank = int(route.transport.peer_rank)
            peer_slot = peer_slots[peer_rank]
            row_count = len(route.req_indices)
            host_counts[peer_slot] = row_count
            host_indices[peer_slot, :row_count].copy_(
                torch.tensor(route.req_indices, dtype=torch.int64)
            )
            routed_rows += row_count
        if routed_rows != batch_rows:
            raise RuntimeError("PAP Projection routed graph row count mismatch")
        buffers.indices.copy_(host_indices, non_blocking=True)
        buffers.counts.copy_(host_counts, non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(torch.cuda.current_stream(device))
        buffers.copy_done_events[host_slot] = copy_done
        return buffers


def prepare_projection_step_graph(
    additional_kwargs: dict[str, Any],
    dtype: torch.dtype,
) -> PAPProjectionStepPreparation | None:
    """Publish all Attention step plans before one model graph replay."""
    if not pap_projection_step_graph_enabled() or not additional_kwargs.get(
        "pap_enabled"
    ):
        return None
    if not _PROJECTION_ADAPTERS:
        raise RuntimeError("PAP Projection step graph has no model adapters")
    ordered_adapters = sorted(
        _PROJECTION_ADAPTERS.items(),
        key=lambda item: PAPProjectionStepGraphContext.layer_index(item[0]),
    )
    layer_count = len(ordered_adapters)
    first_layer_name, first_adapter = ordered_adapters[0]
    if PAPProjectionStepGraphContext.layer_index(first_layer_name) != 0:
        raise RuntimeError("PAP Projection graph is missing layer zero")
    if any(
        PAPProjectionStepGraphContext.layer_index(layer_name) != index
        for index, (layer_name, _adapter) in enumerate(ordered_adapters)
    ):
        raise RuntimeError("PAP Projection graph layer registration is incomplete")
    expected_layers = int(first_adapter.num_hidden_layers or 0)
    if expected_layers != layer_count:
        raise RuntimeError("PAP Projection graph model layer count mismatch")
    num_reqs = int(additional_kwargs.get("pap_num_reqs") or 0)
    step_groups = _pap_offload_exec_step_groups(
        additional_kwargs,
        num_reqs=num_reqs,
        scaling=float(first_adapter.scaling),
    )
    qkv_width = int(first_adapter._qkv_width)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    routes: list[PAPProjectionStepGraphRoute] = []
    for step_group in step_groups:
        transport = _pap_offload_exec_transport_for_attention_endpoint(
            step_group.attention_endpoint,
        )
        _pap_bind_offload_exec_nvshmem_peer(
            transport,
            step_group.attention_endpoint,
        )
        descriptor = PAPOffloadExecBatchDescriptor(
            layer_name=first_layer_name,
            items=(),
            batch_id_suffix=step_group.batch_id_suffix,
            metadata_template=step_group.metadata_template,
        )
        transport.send_step_prepare(
            descriptor,
            dtype=dtype,
            qkv_width=qkv_width,
            layer_count=layer_count,
        )
        routes.append(
            PAPProjectionStepGraphRoute(
                endpoint=step_group.attention_endpoint,
                req_indices=step_group.req_indices,
                transport=transport,
            )
        )
    route_tuple = tuple(routes)
    routes[0].transport.tracing.prepare_projection_pa_trace(layer_count)
    context = PAPProjectionStepGraphContext(
        layer_count=layer_count,
        routed=_update_routed_graph_buffers(
            routes=route_tuple,
            batch_rows=num_reqs,
            qkv_width=qkv_width,
            dtype=dtype,
            device=device,
        ),
    )
    key = (
        num_reqs,
        str(dtype),
        layer_count,
        qkv_width,
        routes[0].transport.world.world_size,
    )
    return PAPProjectionStepPreparation(key=key, context=context)


@dataclass
class _PAPProjectionGraphEntry:
    graph: torch.cuda.CUDAGraph
    stream: torch.Stream
    output: Any


class PAPProjectionStepGraphManager:
    """Own graph specializations for complete Projection model forwards."""

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], _PAPProjectionGraphEntry] = {}
        self._route_keys: set[tuple[Any, ...]] = set()
        self._address_keys: set[tuple[int, ...]] = set()

    def run(
        self,
        preparation: PAPProjectionStepPreparation,
        *,
        inputs: tuple[torch.Tensor, ...],
        forward: Callable[[], Any],
    ) -> Any:
        """Capture or replay one complete Projection decode step."""
        if not inputs:
            raise RuntimeError("PAP Projection graph has no CUDA inputs")
        input_signatures = tuple(
            (
                int(tensor.data_ptr()),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                str(tensor.device),
            )
            for tensor in inputs
        )
        key = (*preparation.key, input_signatures)
        entry = self._entries.get(key)
        current_stream = torch.cuda.current_stream(inputs[0].device)
        if entry is not None:
            entry.stream.wait_stream(current_stream)
            with torch.cuda.stream(entry.stream):
                entry.graph.replay()
            current_stream.wait_stream(entry.stream)
            return entry.output

        graph_stream = torch.cuda.Stream(device=inputs[0].device)
        graph_stream.wait_stream(current_stream)
        graph = torch.cuda.CUDAGraph()
        route_is_new = preparation.key not in self._route_keys
        addresses = tuple(int(tensor.data_ptr()) for tensor in inputs)
        addresses_are_new = addresses not in self._address_keys
        self._route_keys.add(preparation.key)
        self._address_keys.add(addresses)
        logger.info(
            "PAP Projection whole-step CUDA Graph capture begin "
            "route_new=%d addresses_new=%d route_keys=%d address_keys=%d "
            "batch_rows=%s",
            int(route_is_new),
            int(addresses_are_new),
            len(self._route_keys),
            len(self._address_keys),
            preparation.key[0],
        )
        with (
            torch.cuda.stream(graph_stream),
            projection_step_graph_capture_context(preparation.context),
            torch.cuda.graph(
                graph,
                stream=graph_stream,
                capture_error_mode="thread_local",
            ),
        ):
            output = forward()
        with torch.cuda.stream(graph_stream):
            graph.replay()
        current_stream.wait_stream(graph_stream)
        entry = _PAPProjectionGraphEntry(
            graph=graph,
            stream=graph_stream,
            output=output,
        )
        self._entries[key] = entry
        logger.info("PAP Projection whole-step CUDA Graph capture complete")
        return output

    def shutdown(self) -> None:
        """Synchronize and release all runner-owned Graph specializations."""
        for entry in self._entries.values():
            entry.stream.synchronize()
        self._entries.clear()
        self._route_keys.clear()
        self._address_keys.clear()


def shutdown_projection_step_graph() -> None:
    """Release process-wide layer registrations and routed scratch buffers."""
    exported_transports: set[int] = set()
    for buffers in _ROUTED_GRAPH_BUFFERS.values():
        transport = buffers.controller
        if id(transport) not in exported_transports:
            transport.tracing.export_projection_pa_trace()
            exported_transports.add(id(transport))
    with _ROUTE_INDEX_LOCK:
        _PROJECTION_ADAPTERS.clear()
        _ROUTED_GRAPH_BUFFERS.clear()


__all__ = [
    "PAPProjectionStepGraphContext",
    "PAPProjectionStepGraphManager",
    "current_projection_step_graph_context",
    "pap_projection_step_graph_enabled",
    "prepare_projection_step_graph",
    "register_projection_step_graph_adapter",
    "shutdown_projection_step_graph",
]
