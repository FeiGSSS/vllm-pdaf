# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-step CUDA Graph execution for PAP Attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.attention.kernels import run_paged_decode_attention
from vllm.pap.kv.layout import split_paged_kv_cache
from vllm.pap.kv.models import PAPAttentionStepContext
from vllm.pap.protocol.offload_exec import layer_index_and_template

logger = init_logger(__name__)


def _ordered_layers(context: PAPAttentionStepContext) -> tuple[str, ...]:
    def index(layer_name: str) -> int:
        layer_info = layer_index_and_template(layer_name)
        if layer_info is None:
            raise RuntimeError(
                f"PAP Attention graph layer name is invalid: {layer_name}"
            )
        return int(layer_info[0])

    return tuple(sorted(context.expected_layers, key=index))


@dataclass
class _PAPAttentionGraphEntry:
    graph: torch.cuda.CUDAGraph
    stream: torch.cuda.Stream
    bound_tensors: tuple[torch.Tensor, ...]


class PAPAttentionStepGraphExecutor:
    """Capture and replay all remote Attention layers with one CPU launch."""

    def __init__(self, registry: Any, transport: Any) -> None:
        self.registry = registry
        self.transport = transport
        self._entries: dict[tuple[Any, ...], _PAPAttentionGraphEntry] = {}

    def execute(
        self,
        *,
        descriptor: Any,
        qkv_batch: torch.Tensor,
        context: PAPAttentionStepContext,
    ) -> bool:
        """Replay one complete Attention step and commit its KV state."""
        if context is None:
            raise RuntimeError("PAP Attention graph has no prepared step context")
        layer_names = _ordered_layers(context)
        layer_count = len(layer_names)
        expected_layer_count = int(
            getattr(descriptor, "expected_layer_count", 0) or layer_count
        )
        if layer_count <= 0 or layer_count != expected_layer_count:
            raise RuntimeError("PAP Attention graph layer count mismatch")
        qkv_shape = tuple(int(value) for value in qkv_batch.shape)
        if qkv_shape != (
            len(context.request_ids),
            context.q_size + 2 * context.kv_size,
        ):
            raise RuntimeError("PAP Attention graph QKV shape mismatch")
        metadata = context.metadata
        workspace = context.paged_decode_workspace
        if metadata is None or workspace is None:
            raise RuntimeError("PAP Attention graph metadata was not prepared")

        key = self._entry_key(
            context=context,
            layer_names=layer_names,
            qkv_batch=qkv_batch,
        )
        entry = self._entries.get(key)
        if entry is None:
            entry = self._capture(
                context=context,
                layer_names=layer_names,
                qkv_batch=qkv_batch,
            )

        if context.prepare_event is not None:
            entry.stream.wait_event(context.prepare_event)
        with torch.cuda.stream(entry.stream):
            entry.graph.replay()
        entry.stream.synchronize()
        return self.transport.commit_received_step(
            lambda: self._commit_context(context, layer_names)
        )

    def _entry_key(
        self,
        *,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
        qkv_batch: torch.Tensor,
    ) -> tuple[Any, ...]:
        metadata = context.metadata
        workspace = context.paged_decode_workspace
        graph_slot_tensor = context.graph_slot_tensor
        assert (
            metadata is not None
            and workspace is not None
            and graph_slot_tensor is not None
        )
        kv_addresses = tuple(
            int(context.layer_states[layer_name][0].kv_cache.data_ptr())
            for layer_name in layer_names
        )
        return (
            tuple(qkv_batch.shape),
            str(qkv_batch.dtype),
            kv_addresses,
            int(qkv_batch.data_ptr()),
            int(metadata.block_table.data_ptr()),
            int(metadata.seq_lens.data_ptr()),
            int(graph_slot_tensor.data_ptr()),
            int(workspace.output.data_ptr()),
            int(workspace.partial.data_ptr()),
        )

    def _capture(
        self,
        *,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
        qkv_batch: torch.Tensor,
    ) -> _PAPAttentionGraphEntry:
        device = qkv_batch.device
        stream = torch.cuda.Stream(device=device)
        if context.prepare_event is not None:
            stream.wait_event(context.prepare_event)
        self._warm_compute(context, layer_names[0], qkv_batch, stream)
        graph = torch.cuda.CUDAGraph()
        logger.info(
            "PAP Attention whole-step CUDA Graph capture begin rows=%d layers=%d",
            qkv_batch.shape[0],
            len(layer_names),
        )
        # Prefill KV publication remains active on independent HTTP threads
        # while decode-step shapes are captured.  Restrict capture safety
        # checks to this executor thread so those unrelated CUDA submissions
        # cannot invalidate the decode graph.
        with (
            torch.cuda.stream(stream),
            torch.cuda.graph(
                graph,
                stream=stream,
                capture_error_mode="thread_local",
            ),
        ):
            self._capture_body(context, layer_names, qkv_batch, stream)
        entry = _PAPAttentionGraphEntry(
            graph=graph,
            stream=stream,
            bound_tensors=self._bound_tensors(
                context=context,
                layer_names=layer_names,
                qkv_batch=qkv_batch,
            ),
        )
        key = self._entry_key(
            context=context,
            layer_names=layer_names,
            qkv_batch=qkv_batch,
        )
        self._entries[key] = entry
        logger.info(
            "PAP Attention whole-step CUDA Graph capture complete "
            "rows=%d active_rows=%d layers=%d graphs=%d",
            qkv_batch.shape[0],
            len(context.active_indices),
            len(layer_names),
            len(self._entries),
        )
        return entry

    @staticmethod
    def _bound_tensors(
        *,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
        qkv_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Keep every address captured by the graph alive with the entry."""
        metadata = context.metadata
        workspace = context.paged_decode_workspace
        graph_slot_tensor = context.graph_slot_tensor
        assert (
            metadata is not None
            and workspace is not None
            and graph_slot_tensor is not None
        )
        tensors = [
            qkv_batch,
            metadata.block_table,
            metadata.seq_lens,
            metadata.cu_seqlens_q,
            graph_slot_tensor,
            workspace.output,
            workspace.partial,
            workspace.lse,
            workspace.k_scale,
            workspace.v_scale,
        ]
        tensors.extend(
            context.layer_states[layer_name][0].kv_cache for layer_name in layer_names
        )
        return tuple(tensors)

    def _warm_compute(
        self,
        context: PAPAttentionStepContext,
        layer_name: str,
        qkv_batch: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        with torch.cuda.stream(stream):
            self._compute_layer(context, layer_name, qkv_batch)
        stream.synchronize()

    def _capture_body(
        self,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
        qkv_batch: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        layer_count = len(layer_names)
        self.transport.graph_begin_step(
            layer_count=layer_count,
            stream=stream,
        )
        for layer_index, layer_name in enumerate(layer_names):
            self.transport.graph_wait_qkv(
                layer_index=layer_index,
                layer_count=layer_count,
                stream=stream,
            )
            output = self._compute_layer(context, layer_name, qkv_batch)
            self.transport.graph_send_output(
                output.reshape(qkv_batch.shape[0], -1),
                layer_index=layer_index,
                layer_count=layer_count,
                stream=stream,
            )

    @staticmethod
    def _compute_layer(
        context: PAPAttentionStepContext,
        layer_name: str,
        qkv_batch: torch.Tensor,
    ) -> torch.Tensor:
        query_width = context.q_size
        kv_width = context.kv_size
        query, key, value = qkv_batch.split(
            (query_width, kv_width, kv_width),
            dim=-1,
        )
        query = query.view(-1, context.num_heads, context.head_dim)
        key = key.view(-1, context.num_kv_heads, context.head_dim)
        value = value.view(-1, context.num_kv_heads, context.head_dim)
        graph_slot_tensor = context.graph_slot_tensor
        if graph_slot_tensor is None:
            raise RuntimeError("PAP Attention graph KV slots are missing")
        kv_cache = context.layer_states[layer_name][0].kv_cache
        key_cache, value_cache = split_paged_kv_cache(kv_cache, context.head_dim)
        workspace = context.paged_decode_workspace
        assert workspace is not None
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            graph_slot_tensor,
            "auto",
            workspace.k_scale,
            workspace.v_scale,
        )
        states = context.layer_states[layer_name]
        kv_cache = states[0].kv_cache
        key_cache, value_cache = split_paged_kv_cache(kv_cache, context.head_dim)
        metadata = context.metadata
        workspace = context.paged_decode_workspace
        assert metadata is not None and workspace is not None
        return run_paged_decode_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            workspace=workspace,
            scale=context.scale,
            block_size=int(states[0].block_size),
        )

    def _commit_context(
        self,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
    ) -> None:
        with self.registry._lock:
            for layer_name in layer_names:
                states = context.layer_states[layer_name]
                for index in context.active_indices:
                    state = states[index]
                    expected = context.prior_seq_lens[index]
                    if int(state.seq_len) != int(expected):
                        raise RuntimeError(
                            "PAP Attention graph observed KV sequence drift"
                        )
                    state.seq_len = context.result_seq_lens[index]
        for layer_name in layer_names:
            self.registry.complete_attention_step_layer(
                context=context,
                layer_name=layer_name,
            )


__all__ = ["PAPAttentionStepGraphExecutor"]
