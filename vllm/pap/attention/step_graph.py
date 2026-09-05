# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-step CUDA Graph execution for PAP Attention."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.attention.dispatch import run_pap_decode_attention
from vllm.pap.config import read_env_bool, read_env_int
from vllm.pap.kv.layout import split_paged_kv_cache
from vllm.pap.kv.models import PAPAttentionStepContext
from vllm.pap.protocol.offload_exec import layer_index_and_template

logger = init_logger(__name__)


def _device_launch_probe_enabled() -> bool:
    return read_env_bool(os.environ, "PAP_ATTENTION_DEVICE_GRAPH_PROBE")


def _device_graph_launch_enabled() -> bool:
    return read_env_bool(os.environ, "PAP_ATTENTION_DEVICE_GRAPH_LAUNCH")


def _resident_dispatch_enabled() -> bool:
    return read_env_bool(os.environ, "PAP_ATTENTION_GPU_RESIDENT_DISPATCH")


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
    attention_plan: Any | None
    device_graph_handle: int | None = None


class PAPAttentionStepGraphExecutor:
    """Capture and replay all remote Attention layers with one CPU launch."""

    def __init__(
        self,
        registry: Any,
        transport: Any,
        *,
        max_entries: int = 32,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("PAP Attention Graph cache must be non-empty")
        self.registry = registry
        self.transport = transport
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[Any, ...], _PAPAttentionGraphEntry] = (
            OrderedDict()
        )
        self._resident_dispatch = _resident_dispatch_enabled()
        self._resident_dispatch_handle: int | None = None
        self._resident_dispatch_stream: torch.cuda.Stream | None = None
        self._resident_generation = 0

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
        if self._resident_dispatch and self._resident_dispatch_handle is None:
            self._create_resident_dispatcher(qkv_batch.device)

        trace_host = getattr(context, "_pap_trace_control_wait_ns", None) is not None
        graph_lookup_started_ns = perf_counter_ns() if trace_host else 0
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
        else:
            self._entries.move_to_end(key)
        if trace_host:
            context._pap_trace_graph_lookup_ns = (
                perf_counter_ns() - graph_lookup_started_ns
            )

        replay_started_ns = perf_counter_ns() if trace_host else 0
        if self._resident_dispatch:
            if context.prepare_event is not None and not context.prepare_event.query():
                context.prepare_event.synchronize()
            dispatcher_handle = self._resident_dispatch_handle
            executable_handle = entry.device_graph_handle
            if dispatcher_handle is None or executable_handle is None:
                raise RuntimeError("PAP resident Attention graph is incomplete")
            self._resident_generation += 1
            self.transport.run_resident_device_graph(
                dispatcher_handle=dispatcher_handle,
                executable_handle=executable_handle,
                generation=self._resident_generation,
            )
        else:
            if context.prepare_event is not None:
                entry.stream.wait_event(context.prepare_event)
            with torch.cuda.stream(entry.stream):
                self.transport.graph_attention_replay_trace_start(
                    layer_count=layer_count,
                    stream=entry.stream,
                )
                if entry.device_graph_handle is None:
                    entry.graph.replay()
                else:
                    self.transport.launch_device_graph(
                        executable_handle=entry.device_graph_handle,
                        stream=entry.stream,
                    )
            entry.stream.synchronize()
        if trace_host:
            context._pap_trace_graph_replay_submit_ns = (
                perf_counter_ns() - replay_started_ns
            )
        return self.transport.commit_received_step(
            lambda: self._commit_context(context, layer_names)
        )

    def _create_resident_dispatcher(self, device: torch.device) -> None:
        window_size = read_env_int(
            os.environ, "PAP_ATTENTION_DISPATCH_WINDOW_STEPS", 1, minimum=1
        )
        if window_size != 1:
            raise RuntimeError(
                "PAP Attention dynamic dispatch currently requires a one-step window"
            )
        stream = torch.cuda.Stream(device=device)
        self._resident_dispatch_handle = (
            self.transport.create_resident_graph_dispatcher(
                stream=stream,
                window_size=window_size,
            )
        )
        self._resident_dispatch_stream = stream
        logger.info(
            "PAP Attention GPU-resident dispatcher started window_steps=%d",
            window_size,
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
        attention_plan = context.attention_kernel_plan
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
            attention_plan.graph_key if attention_plan is not None else None,
        )

    def _capture(
        self,
        *,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
        qkv_batch: torch.Tensor,
    ) -> _PAPAttentionGraphEntry:
        device = qkv_batch.device
        self.transport.tracing.prepare_attention_kernel_trace(len(layer_names))
        stream = torch.cuda.Stream(device=device)
        if context.prepare_event is not None:
            stream.wait_event(context.prepare_event)
        self._warm_compute(
            context,
            layer_names[0],
            qkv_batch,
            stream,
            layer_count=len(layer_names),
        )
        device_graph_launch = _device_graph_launch_enabled() or self._resident_dispatch
        probe_device_launch = _device_launch_probe_enabled()
        graph = torch.cuda.CUDAGraph(
            keep_graph=device_graph_launch or probe_device_launch
        )
        attention_plan = context.attention_kernel_plan
        logger.info(
            "PAP Attention whole-step CUDA Graph capture begin "
            "rows=%d layers=%d attention_backend=%s reused_kv_tokens=%d",
            qkv_batch.shape[0],
            len(layer_names),
            (attention_plan.backend_name if attention_plan is not None else "triton"),
            (attention_plan.reused_kv_tokens if attention_plan is not None else 0),
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
            captured_outputs = self._capture_body(
                context,
                layer_names,
                qkv_batch,
                stream,
            )
        device_graph_handle = None
        if device_graph_launch:
            device_graph_handle = self.transport.create_device_graph_launch(
                graph_handle=int(graph.raw_cuda_graph()),
                stream=stream,
            )
            logger.info("PAP Attention CUDA Graph uses device launch")
        elif probe_device_launch:
            self.transport.probe_device_graph_launch(
                graph_handle=int(graph.raw_cuda_graph()),
                stream=stream,
            )
            logger.info("PAP Attention CUDA Graph supports device launch")
        entry = _PAPAttentionGraphEntry(
            graph=graph,
            stream=stream,
            bound_tensors=(
                *self._bound_tensors(
                    context=context,
                    layer_names=layer_names,
                    qkv_batch=qkv_batch,
                ),
                *captured_outputs,
            ),
            attention_plan=attention_plan,
            device_graph_handle=device_graph_handle,
        )
        key = self._entry_key(
            context=context,
            layer_names=layer_names,
            qkv_batch=qkv_batch,
        )
        self._store_entry(key, entry)
        logger.info(
            "PAP Attention whole-step CUDA Graph capture complete "
            "rows=%d active_rows=%d layers=%d graphs=%d",
            qkv_batch.shape[0],
            len(context.active_indices),
            len(layer_names),
            len(self._entries),
        )
        return entry

    def _store_entry(
        self,
        key: tuple[Any, ...],
        entry: _PAPAttentionGraphEntry,
    ) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            evicted.stream.synchronize()
            if evicted.device_graph_handle is not None:
                self.transport.destroy_device_graph_launch(evicted.device_graph_handle)

    def shutdown(self) -> None:
        """Synchronize and release all captured Graph specializations."""
        if self._resident_dispatch_handle is not None:
            self.transport.destroy_resident_graph_dispatcher(
                self._resident_dispatch_handle
            )
            self._resident_dispatch_handle = None
        for entry in self._entries.values():
            entry.stream.synchronize()
            if entry.device_graph_handle is not None:
                self.transport.destroy_device_graph_launch(entry.device_graph_handle)
        self._entries.clear()

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
        attention_plan = context.attention_kernel_plan
        if attention_plan is not None:
            tensors.extend(attention_plan.bound_tensors)
        return tuple(tensors)

    def _warm_compute(
        self,
        context: PAPAttentionStepContext,
        layer_name: str,
        qkv_batch: torch.Tensor,
        stream: torch.cuda.Stream,
        *,
        layer_count: int,
    ) -> None:
        with torch.cuda.stream(stream):
            self._compute_layer(
                context,
                layer_name,
                qkv_batch,
                layer_index=0,
                layer_count=layer_count,
                trace_kernel=False,
            )
        stream.synchronize()

    def _capture_body(
        self,
        context: PAPAttentionStepContext,
        layer_names: tuple[str, ...],
        qkv_batch: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, ...]:
        layer_count = len(layer_names)
        outputs = []
        if self._resident_dispatch:
            self.transport.graph_attention_replay_trace_start(
                layer_count=layer_count,
                stream=stream,
            )
        self.transport.graph_begin_step(
            layer_count=layer_count,
            stream=stream,
        )
        self.transport.graph_attention_step_trace_start(
            layer_count=layer_count,
            stream=stream,
        )
        for layer_index, layer_name in enumerate(layer_names):
            self.transport.graph_wait_qkv(
                layer_index=layer_index,
                layer_count=layer_count,
                stream=stream,
            )
            output = self._compute_layer(
                context,
                layer_name,
                qkv_batch,
                layer_index=layer_index,
                layer_count=layer_count,
                trace_kernel=True,
            )
            outputs.append(output)
            self.transport.graph_send_output(
                output.reshape(qkv_batch.shape[0], -1),
                layer_index=layer_index,
                layer_count=layer_count,
                stream=stream,
            )
        return tuple(outputs)

    def _compute_layer(
        self,
        context: PAPAttentionStepContext,
        layer_name: str,
        qkv_batch: torch.Tensor,
        *,
        layer_index: int,
        layer_count: int,
        trace_kernel: bool,
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
        stream = torch.cuda.current_stream(qkv_batch.device)
        if trace_kernel:
            self.transport.graph_attention_kernel_trace_start(
                layer_index=layer_index,
                layer_count=layer_count,
                stream=stream,
            )
        output = run_pap_decode_attention(
            attention_plan=context.attention_kernel_plan,
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            workspace=workspace,
            scale=context.scale,
            block_size=int(states[0].block_size),
        )
        if trace_kernel:
            self.transport.graph_attention_kernel_trace_end(
                layer_index=layer_index,
                layer_count=layer_count,
                stream=stream,
            )
        return output

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
