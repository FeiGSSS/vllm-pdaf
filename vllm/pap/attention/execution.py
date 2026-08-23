# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVSHMEM whole-step Graph execution loop for PAP Attention."""

from __future__ import annotations

from typing import Any

from vllm.pap.attention.step_graph import PAPAttentionStepGraphExecutor
from vllm.pap.kv.registry import PAPAttentionRegistry
from vllm.pap.protocol import PAPOffloadExecTransportClosed


def run_offload_exec_nvshmem_graph_loop(
    *,
    registry: PAPAttentionRegistry,
    transport: Any,
    peer_id: str | None = None,
) -> None:
    """Receive step plans and replay complete Attention CUDA Graphs."""
    peer_id = peer_id or str(getattr(transport, "actor_id", type(transport).__name__))
    graph_executor = PAPAttentionStepGraphExecutor(registry, transport)
    while True:
        try:
            descriptor, qkv_batch, step_context = transport.recv_graph_step_plan()
        except PAPOffloadExecTransportClosed:
            return
        registry.record_offload_exec_peer_batch(
            peer_id=peer_id,
            rows=descriptor.item_count,
        )
        graph_executor.execute(
            descriptor=descriptor,
            qkv_batch=qkv_batch,
            context=step_context,
        )
        for layer_name in step_context.expected_layers:
            registry.record_offload_exec_compute(
                layer_name=layer_name,
                rows=descriptor.item_count,
                source_batches=1,
            )


__all__ = ["run_offload_exec_nvshmem_graph_loop"]
