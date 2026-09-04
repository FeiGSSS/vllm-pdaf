# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection peer and NVSHMEM transport lifecycle for Attention."""

from __future__ import annotations

import hashlib
import time
from threading import Lock, Thread
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.pap.attention.compute import prepare_offload_exec_step
from vllm.pap.attention.execution import run_offload_exec_nvshmem_graph_loop
from vllm.pap.attention.kernels import (
    PAPAttentionStepTensorCache,
    PAPPagedDecodeWorkspaceCache,
)
from vllm.pap.attention.runtime import PAPAttentionRuntime
from vllm.pap.config import PAPAttentionKernelPolicy, PAPRuntimeConfig
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    end_deferred_cuda_span,
)
from vllm.pap.kv.metadata import PAPPagedBlockTableBuffer
from vllm.pap.transport.factory import build_offload_exec_transport

logger = init_logger(__name__)


class PAPAttentionPeerConflict(ValueError):
    """Raised when a peer reuses an existing identity inconsistently."""


_RECEIVER_JOIN_TIMEOUT_SECONDS = 5.0


class PAPAttentionPeerManager:
    """Own the Attention-side NVSHMEM transport and Graph receiver thread."""

    def __init__(
        self,
        *,
        runtime: PAPAttentionRuntime,
        config: PAPRuntimeConfig,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.initial_transport: Any | None = None
        self.transports: dict[str, Any] = {}
        self.source_ids: dict[str, str] = {}
        self.receiver_threads: dict[str, Thread] = {}
        self.local_rank = config.attention.local_rank
        self.actor_base = config.attention.actor_id
        self._lock = Lock()
        self._stopping = False
        self._stopped = False
        self.attention_kernel_selector: Any | None = None

    def initialize(self, *, enabled: bool) -> None:
        """Create the transport used by the single Projection peer."""
        if enabled:
            self.initial_transport = self._build_transport(actor_id=self.actor_base)

    def _build_transport(self, *, actor_id: str) -> Any:
        return build_offload_exec_transport(
            actor_id=actor_id,
            local_rank=self.local_rank,
        )

    def membership_stats(self) -> dict[str, Any]:
        """Return a stable peer snapshot for the Attention stats endpoint."""
        with self._lock:
            stats = {
                "attention_bound_source_ids": sorted(self.source_ids.values()),
            }
            if self.attention_kernel_selector is not None:
                stats.update(self.attention_kernel_selector.stats())
            return stats

    def health(self) -> dict[str, Any]:
        """Report runtime health together with the bound receiver state."""
        with self._lock:
            threads = tuple(self.receiver_threads.values())
            bound = bool(self.transports)
            stopping = self._stopping or self._stopped
        dead = sorted(thread.name for thread in threads if not thread.is_alive())
        health = self.runtime.health()
        health["receiver_threads"] = len(threads)
        if dead or (bound and not threads) or stopping:
            health["status"] = "error"
            health["receiver_state"] = "stopping" if stopping else "dead"
            if dead:
                health["dead_receiver_threads"] = dead
        else:
            health["receiver_state"] = "running" if threads else "unbound"
        return health

    def bind(self, *, peer_metadata: bytes, source_id: str | None) -> bytes:
        """Bind the Projection peer and start its Graph receiver exactly once."""
        peer_key = hashlib.sha1(peer_metadata).hexdigest()[:16]
        stable_source_id = str(source_id or peer_key)
        with self._lock:
            if self._stopping or self._stopped:
                raise PAPAttentionPeerConflict("PAP Attention peer manager is stopping")
            existing_source_id = self.source_ids.get(peer_key)
            if existing_source_id not in (None, stable_source_id):
                raise PAPAttentionPeerConflict(
                    "PAP NVSHMEM peer changed its stable source id"
                )
            if any(
                existing_key != peer_key and value == stable_source_id
                for existing_key, value in self.source_ids.items()
            ):
                raise PAPAttentionPeerConflict("PAP NVSHMEM source id is already bound")
            transport = self.transports.get(peer_key)
            if transport is None:
                if self.transports:
                    raise PAPAttentionPeerConflict(
                        "PAP NVSHMEM Attention accepts one Projection peer"
                    )
                transport = self.initial_transport or self._build_transport(
                    actor_id=self.actor_base
                )
                transport.bind_peer(peer_metadata)
                transport._pap_nvshmem_bound = True
                self._install_step_prepare_handler(transport)
                self.transports[peer_key] = transport
            self.source_ids[peer_key] = stable_source_id
            if peer_key not in self.receiver_threads:
                thread = Thread(
                    target=run_offload_exec_nvshmem_graph_loop,
                    kwargs={
                        "registry": self.runtime.registry,
                        "transport": transport,
                        "peer_id": stable_source_id,
                    },
                    daemon=True,
                    name=f"pap-offload-exec-nvshmem-graph-{peer_key}",
                )
                self.receiver_threads[peer_key] = thread
                thread.start()
            return transport.local_agent_metadata

    def _install_step_prepare_handler(self, transport: Any) -> None:
        stream = transport.step_prepare_stream()
        workspace_cache = PAPPagedDecodeWorkspaceCache(max_entries=64)
        step_tensor_cache = PAPAttentionStepTensorCache(max_entries=256)
        block_table_buffer = PAPPagedBlockTableBuffer()
        from vllm.pap.attention.pat import PAPPATOrTritonSelector

        if self.config.attention.kernel_policy is PAPAttentionKernelPolicy.AUTO:
            attention_kernel_selector, unavailable_reason = (
                PAPPATOrTritonSelector.create_if_available()
            )
            if unavailable_reason is not None:
                logger.warning_once(
                    "PAP PAT is unavailable; auto policy is using Triton: %s",
                    unavailable_reason,
                )
        elif self.config.attention.kernel_policy is PAPAttentionKernelPolicy.TRITON:
            attention_kernel_selector = None
        else:
            raise RuntimeError("PAP Attention kernel policy must be auto or triton")
        self.attention_kernel_selector = attention_kernel_selector

        def prepare_step(descriptor: Any, dtype: torch.dtype) -> Any:
            with torch.cuda.stream(stream):
                trace = begin_deferred_cuda_span(
                    "attention_step_prepare_gpu_ms",
                    stream,
                )
                try:
                    return prepare_offload_exec_step(
                        registry=self.runtime.registry,
                        descriptor=descriptor,
                        dtype=dtype,
                        workspace_cache=workspace_cache,
                        step_tensor_cache=step_tensor_cache,
                        block_table_buffer=block_table_buffer,
                        attention_kernel_selector=attention_kernel_selector,
                    )
                finally:
                    end_deferred_cuda_span(trace)

        transport.set_step_prepare_handler(prepare_step)

    def stop(self) -> None:
        """Quiesce the Graph receiver before releasing NVSHMEM resources."""
        with self._lock:
            if self._stopped:
                return
            if self._stopping:
                raise RuntimeError("PAP Attention peer manager is already stopping")
            self._stopping = True
            threads = tuple(self.receiver_threads.values())
            candidates = [self.initial_transport, *self.transports.values()]
            transports: list[Any] = []
            seen: set[int] = set()
            for transport in candidates:
                if transport is None or id(transport) in seen:
                    continue
                seen.add(id(transport))
                transports.append(transport)

        for transport in transports:
            transport.stop_receiving()
        deadline = time.monotonic() + _RECEIVER_JOIN_TIMEOUT_SECONDS
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise RuntimeError(
                "PAP Attention receiver threads did not stop: " + ", ".join(alive)
            )

        for transport in transports:
            export_trace = getattr(transport, "export_attention_kernel_trace", None)
            if export_trace is not None:
                export_trace()
        self.runtime.stop()
        close_error: BaseException | None = None
        for transport in transports:
            try:
                transport.close()
            except BaseException as exc:
                close_error = close_error or exc

        with self._lock:
            self.initial_transport = None
            self.transports.clear()
            self.source_ids.clear()
            self.receiver_threads.clear()
            self._stopped = True
        if close_error is not None:
            raise close_error
