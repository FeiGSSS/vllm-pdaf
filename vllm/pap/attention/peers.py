# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection peer membership and transport lifecycle for Attention."""

from __future__ import annotations

import hashlib
from threading import Lock, Thread
from typing import Any

import torch

from vllm.pap.attention.compute import prepare_offload_exec_step
from vllm.pap.attention.execution import (
    run_offload_exec_mailbox_loop,
    run_offload_exec_mailbox_receiver_loop,
)
from vllm.pap.attention.runtime import PAPAttentionRuntime
from vllm.pap.config import PAPAttentionDispatchMode, PAPRuntimeConfig
from vllm.pap.transport.factory import build_offload_exec_transport


class PAPAttentionPeerConflict(ValueError):
    """Raised when a peer reuses an existing identity inconsistently."""


class PAPAttentionPeerManager:
    """Own Attention transports, Projection membership, and receiver threads."""

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
        self.active_source_ids: set[str] = set()
        self.membership_generations: dict[str, int] = {}
        self.membership_updates = 0
        self.membership_stale_updates = 0
        self.mailbox_loop_peers: set[str] = set()
        self.local_rank = config.attention.local_rank
        self.actor_base = config.attention.actor_id
        self._lock = Lock()

    @property
    def dispatch_mode(self) -> str:
        return self.runtime.dispatch_mode

    @property
    def dispatcher(self) -> Any | None:
        return self.runtime.dispatcher

    def initialize(self, *, enabled: bool) -> None:
        """Create the first transport used by the first bound Projection peer."""
        if not enabled:
            return
        self.initial_transport = self._build_transport(actor_id=self.actor_base)

    def _build_transport(self, *, actor_id: str) -> Any:
        return build_offload_exec_transport(
            transport=self.config.offload_exec_transport,
            actor_id=actor_id,
            local_rank=self.local_rank,
        )

    def _sync_dispatcher_membership_locked(self) -> None:
        if self.dispatch_mode != PAPAttentionDispatchMode.CENTRAL_COMBINE.value:
            return
        if self.runtime.active_peer_tracking:
            source_ids = set(self.active_source_ids)
        else:
            source_ids = set(self.source_ids.values())
        self.runtime.sync_dispatcher_membership(source_ids)

    def membership_stats(self) -> dict[str, Any]:
        """Return a stable snapshot for the Attention stats endpoint."""
        with self._lock:
            return {
                "attention_active_source_ids": sorted(self.active_source_ids),
                "attention_membership_generations": dict(
                    sorted(self.membership_generations.items())
                ),
                "attention_membership_updates": self.membership_updates,
                "attention_membership_stale_updates": (
                    self.membership_stale_updates
                ),
            }

    def update_activity(
        self,
        *,
        source_id: str,
        generation: int,
        active: bool,
    ) -> dict[str, Any]:
        """Apply one monotonic Projection membership update."""
        with self._lock:
            previous_generation = self.membership_generations.get(source_id)
            previous_active = source_id in self.active_source_ids
            if previous_generation is not None and generation < previous_generation:
                self.membership_stale_updates += 1
                return {
                    "source_id": source_id,
                    "active": previous_active,
                    "membership_generation": previous_generation,
                    "applied": False,
                    "stale": True,
                }
            if previous_generation == generation:
                if previous_active != active:
                    raise PAPAttentionPeerConflict(
                        "PAP mailbox membership generation changed activity"
                    )
                return {
                    "source_id": source_id,
                    "active": active,
                    "membership_generation": generation,
                    "applied": False,
                    "stale": False,
                }
            self.membership_generations[source_id] = generation
            if active:
                self.active_source_ids.add(source_id)
            else:
                self.active_source_ids.discard(source_id)
            self.membership_updates += 1
            self._sync_dispatcher_membership_locked()
        return {
            "source_id": source_id,
            "active": active,
            "membership_generation": generation,
            "applied": True,
            "stale": False,
        }

    def bind(self, *, peer_metadata: bytes, source_id: str | None) -> bytes:
        """Bind one Projection peer and start its receiver exactly once."""
        peer_key = hashlib.sha1(peer_metadata).hexdigest()[:16]
        stable_source_id = str(source_id or peer_key)
        with self._lock:
            existing_source_id = self.source_ids.get(peer_key)
            if (
                existing_source_id is not None
                and existing_source_id != stable_source_id
            ):
                raise PAPAttentionPeerConflict(
                    "PAP mailbox peer changed its stable source id"
                )
            if any(
                existing_peer_key != peer_key
                and bound_source_id == stable_source_id
                for existing_peer_key, bound_source_id in self.source_ids.items()
            ):
                raise PAPAttentionPeerConflict(
                    "PAP mailbox source id is already bound"
                )
            transport = self.transports.get(peer_key)
            if transport is None:
                if not self.transports and self.initial_transport is not None:
                    transport = self.initial_transport
                else:
                    transport = self._build_transport(
                        actor_id=f"{self.actor_base}-{peer_key}"
                    )
                if not hasattr(transport, "local_agent_metadata"):
                    raise PAPAttentionPeerConflict(
                        "PAP OFFLOAD_EXEC mailbox transport is not initialized"
                    )
                transport.bind_peer(peer_metadata)
                transport._pap_mailbox_bound = True
                set_step_prepare_handler = getattr(
                    transport,
                    "set_step_prepare_handler",
                    None,
                )
                step_prepare_stream = getattr(
                    transport,
                    "step_prepare_stream",
                    None,
                )
                if callable(set_step_prepare_handler) and callable(
                    step_prepare_stream
                ):
                    stream = step_prepare_stream()

                    def prepare_step(
                        descriptor: Any,
                        dtype: torch.dtype,
                        *,
                        _stream: torch.cuda.Stream = stream,
                    ) -> None:
                        with torch.cuda.stream(_stream):
                            prepare_offload_exec_step(
                                registry=self.runtime.registry,
                                descriptor=descriptor,
                                dtype=dtype,
                            )

                    set_step_prepare_handler(prepare_step)
                self.transports[peer_key] = transport
            self.source_ids[peer_key] = stable_source_id
            self._sync_dispatcher_membership_locked()
            if peer_key not in self.mailbox_loop_peers:
                dispatcher = self.dispatcher
                if self.dispatch_mode == (
                    PAPAttentionDispatchMode.CENTRAL_COMBINE.value
                ):
                    assert dispatcher is not None
                    dispatcher.start()
                    target = run_offload_exec_mailbox_receiver_loop
                    kwargs = {
                        "registry": self.runtime.registry,
                        "transport": transport,
                        "dispatcher": dispatcher,
                        "peer_id": stable_source_id,
                    }
                    thread_kind = "receiver"
                else:
                    target = run_offload_exec_mailbox_loop
                    kwargs = {
                        "registry": self.runtime.registry,
                        "transport": transport,
                        "peer_id": peer_key,
                    }
                    thread_kind = "loop"
                Thread(
                    target=target,
                    kwargs=kwargs,
                    daemon=True,
                    name=f"pap-offload-exec-mailbox-{thread_kind}-{peer_key}",
                ).start()
                self.mailbox_loop_peers.add(peer_key)
            return transport.local_agent_metadata

    def stop(self) -> None:
        """Stop the runtime-owned dispatcher."""
        self.runtime.stop()
