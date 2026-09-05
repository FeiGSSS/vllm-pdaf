# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-side updates to Prefill-owned unified KV state."""

from __future__ import annotations

import logging
import time
from _thread import LockType
from collections.abc import Sequence
from threading import Condition

from vllm.pap.deferred_cuda_trace import (
    deferred_cuda_trace_enabled,
)
from vllm.pap.kv.models import (
    PAPAttentionSession,
    PAPPrefillLayerReadiness,
    PAPUnifiedPagedKVState,
    PAPUnifiedSlotActivation,
)
from vllm.pap.kv.models import (
    allocate_unified_slot_topology_id as _allocate_unified_slot_topology_id,
)

logger = logging.getLogger("pap_attention")

_DEFERRED_CUDA_TRACE_ENABLED = deferred_cuda_trace_enabled()


class _PAPDecodeStateMixin:
    """Own unified-KV slot topology and read readiness."""

    _lock: LockType
    _prefill_condition: Condition
    _prefill_wait_timeout_s: float
    _sessions: dict[str, PAPAttentionSession]
    _prefill_readiness: dict[str, dict[str, PAPPrefillLayerReadiness]]
    _session_manifest_prefix_lens: dict[str, int]
    _session_manifest_claimed: set[str]
    _session_epochs: dict[str, int]
    _unified_paged_kv: dict[str, dict[str, PAPUnifiedPagedKVState]]
    _unified_slot_activations: dict[str, PAPUnifiedSlotActivation]
    _decode_slot_topology_mismatches: int

    def _record_unified_slot_topology_locked(
        self,
        *,
        session_request_id: str,
        layer_name: str,
        state: PAPUnifiedPagedKVState,
    ) -> None:
        topology = (
            state.block_ids,
            int(state.block_size),
            str(state.kv_cache.device),
        )
        prefix_len = int(state.prefix_len)
        activation = self._unified_slot_activations.get(session_request_id)
        if activation is None:
            topology_id = _allocate_unified_slot_topology_id()
            activation = PAPUnifiedSlotActivation(
                prefix_len=prefix_len,
                generation=1,
                canonical_topology_id=topology_id,
                topology_ids={topology: topology_id},
                layer_observations={layer_name: topology_id},
            )
            self._unified_slot_activations[session_request_id] = activation
            state.slot_generation = activation.generation
            state.slot_topology_id = topology_id
            return

        if prefix_len < activation.prefix_len:
            raise RuntimeError(
                "PAP unified KV stale slot activation "
                f"request_id={session_request_id} layer={layer_name} "
                f"prefix_len={prefix_len} current={activation.prefix_len}"
            )

        if prefix_len > activation.prefix_len:
            expected_layers = activation.expected_layers
            if expected_layers is None:
                expected_layers = frozenset(activation.layer_observations)
            elif not activation.complete:
                raise RuntimeError(
                    "PAP unified KV slot activation advanced before all layers "
                    f"request_id={session_request_id} "
                    f"prefix_len={activation.prefix_len} next={prefix_len}"
                )
            if expected_layers and layer_name not in expected_layers:
                raise RuntimeError(
                    "PAP unified KV slot activation contains unexpected layer "
                    f"request_id={session_request_id} layer={layer_name}"
                )
            topology_id = _allocate_unified_slot_topology_id()
            activation = PAPUnifiedSlotActivation(
                prefix_len=prefix_len,
                generation=activation.generation + 1,
                canonical_topology_id=topology_id,
                topology_ids={topology: topology_id},
                layer_observations={layer_name: topology_id},
                expected_layers=expected_layers,
                complete=not expected_layers or expected_layers == {layer_name},
            )
            self._unified_slot_activations[session_request_id] = activation
            state.slot_generation = activation.generation
            state.slot_topology_id = topology_id
            return

        if activation.expected_layers and layer_name not in activation.expected_layers:
            raise RuntimeError(
                "PAP unified KV slot activation contains unexpected layer "
                f"request_id={session_request_id} layer={layer_name}"
            )
        observed_topology_id = activation.topology_ids.get(topology)
        if observed_topology_id is None:
            observed_topology_id = _allocate_unified_slot_topology_id()
            activation.topology_ids[topology] = observed_topology_id
        previous_topology_id = activation.layer_observations.get(layer_name)
        has_conflict = (
            observed_topology_id != activation.canonical_topology_id
            or previous_topology_id not in (None, observed_topology_id)
        )
        if has_conflict and not activation.conflict_latched:
            self._decode_slot_topology_mismatches += 1
            logger.warning(
                "PAP unified KV slot topology conflicts within one activation; "
                "disabling cross-layer slot-plan cache request_id=%s",
                session_request_id,
            )
        activation.conflict_latched = activation.conflict_latched or has_conflict
        activation.layer_observations[layer_name] = observed_topology_id
        if activation.expected_layers:
            activation.complete = activation.expected_layers.issubset(
                activation.layer_observations
            )
        state.slot_generation = activation.generation
        state.slot_topology_id = observed_topology_id

    def slot_topology_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "slot_topology_mismatches": (self._decode_slot_topology_mismatches),
            }

    def get_unified_paged_states(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
    ) -> list[PAPUnifiedPagedKVState] | None:
        """Return per-row unified states if every row has unified state."""
        with self._lock:
            deadline = time.monotonic() + self._prefill_wait_timeout_s
            while True:
                states: list[PAPUnifiedPagedKVState] = []
                pending_request_id = ""
                pending_prefix_len = 0
                pending_state_len = -1
                for session_request_id in session_request_ids:
                    session = self._sessions.get(session_request_id)
                    if session is None:
                        raise KeyError(session_request_id)
                    readiness = self._prefill_readiness.get(
                        session_request_id,
                        {},
                    ).get(layer_name)
                    if readiness is not None and readiness.failed:
                        raise RuntimeError(
                            "prefill KV import failed before unified decode "
                            f"attention: {readiness.error}"
                        )
                    if session_request_id in self._session_manifest_prefix_lens and (
                        readiness is None or not readiness.ready
                    ):
                        raise RuntimeError(
                            "prefill KV entered unified decode before Gateway "
                            "readiness admission "
                            f"request_id={session_request_id} layer={layer_name}"
                        )
                    state = self._unified_paged_kv.get(
                        session_request_id,
                        {},
                    ).get(layer_name)
                    required_prefix_len = int(session.prefix_len or 0)
                    if state is None or int(state.seq_len) < required_prefix_len:
                        pending_request_id = session_request_id
                        pending_prefix_len = required_prefix_len
                        pending_state_len = -1 if state is None else int(state.seq_len)
                        break
                    states.append(state)
                if not pending_request_id:
                    for session_request_id in session_request_ids:
                        if session_request_id not in self._session_manifest_prefix_lens:
                            continue
                        self._session_manifest_claimed.add(session_request_id)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "prefill KV must reach the registered prefix before "
                        "unified decode attention "
                        f"request_id={pending_request_id} layer={layer_name} "
                        f"state_seq_len={pending_state_len} "
                        f"required_prefix_len={pending_prefix_len}"
                    )
                self._prefill_condition.wait(timeout=remaining)
        return states
