# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-side updates to Prefill-owned unified KV state."""

from __future__ import annotations

import logging
import time
from _thread import LockType
from collections import OrderedDict
from collections.abc import Sequence
from threading import Condition
from typing import Any, Protocol, cast

import torch

from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    deferred_cuda_trace_enabled,
    end_deferred_cuda_span,
)
from vllm.pap.kv.metadata import _coerce_block_id
from vllm.pap.kv.models import (
    PAPAttentionSession,
    PAPAttentionStepContext,
    PAPPrefillLayerReadiness,
    PAPUnifiedPagedKVState,
    PAPUnifiedSlotActivation,
)
from vllm.pap.kv.models import (
    allocate_unified_slot_topology_id as _allocate_unified_slot_topology_id,
)

logger = logging.getLogger("pap_attention")

_DEFERRED_CUDA_TRACE_ENABLED = deferred_cuda_trace_enabled()


class _PAPDecodeStateOwner(Protocol):
    def _decode_slot_plan_cache_limit(self) -> int: ...

    def record_attention_step_slot_plan_build(self) -> None: ...


class _PAPDecodeStateMixin:
    """Own unified-KV slot planning, append, and read synchronization."""

    _lock: LockType
    _decode_append_lock: LockType
    _prefill_condition: Condition
    _prefill_wait_timeout_s: float
    _sessions: dict[str, PAPAttentionSession]
    _prefill_readiness: dict[str, dict[str, PAPPrefillLayerReadiness]]
    _session_manifest_prefix_lens: dict[str, int]
    _session_manifest_claimed: set[str]
    _session_epochs: dict[str, int]
    _unified_paged_kv: dict[str, dict[str, PAPUnifiedPagedKVState]]
    _unified_slot_activations: dict[str, PAPUnifiedSlotActivation]
    _decode_slot_plan_cache: OrderedDict[tuple[Any, ...], torch.Tensor]
    _reshape_cache_scales: dict[str, tuple[torch.Tensor, torch.Tensor]]
    _decode_append_fast_path_hits: int
    _decode_append_fallbacks: int
    _decode_slot_plan_cache_hits: int
    _decode_slot_plan_cache_misses: int
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
                canonical_topology=topology,
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
                canonical_topology=topology,
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

    def _decode_slot_plan_key_locked(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
        decode_seq_lens: Sequence[int],
        device: torch.device,
    ) -> tuple[Any, ...] | None:
        session_generations: list[tuple[str, int, int, int, int]] = []
        for session_request_id in session_request_ids:
            activation = self._unified_slot_activations.get(session_request_id)
            if (
                activation is None
                or activation.conflict_latched
                or not activation.complete
            ):
                return None
            epoch = self._session_epochs.get(session_request_id)
            if epoch is None:
                return None
            state = self._unified_paged_kv.get(session_request_id, {}).get(layer_name)
            if (
                state is None
                or state.slot_generation != activation.generation
                or state.slot_topology_id != activation.canonical_topology_id
            ):
                return None
            session_generations.append(
                (
                    session_request_id,
                    int(epoch),
                    activation.prefix_len,
                    activation.generation,
                    activation.canonical_topology_id,
                )
            )
        return (
            tuple(session_generations),
            tuple(int(seq_len) for seq_len in decode_seq_lens),
            str(device),
        )

    def _store_decode_slot_plan_locked(
        self,
        key: tuple[Any, ...],
        slot_tensor: torch.Tensor,
    ) -> None:
        owner = cast(_PAPDecodeStateOwner, self)
        limit = owner._decode_slot_plan_cache_limit()
        if limit <= 0:
            return
        self._decode_slot_plan_cache[key] = slot_tensor
        self._decode_slot_plan_cache.move_to_end(key)
        while len(self._decode_slot_plan_cache) > limit:
            self._decode_slot_plan_cache.popitem(last=False)

    def append_decode_kv_to_unified_prefill_cache(
        self,
        *,
        session_request_ids: Sequence[str],
        layer_name: str,
        key_batch: torch.Tensor,
        value_batch: torch.Tensor,
        decode_seq_lens: Sequence[int],
        step_context: PAPAttentionStepContext | None = None,
        trace_stats: dict[str, float] | None = None,
    ) -> int:
        """Write a same-layer decode batch into Prefill-owned unified KV blocks.

        Uses a single ``reshape_and_cache_flash`` call over the batch (one GPU
        kernel launch), mirroring the legacy local-pool fast path.
        """
        if len(session_request_ids) != len(decode_seq_lens):
            raise ValueError(
                "PAP unified KV append session_request_ids/decode_seq_lens "
                "length mismatch"
            )
        if step_context is not None:
            session_ids_match = (
                session_request_ids is step_context.session_request_ids
                or tuple(session_request_ids) == step_context.session_request_ids
            )
            seq_lens_match = (
                decode_seq_lens is step_context.decode_seq_lens
                or tuple(int(value) for value in decode_seq_lens)
                == step_context.decode_seq_lens
            )
            if not session_ids_match or not seq_lens_match:
                raise RuntimeError("PAP Attention step context batch mismatch")

        decode_lock_start = time.perf_counter() if trace_stats is not None else 0.0
        with self._decode_append_lock:
            decode_lock_acquired = (
                time.perf_counter() if trace_stats is not None else 0.0
            )
            registry_lock_start = (
                time.perf_counter() if trace_stats is not None else 0.0
            )
            with self._lock:
                if trace_stats is not None:
                    trace_stats["append_lock_wait_ms"] += (
                        decode_lock_acquired - decode_lock_start
                    ) * 1000.0
                    trace_stats["append_lock_wait_ms"] += (
                        time.perf_counter() - registry_lock_start
                    ) * 1000.0
                prepare_start = time.perf_counter() if trace_stats is not None else 0.0
                active_indices: Sequence[int]
                active_states: Sequence[PAPUnifiedPagedKVState]
                expected_positions: Sequence[int]
                base_v_cache: torch.Tensor | None = None
                if step_context is not None:
                    layer_states = step_context.layer_states.get(layer_name)
                    if layer_states is None:
                        raise RuntimeError(
                            f"PAP Attention step has no layer state for {layer_name}"
                        )
                    if layer_name in step_context.completed_layers:
                        active_indices = ()
                        active_states = ()
                        expected_positions = ()
                    else:
                        active_indices = step_context.active_indices
                        active_states = tuple(
                            layer_states[index] for index in active_indices
                        )
                        expected_positions = step_context.active_prior_seq_lens
                        for index, state, position in zip(
                            active_indices,
                            active_states,
                            expected_positions,
                        ):
                            session_request_id = session_request_ids[index]
                            if (
                                self._unified_paged_kv.get(
                                    session_request_id,
                                    {},
                                ).get(layer_name)
                                is not state
                            ):
                                raise RuntimeError(
                                    "PAP unified KV state changed during decode "
                                    f"append request_id={session_request_id} "
                                    f"layer={layer_name}"
                                )
                            decode_len = int(step_context.decode_seq_lens[index])
                            if int(state.seq_len) != position:
                                raise RuntimeError(
                                    "PAP Attention step slot plan sequence changed"
                                )
                            if decode_len != position + 1:
                                raise RuntimeError(
                                    "PAP Attention step decode position changed"
                                )
                            if position < int(
                                state.writable_start_token
                            ) or position >= int(state.writable_end_token):
                                raise RuntimeError(
                                    "PAP unified KV append is outside the "
                                    "writable range"
                                )
                            if int(state.slot_topology_id) != int(
                                step_context.topology_ids[index]
                            ):
                                raise RuntimeError(
                                    "PAP Attention step slot topology changed"
                                )
                        if active_states:
                            base_v_cache = active_states[0].kv_cache
                else:
                    mutable_active_indices: list[int] = []
                    mutable_active_states: list[PAPUnifiedPagedKVState] = []
                    mutable_expected_positions: list[int] = []
                    for index, session_request_id in enumerate(session_request_ids):
                        decode_len = int(decode_seq_lens[index])
                        if decode_len <= 0:
                            continue
                        layer_state_by_name = self._unified_paged_kv.get(
                            session_request_id, {}
                        )
                        current_state = layer_state_by_name.get(layer_name)
                        if current_state is None:
                            raise RuntimeError(
                                "PAP unified KV state missing for request_id="
                                f"{session_request_id} layer={layer_name}"
                            )
                        base_v_cache = current_state.kv_cache
                        position = int(current_state.seq_len)
                        if decode_len <= position:
                            continue
                        if decode_len != position + 1:
                            raise RuntimeError(
                                "PAP unified KV append out of order request_id="
                                f"{session_request_id} layer={layer_name} "
                                f"current_seq_len={position} "
                                f"decode_seq_len={decode_len}"
                            )
                        if position < int(
                            current_state.writable_start_token
                        ) or position >= int(current_state.writable_end_token):
                            raise RuntimeError(
                                "PAP unified KV append out of range request_id="
                                f"{session_request_id} layer={layer_name} "
                                f"position={position} writable=["
                                f"{current_state.writable_start_token},"
                                f"{current_state.writable_end_token})"
                            )
                        mutable_active_indices.append(index)
                        mutable_active_states.append(current_state)
                        mutable_expected_positions.append(position)
                    active_indices = mutable_active_indices
                    active_states = mutable_active_states
                    expected_positions = mutable_expected_positions

                if not active_indices or base_v_cache is None:
                    if (
                        step_context is not None
                        and layer_name not in step_context.completed_layers
                        and tuple(active_indices) != step_context.active_indices
                    ):
                        raise RuntimeError(
                            "PAP Attention step active rows changed across layers"
                        )
                    if trace_stats is not None:
                        trace_stats["append_prepare_ms"] += (
                            time.perf_counter() - prepare_start
                        ) * 1000.0
                    return 0

                all_rows_active = len(active_indices) == len(session_request_ids)
                if all_rows_active:
                    self._decode_append_fast_path_hits += 1
                else:
                    self._decode_append_fallbacks += 1

                slot_plan_key = None
                slot_tensor = None
                use_step_slot_plan = False
                if step_context is not None:
                    use_step_slot_plan = True
                    slot_tensor = step_context.slot_tensor
                elif all_rows_active:
                    slot_plan_key = self._decode_slot_plan_key_locked(
                        session_request_ids=session_request_ids,
                        layer_name=layer_name,
                        decode_seq_lens=decode_seq_lens,
                        device=base_v_cache.device,
                    )
                    if slot_plan_key is not None:
                        slot_tensor = self._decode_slot_plan_cache.get(slot_plan_key)
                        if slot_tensor is not None:
                            self._decode_slot_plan_cache_hits += 1
                            self._decode_slot_plan_cache.move_to_end(slot_plan_key)
                        else:
                            self._decode_slot_plan_cache_misses += 1
                slots: list[int] = []
                if slot_tensor is None:
                    for state, position in zip(
                        active_states,
                        expected_positions,
                    ):
                        block_size = int(state.block_size)
                        logical_block = position // block_size
                        if logical_block >= len(state.block_ids):
                            raise RuntimeError(
                                f"PAP unified KV logical_block={logical_block} "
                                f"exceeds block_ids len={len(state.block_ids)} "
                                f"(layer={layer_name})"
                            )
                        physical_block = _coerce_block_id(
                            state.block_ids[logical_block]
                        )
                        slots.append(
                            physical_block * block_size + position % block_size
                        )
                scale_key = str(base_v_cache.device)
                scales = self._reshape_cache_scales.get(scale_key)
                if trace_stats is not None:
                    trace_stats["append_prepare_ms"] += (
                        time.perf_counter() - prepare_start
                    ) * 1000.0

            tensor_start = time.perf_counter() if trace_stats is not None else 0.0
            if all_rows_active:
                kb = key_batch
                vb = value_batch
            elif step_context is not None and (
                step_context.active_index_tensor is not None
            ):
                kb = torch.index_select(
                    key_batch,
                    0,
                    step_context.active_index_tensor,
                )
                vb = torch.index_select(
                    value_batch,
                    0,
                    step_context.active_index_tensor,
                )
            else:
                row_indices = list(active_indices)
                kb = key_batch[row_indices]
                vb = value_batch[row_indices]
            if slot_tensor is None:
                slot_tensor = torch.tensor(
                    slots,
                    dtype=torch.int64,
                    device=base_v_cache.device,
                )
                if use_step_slot_plan:
                    assert step_context is not None
                    step_context.slot_tensor = slot_tensor
                    cast(
                        _PAPDecodeStateOwner, self
                    ).record_attention_step_slot_plan_build()
                elif slot_plan_key is not None:
                    with self._lock:
                        self._store_decode_slot_plan_locked(
                            slot_plan_key,
                            slot_tensor,
                        )
            if scales is None:
                scales = (
                    torch.ones(
                        1,
                        dtype=torch.float32,
                        device=base_v_cache.device,
                    ),
                    torch.ones(
                        1,
                        dtype=torch.float32,
                        device=base_v_cache.device,
                    ),
                )
                with self._lock:
                    self._reshape_cache_scales[scale_key] = scales
            if trace_stats is not None:
                trace_stats["append_tensor_ms"] += (
                    time.perf_counter() - tensor_start
                ) * 1000.0

            copy_start = time.perf_counter() if trace_stats is not None else 0.0
            if kb.device != base_v_cache.device or kb.dtype != base_v_cache.dtype:
                kb = kb.to(device=base_v_cache.device, dtype=base_v_cache.dtype)
            if vb.device != base_v_cache.device or vb.dtype != base_v_cache.dtype:
                vb = vb.to(device=base_v_cache.device, dtype=base_v_cache.dtype)
            if trace_stats is not None:
                trace_stats["append_copy_ms"] += (
                    time.perf_counter() - copy_start
                ) * 1000.0

            record_start = time.perf_counter() if trace_stats is not None else 0.0
            k_scale, v_scale = scales
            key_cache, value_cache = base_v_cache.unbind(1)
            if (
                _DEFERRED_CUDA_TRACE_ENABLED
                and base_v_cache.is_cuda
                and torch.cuda.is_available()
            ):
                append_trace = begin_deferred_cuda_span(
                    "kv_append_gpu_ms",
                    torch.cuda.current_stream(base_v_cache.device),
                )
                try:
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        kb,
                        vb,
                        key_cache,
                        value_cache,
                        slot_tensor,
                        "auto",
                        k_scale,
                        v_scale,
                    )
                finally:
                    end_deferred_cuda_span(append_trace)
            else:
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    kb,
                    vb,
                    key_cache,
                    value_cache,
                    slot_tensor,
                    "auto",
                    k_scale,
                    v_scale,
                )
            if trace_stats is not None:
                trace_stats["append_record_ms"] += (
                    time.perf_counter() - record_start
                ) * 1000.0

            state_start = time.perf_counter() if trace_stats is not None else 0.0
            with self._lock:
                for index, expected_state, expected_position in zip(
                    active_indices,
                    active_states,
                    expected_positions,
                ):
                    session_request_id = session_request_ids[index]
                    current_state = self._unified_paged_kv.get(
                        session_request_id,
                        {},
                    ).get(layer_name)
                    if current_state is not expected_state:
                        raise RuntimeError(
                            "PAP unified KV state changed during decode append "
                            f"request_id={session_request_id} layer={layer_name}"
                        )
                    if int(current_state.seq_len) != expected_position:
                        raise RuntimeError(
                            "PAP unified KV seq_len changed during decode append "
                            f"request_id={session_request_id} layer={layer_name}"
                        )
                for state in active_states:
                    state.seq_len = int(state.seq_len) + 1
            if trace_stats is not None:
                trace_stats["append_state_ms"] += (
                    time.perf_counter() - state_start
                ) * 1000.0
            return len(active_indices)

    def decode_append_fast_path_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "fast_path_hits": self._decode_append_fast_path_hits,
                "fallbacks": self._decode_append_fallbacks,
                "scale_cache_entries": len(self._reshape_cache_scales),
                "slot_plan_hits": self._decode_slot_plan_cache_hits,
                "slot_plan_misses": self._decode_slot_plan_cache_misses,
                "slot_plan_entries": len(self._decode_slot_plan_cache),
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
