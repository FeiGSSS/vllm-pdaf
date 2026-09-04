# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-context planning and cache ownership for PAP Attention."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from vllm.pap.kv.models import (
    PAPAttentionStepContext,
    PAPOffloadExecSessionEntry,
    PAPUnifiedPagedKVState,
)
from vllm.pap.kv.session_registry import (
    _DECODE_COMMIT_PATH,
    _prefill_control_endpoint,
)


class _PAPAttentionStepContextMixin:
    """Own reusable per-step metadata and OFFLOAD_EXEC entry planning."""

    _last_attention_step_context: PAPAttentionStepContext | None

    def _successor_attention_step_context(
        self,
        *,
        cache_key: tuple[object, ...],
        request_ids: tuple[str, ...],
        decode_seq_lens: tuple[int, ...],
        scales: tuple[float, ...],
        default_q_size: int,
        default_kv_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> PAPAttentionStepContext | None:
        previous = self._last_attention_step_context
        if (
            previous is None
            or previous.request_ids != request_ids
            or previous.completed_layers != set(previous.expected_layers)
            or not previous.kv_ready_published
            or previous.q_size != int(default_q_size)
            or previous.kv_size != int(default_kv_size)
            or previous.num_heads != int(num_heads)
            or previous.num_kv_heads != int(num_kv_heads)
            or previous.head_dim != int(head_dim)
            or not scales
            or any(float(value) != previous.scale for value in scales)
        ):
            return None
        for entry, topology_id, session_request_id in zip(
            previous.session_entries,
            previous.topology_ids,
            previous.session_request_ids,
            strict=True,
        ):
            if self._session_epochs.get(session_request_id) != entry.session_epoch:
                return None
            activation = self._unified_slot_activations.get(session_request_id)
            if (
                activation is None
                or not activation.complete
                or activation.conflict_latched
                or activation.canonical_topology_id != topology_id
            ):
                return None
        first_layer = next(iter(previous.expected_layers))
        prior_seq_lens = tuple(
            int(state.seq_len) for state in previous.layer_states[first_layer]
        )
        if prior_seq_lens != previous.result_seq_lens:
            return None
        if any(
            int(decode_seq_len) > int(prior_seq_len) + 1
            for decode_seq_len, prior_seq_len in zip(
                decode_seq_lens,
                prior_seq_lens,
                strict=True,
            )
        ):
            return None
        result_seq_lens = tuple(
            max(int(prior_seq_len), int(decode_seq_len))
            for decode_seq_len, prior_seq_len in zip(
                decode_seq_lens,
                prior_seq_lens,
                strict=True,
            )
        )
        commit_new_seq_lens = tuple(
            int(decode_seq_len) if int(decode_seq_len) > int(prior_seq_len) else None
            for decode_seq_len, prior_seq_len in zip(
                decode_seq_lens,
                prior_seq_lens,
                strict=True,
            )
        )
        active_indices = tuple(
            index
            for index, new_seq_len in enumerate(commit_new_seq_lens)
            if new_seq_len is not None
        )
        return PAPAttentionStepContext(
            cache_key=cache_key,
            request_ids=request_ids,
            decode_seq_lens=decode_seq_lens,
            session_entries=previous.session_entries,
            session_request_ids=previous.session_request_ids,
            prior_seq_lens=prior_seq_lens,
            result_seq_lens=result_seq_lens,
            commit_new_seq_lens=commit_new_seq_lens,
            active_indices=active_indices,
            active_prior_seq_lens=tuple(
                prior_seq_lens[index] for index in active_indices
            ),
            expected_layers=previous.expected_layers,
            layer_states=previous.layer_states,
            topology_ids=previous.topology_ids,
            q_size=previous.q_size,
            kv_size=previous.kv_size,
            num_heads=previous.num_heads,
            num_kv_heads=previous.num_kv_heads,
            head_dim=previous.head_dim,
            scale=previous.scale,
        )

    def get_or_create_attention_step_context(
        self,
        *,
        request_ids: tuple[str, ...],
        decode_seq_lens: tuple[int, ...],
        scales: tuple[float, ...],
        layer_name: str,
        default_q_size: int,
        default_kv_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> PAPAttentionStepContext:
        """Return one decode-step context shared by all model layers."""
        cache_key = (
            tuple(str(request_id) for request_id in request_ids),
            tuple(int(seq_len) for seq_len in decode_seq_lens),
            tuple(float(scale) for scale in scales),
            int(default_q_size),
            int(default_kv_size),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            str(self.storage_device),
        )
        with self._lock:
            cached = self._attention_step_context_cache.get(cache_key)
            if cached is not None:
                self._attention_step_context_hits += 1
                self._attention_step_context_cache.move_to_end(cache_key)
                return cached
            successor = self._successor_attention_step_context(
                cache_key=cache_key,
                request_ids=request_ids,
                decode_seq_lens=decode_seq_lens,
                scales=scales,
                default_q_size=default_q_size,
                default_kv_size=default_kv_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            if successor is not None:
                self._attention_step_context_successor_hits += 1
                self._attention_step_context_hits += 1
                self._last_attention_step_context = successor
                limit = self._decode_slot_plan_cache_limit()
                if limit > 0:
                    self._attention_step_context_cache[cache_key] = successor
                    self._attention_step_context_cache.move_to_end(cache_key)
                    while len(self._attention_step_context_cache) > limit:
                        self._attention_step_context_cache.popitem(last=False)
                return successor

        session_entries = tuple(
            self.offload_exec_batch_session_entries(
                request_ids,
                default_q_size=default_q_size,
                default_kv_size=default_kv_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        )
        session_request_ids = tuple(
            entry.session_request_id for entry in session_entries
        )
        self.get_unified_paged_states(
            session_request_ids=session_request_ids,
            layer_name=layer_name,
        )

        with self._lock:
            cached = self._attention_step_context_cache.get(cache_key)
            if cached is not None:
                self._attention_step_context_hits += 1
                self._attention_step_context_cache.move_to_end(cache_key)
                return cached

            expected_layers: frozenset[str] | None = None
            for session_request_id in session_request_ids:
                states = self._unified_paged_kv.get(session_request_id, {})
                activation = self._unified_slot_activations.get(session_request_id)
                session_layers = (
                    activation.expected_layers
                    if activation is not None and activation.expected_layers is not None
                    else frozenset(states)
                )
                if not session_layers or layer_name not in session_layers:
                    raise RuntimeError(
                        "PAP Attention step has no complete layer set for "
                        f"request_id={session_request_id} layer={layer_name}"
                    )
                if expected_layers is None:
                    expected_layers = frozenset(session_layers)
                elif expected_layers != session_layers:
                    raise RuntimeError(
                        "PAP Attention step mixes incompatible layer sets"
                    )
            assert expected_layers is not None

            layer_states: dict[str, tuple[PAPUnifiedPagedKVState, ...]] = {}
            for expected_layer in expected_layers:
                row_states: list[PAPUnifiedPagedKVState] = []
                for session_request_id in session_request_ids:
                    state = self._unified_paged_kv.get(session_request_id, {}).get(
                        expected_layer
                    )
                    if state is None:
                        raise RuntimeError(
                            "PAP Attention step is missing sealed layer state "
                            f"request_id={session_request_id} "
                            f"layer={expected_layer}"
                        )
                    row_states.append(state)
                layer_states[expected_layer] = tuple(row_states)

            prior_seq_lens: list[int] = []
            topology_ids: list[int] = []
            for row_index, session_request_id in enumerate(session_request_ids):
                row_seq_lens = {
                    int(layer_states[name][row_index].seq_len)
                    for name in expected_layers
                }
                if len(row_seq_lens) != 1:
                    raise RuntimeError(
                        "PAP Attention step observed divergent layer sequence "
                        f"lengths request_id={session_request_id}"
                    )
                prior_seq_lens.append(next(iter(row_seq_lens)))
                row_topologies = {
                    (
                        state.block_ids,
                        int(state.block_size),
                        str(state.kv_cache.device),
                        str(state.layout),
                    )
                    for state in (
                        layer_states[name][row_index] for name in expected_layers
                    )
                }
                if len(row_topologies) != 1:
                    raise RuntimeError(
                        "PAP sealed step topology differs across layers "
                        f"request_id={session_request_id}"
                    )
                activation = self._unified_slot_activations.get(session_request_id)
                if activation is None:
                    topology_ids.append(0)
                    continue
                if activation.conflict_latched or not activation.complete:
                    raise RuntimeError(
                        "PAP sealed step topology activation is incomplete "
                        f"request_id={session_request_id}"
                    )
                topology_id = int(activation.canonical_topology_id)
                topology_ids.append(topology_id)
                if any(
                    state.slot_generation != activation.generation
                    or state.slot_topology_id != topology_id
                    for state in (
                        layer_states[name][row_index] for name in expected_layers
                    )
                ):
                    raise RuntimeError(
                        "PAP sealed step topology identity differs across "
                        f"layers request_id={session_request_id}"
                    )

            result_seq_lens: list[int] = []
            commit_new_seq_lens: list[int | None] = []
            for request_id, prior_seq_len, decode_seq_len in zip(
                request_ids,
                prior_seq_lens,
                decode_seq_lens,
            ):
                if int(decode_seq_len) > int(prior_seq_len) + 1:
                    raise RuntimeError(
                        "PAP unified KV append out of order request_id="
                        f"{request_id} current_seq_len={prior_seq_len} "
                        f"decode_seq_len={decode_seq_len}"
                    )
                result_seq_lens.append(max(int(prior_seq_len), int(decode_seq_len)))
                commit_new_seq_lens.append(
                    int(decode_seq_len)
                    if int(decode_seq_len) > int(prior_seq_len)
                    else None
                )

            shapes = {
                (
                    entry.q_size,
                    entry.kv_size,
                    entry.num_heads,
                    entry.num_kv_heads,
                    entry.head_dim,
                )
                for entry in session_entries
            }
            if len(shapes) != 1 or not scales:
                raise RuntimeError("PAP OFFLOAD_EXEC batch has mixed shapes or scales")
            shape = next(iter(shapes))
            scale = float(scales[0])
            if any(float(value) != scale for value in scales):
                raise RuntimeError("PAP OFFLOAD_EXEC batch has mixed shapes or scales")

            active_indices = tuple(
                index
                for index, seq_len in enumerate(commit_new_seq_lens)
                if seq_len is not None
            )
            context = PAPAttentionStepContext(
                cache_key=cache_key,
                request_ids=tuple(request_ids),
                decode_seq_lens=tuple(decode_seq_lens),
                session_entries=session_entries,
                session_request_ids=session_request_ids,
                prior_seq_lens=tuple(prior_seq_lens),
                result_seq_lens=tuple(result_seq_lens),
                commit_new_seq_lens=tuple(commit_new_seq_lens),
                active_indices=active_indices,
                active_prior_seq_lens=tuple(
                    prior_seq_lens[index] for index in active_indices
                ),
                expected_layers=expected_layers,
                layer_states=layer_states,
                topology_ids=tuple(topology_ids),
                q_size=int(shape[0]),
                kv_size=int(shape[1]),
                num_heads=int(shape[2]),
                num_kv_heads=int(shape[3]),
                head_dim=int(shape[4]),
                scale=scale,
            )
            self._attention_step_context_misses += 1
            self._last_attention_step_context = context
            limit = self._decode_slot_plan_cache_limit()
            if limit > 0:
                self._attention_step_context_cache[cache_key] = context
                self._attention_step_context_cache.move_to_end(cache_key)
                while len(self._attention_step_context_cache) > limit:
                    self._attention_step_context_cache.popitem(last=False)
            return context

    def record_attention_step_slot_plan_build(self) -> None:
        """Record construction of one step-owned decode slot tensor."""
        with self._lock:
            self._attention_step_slot_plan_builds += 1

    def record_attention_step_metadata_build(self) -> None:
        """Record construction of one step-owned paged-attention plan."""
        with self._lock:
            self._attention_step_metadata_builds += 1

    def complete_attention_step_layer(
        self,
        *,
        context: PAPAttentionStepContext,
        layer_name: str,
    ) -> int:
        """Publish KV readiness once every layer in the step has completed."""
        with context.lock:
            if layer_name not in context.expected_layers:
                raise RuntimeError(
                    f"PAP Attention step received unexpected layer {layer_name}"
                )
            context.completed_layers.add(layer_name)
            if context.kv_ready_published or len(context.completed_layers) != len(
                context.expected_layers
            ):
                return 0

            published = 0
            for request_id, entry, new_seq_len in zip(
                context.request_ids,
                context.session_entries,
                context.commit_new_seq_lens,
            ):
                if new_seq_len is None:
                    continue
                endpoint = _prefill_control_endpoint(
                    entry.prefill_endpoint,
                    _DECODE_COMMIT_PATH,
                )
                status = self.record_decode_kv_ready(
                    request_id=entry.session_request_id,
                    session_epoch=entry.session_epoch,
                    new_seq_len=new_seq_len,
                    endpoint=endpoint,
                    commit_request_id=request_id,
                )
                if status != "released":
                    published += 1
            context.kv_ready_published = True
        if published:
            with self._lock:
                self._attention_step_kv_ready_publishes += published
        return published

    def attention_step_context_stats(self) -> dict[str, int]:
        """Return step-context reuse counters."""
        with self._lock:
            return {
                "step_context_hits": self._attention_step_context_hits,
                "step_context_misses": self._attention_step_context_misses,
                "step_context_successor_hits": (
                    self._attention_step_context_successor_hits
                ),
                "step_context_entries": len(self._attention_step_context_cache),
                "step_slot_plan_builds": (self._attention_step_slot_plan_builds),
                "step_metadata_builds": self._attention_step_metadata_builds,
                "step_kv_ready_publishes": (self._attention_step_kv_ready_publishes),
            }

    def offload_exec_batch_session_entries(
        self,
        request_ids: tuple[str, ...],
        *,
        default_q_size: int,
        default_kv_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> list[PAPOffloadExecSessionEntry]:
        if int(num_heads) <= 0 or int(num_kv_heads) <= 0 or int(head_dim) <= 0:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC requires PAP_OFFLOAD_EXEC_NUM_HEADS, "
                "PAP_OFFLOAD_EXEC_NUM_KV_HEADS, and PAP_OFFLOAD_EXEC_HEAD_DIM"
            )
        with self._lock:
            entries: list[PAPOffloadExecSessionEntry] = []
            for request_id in request_ids:
                session_request_id = self._resolve_session_request_id_locked(request_id)
                if session_request_id is None:
                    raise KeyError(request_id)
                session = self._sessions[session_request_id]
                session_epoch = self._session_epochs.get(session_request_id)
                if session_epoch is None:
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC session has no active epoch "
                        f"request_id={session_request_id}"
                    )
                q_size = int(session.q_size or default_q_size)
                kv_size = int(session.kv_size or default_kv_size)
                if q_size <= 0 or kv_size <= 0:
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC requires q_size and kv_size in "
                        "attention registration or PAP_OFFLOAD_EXEC_Q_SIZE/"
                        "PAP_OFFLOAD_EXEC_KV_SIZE"
                    )
                cache_key = (
                    session_request_id,
                    int(session_epoch),
                    q_size,
                    kv_size,
                    int(num_heads),
                    int(num_kv_heads),
                    int(head_dim),
                )
                session_entry = self._offload_exec_session_entry_cache.get(cache_key)
                if session_entry is None:
                    session_entry = PAPOffloadExecSessionEntry(
                        session_request_id=session_request_id,
                        session_epoch=int(session_epoch),
                        prefill_endpoint=session.prefill_endpoint,
                        q_size=q_size,
                        kv_size=kv_size,
                        num_heads=int(num_heads),
                        num_kv_heads=int(num_kv_heads),
                        head_dim=int(head_dim),
                    )
                    self._offload_exec_session_entry_cache[cache_key] = session_entry
                entries.append(session_entry)
            return entries
