# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable PAT planning for PAP decode Attention."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count as itertools_count
from typing import Any

import torch

from vllm.pap.config import read_env_int

_PAT_GRAPH_TOKENS = itertools_count(1)


@dataclass(frozen=True)
class _PATTail:
    group: int
    row: int
    base_table_blocks: int
    base_used_blocks: int
    private: bool


@dataclass
class PAPPATPlan:
    """One fixed-address PAT schedule and its GPU metadata."""

    q_tables: tuple[torch.Tensor, ...]
    block_tables: tuple[torch.Tensor, ...]
    num_seqs_per_ctas: tuple[torch.Tensor, ...]
    cta_ranks: tuple[torch.Tensor, ...]
    kv_in_ctas: tuple[torch.Tensor, ...]
    mnws: tuple[tuple[int, int, int], ...]
    num_split_per_seq: torch.Tensor
    max_split_per_seq: int
    max_seqs_in_cta: int
    max_blocks_in_cta: int
    output: torch.Tensor
    scale: float
    reused_kv_tokens: int
    base_seq_lens: tuple[int, ...]
    block_size: int
    base_kv_in_ctas: tuple[torch.Tensor, ...]
    host_kv_in_ctas: tuple[torch.Tensor, ...]
    host_block_tables: tuple[torch.Tensor, ...]
    kv_in_cta_deltas: tuple[tuple[torch.Tensor, ...], ...]
    request_tails: tuple[_PATTail, ...]
    incremental_updates_supported: bool
    graph_token: int = field(init=False)
    backend_name: str = "pat"

    def __post_init__(self) -> None:
        self.graph_token = next(_PAT_GRAPH_TOKENS)

    @property
    def graph_key(self) -> tuple[Any, ...]:
        return (self.graph_token, float(self.scale))

    @property
    def bound_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *self.q_tables,
            *self.block_tables,
            *self.num_seqs_per_ctas,
            *self.cta_ranks,
            *self.kv_in_ctas,
            self.num_split_per_seq,
            self.output,
        )

    def copy_from(self, info: Any) -> None:
        groups = (
            (self.q_tables, info.q_tables),
            (self.num_seqs_per_ctas, info.num_seqs_per_CTAs),
            (self.cta_ranks, info.CTA_ranks),
            (self.kv_in_ctas, info.kv_in_CTAs),
        )
        for destinations, sources in groups:
            for destination, source in zip(destinations, sources):
                destination.copy_(source, non_blocking=True)
        for destination, host, source in zip(
            self.block_tables,
            self.host_block_tables,
            info.block_tables,
            strict=True,
        ):
            host.zero_()
            host[:, : source.shape[1]].copy_(source)
            destination.copy_(host, non_blocking=True)
        self.num_split_per_seq.copy_(info.num_split_per_seq, non_blocking=True)

    def update_decode_state(
        self,
        states: Sequence[Any],
        seq_lens: Sequence[int],
    ) -> bool:
        if (
            not self.incremental_updates_supported
            or len(seq_lens) != len(self.base_seq_lens)
            or len(states) != len(seq_lens)
        ):
            return False
        deltas = tuple(
            int(seq_len) - base_seq_len
            for seq_len, base_seq_len in zip(
                seq_lens,
                self.base_seq_lens,
                strict=True,
            )
        )
        block_updates: list[tuple[_PATTail, tuple[int, ...]]] = []
        for state, seq_len, base_seq_len, delta, tail in zip(
            states,
            seq_lens,
            self.base_seq_lens,
            deltas,
            self.request_tails,
            strict=True,
        ):
            if delta < 0 or int(state.block_size) != self.block_size:
                return False
            used_blocks = math.ceil(int(seq_len) / self.block_size)
            if used_blocks > len(state.block_ids):
                return False
            added_blocks = used_blocks - tail.base_used_blocks
            if added_blocks < 0:
                return False
            if added_blocks:
                if not tail.private:
                    return False
                table_end = tail.base_table_blocks + added_blocks
                if table_end > self.host_block_tables[tail.group].shape[1]:
                    return False
                block_updates.append(
                    (
                        tail,
                        tuple(
                            map(
                                int,
                                state.block_ids[tail.base_used_blocks : used_blocks],
                            )
                        ),
                    )
                )
        for host, base in zip(
            self.host_kv_in_ctas,
            self.base_kv_in_ctas,
            strict=True,
        ):
            host.copy_(base)
        for delta, request_deltas in zip(
            deltas,
            self.kv_in_cta_deltas,
            strict=True,
        ):
            if delta == 0:
                continue
            for host, direction in zip(
                self.host_kv_in_ctas,
                request_deltas,
                strict=True,
            ):
                host.add_(direction, alpha=delta)
        for destination, source in zip(
            self.kv_in_ctas,
            self.host_kv_in_ctas,
            strict=True,
        ):
            destination.copy_(source, non_blocking=True)
        changed_groups: set[int] = set()
        for tail, block_ids in block_updates:
            start = tail.base_table_blocks
            end = start + len(block_ids)
            self.host_block_tables[tail.group][tail.row, start:end] = torch.tensor(
                block_ids,
                dtype=torch.int32,
            )
            changed_groups.add(tail.group)
        for group in changed_groups:
            self.block_tables[group].copy_(
                self.host_block_tables[group],
                non_blocking=True,
            )
        return True

    def run_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        from prefix_attn import prefix_attn_with_kvcache

        prefix_attn_with_kvcache(
            query.unsqueeze(1),
            key_cache,
            value_cache,
            self.num_split_per_seq,
            list(self.q_tables),
            list(self.block_tables),
            list(self.num_seqs_per_ctas),
            list(self.cta_ranks),
            list(self.kv_in_ctas),
            [list(mnw) for mnw in self.mnws],
            self.max_split_per_seq,
            self.max_seqs_in_cta,
            self.max_blocks_in_cta,
            self.scale,
            self.output,
            None,
        )
        return self.output.squeeze(1)


class PAPPATPlanner:
    """Build PAT plans and reuse fixed GPU metadata addresses."""

    _TILES = ((16, 32, 1), (32, 32, 2))

    @staticmethod
    def unavailable_reason() -> str | None:
        if os.environ.get("DISABLE_STREAM") != "1":
            return "DISABLE_STREAM=1 is required"
        try:
            from prefix_attn import PrefixTreeCPP  # noqa: F401
        except (ImportError, OSError) as error:
            return f"prefix_attn is unavailable: {error}"
        return None

    def __init__(self) -> None:
        unavailable_reason = self.unavailable_reason()
        if unavailable_reason is not None:
            raise RuntimeError(f"PAP PAT is unavailable: {unavailable_reason}")

        self.max_entries = read_env_int(
            os.environ, "PAP_PAT_PLAN_CACHE_ENTRIES", 32, minimum=0
        )
        self._plans: OrderedDict[tuple[Any, ...], PAPPATPlan] = OrderedDict()

    @staticmethod
    def _tensor_shapes(values: Sequence[torch.Tensor]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(dim) for dim in value.shape) for value in values)

    @staticmethod
    def _capacity(value: int) -> int:
        return 1 << max(0, int(value - 1).bit_length())

    def _build_info(
        self,
        *,
        block_size: int,
        seq_lens: Sequence[int],
        block_table: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> Any:
        from prefix_attn import PrefixTreeCPP

        tree = PrefixTreeCPP(block_size)
        tree.build_radix_tree(list(map(int, seq_lens)), block_table)
        tree.pack_schedule(
            [list(tile) for tile in self._TILES],
            num_heads // num_kv_heads,
            num_kv_heads,
        )
        return tree.kernel_info

    @staticmethod
    def _incremental_metadata(
        *,
        info: Any,
        block_size: int,
        seq_lens: Sequence[int],
        states: Sequence[Any],
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[tuple[torch.Tensor, ...], ...],
        tuple[_PATTail, ...],
        bool,
    ]:
        base_values = tuple(value.clone() for value in info.kv_in_CTAs)
        host_values = tuple(value.clone() for value in info.kv_in_CTAs)
        request_deltas: list[tuple[torch.Tensor, ...]] = []
        request_tails: list[_PATTail] = []
        supported = True
        for request_index, (state, seq_len) in enumerate(
            zip(states, seq_lens, strict=True)
        ):
            used_blocks = math.ceil(int(seq_len) / block_size)
            final_block = int(state.block_ids[used_blocks - 1])
            matches: list[tuple[int, int, int, bool]] = []
            for group, (q_table, block_table, num_seqs, kv_in_cta) in enumerate(
                zip(
                    info.q_tables,
                    info.block_tables,
                    info.num_seqs_per_CTAs,
                    info.kv_in_CTAs,
                    strict=True,
                )
            ):
                for row in range(int(q_table.shape[0])):
                    active_queries = tuple(
                        map(int, q_table[row, : int(num_seqs[row])].tolist())
                    )
                    table_blocks = math.ceil(int(kv_in_cta[row]) / block_size)
                    if (
                        request_index in active_queries
                        and table_blocks > 0
                        and int(block_table[row, table_blocks - 1]) == final_block
                    ):
                        matches.append(
                            (group, row, table_blocks, len(active_queries) == 1)
                        )
            if len(matches) != 1:
                supported = False
                break
            group, row, table_blocks, private = matches[0]
            request_tails.append(
                _PATTail(
                    group=group,
                    row=row,
                    base_table_blocks=table_blocks,
                    base_used_blocks=used_blocks,
                    private=private,
                )
            )
            deltas = tuple(torch.zeros_like(value) for value in info.kv_in_CTAs)
            deltas[group][row] = 1
            request_deltas.append(deltas)
        if not supported:
            request_deltas = [
                tuple(torch.zeros_like(value) for value in info.kv_in_CTAs)
                for _ in seq_lens
            ]
            request_tails = [_PATTail(0, 0, 0, 0, False) for _ in seq_lens]
        return (
            base_values,
            host_values,
            tuple(request_deltas),
            tuple(request_tails),
            supported,
        )

    def _block_table_shapes(
        self,
        values: Sequence[torch.Tensor],
        request_tails: Sequence[_PATTail],
        states: Sequence[Any],
    ) -> tuple[tuple[int, int], ...]:
        widths = [int(value.shape[1]) for value in values]
        for tail, state in zip(request_tails, states, strict=True):
            if not tail.private:
                continue
            future_blocks = len(state.block_ids) - tail.base_used_blocks
            widths[tail.group] = max(
                widths[tail.group],
                tail.base_table_blocks + future_blocks,
            )
        return tuple(
            (int(value.shape[0]), self._capacity(width))
            for value, width in zip(values, widths, strict=True)
        )

    def _key(
        self,
        info: Any,
        *,
        block_table_shapes: tuple[tuple[int, int], ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Any, ...]:
        return (
            self._tensor_shapes(info.q_tables),
            block_table_shapes,
            self._tensor_shapes(info.num_seqs_per_CTAs),
            self._tensor_shapes(info.CTA_ranks),
            self._tensor_shapes(info.kv_in_CTAs),
            tuple(int(dim) for dim in info.num_split_per_seq.shape),
            tuple(tuple(int(value) for value in mnw) for mnw in info.MNWs),
            int(info.max_split_per_seq),
            int(info.max_seqs_in_CTA),
            max(shape[1] for shape in block_table_shapes),
            dtype,
            device,
        )

    @staticmethod
    def _gpu_buffers(
        values: Sequence[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.empty(tuple(value.shape), dtype=value.dtype, device=device)
            for value in values
        )

    @staticmethod
    def _block_table_buffers(
        values: Sequence[torch.Tensor],
        shapes: Sequence[tuple[int, int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.empty(shape, dtype=value.dtype, device=device)
            for value, shape in zip(values, shapes, strict=True)
        )

    def _create_plan(
        self,
        info: Any,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        reused_kv_tokens: int,
        base_seq_lens: tuple[int, ...],
        block_size: int,
        base_kv_in_ctas: tuple[torch.Tensor, ...],
        host_kv_in_ctas: tuple[torch.Tensor, ...],
        block_table_shapes: tuple[tuple[int, int], ...],
        kv_in_cta_deltas: tuple[tuple[torch.Tensor, ...], ...],
        request_tails: tuple[_PATTail, ...],
        incremental_updates_supported: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PAPPATPlan:
        return PAPPATPlan(
            q_tables=self._gpu_buffers(info.q_tables, device),
            block_tables=self._block_table_buffers(
                info.block_tables,
                block_table_shapes,
                device,
            ),
            num_seqs_per_ctas=self._gpu_buffers(info.num_seqs_per_CTAs, device),
            cta_ranks=self._gpu_buffers(info.CTA_ranks, device),
            kv_in_ctas=self._gpu_buffers(info.kv_in_CTAs, device),
            mnws=tuple((int(m), int(n), int(w)) for m, n, w in info.MNWs),
            num_split_per_seq=torch.empty(
                tuple(info.num_split_per_seq.shape),
                dtype=info.num_split_per_seq.dtype,
                device=device,
            ),
            max_split_per_seq=int(info.max_split_per_seq),
            max_seqs_in_cta=int(info.max_seqs_in_CTA),
            max_blocks_in_cta=max(shape[1] for shape in block_table_shapes),
            output=torch.empty(
                (batch_size, 1, num_heads, head_dim),
                dtype=dtype,
                device=device,
            ),
            scale=scale,
            reused_kv_tokens=reused_kv_tokens,
            base_seq_lens=base_seq_lens,
            block_size=block_size,
            base_kv_in_ctas=base_kv_in_ctas,
            host_kv_in_ctas=host_kv_in_ctas,
            host_block_tables=tuple(
                torch.empty(shape, dtype=value.dtype, device="cpu")
                for value, shape in zip(
                    info.block_tables,
                    block_table_shapes,
                    strict=True,
                )
            ),
            kv_in_cta_deltas=kv_in_cta_deltas,
            request_tails=request_tails,
            incremental_updates_supported=incremental_updates_supported,
        )

    def plan(
        self,
        *,
        states: Sequence[Any],
        seq_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        reused_kv_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PAPPATPlan | None:
        batch_size = len(states)
        if batch_size < 2 or len(seq_lens) != batch_size:
            return None
        block_size = int(states[0].block_size)
        if (
            block_size != 16
            or num_heads != 32
            or num_kv_heads != 8
            or head_dim != 128
            or dtype != torch.float16
            or any(int(state.block_size) != block_size for state in states)
        ):
            return None
        if reused_kv_tokens <= 0:
            return None

        used_blocks = [math.ceil(int(seq_len) / block_size) for seq_len in seq_lens]
        block_table = torch.zeros(
            (batch_size, max(used_blocks)),
            dtype=torch.int32,
            device="cpu",
        )
        for row, (state, count) in enumerate(zip(states, used_blocks, strict=True)):
            block_table[row, :count] = torch.tensor(
                state.block_ids[:count],
                dtype=torch.int32,
                device="cpu",
            )
        info = self._build_info(
            block_size=block_size,
            seq_lens=seq_lens,
            block_table=block_table,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        (
            base_kv_in_ctas,
            host_kv_in_ctas,
            kv_in_cta_deltas,
            request_tails,
            incremental_updates_supported,
        ) = self._incremental_metadata(
            info=info,
            block_size=block_size,
            seq_lens=seq_lens,
            states=states,
        )
        block_table_shapes = self._block_table_shapes(
            info.block_tables,
            request_tails,
            states,
        )
        key = self._key(
            info,
            block_table_shapes=block_table_shapes,
            dtype=dtype,
            device=device,
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = self._create_plan(
                info,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                reused_kv_tokens=reused_kv_tokens,
                base_seq_lens=tuple(map(int, seq_lens)),
                block_size=block_size,
                base_kv_in_ctas=base_kv_in_ctas,
                host_kv_in_ctas=host_kv_in_ctas,
                block_table_shapes=block_table_shapes,
                kv_in_cta_deltas=kv_in_cta_deltas,
                request_tails=request_tails,
                incremental_updates_supported=incremental_updates_supported,
                dtype=dtype,
                device=device,
            )
            self._plans[key] = plan
            while len(self._plans) > self.max_entries:
                self._plans.popitem(last=False)
        else:
            self._plans.move_to_end(key)
            plan.scale = scale
            plan.reused_kv_tokens = reused_kv_tokens
            plan.base_seq_lens = tuple(map(int, seq_lens))
            plan.block_size = block_size
            plan.base_kv_in_ctas = base_kv_in_ctas
            plan.host_kv_in_ctas = host_kv_in_ctas
            plan.kv_in_cta_deltas = kv_in_cta_deltas
            plan.request_tails = request_tails
            plan.incremental_updates_supported = incremental_updates_supported
        plan.copy_from(info)
        return plan


class PAPPATOrTritonSelector:
    """Reuse the previous decision and select PAT for any physical KV reuse."""

    def __init__(self, pat_planner: PAPPATPlanner | None = None) -> None:
        self.pat = pat_planner or PAPPATPlanner()
        self._has_previous_step = False
        self._last_structure_signature: tuple[Any, ...] | None = None
        self._last_metadata_signature: tuple[Any, ...] | None = None
        self._last_reused_kv_tokens = 0
        self._last_plan: PAPPATPlan | None = None
        self.metadata_reuses = 0
        self.incremental_metadata_reuses = 0
        self.prefix_rechecks = 0
        self.pat_rebuilds = 0
        self.triton_selections = 0

    @classmethod
    def create_if_available(
        cls,
    ) -> tuple[PAPPATOrTritonSelector | None, str | None]:
        unavailable_reason = PAPPATPlanner.unavailable_reason()
        if unavailable_reason is not None:
            return None, unavailable_reason
        return cls(), None

    def stats(self) -> dict[str, int]:
        return {
            "attention_kernel_metadata_reuses": self.metadata_reuses,
            "attention_kernel_incremental_metadata_reuses": (
                self.incremental_metadata_reuses
            ),
            "attention_kernel_prefix_rechecks": self.prefix_rechecks,
            "attention_kernel_pat_rebuilds": self.pat_rebuilds,
            "attention_kernel_triton_selections": self.triton_selections,
        }

    @staticmethod
    def _storage_id(state: Any) -> int:
        return int(state.kv_cache.untyped_storage().data_ptr())

    @classmethod
    def _structure_signature(
        cls,
        *,
        request_ids: Sequence[str],
        topology_ids: Sequence[int],
        states: Sequence[Any],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Any, ...]:
        return (
            tuple(map(str, request_ids)),
            tuple(map(int, topology_ids)),
            tuple(cls._storage_id(state) for state in states),
            tuple(int(state.block_size) for state in states),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            dtype,
            device,
        )

    @classmethod
    def _reused_kv_tokens(
        cls,
        states: Sequence[Any],
        seq_lens: Sequence[int],
    ) -> int:
        logical_blocks = 0
        unique_blocks_by_storage: dict[int, set[int]] = {}
        block_size = int(states[0].block_size) if states else 0
        for state, raw_seq_len in zip(states, seq_lens, strict=True):
            seq_len = int(raw_seq_len)
            if int(state.block_size) != block_size:
                raise ValueError("PAP Attention selector received mixed block sizes")
            used_blocks = math.ceil(seq_len / block_size)
            storage_id = cls._storage_id(state)
            logical_blocks += used_blocks
            unique_blocks_by_storage.setdefault(storage_id, set()).update(
                state.block_ids[:used_blocks]
            )
        unique_blocks = sum(map(len, unique_blocks_by_storage.values()))
        return (logical_blocks - unique_blocks) * block_size

    def plan(
        self,
        *,
        step_signature: tuple[Any, ...],
        request_ids: Sequence[str],
        topology_ids: Sequence[int],
        states: Sequence[Any],
        seq_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> PAPPATPlan | None:
        batch_size = len(states)
        if not (len(request_ids) == len(topology_ids) == len(seq_lens) == batch_size):
            raise ValueError("PAP Attention selector batch metadata length mismatch")
        structure_signature = self._structure_signature(
            request_ids=request_ids,
            topology_ids=topology_ids,
            states=states,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        metadata_signature = (
            step_signature,
            structure_signature,
            tuple(map(int, seq_lens)),
        )
        if (
            self._has_previous_step
            and metadata_signature == self._last_metadata_signature
        ):
            self.metadata_reuses += 1
            return self._last_plan
        if (
            self._has_previous_step
            and structure_signature == self._last_structure_signature
        ):
            if self._last_plan is None:
                self.incremental_metadata_reuses += 1
                self._last_metadata_signature = metadata_signature
                return None
            if self._last_plan.update_decode_state(states, seq_lens):
                self.incremental_metadata_reuses += 1
                self._last_plan.reused_kv_tokens = self._last_reused_kv_tokens
                self._last_metadata_signature = metadata_signature
                return self._last_plan

        self.prefix_rechecks += 1
        reused_kv_tokens = self._reused_kv_tokens(states, seq_lens)

        plan = None
        if reused_kv_tokens > 0:
            plan = self.pat.plan(
                states=states,
                seq_lens=seq_lens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scale=scale,
                reused_kv_tokens=reused_kv_tokens,
                dtype=dtype,
                device=device,
            )
        if plan is not None:
            self.pat_rebuilds += 1
        else:
            self.triton_selections += 1

        self._has_previous_step = True
        self._last_structure_signature = structure_signature
        self._last_metadata_signature = metadata_signature
        self._last_reused_kv_tokens = reused_kv_tokens
        self._last_plan = plan
        return plan


__all__ = ["PAPPATOrTritonSelector", "PAPPATPlan", "PAPPATPlanner"]
