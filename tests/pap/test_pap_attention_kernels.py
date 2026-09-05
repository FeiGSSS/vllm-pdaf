# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import vllm.pap.attention.dispatch as dispatch
import vllm.pap.attention.kernels as kernels
import vllm.pap.attention.pat as pat
from vllm.pap.attention.dispatch import run_pap_decode_attention
from vllm.pap.attention.kernels import (
    PAP_TRITON_DECODE_DEFAULT_CONFIG,
    PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG,
    PAPAttentionStepTensorCache,
    PAPPagedDecodeWorkspace,
    PAPPagedDecodeWorkspaceCache,
    paged_decode_kernel_config_for_sms,
    run_triton_paged_decode_attention,
)
from vllm.pap.attention.pat import PAPPATOrTritonSelector, PAPPATPlan
from vllm.pap.config import PAPConfigError
from vllm.pap.kv.metadata import PAPPagedFlashMetadata


def test_pat_rejects_negative_plan_cache_budget_before_building(monkeypatch):
    monkeypatch.setattr(
        pat.PAPPATPlanner, "unavailable_reason", staticmethod(lambda: None)
    )
    monkeypatch.setenv("PAP_PAT_PLAN_CACHE_ENTRIES", "-1")
    with pytest.raises(PAPConfigError, match="PAP_PAT_PLAN_CACHE_ENTRIES"):
        pat.PAPPATPlanner()


class _FakePATPlan:
    def __init__(self) -> None:
        self.reused_kv_tokens = 0
        self.length_updates: list[tuple[int, ...]] = []
        self.allow_length_updates = False

    def update_decode_state(self, states, seq_lens) -> bool:
        del states
        self.length_updates.append(tuple(seq_lens))
        return self.allow_length_updates


class _FakePATPlanner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result = _FakePATPlan()

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _selector_plan(selector, *, step_signature, states, seq_lens):
    return selector.plan(
        step_signature=step_signature,
        request_ids=tuple(f"request-{index}" for index in range(len(states))),
        topology_ids=tuple(range(len(states))),
        states=states,
        seq_lens=seq_lens,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        scale=128**-0.5,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )


def test_attention_selector_reuses_exact_previous_pat_metadata() -> None:
    storage = torch.empty(1)
    states = tuple(
        SimpleNamespace(kv_cache=storage, block_ids=(0,), block_size=16)
        for _ in range(2)
    )
    pat = _FakePATPlanner()
    selector = PAPPATOrTritonSelector(pat)

    first = _selector_plan(
        selector,
        step_signature=("same-step",),
        states=states,
        seq_lens=(16, 16),
    )
    second = _selector_plan(
        selector,
        step_signature=("same-step",),
        states=states,
        seq_lens=(16, 16),
    )

    assert first is pat.result
    assert second is first
    assert len(pat.calls) == 1
    assert selector.stats()["attention_kernel_metadata_reuses"] == 1


def test_auto_selector_falls_back_without_required_graph_stream(monkeypatch) -> None:
    monkeypatch.delenv("DISABLE_STREAM", raising=False)

    selector, unavailable_reason = PAPPATOrTritonSelector.create_if_available()

    assert selector is None
    assert unavailable_reason == "DISABLE_STREAM=1 is required"


def test_attention_selector_rebuilds_when_incremental_update_is_unsupported() -> None:
    storage = torch.empty(1)
    states = tuple(
        SimpleNamespace(kv_cache=storage, block_ids=(0,), block_size=16)
        for _ in range(2)
    )
    pat = _FakePATPlanner()
    selector = PAPPATOrTritonSelector(pat)

    _selector_plan(
        selector,
        step_signature=("step-1",),
        states=states,
        seq_lens=(16, 16),
    )
    _selector_plan(
        selector,
        step_signature=("step-2",),
        states=states,
        seq_lens=(16, 16),
    )

    assert len(pat.calls) == 2
    assert pat.calls[0]["reused_kv_tokens"] == 16
    assert pat.calls[1]["reused_kv_tokens"] == 16


def test_attention_selector_updates_pat_metadata_for_same_topology() -> None:
    storage = torch.empty(1)
    states = tuple(
        SimpleNamespace(kv_cache=storage, block_ids=(0,), block_size=16)
        for _ in range(2)
    )
    pat = _FakePATPlanner()
    pat.result.allow_length_updates = True
    selector = PAPPATOrTritonSelector(pat)

    first = _selector_plan(
        selector,
        step_signature=("step-1",),
        states=states,
        seq_lens=(15, 15),
    )
    second = _selector_plan(
        selector,
        step_signature=("step-2",),
        states=states,
        seq_lens=(16, 16),
    )

    assert second is first
    assert len(pat.calls) == 1
    assert pat.result.length_updates == [(16, 16)]
    assert selector.stats()["attention_kernel_incremental_metadata_reuses"] == 1


def test_pat_plan_updates_only_dynamic_kv_lengths() -> None:
    destination = torch.empty(2, dtype=torch.int32)
    base = torch.tensor([100, 200], dtype=torch.int32)
    host = torch.empty_like(base)
    host_block_table = torch.tensor(
        [[0, 1, 0, 0], [0, 1, 0, 0]],
        dtype=torch.int32,
    )
    block_table = torch.empty_like(host_block_table)
    plan = PAPPATPlan(
        q_tables=(),
        block_tables=(block_table,),
        num_seqs_per_ctas=(),
        cta_ranks=(),
        kv_in_ctas=(destination,),
        mnws=(),
        num_split_per_seq=torch.empty(0, dtype=torch.int32),
        max_split_per_seq=0,
        max_seqs_in_cta=0,
        max_blocks_in_cta=0,
        output=torch.empty(0),
        scale=1.0,
        reused_kv_tokens=16,
        base_seq_lens=(32, 32),
        block_size=16,
        base_kv_in_ctas=(base,),
        host_kv_in_ctas=(host,),
        host_block_tables=(host_block_table,),
        kv_in_cta_deltas=(
            (torch.tensor([1, 0], dtype=torch.int32),),
            (torch.tensor([0, 1], dtype=torch.int32),),
        ),
        request_tails=(
            pat._PATTail(0, 0, 2, 2, True),
            pat._PATTail(0, 1, 2, 2, True),
        ),
        incremental_updates_supported=True,
    )

    states = (
        SimpleNamespace(block_ids=(0, 1, 2), block_size=16),
        SimpleNamespace(block_ids=(0, 1, 3), block_size=16),
    )
    assert plan.update_decode_state(states, (32, 33))
    assert destination.tolist() == [100, 201]
    assert plan.update_decode_state(states, (33, 33))
    assert block_table[0, 2].item() == 2
    assert replace(plan).graph_key != plan.graph_key


def test_pat_reuses_tree_when_private_decode_tails_cross_pages(monkeypatch) -> None:
    pytest.importorskip("prefix_attn")
    monkeypatch.setenv("DISABLE_STREAM", "1")
    storage = torch.empty(1)
    states = tuple(
        SimpleNamespace(
            kv_cache=storage,
            block_ids=(0, 1, *range(100 * index, 100 * index + 8)),
            block_size=16,
        )
        for index in range(1, 7)
    )
    selector = PAPPATOrTritonSelector(pat.PAPPATPlanner())
    initial_seq_lens = (50, 53, 55, 58, 61, 63)

    first = _selector_plan(
        selector,
        step_signature=("step-0",),
        states=states,
        seq_lens=initial_seq_lens,
    )
    assert first is not None
    for step in range(1, 33):
        seq_lens = tuple(seq_len + step for seq_len in initial_seq_lens)
        current = _selector_plan(
            selector,
            step_signature=(f"step-{step}",),
            states=states,
            seq_lens=seq_lens,
        )
        assert current is first
        for request_index, (state, seq_len) in enumerate(
            zip(states, seq_lens, strict=True)
        ):
            actual_blocks: list[int] = []
            actual_tokens = 0
            for q_table, block_table, num_seqs, kv_in_cta in zip(
                current.q_tables,
                current.block_tables,
                current.num_seqs_per_ctas,
                current.kv_in_ctas,
                strict=True,
            ):
                for row in range(int(q_table.shape[0])):
                    active_queries = q_table[row, : int(num_seqs[row])].tolist()
                    if request_index not in active_queries:
                        continue
                    kv_tokens = int(kv_in_cta[row])
                    actual_tokens += kv_tokens
                    block_count = math.ceil(kv_tokens / 16)
                    actual_blocks.extend(map(int, block_table[row, :block_count]))
            expected_blocks = state.block_ids[: math.ceil(seq_len / 16)]
            assert actual_tokens == seq_len
            assert sorted(actual_blocks) == sorted(expected_blocks)

    assert selector.stats()["attention_kernel_pat_rebuilds"] == 1
    assert selector.stats()["attention_kernel_incremental_metadata_reuses"] == 32


def test_attention_selector_uses_pat_for_pairwise_prefix_reuse() -> None:
    storage = torch.empty(1)
    states = (
        SimpleNamespace(kv_cache=storage, block_ids=(0, 1), block_size=16),
        SimpleNamespace(kv_cache=storage, block_ids=(0, 2), block_size=16),
        SimpleNamespace(kv_cache=storage, block_ids=(3, 4), block_size=16),
    )
    pat = _FakePATPlanner()
    selector = PAPPATOrTritonSelector(pat)

    result = _selector_plan(
        selector,
        step_signature=("pairwise-prefix",),
        states=states,
        seq_lens=(32, 32, 32),
    )

    assert result is pat.result
    assert pat.calls[0]["reused_kv_tokens"] == 16
    assert selector.stats()["attention_kernel_pat_rebuilds"] == 1


def test_attention_selector_uses_triton_without_physical_reuse() -> None:
    storage = torch.empty(1)
    states = (
        SimpleNamespace(kv_cache=storage, block_ids=(0,), block_size=16),
        SimpleNamespace(kv_cache=storage, block_ids=(1,), block_size=16),
    )
    pat = _FakePATPlanner()
    selector = PAPPATOrTritonSelector(pat)

    result = _selector_plan(
        selector,
        step_signature=("unique",),
        states=states,
        seq_lens=(16, 16),
    )

    assert result is None
    assert not pat.calls
    assert selector.stats()["attention_kernel_triton_selections"] == 1


def test_decode_dispatch_uses_attention_plan(monkeypatch) -> None:
    expected = torch.empty(1)

    class Plan:
        def run_attention(self, query, key_cache, value_cache):
            return expected

    def fail_triton(**kwargs):
        raise AssertionError("Triton fallback must not run")

    monkeypatch.setattr(dispatch, "run_triton_paged_decode_attention", fail_triton)
    actual = run_pap_decode_attention(
        attention_plan=Plan(),
        query=torch.empty(1),
        key_cache=torch.empty(1),
        value_cache=torch.empty(1),
        metadata=None,
        workspace=None,
        scale=1.0,
        block_size=16,
    )

    assert actual is expected


def test_decode_dispatch_uses_triton_fallback(monkeypatch) -> None:
    expected = torch.empty(1)
    received = {}

    def fake_triton(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(dispatch, "run_triton_paged_decode_attention", fake_triton)
    actual = run_pap_decode_attention(
        attention_plan=None,
        query=torch.empty(1),
        key_cache=torch.empty(1),
        value_cache=torch.empty(1),
        metadata="metadata",
        workspace="workspace",
        scale=0.125,
        block_size=16,
    )

    assert actual is expected
    assert received["metadata"] == "metadata"
    assert received["workspace"] == "workspace"


def test_paged_decode_kernel_config_is_low_sm_specific() -> None:
    assert (
        paged_decode_kernel_config_for_sms(12) is PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    )
    assert (
        paged_decode_kernel_config_for_sms(20) is PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    )
    assert paged_decode_kernel_config_for_sms(21) is PAP_TRITON_DECODE_DEFAULT_CONFIG


def test_paged_decode_workspace_cache_reuses_shape() -> None:
    cache = PAPPagedDecodeWorkspaceCache(max_entries=2)
    first_query = torch.empty((2, 4, 8))
    second_query = torch.empty((2, 4, 8))

    first = cache.get(first_query)
    second = cache.get(second_query)

    assert second is first


def test_paged_decode_workspace_cache_is_bounded() -> None:
    cache = PAPPagedDecodeWorkspaceCache(max_entries=2)
    first = cache.get(torch.empty((1, 4, 8)))
    cache.get(torch.empty((2, 4, 8)))
    cache.get(torch.empty((3, 4, 8)))

    assert cache.get(torch.empty((1, 4, 8))) is not first


def test_step_tensor_cache_reuses_shape_and_updates_values() -> None:
    cache = PAPAttentionStepTensorCache()

    first = cache.copy(
        kind="seq_lens",
        values=(10, 20),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )
    second = cache.copy(
        kind="seq_lens",
        values=(11, 21),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    assert first is second
    assert second.tolist() == [11, 21]


def test_step_tensor_cache_is_bounded() -> None:
    cache = PAPAttentionStepTensorCache(max_entries=2)

    for size in (1, 2, 3):
        cache.copy(
            kind="slots",
            values=tuple(range(size)),
            dtype=torch.int64,
            device=torch.device("cpu"),
        )

    assert len(cache._entries) == 2


def _decode_inputs(
    kernel_config=PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    PAPPagedFlashMetadata,
    PAPPagedDecodeWorkspace,
]:
    query = torch.empty((1, 4, 8))
    kv_cache = torch.empty((1, 1, 1, 8))
    metadata = PAPPagedFlashMetadata(
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        cu_seqlens_q=torch.arange(2, dtype=torch.int32),
        max_seq_len=1,
    )
    workspace = PAPPagedDecodeWorkspace(
        output=torch.empty_like(query),
        partial=torch.empty((1, 4, kernel_config.num_splits, 9)),
        lse=torch.empty((1, 4)),
        k_scale=torch.ones(()),
        v_scale=torch.ones(()),
        batch_size=1,
        num_heads=4,
        head_dim=8,
        dtype=query.dtype,
        device=query.device,
        kernel_config=kernel_config,
    )
    return query, kv_cache, metadata, workspace


def test_low_sm_decode_uses_pap_launch_specialization(monkeypatch) -> None:
    query, kv_cache, metadata, workspace = _decode_inputs()
    calls = []

    def fake_launch(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(kernels, "_run_grouped_paged_decode_attention", fake_launch)

    output = run_triton_paged_decode_attention(
        query=query,
        key_cache=kv_cache,
        value_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        scale=0.125,
        block_size=1,
    )

    assert output is workspace.output
    assert len(calls) == 1
    assert calls[0]["workspace"].kernel_config is (
        PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    )


def test_low_sm_launch_preserves_tuned_triton_parameters(monkeypatch) -> None:
    from vllm.v1.attention.ops import triton_decode_attention as decode_ops

    query, kv_cache, metadata, workspace = _decode_inputs()
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs) -> None:
                launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(decode_ops, "is_hip_", False)
    monkeypatch.setattr(decode_ops, "_fwd_grouped_kernel_stage1", FakeKernel())
    monkeypatch.setattr(decode_ops, "_page_stride", lambda *_args: 8)
    monkeypatch.setattr(
        decode_ops,
        "_decode_softmax_reducev_fwd",
        lambda *_args: None,
    )

    kernels._run_grouped_paged_decode_attention(
        query=query,
        key_cache=kv_cache,
        value_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        scale=0.125,
        block_size=1,
    )

    assert len(launches) == 1
    _grid, _args, launch_options = launches[0]
    assert launch_options["BLOCK_H"] == 4
    assert launch_options["NUM_KV_SPLITS"] == 8
    assert launch_options["num_warps"] == 4
    assert launch_options["num_stages"] == 1


def test_default_decode_uses_v026_public_abi(monkeypatch) -> None:
    query, kv_cache, metadata, workspace = _decode_inputs(
        PAP_TRITON_DECODE_DEFAULT_CONFIG
    )
    calls = []

    def fake_decode(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "vllm.v1.attention.ops.triton_decode_attention.decode_attention_fwd",
        fake_decode,
    )

    run_triton_paged_decode_attention(
        query=query,
        key_cache=kv_cache,
        value_cache=kv_cache,
        metadata=metadata,
        workspace=workspace,
        scale=0.125,
        block_size=1,
    )

    assert len(calls) == 1
    assert calls[0][1] == {
        "page_size": 1,
        "k_scale": workspace.k_scale,
        "v_scale": workspace.v_scale,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen3_gqa_paged_decode_matches_reference() -> None:
    """Check the low-SM PAP kernel on the Qwen3 attention shape."""
    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    dtype = torch.float16
    batch_size = 3
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    block_size = 16
    seq_lens = (17, 33, 49)
    block_rows = (
        (7, 1, 6, 4),
        (10, 3, 5, 8),
        (9, 0, 11, 2),
    )
    query = torch.randn(
        (batch_size, num_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    key_cache = torch.randn(
        (12, block_size, num_kv_heads, head_dim),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor(block_rows, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    metadata = PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens_tensor,
        cu_seqlens_q=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        max_seq_len=max(seq_lens),
    )
    config = PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    workspace = PAPPagedDecodeWorkspace(
        output=torch.empty_like(query),
        partial=torch.empty(
            (batch_size, num_heads, config.num_splits, head_dim + 1),
            dtype=torch.float32,
            device=device,
        ),
        lse=torch.empty((batch_size, num_heads), dtype=torch.float32, device=device),
        k_scale=torch.ones((), dtype=torch.float32, device=device),
        v_scale=torch.ones((), dtype=torch.float32, device=device),
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        kernel_config=config,
    )

    actual = run_triton_paged_decode_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        metadata=metadata,
        workspace=workspace,
        scale=1.0 / math.sqrt(head_dim),
        block_size=block_size,
    )

    references = []
    repeats = num_heads // num_kv_heads
    for batch_index, seq_len in enumerate(seq_lens):
        block_count = math.ceil(seq_len / block_size)
        block_ids = block_table[batch_index, :block_count].long()
        keys = key_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        values = value_cache[block_ids].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        keys = keys.repeat_interleave(repeats, dim=1).float()
        values = values.repeat_interleave(repeats, dim=1).float()
        scores = torch.einsum("hd,thd->ht", query[batch_index].float(), keys)
        probabilities = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
        references.append(torch.einsum("ht,thd->hd", probabilities, values))
    reference = torch.stack(references).to(dtype)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
