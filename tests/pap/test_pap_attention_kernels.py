# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

import vllm.pap.attention.kernels as kernels
from vllm.pap.attention.kernels import (
    PAP_TRITON_DECODE_DEFAULT_CONFIG,
    PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG,
    PAPAttentionStepTensorCache,
    PAPPagedDecodeWorkspace,
    PAPPagedDecodeWorkspaceCache,
    paged_decode_kernel_config_for_sms,
    run_paged_decode_attention,
)
from vllm.pap.kv.metadata import PAPPagedFlashMetadata


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

    output = run_paged_decode_attention(
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

    run_paged_decode_attention(
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

    actual = run_paged_decode_attention(
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
