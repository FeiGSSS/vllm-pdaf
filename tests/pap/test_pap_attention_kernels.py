# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

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
