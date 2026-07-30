from __future__ import annotations

import torch

from vllm.pap.attention.kernels import (
    PAPAttentionStepTensorCache,
    PAP_TRITON_DECODE_DEFAULT_CONFIG,
    PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG,
    PAPPagedDecodeWorkspaceCache,
    paged_decode_kernel_config_for_sms,
)


def test_paged_decode_kernel_config_is_low_sm_specific() -> None:
    assert (
        paged_decode_kernel_config_for_sms(12)
        is PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    )
    assert (
        paged_decode_kernel_config_for_sms(20)
        is PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    )
    assert (
        paged_decode_kernel_config_for_sms(21)
        is PAP_TRITON_DECODE_DEFAULT_CONFIG
    )


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
