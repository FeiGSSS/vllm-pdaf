# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.pap.remote_attention import (
    compute_attention_output,
    deserialize_attention_result,
    gather_paged_kv,
    serialize_attention_result,
)


def test_compute_attention_output_matches_manual_gqa() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]])
    key = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )
    value = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ]
    )

    output = compute_attention_output(
        query=query,
        key=key,
        value=value,
        scale=1 / math.sqrt(2),
    )

    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    scores = torch.einsum("qhd,khd->qhk", query, expanded_key) / math.sqrt(2)
    expected = torch.einsum(
        "qhk,khd->qhd", torch.softmax(scores, dim=-1), expanded_value
    )
    assert torch.allclose(output, expected)


def test_gather_paged_kv_supports_nhd_layout() -> None:
    kv_cache = torch.zeros((2, 2, 4, 2, 2))
    for block in range(2):
        for offset in range(4):
            kv_cache[0, block, offset] = block * 100 + offset * 10 + 1
            kv_cache[1, block, offset] = block * 100 + offset * 10 + 2

    key, value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_len=5,
        num_kv_heads=2,
        layout="NHD",
    )

    assert key.shape == (5, 2, 2)
    assert value.shape == (5, 2, 2)
    assert torch.equal(key[0], torch.full((2, 2), 1.0))
    assert torch.equal(key[4], torch.full((2, 2), 101.0))
    assert torch.equal(value[4], torch.full((2, 2), 102.0))


def test_gather_paged_kv_supports_hnd_layout() -> None:
    kv_cache = torch.zeros((2, 2, 2, 4, 2))
    for block in range(2):
        for offset in range(4):
            kv_cache[0, block, :, offset] = block * 100 + offset * 10 + 1
            kv_cache[1, block, :, offset] = block * 100 + offset * 10 + 2

    key, value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_len=5,
        num_kv_heads=2,
        layout="HND",
    )

    assert key.shape == (5, 2, 2)
    assert value.shape == (5, 2, 2)
    assert torch.equal(key[3], torch.full((2, 2), 31.0))
    assert torch.equal(key[4], torch.full((2, 2), 101.0))
    assert torch.equal(value[4], torch.full((2, 2), 102.0))


def test_gather_paged_kv_supports_logical_shape_with_hnd_strides() -> None:
    physical = torch.zeros((2, 2, 2, 4, 2))
    for block in range(2):
        for offset in range(4):
            physical[0, block, :, offset] = block * 100 + offset * 10 + 1
            physical[1, block, :, offset] = block * 100 + offset * 10 + 2
    kv_cache = physical.permute(0, 1, 3, 2, 4)

    key, value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_len=5,
        num_kv_heads=2,
        layout="HND",
    )

    assert kv_cache.shape == (2, 2, 4, 2, 2)
    assert not kv_cache.is_contiguous()
    assert torch.equal(key[3], torch.full((2, 2), 31.0))
    assert torch.equal(key[4], torch.full((2, 2), 101.0))
    assert torch.equal(value[4], torch.full((2, 2), 102.0))


def test_attention_result_round_trips_bfloat16() -> None:
    result = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2).to(torch.bfloat16)

    encoded = serialize_attention_result(result)
    decoded = deserialize_attention_result(encoded)

    assert decoded.dtype == torch.bfloat16
    assert torch.equal(decoded, result.cpu())
