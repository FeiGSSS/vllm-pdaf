# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.pap.remote_attention import (
    combine_segmented_attention_partial_states,
    compute_attention_output,
    compute_segmented_attention_output,
    compute_segmented_attention_partial_state,
    deserialize_attention_result,
    deserialize_compact_attention_batch,
    deserialize_compact_attention_response,
    deserialize_compact_offload_exec_ack,
    deserialize_compact_offload_exec_batch_command,
    deserialize_compact_offload_exec_command,
    deserialize_tensor_bundle,
    gather_paged_kv,
    paged_kv_segments,
    serialize_attention_result,
    serialize_compact_attention_batch,
    serialize_compact_attention_response,
    serialize_compact_offload_exec_ack,
    serialize_compact_offload_exec_batch_command,
    serialize_compact_offload_exec_command,
    serialize_tensor_bundle,
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


def test_compute_segmented_attention_output_matches_full_kv() -> None:
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

    segmented = compute_segmented_attention_output(
        query=query,
        segments=[(key[:2], value[:2]), (key[2:], value[2:])],
        scale=1 / math.sqrt(2),
    )
    full = compute_attention_output(
        query=query,
        key=key,
        value=value,
        scale=1 / math.sqrt(2),
    )

    assert torch.allclose(segmented, full)


def test_segmented_attention_partial_states_combine_to_full_attention() -> None:
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

    prev_state = compute_segmented_attention_partial_state(
        query=query,
        segments=[(key[:2], value[:2])],
        scale=1 / math.sqrt(2),
    )
    current_state = compute_segmented_attention_partial_state(
        query=query,
        segments=[(key[2:], value[2:])],
        scale=1 / math.sqrt(2),
    )

    combined = combine_segmented_attention_partial_states([prev_state, current_state])
    full = compute_attention_output(
        query=query,
        key=key,
        value=value,
        scale=1 / math.sqrt(2),
    )

    assert torch.allclose(combined, full)


def test_compute_segmented_attention_output_honors_segmented_flag(
    monkeypatch,
) -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    monkeypatch.setenv("PAP_REMOTE_ATTENTION_CUDA_SEGMENTED", "1")

    segmented = compute_segmented_attention_output(
        query=query,
        segments=[(key[:1], value[:1]), (key[1:], value[1:])],
        scale=1.0,
    )
    full = compute_attention_output(
        query=query,
        key=key,
        value=value,
        scale=1.0,
    )

    assert torch.allclose(segmented, full)


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


def test_paged_kv_segments_match_gathered_kv_and_share_storage() -> None:
    kv_cache = torch.zeros((2, 2, 4, 2, 2))
    for block in range(2):
        for offset in range(4):
            kv_cache[0, block, offset] = block * 100 + offset * 10 + 1
            kv_cache[1, block, offset] = block * 100 + offset * 10 + 2

    gathered_key, gathered_value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_len=5,
        num_kv_heads=2,
        layout="NHD",
    )

    segments = paged_kv_segments(
        kv_cache=kv_cache,
        block_ids=[0, 1],
        seq_len=5,
        num_kv_heads=2,
        layout="NHD",
    )

    assert torch.equal(torch.cat([key for key, _ in segments], dim=0), gathered_key)
    assert torch.equal(
        torch.cat([value for _, value in segments], dim=0), gathered_value
    )
    assert (
        segments[0][0].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )
    assert (
        segments[0][1].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )


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


def test_tensor_bundle_round_trips_multiple_tensors() -> None:
    metadata = {"request_id": "cmpl-1", "seq_len": 7}
    query = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    key = torch.arange(4, dtype=torch.bfloat16).reshape(1, 2, 2)

    decoded_metadata, tensors = deserialize_tensor_bundle(
        serialize_tensor_bundle(metadata, {"query": query, "key": key})
    )

    assert decoded_metadata == metadata
    assert torch.equal(tensors["query"], query)
    assert torch.equal(tensors["key"], key)


def test_compact_attention_batch_round_trips() -> None:
    qkv = torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]], dtype=torch.float32)
    payload = serialize_compact_attention_batch(
        [
            {
                "request_id": "cmpl-req",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
                "block_id": 4,
                "slot": 64,
                "seq_len": 1,
                "q_size": 2,
                "kv_size": 2,
                "num_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 2,
            }
        ],
        [qkv],
    )

    items, qkv_tensors = deserialize_compact_attention_batch(payload)

    assert items[0]["request_id"] == "cmpl-req"
    assert items[0]["block_id"] == 4
    assert items[0]["q_size"] == 2
    assert torch.equal(qkv_tensors[0], qkv)

    response = serialize_compact_attention_response(
        [torch.tensor([[[3.0, 5.0]]], dtype=torch.float32)]
    )

    assert torch.equal(
        deserialize_compact_attention_response(response)[0],
        torch.tensor([[[3.0, 5.0]]], dtype=torch.float32),
    )


def test_compact_offload_exec_command_round_trips() -> None:
    payload = serialize_compact_offload_exec_command(
        request_id="req-1",
        layer_name="model.layers.0.self_attn.attn",
        step=9,
        scale=0.5,
        remote_address="127.0.0.1:11300",
    )

    assert deserialize_compact_offload_exec_command(payload) == {
        "request_id": "req-1",
        "layer_name": "model.layers.0.self_attn.attn",
        "step": 9,
        "scale": 0.5,
        "remote_address": "127.0.0.1:11300",
    }
    deserialize_compact_offload_exec_ack(serialize_compact_offload_exec_ack())


def test_compact_offload_exec_batch_command_round_trips() -> None:
    payload = serialize_compact_offload_exec_batch_command(
        layer_name="model.layers.0.self_attn.attn",
        remote_address="127.0.0.1:11300",
        items=[
            {"request_id": "req-1", "step": 9, "scale": 0.5},
            {"request_id": "req-2", "step": 11, "scale": 0.25},
        ],
    )

    assert deserialize_compact_offload_exec_batch_command(payload) == {
        "layer_name": "model.layers.0.self_attn.attn",
        "remote_address": "127.0.0.1:11300",
        "items": [
            {"request_id": "req-1", "step": 9, "scale": 0.5},
            {"request_id": "req-2", "step": 11, "scale": 0.25},
        ],
    }
