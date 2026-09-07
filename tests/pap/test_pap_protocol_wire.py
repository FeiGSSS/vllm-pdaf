# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.pap.protocol.descriptors import (
    PAPCudaIPCTensorHandle,
    PAPPrefillKVSessionManifest,
)
from vllm.pap.protocol.wire import (
    deserialize_tensor_bundle,
    serialize_tensor_bundle,
)


def test_tensor_bundle_round_trips_metadata_and_tensors() -> None:
    metadata = {"request_id": "cmpl-1", "seq_len": 7}
    query = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    key = torch.arange(4, dtype=torch.bfloat16).reshape(1, 2, 2)

    decoded_metadata, tensors = deserialize_tensor_bundle(
        serialize_tensor_bundle(metadata, {"query": query, "key": key})
    )

    assert decoded_metadata == metadata
    assert torch.equal(tensors["query"], query)
    assert torch.equal(tensors["key"], key)


@pytest.mark.parametrize("payload", [b"", b"header"])
def test_tensor_bundle_rejects_missing_header_length(payload: bytes) -> None:
    with pytest.raises(ValueError, match="header length"):
        deserialize_tensor_bundle(payload)


def test_cuda_ipc_descriptor_round_trips_through_safe_json() -> None:
    ipc_args = (
        torch.Tensor,
        torch.Size((2, 3)),
        (3, 1),
        0,
        torch.storage.TypedStorage,
        torch.bfloat16,
        torch.device("cuda:2"),
        b"memory-handle",
        4096,
        128,
        False,
        b"ref-counter-handle",
        7,
        b"event-handle",
        True,
    )
    descriptor = PAPCudaIPCTensorHandle(
        dtype="bfloat16",
        shape=(2, 3),
        ipc_handle={"GPU-b": ipc_args, "GPU-a": ipc_args},
    )

    wire_data = json.loads(json.dumps(descriptor.to_dict()))
    restored = PAPCudaIPCTensorHandle.from_dict(wire_data)

    assert list(wire_data["ipc_handle"]) == ["GPU-a", "GPU-b"]
    assert restored == descriptor


def test_cuda_ipc_descriptor_rejects_legacy_pickle_payload() -> None:
    with pytest.raises(ValueError, match="pickled CUDA IPC handles"):
        PAPCudaIPCTensorHandle.from_dict(
            {
                "dtype": "bfloat16",
                "shape": [2, 3],
                "ipc_handle_pickled": "malicious-payload",
            }
        )


def test_prefill_manifest_round_trips_generation_and_rejects_aliases() -> None:
    manifest = PAPPrefillKVSessionManifest(
        request_id="req",
        session_handle="session",
        catalog_id="catalog",
        prefix_len=17,
        block_ids=(1, 2),
        block_size=16,
        expected_layer_count=1,
        lease_id="lease",
        leased_block_ids=(1, 2),
        lease_capacity_tokens=32,
        writable_start_token=17,
        writable_end_token=32,
        allocation_limit_token=49,
        generation=3,
    )

    assert PAPPrefillKVSessionManifest.from_dict(manifest.to_dict()) == manifest
    with pytest.raises(ValueError, match="must not alias"):
        PAPPrefillKVSessionManifest.from_dict(
            {**manifest.to_dict(), "block_ids": [1, 1]}
        )
