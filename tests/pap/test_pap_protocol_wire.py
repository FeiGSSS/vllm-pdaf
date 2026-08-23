# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.pap.protocol.descriptors import PAPCudaIPCTensorHandle
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
