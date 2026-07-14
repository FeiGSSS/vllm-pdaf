# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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
