# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for the PAP sealed KV wire codec."""

from vllm.pap.protocol.wire import (
    deserialize_tensor_bundle,
    serialize_tensor_bundle,
)

__all__ = ["deserialize_tensor_bundle", "serialize_tensor_bundle"]
