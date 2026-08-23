# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.pap.kv.ipc import _resolve_ipc_gpu_uuid


def test_cuda_ipc_uuid_accepts_nvml_prefix_difference() -> None:
    uuid = "5954ae35-b1fa-1499-e9f2-634b5517efa6"

    assert _resolve_ipc_gpu_uuid(uuid, [f"GPU-{uuid}"]) == f"GPU-{uuid}"
    assert _resolve_ipc_gpu_uuid(f"GPU-{uuid}", [uuid]) == uuid


def test_cuda_ipc_uuid_rejects_different_device() -> None:
    assert _resolve_ipc_gpu_uuid("aaaa", ["GPU-bbbb"]) is None
