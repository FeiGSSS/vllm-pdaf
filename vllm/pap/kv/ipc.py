# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA IPC open helpers for sealed Prefill KV handoff."""

from __future__ import annotations

from typing import Any

import torch

from vllm.pap.protocol import (
    PAPCudaIPCTensorHandle,
    PAPPrefillKVSessionManifest,
)


def _normalize_gpu_uuid(value: object) -> str:
    return str(value).removeprefix("GPU-").lower()


def _resolve_ipc_gpu_uuid(
    physical_gpu_id: object,
    available_gpu_ids: list[str],
) -> str | None:
    normalized = _normalize_gpu_uuid(physical_gpu_id)
    return next(
        (
            gpu_id
            for gpu_id in available_gpu_ids
            if _normalize_gpu_uuid(gpu_id) == normalized
        ),
        None,
    )


def open_ipc_tensor_handle(handle: PAPCudaIPCTensorHandle) -> torch.Tensor:
    """Open one CUDA IPC tensor handle on the current physical GPU."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    device_index = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device_index)
    physical_gpu_id = str(props.uuid)
    ipc_handle = handle.ipc_handle
    ipc_gpu_id = _resolve_ipc_gpu_uuid(physical_gpu_id, list(ipc_handle))
    if ipc_gpu_id is None:
        raise ValueError(
            f"IPC handle not found for GPU UUID {physical_gpu_id}. "
            f"Available UUIDs: {list(ipc_handle.keys())}"
        )
    args = list(ipc_handle[ipc_gpu_id])
    args[6] = device_index
    return rebuild_cuda_tensor(*args)


def open_prefill_manifest_event(
    manifest: PAPPrefillKVSessionManifest,
) -> Any | None:
    """Open an interprocess CUDA event carried by a Prefill manifest."""
    if manifest.ready_event_handle is None:
        return None
    device_index = torch.accelerator.current_device_index()
    return torch.cuda.Event.from_ipc_handle(
        device_index,
        manifest.ready_event_handle,
    )
