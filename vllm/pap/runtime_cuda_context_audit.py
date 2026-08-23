# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime CUDA-context audit for PAP diagnostic runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from vllm.platforms import current_platform


def write_runtime_cuda_context_audit(*, role: str) -> dict[str, Any] | None:
    """Write the live CUDA context's visible resources when configured."""

    output = os.environ.get("PAP_RUNTIME_CUDA_CONTEXT_AUDIT_PATH", "").strip()
    if not output:
        return None
    device_index = int(torch.accelerator.current_device_index())
    payload: dict[str, Any] = {
        "pid": os.getpid(),
        "role": str(role),
        "device_index": device_index,
        "device_name": str(current_platform.get_device_name(device_index)),
        "device_uuid": str(current_platform.get_device_uuid(device_index)),
        "multiprocessor_count": int(current_platform.num_compute_units(device_index)),
        "cuda_mps_sm_partition": os.environ.get("CUDA_MPS_SM_PARTITION"),
        "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return payload
