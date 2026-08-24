# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan the KV-unaware PAP Projection runtime memory budget."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

_MIB = 1024 * 1024
_UTILIZATION_SCALE = 10_000
_HEADROOM_NUMERATOR = 6
_HEADROOM_DENOMINATOR = 5
_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_WEIGHT_PATTERNS = ("model*.safetensors", "pytorch_model*.bin")


@dataclass(frozen=True, slots=True)
class ProjectionMemoryBudget:
    """Computed memory budget for one Projection replica."""

    utilization: float
    model_weight_bytes: int
    per_rank_weight_bytes: int
    validation_kv_bytes: int
    target_bytes: int
    gpu_total_bytes: int


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def discover_model_weight_bytes(model_path: Path) -> int:
    """Return checkpoint weight bytes for a local Hugging Face model."""
    for index_name in _INDEX_NAMES:
        index_path = model_path / index_name
        if not index_path.is_file():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        total_size = payload.get("metadata", {}).get("total_size")
        if total_size is not None and int(total_size) > 0:
            return int(total_size)
        weight_map = payload.get("weight_map", {})
        shard_paths = {model_path / name for name in weight_map.values()}
        if shard_paths and all(path.is_file() for path in shard_paths):
            return sum(path.stat().st_size for path in shard_paths)

    for pattern in _WEIGHT_PATTERNS:
        weight_paths = sorted(model_path.glob(pattern))
        if weight_paths:
            return sum(path.stat().st_size for path in weight_paths)

    raise ValueError(
        f"cannot determine model weight size below {model_path}; "
        "expected a safetensors or PyTorch weight index"
    )


def query_smallest_gpu_total_bytes(
    gpu_ids: list[str],
    *,
    nvidia_smi: str = "nvidia-smi",
) -> int:
    """Return total bytes of the smallest selected Projection GPU."""
    if not gpu_ids:
        raise ValueError("at least one Projection GPU is required")
    totals_mib: list[int] = []
    for gpu_id in gpu_ids:
        result = subprocess.run(
            [
                nvidia_smi,
                "-i",
                gpu_id,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(
                f"invalid total-memory result for Projection GPU {gpu_id}: {value!r}"
            )
        totals_mib.append(int(value))
    return min(totals_mib) * _MIB


def plan_projection_memory(
    *,
    model_weight_bytes: int,
    tensor_parallel_size: int,
    gpu_total_bytes: int,
    validation_kv_bytes: int = 0,
    runtime_headroom_bytes: int = 0,
) -> ProjectionMemoryBudget:
    """Reserve model headroom plus vLLM's temporary KV capacity check."""
    if model_weight_bytes <= 0:
        raise ValueError("model weight size must be positive")
    if tensor_parallel_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    if gpu_total_bytes <= 0:
        raise ValueError("GPU total memory must be positive")
    if validation_kv_bytes < 0:
        raise ValueError("Projection validation KV bytes must be non-negative")
    if runtime_headroom_bytes < 0:
        raise ValueError("Projection runtime headroom must be non-negative")

    per_rank_weight_bytes = _ceil_div(model_weight_bytes, tensor_parallel_size)
    model_target_bytes = _ceil_div(
        per_rank_weight_bytes * _HEADROOM_NUMERATOR,
        _HEADROOM_DENOMINATOR,
    )
    target_bytes = model_target_bytes + validation_kv_bytes + runtime_headroom_bytes
    utilization_steps = _ceil_div(
        target_bytes * _UTILIZATION_SCALE,
        gpu_total_bytes,
    )
    if utilization_steps > _UTILIZATION_SCALE:
        raise ValueError(
            "Projection model weights plus 20% headroom exceed the smallest "
            "selected GPU"
        )
    return ProjectionMemoryBudget(
        utilization=utilization_steps / _UTILIZATION_SCALE,
        model_weight_bytes=model_weight_bytes,
        per_rank_weight_bytes=per_rank_weight_bytes,
        validation_kv_bytes=validation_kv_bytes,
        target_bytes=target_bytes,
        gpu_total_bytes=gpu_total_bytes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--gpu-id", action="append", required=True)
    parser.add_argument("--validation-kv-bytes", type=int, default=0)
    parser.add_argument("--runtime-headroom-bytes", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_weight_bytes = discover_model_weight_bytes(args.model_path)
    gpu_total_bytes = query_smallest_gpu_total_bytes(args.gpu_id)
    budget = plan_projection_memory(
        model_weight_bytes=model_weight_bytes,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_total_bytes=gpu_total_bytes,
        validation_kv_bytes=args.validation_kv_bytes,
        runtime_headroom_bytes=args.runtime_headroom_bytes,
    )
    print(
        f"{budget.utilization:.4f}",
        budget.model_weight_bytes,
        budget.per_rank_weight_bytes,
        budget.validation_kv_bytes,
        budget.target_bytes,
        budget.gpu_total_bytes,
    )


if __name__ == "__main__":
    main()
