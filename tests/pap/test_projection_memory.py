# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm.pap.model import memory as projection_memory
from vllm.pap.model.memory import (
    discover_model_weight_bytes,
    plan_projection_memory,
    query_smallest_gpu_total_bytes,
)


def test_discovers_weight_size_from_safetensors_index(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 16_381_470_720}}),
        encoding="utf-8",
    )

    assert discover_model_weight_bytes(tmp_path) == 16_381_470_720


def test_discovers_weight_size_from_local_shards(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"a" * 7)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"b" * 11)

    assert discover_model_weight_bytes(tmp_path) == 18


def test_projection_budget_reserves_120_percent_per_tp_rank() -> None:
    budget = plan_projection_memory(
        model_weight_bytes=16_381_470_720,
        tensor_parallel_size=1,
        gpu_total_bytes=46_068 * 1024 * 1024,
    )

    assert budget.target_bytes == 19_657_764_864
    assert budget.utilization == 0.407


def test_projection_budget_rounds_up_and_accounts_for_tp() -> None:
    budget = plan_projection_memory(
        model_weight_bytes=101,
        tensor_parallel_size=2,
        gpu_total_bytes=1000,
    )

    assert budget.per_rank_weight_bytes == 51
    assert budget.target_bytes == 62
    assert budget.utilization == 0.062


def test_projection_budget_rejects_model_larger_than_gpu() -> None:
    with pytest.raises(ValueError, match="exceed the smallest selected GPU"):
        plan_projection_memory(
            model_weight_bytes=1000,
            tensor_parallel_size=1,
            gpu_total_bytes=1100,
        )


def test_gpu_query_uses_smallest_projection_device(monkeypatch) -> None:
    totals = {"2": "46068\n", "3": "49140\n"}

    def fake_run(command, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        return projection_memory.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=totals[command[2]],
            stderr="",
        )

    monkeypatch.setattr(projection_memory.subprocess, "run", fake_run)

    assert query_smallest_gpu_total_bytes(["2", "3"]) == 46068 * 1024 * 1024
