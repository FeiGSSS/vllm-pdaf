# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from benchmarks.multi_turn.long_context_common import (
    CapacityAdmission,
    TokenAccounting,
    calculate_tpot_s,
    decide_capacity_admission,
    parse_kv_capacities,
    stable_cell_id,
    validate_token_accounting,
)


COMMON_MODULE = (
    Path(__file__).parents[2]
    / "benchmarks"
    / "multi_turn"
    / "long_context_common.py"
)


def test_parse_kv_capacities_accepts_stable_lines_from_multiple_logs(
    tmp_path: Path,
) -> None:
    prefill_log = tmp_path / "prefill.log"
    decode_log = tmp_path / "decode.log"
    prefill_log.write_text(
        "INFO startup\nGPU KV cache size: 145,632 tokens\n",
        encoding="utf-8",
    )
    decode_log.write_text(
        "INFO GPU KV cache size: 98,304 tokens\n",
        encoding="utf-8",
    )

    capacities = parse_kv_capacities(
        {"prefill": prefill_log, "decode": decode_log}
    )

    assert capacities == {"prefill": 145_632, "decode": 98_304}


def test_parse_kv_capacities_accepts_duplicate_consistent_lines(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.log"
    service_log.write_text(
        "GPU KV cache size: 145,632 tokens\n"
        "GPU KV cache size: 145,632 tokens\n",
        encoding="utf-8",
    )

    assert parse_kv_capacities({"service": service_log}) == {
        "service": 145_632
    }


@pytest.mark.parametrize(
    "contents",
    [
        "INFO startup completed without a capacity line\n",
        "Available GPU KV cache size: unknown\n",
        "GPU KV cache size: 0 tokens\n",
        "GPU KV cache size: -1 tokens\n",
    ],
)
def test_parse_kv_capacities_rejects_missing_or_non_positive_capacity(
    tmp_path: Path, contents: str
) -> None:
    service_log = tmp_path / "service.log"
    service_log.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="service"):
        parse_kv_capacities({"service": service_log})


def test_parse_kv_capacities_rejects_conflicting_lines(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.log"
    service_log.write_text(
        "GPU KV cache size: 145,632 tokens\n"
        "GPU KV cache size: 98,304 tokens\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting.*service"):
        parse_kv_capacities({"service": service_log})


def test_parse_kv_capacities_rejects_an_empty_log_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        parse_kv_capacities({})


def test_capacity_admission_uses_required_service_minimum_at_boundary() -> None:
    admission = decide_capacity_admission(
        capacities={"prefill": 1_000, "decode": 900, "unused": 100},
        required_services=["prefill", "decode"],
        active_conversations=3,
        max_rendered_context_tokens=210,
    )

    assert isinstance(admission, CapacityAdmission)
    assert admission.to_dict() == {
        "schema_version": 1,
        "reported_capacity_tokens_by_service": {
            "prefill": 1_000,
            "decode": 900,
            "unused": 100,
        },
        "required_services": ["prefill", "decode"],
        "usable_kv_token_capacity": 900,
        "active_conversations": 3,
        "max_rendered_context_tokens_per_conversation": 210,
        "required_live_tokens": 630,
        "safety_fraction": 0.7,
        "budget_tokens": 630,
        "decision": "admitted",
        "reason": admission.reason,
    }
    assert "630" in admission.reason


def test_capacity_admission_rejects_one_token_over_budget() -> None:
    admission = decide_capacity_admission(
        capacities={"prefill": 900},
        required_services=["prefill"],
        active_conversations=1,
        max_rendered_context_tokens=631,
    )

    assert admission.required_live_tokens == 631
    assert admission.budget_tokens == 630
    assert admission.decision == "admission-limited"
    assert "631" in admission.reason
    assert "630" in admission.reason


def test_capacity_admission_floors_decimal_safety_budget() -> None:
    admission = decide_capacity_admission(
        capacities={"prefill": 101},
        required_services=["prefill"],
        active_conversations=1,
        max_rendered_context_tokens=70,
        safety_fraction=0.70,
    )

    assert admission.budget_tokens == 70
    assert admission.decision == "admitted"


@pytest.mark.parametrize(
    (
        "capacities",
        "required_services",
        "active_conversations",
        "max_context_tokens",
        "safety_fraction",
    ),
    [
        ({"prefill": 100}, [], 1, 1, 0.7),
        ({"prefill": 100}, ["decode"], 1, 1, 0.7),
        ({"prefill": 0}, ["prefill"], 1, 1, 0.7),
        ({"prefill": -1}, ["prefill"], 1, 1, 0.7),
        ({"prefill": 100}, ["prefill"], 0, 1, 0.7),
        ({"prefill": 100}, ["prefill"], -1, 1, 0.7),
        ({"prefill": 100}, ["prefill"], 1, 0, 0.7),
        ({"prefill": 100}, ["prefill"], 1, -1, 0.7),
        ({"prefill": 100}, ["prefill"], 1, 1, 0.0),
        ({"prefill": 100}, ["prefill"], 1, 1, -0.1),
        ({"prefill": 100}, ["prefill"], 1, 1, 1.01),
        ({"prefill": 100}, ["prefill"], 1, 1, float("nan")),
    ],
)
def test_capacity_admission_rejects_invalid_arguments(
    capacities: dict[str, int],
    required_services: list[str],
    active_conversations: int,
    max_context_tokens: int,
    safety_fraction: float,
) -> None:
    with pytest.raises(ValueError):
        decide_capacity_admission(
            capacities=capacities,
            required_services=required_services,
            active_conversations=active_conversations,
            max_rendered_context_tokens=max_context_tokens,
            safety_fraction=safety_fraction,
        )


def test_capacity_admission_is_frozen() -> None:
    admission = decide_capacity_admission(
        capacities={"prefill": 100},
        required_services=["prefill"],
        active_conversations=1,
        max_rendered_context_tokens=1,
    )

    with pytest.raises(FrozenInstanceError):
        admission.decision = "admission-limited"  # type: ignore[misc]


def test_validate_token_accounting_accepts_exclusive_partition() -> None:
    accounting = TokenAccounting(
        prompt_tokens=100,
        local_reused_tokens=40,
        remote_loaded_tokens=30,
        recomputed_tokens=30,
    )

    assert validate_token_accounting(accounting) is None


def test_validate_token_accounting_reports_all_counts_on_mismatch() -> None:
    accounting = TokenAccounting(
        prompt_tokens=100,
        local_reused_tokens=40,
        remote_loaded_tokens=30,
        recomputed_tokens=29,
    )

    with pytest.raises(ValueError) as exc_info:
        validate_token_accounting(accounting)

    message = str(exc_info.value)
    for expected in (
        "prompt_tokens=100",
        "local_reused_tokens=40",
        "remote_loaded_tokens=30",
        "recomputed_tokens=29",
    ):
        assert expected in message


def test_validate_token_accounting_rejects_negative_counts() -> None:
    accounting = TokenAccounting(
        prompt_tokens=10,
        local_reused_tokens=-1,
        remote_loaded_tokens=1,
        recomputed_tokens=10,
    )

    with pytest.raises(ValueError, match="local_reused_tokens=-1"):
        validate_token_accounting(accounting)


def test_token_accounting_is_frozen() -> None:
    accounting = TokenAccounting(
        prompt_tokens=1,
        local_reused_tokens=0,
        remote_loaded_tokens=0,
        recomputed_tokens=1,
    )

    with pytest.raises(FrozenInstanceError):
        accounting.prompt_tokens = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("turn_latency_s", "ttft_s", "output_tokens", "expected"),
    [
        (1.5, 0.4, 1, 1.1),
        (1.5, 0.4, 5, 0.275),
    ],
)
def test_calculate_tpot_uses_actual_output_token_count(
    turn_latency_s: float,
    ttft_s: float,
    output_tokens: int,
    expected: float,
) -> None:
    assert calculate_tpot_s(
        turn_latency_s, ttft_s, output_tokens
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("turn_latency_s", "ttft_s", "output_tokens"),
    [
        (-0.1, 0.0, 1),
        (1.0, -0.1, 1),
        (0.5, 0.6, 1),
        (1.0, 0.1, 0),
        (1.0, 0.1, -1),
    ],
)
def test_calculate_tpot_rejects_invalid_arguments(
    turn_latency_s: float, ttft_s: float, output_tokens: int
) -> None:
    with pytest.raises(ValueError):
        calculate_tpot_s(turn_latency_s, ttft_s, output_tokens)


def test_stable_cell_id_uses_the_frozen_artifact_format() -> None:
    assert stable_cell_id(
        matrix="m2",
        lane="PD-bidir",
        context_tokens=32_768,
        decode_tokens=512,
        rounds=4,
        active_conversations=2,
        repetition=3,
    ) == "MT-m2-PD-bidir-ctx32768-d512-r4-c2-rep3"


def _run_capacity_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(COMMON_MODULE), "capacity", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_capacity_cli_atomically_writes_exact_admitted_json(
    tmp_path: Path,
) -> None:
    prefill_log = tmp_path / "prefill.log"
    decode_log = tmp_path / "decode.log"
    output = tmp_path / "capacity.json"
    prefill_log.write_text(
        "GPU KV cache size: 1,000 tokens\n", encoding="utf-8"
    )
    decode_log.write_text(
        "GPU KV cache size: 900 tokens\n", encoding="utf-8"
    )

    result = _run_capacity_cli(
        "--service-log",
        f"prefill={prefill_log}",
        "--service-log",
        f"decode={decode_log}",
        "--required-service",
        "prefill",
        "--required-service",
        "decode",
        "--active-conversations",
        "2",
        "--max-rendered-context-tokens",
        "300",
        "--safety-fraction",
        "0.70",
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    expected = decide_capacity_admission(
        capacities={"prefill": 1_000, "decode": 900},
        required_services=["prefill", "decode"],
        active_conversations=2,
        max_rendered_context_tokens=300,
    ).to_dict()
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_capacity_cli_writes_rejection_and_returns_nonzero(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.log"
    output = tmp_path / "capacity.json"
    service_log.write_text(
        "GPU KV cache size: 900 tokens\n", encoding="utf-8"
    )

    result = _run_capacity_cli(
        "--service-log",
        f"prefill={service_log}",
        "--required-service",
        "prefill",
        "--active-conversations",
        "1",
        "--max-rendered-context-tokens",
        "631",
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == (
        "admission-limited"
    )


def test_capacity_cli_preserves_existing_output_when_parsing_fails(
    tmp_path: Path,
) -> None:
    service_log = tmp_path / "service.log"
    output = tmp_path / "capacity.json"
    service_log.write_text("startup failed\n", encoding="utf-8")
    output.write_text("sentinel\n", encoding="utf-8")

    result = _run_capacity_cli(
        "--service-log",
        f"prefill={service_log}",
        "--required-service",
        "prefill",
        "--active-conversations",
        "1",
        "--max-rendered-context-tokens",
        "1",
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == "sentinel\n"
