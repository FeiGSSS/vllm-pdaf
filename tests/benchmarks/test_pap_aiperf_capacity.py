import json
import random
from pathlib import Path

import pytest

from benchmarks.pap.aiperf.generate_multiturn_dataset import (
    TokenLengthDistribution,
    build_delay_schedule,
)
from benchmarks.pap.aiperf.summarize_capacity_matrix import (
    build_envelope,
    build_rows,
    write_markdown,
)
from benchmarks.pap.aiperf.summarize_capacity_run import summarize_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_delay_schedule_mixes_think_and_tool_waits() -> None:
    schedule = build_delay_schedule(
        10,
        think_time_ms=3_000,
        tool_time_ms=1_000,
        tool_every=3,
    )

    assert schedule == [
        0,
        3_000,
        3_000,
        1_000,
        3_000,
        3_000,
        1_000,
        3_000,
        3_000,
        1_000,
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"turns": 0}, "turns must be positive"),
        ({"think_time_ms": -1}, "delays must be non-negative"),
        ({"tool_every": 0}, "tool_every must be positive"),
    ],
)
def test_delay_schedule_rejects_invalid_values(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "turns": 10,
        "think_time_ms": 3_000,
        "tool_time_ms": 1_000,
        "tool_every": 3,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        build_delay_schedule(**values)


def test_lognormal_lengths_use_distinct_mean_and_median() -> None:
    distribution = TokenLengthDistribution(
        mean=32,
        median=30,
        minimum=16,
        maximum=64,
    )
    rng = random.Random(42)
    values = [distribution.sample(rng) for _ in range(1_000)]

    assert min(values) >= 16
    assert max(values) <= 64
    assert len(set(values)) > 1
    assert abs(sum(values) / len(values) - 32) < 2


def test_lognormal_lengths_reject_median_above_mean() -> None:
    with pytest.raises(ValueError, match="median must not exceed mean"):
        TokenLengthDistribution(mean=32, median=33)


def _write_pd_run(
    run_root: Path,
    ttfts: list[float],
    output_tokens: list[int] | None = None,
) -> None:
    output_tokens = output_tokens or [256] * len(ttfts)
    records = []
    for index, (ttft, output_length) in enumerate(
        zip(ttfts, output_tokens, strict=True)
    ):
        records.append(
            {
                "metadata": {
                    "conversation_id": f"session-{index // 2}",
                    "turn_index": index % 2,
                    "was_cancelled": False,
                },
                "metrics": {
                    "time_to_first_token": {"value": ttft},
                    "inter_token_latency": {"value": 40.0},
                    "output_token_count": {"value": output_length},
                },
            }
        )
    records_path = run_root / "aiperf/profile.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(
        run_root / "aiperf/profile.json",
        {
            "request_count": {"avg": 4},
            "request_throughput": {"avg": 2.0},
            "output_token_throughput": {"avg": 512.0},
            "benchmark_duration": {"avg": 2.0},
            "error_summary": [],
            "was_cancelled": False,
        },
    )
    (run_root / "correctness_audit.env").write_text(
        "STATUS=passed\n",
        encoding="utf-8",
    )
    _write_json(
        run_root / "proxy_health.json",
        {
            "status": "ok",
            "prefill_routing": {
                "conversations": 2,
                "assignments": [1, 1],
                "requests": [2, 2],
            },
            "decode_routing": {
                "conversations": 2,
                "assignments": [1, 1],
                "requests": [2, 2],
            },
            "pair_routing": {
                "conversations": 2,
                "labels": ["p0:d0", "p1:d1", "p0:d1", "p1:d0"],
                "assignments": [1, 1, 0, 0],
                "requests": [2, 2, 0, 0],
            },
        },
    )


def _write_dataset(path: Path, output_tokens: list[int]) -> None:
    conversations = []
    for session_index in range(2):
        turns = []
        for output_length in output_tokens[session_index * 2 : (session_index + 1) * 2]:
            turns.append(
                {
                    "text": "prompt",
                    "role": "user",
                    "output_length": output_length,
                    "extra": {
                        "ignore_eos": True,
                        "min_tokens": output_length,
                    },
                }
            )
        conversations.append({"session_id": f"session-{session_index}", "turns": turns})
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in conversations),
        encoding="utf-8",
    )


def test_capacity_summary_applies_three_request_level_slos(tmp_path: Path) -> None:
    _write_pd_run(tmp_path, [1_000.0, 2_000.0, 3_000.0, 6_000.0])

    summary = summarize_run(
        tmp_path,
        architecture="pd",
        topology="2p2d",
        concurrency=2,
        sessions=2,
        turns=2,
        output_tokens=256,
    )

    assert summary["correctness"]["passed"] is True
    assert summary["run_status"]["state"] == "completed"
    assert summary["routing"]["migration_count"] == 0
    assert summary["slo"]["strict"]["good_request_fraction"] == 0.75
    assert summary["slo"]["strict"]["passed"] is False
    assert summary["slo"]["standard"]["passed"] is True
    assert summary["slo"]["relaxed"]["passed"] is True
    assert summary["slo"]["standard"]["goodput_requests_per_second"] == 2.0


def test_capacity_summary_audits_per_request_output_lengths(
    tmp_path: Path,
) -> None:
    output_tokens = [16, 24, 32, 56]
    _write_pd_run(tmp_path, [1_000.0] * 4, output_tokens)
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, output_tokens)

    summary = summarize_run(
        tmp_path,
        architecture="pd",
        topology="2p2d",
        concurrency=2,
        sessions=2,
        turns=2,
        output_tokens=32,
        dataset_file=dataset,
    )

    assert summary["correctness"]["passed"] is True
    assert summary["workload"]["output_tokens_per_turn"] is None
    assert summary["workload"]["requested_output_tokens"]["mean"] == 32


def test_correctness_failure_fails_every_slo(tmp_path: Path) -> None:
    _write_pd_run(tmp_path, [1_000.0] * 4)
    profile_path = tmp_path / "aiperf/profile.jsonl"
    records = [json.loads(line) for line in profile_path.read_text().splitlines()]
    records[0]["metrics"]["output_token_count"]["value"] = 255
    profile_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_run(
        tmp_path,
        architecture="pd",
        topology="2p2d",
        concurrency=2,
        sessions=2,
        turns=2,
        output_tokens=256,
    )

    assert summary["correctness"]["passed"] is False
    assert all(not tier["passed"] for tier in summary["slo"].values())


def test_partial_timeout_records_early_stopped_slo_impossible(
    tmp_path: Path,
) -> None:
    _write_pd_run(tmp_path, [1_000.0] * 4)
    profile_path = tmp_path / "aiperf/profile.jsonl"
    records = [json.loads(line) for line in profile_path.read_text().splitlines()]
    records = records[:3]
    records[-1]["metrics"] = {}
    records[-1]["error"] = {
        "type": "TimeoutError",
        "message": "TimeoutError()",
    }
    profile_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    summary = summarize_run(
        tmp_path,
        architecture="pd",
        topology="2p2d",
        concurrency=2,
        sessions=2,
        turns=2,
        output_tokens=256,
        launcher_exit_code=143,
    )

    status = summary["run_status"]
    assert status["state"] == "early_stopped_slo_impossible"
    assert status["request_error_counts"] == {"TimeoutError": 1}
    assert status["relaxed_slo"] == {
        "observed_bad_requests": 1,
        "maximum_bad_requests": 0,
        "pass_still_possible": False,
    }


def test_matrix_envelope_uses_best_pd_topology() -> None:
    def summary(
        architecture: str,
        topology: str,
        concurrency: int,
        standard: bool,
    ) -> dict[str, object]:
        tiers = {
            name: {
                "passed": standard,
                "good_request_fraction": 1.0,
                "goodput_requests_per_second": 1.0,
            }
            for name in ("strict", "standard", "relaxed")
        }
        return {
            "architecture": architecture,
            "topology": topology,
            "concurrency": concurrency,
            "correctness": {"passed": True},
            "metrics": {
                "ttft_ms": {"p95": 1.0},
                "itl_ms": {"p95": 1.0},
                "request_throughput_per_second": 1.0,
            },
            "slo": tiers,
        }

    rows = build_rows(
        [
            summary("pap", "3pa1p", 24, True),
            summary("pd", "1p3d", 16, True),
            summary("pd", "2p2d", 20, True),
            summary("pd", "3p1d", 24, False),
            summary("dp", "4dp", 18, True),
        ]
    )
    envelope = build_envelope(rows)["capacity_by_slo"]["standard"]

    assert envelope["best_pap"] == {"topology": "3pa1p", "concurrency": 24}
    assert envelope["best_pd"] == {"topology": "2p2d", "concurrency": 20}
    assert envelope["best_dp"] == {"topology": "4dp", "concurrency": 18}
    assert envelope["pap_minus_best_pd"] == 4
    assert envelope["pap_minus_dp"] == 6


def test_matrix_reports_best_compliant_goodput() -> None:
    def summary(
        architecture: str,
        topology: str,
        concurrency: int,
        goodput: float,
        passed: bool = True,
    ) -> dict[str, object]:
        tiers = {
            name: {
                "passed": passed,
                "good_request_fraction": 1.0 if passed else 0.9,
                "goodput_requests_per_second": goodput,
            }
            for name in ("strict", "standard", "relaxed")
        }
        return {
            "architecture": architecture,
            "topology": topology,
            "concurrency": concurrency,
            "correctness": {"passed": True},
            "metrics": {
                "ttft_ms": {"p95": 1.0},
                "itl_ms": {"p95": 1.0},
                "request_throughput_per_second": goodput,
            },
            "slo": tiers,
        }

    rows = build_rows(
        [
            summary("pap", "3pa1p", 12, 2.4),
            summary("pap", "3pa1p", 20, 2.7, passed=False),
            summary("pd", "2p2d", 10, 1.8),
            summary("pd", "3p1d", 8, 1.7),
            summary("dp", "4dp", 12, 2.0),
        ]
    )
    goodput = build_envelope(rows)["compliant_goodput_by_slo"]["standard"]

    assert goodput["best_pap"] == {
        "topology": "3pa1p",
        "concurrency": 12,
        "requests_per_second": 2.4,
    }
    assert goodput["best_pd"] == {
        "topology": "2p2d",
        "concurrency": 10,
        "requests_per_second": 1.8,
    }
    assert goodput["best_dp"] == {
        "topology": "4dp",
        "concurrency": 12,
        "requests_per_second": 2.0,
    }
    assert goodput["pap_over_pd_percent"] == pytest.approx(33.333333)
    assert goodput["pap_over_dp_percent"] == pytest.approx(20.0)


def test_matrix_marks_incomplete_run_as_ineligible(tmp_path: Path) -> None:
    tiers = {
        name: {
            "passed": False,
            "good_request_fraction": 0.5,
            "goodput_requests_per_second": None,
        }
        for name in ("strict", "standard", "relaxed")
    }
    rows = build_rows(
        [
            {
                "architecture": "pd",
                "topology": "2p2d",
                "concurrency": 16,
                "run_status": {"state": "early_stopped_slo_impossible"},
                "workload": {"expected_requests": 960},
                "correctness": {
                    "passed": False,
                    "completed_requests": 628,
                },
                "metrics": {
                    "ttft_ms": {"p95": 315_940.0},
                    "itl_ms": {"p95": 43.0},
                    "request_throughput_per_second": None,
                },
                "slo": tiers,
            }
        ]
    )
    output = tmp_path / "capacity.md"
    write_markdown(rows, build_envelope(rows), output)
    text = output.read_text(encoding="utf-8")

    assert "early-stopped: SLO impossible" in text
    assert "628/960" in text
    assert text.count("ineligible") == 4
