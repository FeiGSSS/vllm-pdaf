import json
from pathlib import Path

from benchmarks.pap.aiperf.summarize_capacity_matrix import (
    build_envelope,
    build_rows,
)
from benchmarks.pap.aiperf.summarize_capacity_run import summarize_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_pd_run(run_root: Path, ttfts: list[float]) -> None:
    records = []
    for index, ttft in enumerate(ttfts):
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
                    "output_token_count": {"value": 256},
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
        },
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
    assert summary["routing"]["migration_count"] == 0
    assert summary["slo"]["strict"]["good_request_fraction"] == 0.75
    assert summary["slo"]["strict"]["passed"] is False
    assert summary["slo"]["standard"]["passed"] is True
    assert summary["slo"]["relaxed"]["passed"] is True
    assert summary["slo"]["standard"]["goodput_requests_per_second"] == 2.0


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


def test_matrix_envelope_uses_best_pd_topology() -> None:
    def summary(
        architecture: str,
        topology: str,
        concurrency: int,
        standard: bool,
    ) -> dict[str, object]:
        tiers = {
            name: {"passed": standard, "good_request_fraction": 1.0}
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
        ]
    )
    envelope = build_envelope(rows)["capacity_by_slo"]["standard"]

    assert envelope["pap_3pa1p"] == 24
    assert envelope["best_pd"] == {"topology": "2p2d", "concurrency": 20}
    assert envelope["pap_minus_best_pd"] == 4
