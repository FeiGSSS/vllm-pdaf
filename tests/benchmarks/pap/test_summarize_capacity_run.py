from __future__ import annotations

import json
from pathlib import Path

from benchmarks.pap.aiperf import summarize_capacity_run


def test_runtime_repetitions_scale_shared_pap_audits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    aiperf_root = run_root / "aiperf"
    aiperf_root.mkdir(parents=True)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "session_id": "session-0",
                "turns": [
                    {
                        "output_length": 1,
                        "extra": {"ignore_eos": True, "min_tokens": 1},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (aiperf_root / "profile.jsonl").write_text(
        json.dumps(
            {
                "metadata": {
                    "conversation_id": "session-0",
                    "turn_index": 0,
                },
                "metrics": {
                    "output_token_count": {"value": 1},
                    "time_to_first_token": {"value": 10.0},
                    "inter_token_latency": {"value": 1.0},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (aiperf_root / "profile.json").write_text(
        json.dumps(
            {
                "request_count": {"avg": 1},
                "request_throughput": {"avg": 2.0},
                "output_token_throughput": {"avg": 2.0},
                "output_token_throughput_per_user": {"avg": 2.0},
                "benchmark_duration": {"avg": 0.5},
            }
        ),
        encoding="utf-8",
    )

    observed: dict[str, int] = {}

    monkeypatch.setattr(
        summarize_capacity_run,
        "_check_runtime_audits",
        lambda *args, **kwargs: None,
    )

    def check_routing(
        run_root: Path,
        *,
        sessions: int,
        turns: int,
        expected_requests: int,
        errors: list[str],
    ) -> dict[str, object]:
        del run_root, turns, errors
        observed["sessions"] = sessions
        observed["expected_requests"] = expected_requests
        return {"passed": True, "migration_count": 0}

    monkeypatch.setattr(
        summarize_capacity_run,
        "_check_pap_routing",
        check_routing,
    )

    summary = summarize_capacity_run.summarize_run(
        run_root,
        aiperf_root=aiperf_root,
        architecture="pap",
        topology="7pa1p",
        concurrency=20,
        sessions=1,
        turns=1,
        output_tokens=1,
        dataset_file=dataset,
        repetition=1,
        runtime_repetitions=2,
    )

    assert summary["correctness"]["passed"] is True
    assert summary["correctness"]["completed_requests"] == 1
    assert observed == {"sessions": 2, "expected_requests": 2}
