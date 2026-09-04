#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare two fixed-duration AI Perf runs without censoring paired requests."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
    "output_token_throughput_per_user",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            metadata = record["metadata"]
            if metadata.get("was_cancelled", False):
                continue
            key = (metadata["conversation_id"], int(metadata["turn_index"]))
            records[key] = record
    return records


def value(record: dict[str, Any], name: str) -> float:
    return float(record["metrics"][name]["value"])


def means(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(value(record, metric) for record in records)
        for metric in METRICS
    }


def compare(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
) -> dict[str, Any]:
    baseline_means = means([baseline[key] for key in keys])
    candidate_means = means([candidate[key] for key in keys])
    return {
        "count": len(keys),
        "baseline_mean": baseline_means,
        "candidate_mean": candidate_means,
        "change_percent": {
            metric: (candidate_means[metric] / baseline_means[metric] - 1.0) * 100.0
            for metric in METRICS
        },
    }


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    common = sorted(baseline.keys() & candidate.keys())
    exact_common = [
        key
        for key in common
        if value(baseline[key], "input_sequence_length")
        == value(candidate[key], "input_sequence_length")
        and value(baseline[key], "output_sequence_length")
        == value(candidate[key], "output_sequence_length")
    ]
    exact_turn_zero = [key for key in exact_common if key[1] == 0]
    input_mismatches = sum(
        value(baseline[key], "input_sequence_length")
        != value(candidate[key], "input_sequence_length")
        for key in common
    )
    output_mismatches = sum(
        value(baseline[key], "output_sequence_length")
        != value(candidate[key], "output_sequence_length")
        for key in common
    )
    result = {
        "baseline_completed": len(baseline),
        "candidate_completed": len(candidate),
        "common_request_keys": len(common),
        "common_input_length_mismatches": input_mismatches,
        "common_output_length_mismatches": output_mismatches,
        "all_common_requests": compare(baseline, candidate, common),
        "strict_exact_requests": compare(baseline, candidate, exact_common),
        "strict_exact_turn_zero": compare(
            baseline,
            candidate,
            exact_turn_zero,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
