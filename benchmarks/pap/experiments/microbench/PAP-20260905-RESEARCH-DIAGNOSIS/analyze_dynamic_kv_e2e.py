# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate growth counts and classify client intervals at page-growth steps."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[int(0.95 * (len(values) - 1))],
        "p99_ms": ordered[int(0.99 * (len(values) - 1))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.run / "aiperf/profile.jsonl").read_text().splitlines()
    ]
    growth_intervals: list[float] = []
    ordinary_intervals: list[float] = []
    expected_growths = 0
    for row in rows:
        metrics = row["metrics"]
        input_tokens = int(metrics["input_sequence_length"]["value"])
        output_tokens = int(metrics["output_sequence_length"]["value"])
        intervals = metrics["inter_chunk_latency"]["value"][: output_tokens - 1]
        capacity = math.ceil(input_tokens / 16) * 16
        for offset, interval in enumerate(intervals, start=1):
            required = input_tokens + offset
            if required > capacity:
                expected_growths += 1
                growth_intervals.append(float(interval))
                capacity = min(
                    input_tokens + output_tokens,
                    max(required, capacity + 256),
                )
            else:
                ordinary_intervals.append(float(interval))

    final_stats = [
        json.loads(Path(path).read_text())
        for path in glob.glob(str(args.run / "attention_fast_path_stats_*.json"))
    ]
    observed_growths = sum(item["decode_capacity_requests"] for item in final_stats)
    payload = {
        "run": str(args.run),
        "records": len(rows),
        "model": "Qwen3-8B",
        "block_size_tokens": 16,
        "growth_chunk_tokens": 256,
        "expected_growth_requests_from_client_lengths": expected_growths,
        "observed_attention_growth_requests": observed_growths,
        "counts_match": observed_growths == expected_growths,
        "observed_attention_growth_installs": sum(
            item["decode_capacity_installs"] for item in final_stats
        ),
        "observed_attention_blocks_added": sum(
            item["decode_capacity_blocks_added"] for item in final_stats
        ),
        "observed_topology_mismatches": sum(
            item["slot_topology_mismatches"] for item in final_stats
        ),
        "growth_step_client_intervals": _summary(growth_intervals),
        "ordinary_client_intervals": _summary(ordinary_intervals),
        "scope": (
            "request-level external token intervals classified by predicted "
            "capacity crossings; not GPU execution time"
        ),
        "limitation": (
            "A crossing stalls a whole PA/Projection step and can delay other "
            "requests in the batch. Do not weight this request-local split to "
            "estimate the system-wide TBT contribution."
        ),
    }
    if not payload["counts_match"]:
        raise RuntimeError("predicted and observed Decode growth counts differ")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
