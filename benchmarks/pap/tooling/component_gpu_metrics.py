#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize Nsight Systems GPU metrics inside PAP component NVTX ranges."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

GPU_METRICS = (
    "DRAM Read Throughput",
    "DRAM Write Throughput",
    "SM Active",
    "SM Issue",
    "Compute Warps In Flight",
    "Tensor Active",
    "GPC Clock Frequency",
)


def range_pattern(component: str) -> str:
    if component == "projection":
        return "pap_projection_stage:%"
    if component == "prefill":
        return "pap_prefill_model_forward_t%"
    if component == "attention":
        return "pap_attention_scaling_probe"
    raise ValueError(f"unknown component: {component}")


def summarize_samples(
    name: str,
    intervals: list[tuple[int, int]],
    samples: list[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    selected = [
        sample
        for timestamp, sample in samples
        if any(start <= timestamp <= end for start, end in intervals)
    ]
    if not selected:
        raise ValueError(f"NVTX range {name} has no GPU metric samples")
    return {
        "range_count": len(intervals),
        "range_ms_total": sum(end - start for start, end in intervals) / 1e6,
        "sample_count": len(selected),
        "mean": {
            metric: statistics.fmean(float(sample[metric]) for sample in selected)
            for metric in GPU_METRICS
        },
        "max": {
            metric: max(float(sample[metric]) for sample in selected)
            for metric in GPU_METRICS
        },
        "min": {
            metric: min(float(sample[metric]) for sample in selected)
            for metric in GPU_METRICS
        },
    }


def summarize(path: Path, component: str) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        ranges = connection.execute(
            "SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE ?",
            (range_pattern(component),),
        ).fetchall()
        raw_samples = connection.execute(
            "SELECT rawTimestamp, data FROM GENERIC_EVENTS ORDER BY rawTimestamp"
        ).fetchall()
    if not ranges:
        raise ValueError(f"no {component} NVTX ranges in {path}")
    filter_metadata: dict[str, Any] = {}
    if component == "projection":
        before_count = len(ranges)
        ranges = [
            item
            for item in ranges
            if item[0] != "pap_projection_stage:attention"
            and int(item[2]) - int(item[1]) <= 10_000_000
        ]
        filter_metadata = {
            "excluded_attention_and_over_10ms_ranges": before_count - len(ranges),
            "maximum_interval_ms": 10.0,
        }
        if not ranges:
            raise ValueError(f"no non-Attention Projection ranges in {path}")
    samples = [
        (int(timestamp), json.loads(payload)) for timestamp, payload in raw_samples
    ]
    by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for name, start, end in ranges:
        by_name[str(name)].append((int(start), int(end)))
    summaries = {
        name: summarize_samples(name, intervals, samples)
        for name, intervals in sorted(by_name.items())
    }
    all_intervals = [
        interval for intervals in by_name.values() for interval in intervals
    ]
    return {
        "schema_version": 1,
        "kind": "pap_component_gpu_metrics",
        "component": component,
        "source": str(path),
        "filters": filter_metadata,
        "ranges": summaries,
        "all_ranges": summarize_samples(f"{component}:all", all_intervals, samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument(
        "--component",
        choices=("projection", "attention", "prefill"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.sqlite.resolve(), args.component)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
