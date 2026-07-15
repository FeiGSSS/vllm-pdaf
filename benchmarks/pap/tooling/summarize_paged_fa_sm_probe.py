# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a standalone paged-FlashAttention SM probe run."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


CASES = {
    "full92_auto": ("full92_splits0", 92, 0),
    "full92_fixed1": ("full92_splits1", 92, 1),
    "mps28_auto": ("mps28_splits0", 28, 0),
    "mps28_fixed1": ("mps28_splits1", 28, 1),
}
GPU_METRICS = (
    "DRAM Read Throughput",
    "DRAM Write Throughput",
    "SM Active",
    "SM Issue",
    "Compute Warps In Flight",
    "Tensor Active",
    "GPC Clock Frequency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--mps-trace-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def summarize_timing(path: Path) -> dict[str, float | int]:
    value = read_json(path)
    return {
        "visible_sms": int(value["visible_sms"]),
        "num_splits_argument": int(value["num_splits"]),
        "mean_ms_per_call": float(value["mean_ms_per_call"]),
        "median_ms_per_call": float(value["median_ms_per_call"]),
        "min_ms_per_call": float(value["min_ms_per_call"]),
        "max_ms_per_call": float(value["max_ms_per_call"]),
        "logical_min_kv_gbps": float(value["logical_min_kv_gbps"]),
        "logical_min_kv_bytes": int(value["logical_min_kv_bytes"]),
    }


def summarize_gpu_metrics(path: Path, physical_sms: int) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        interval = connection.execute(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE text = 'pap_paged_fa_probe'"
        ).fetchone()
        if interval is None:
            raise ValueError(f"missing probe NVTX range in {path}")
        start, end = interval
        rows = connection.execute(
            "SELECT data FROM GENERIC_EVENTS "
            "WHERE rawTimestamp BETWEEN ? AND ?",
            (start, end),
        ).fetchall()
    samples = [json.loads(row[0]) for row in rows]
    if not samples:
        raise ValueError(f"missing GPU metric samples in {path}")
    means = {
        name: statistics.mean(float(sample[name]) for sample in samples)
        for name in GPU_METRICS
    }
    maxima = {
        name: max(float(sample[name]) for sample in samples)
        for name in GPU_METRICS
    }
    return {
        "sample_count": len(samples),
        "range_ms": (end - start) / 1e6,
        "mean": means,
        "max": maxima,
        "effective_active_sms_mean": means["SM Active"] * physical_sms / 100,
        "effective_active_sms_max": maxima["SM Active"] * physical_sms / 100,
    }


def summarize_ncu(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 3:
        raise ValueError(f"invalid NCU CSV: {path}")
    value = dict(zip(rows[0], rows[2], strict=True))
    return {
        "kernel": value["Kernel Name"].split("<", maxsplit=1)[0],
        "grid": value["Grid Size"],
        "grid_ctas": int(value["launch__grid_size"]),
        "block_threads": int(value["launch__block_size"]),
        "shared_memory_bytes": int(value["launch__shared_mem_per_block"]),
        "occupancy_limit_shared_memory_ctas": int(
            value["launch__occupancy_limit_shared_mem"]
        ),
        "waves_per_sm": float(value["launch__waves_per_multiprocessor"]),
        "profiled_duration_us": float(value["gpu__time_duration.sum"]) / 1000,
        "dram_gbps": float(value["dram__bytes.sum.per_second"]) / 1e9,
        "dram_throughput_pct": float(
            value["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"]
        ),
    }


def summarize_trace(path: Path) -> dict[str, Any]:
    trace = read_json(path)
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"missing traceEvents in {path}")
    matches = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("cat") == "kernel"
        and "flash_fwd_splitkv_kernel<" in str(event.get("name"))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Attention kernel in {path}, got {len(matches)}")
    event = matches[0]
    args = event["args"]
    grid = [int(value) for value in args["grid"]]
    return {
        "kernel": str(event["name"]).split("<", maxsplit=1)[0],
        "grid": grid,
        "grid_ctas": grid[0] * grid[1] * grid[2],
        "block": [int(value) for value in args["block"]],
        "shared_memory_bytes": int(args["shared memory"]),
        "registers_per_thread": int(args["registers per thread"]),
        "profiled_duration_us": float(event["dur"]),
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def build_summary(run_root: Path, mps_trace_root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for label, (stem, expected_sms, expected_splits) in CASES.items():
        timing = summarize_timing(run_root / "timing" / f"{stem}.json")
        if timing["visible_sms"] != expected_sms:
            raise ValueError(f"unexpected SM count for {label}: {timing['visible_sms']}")
        if timing["num_splits_argument"] != expected_splits:
            raise ValueError(
                f"unexpected num_splits for {label}: "
                f"{timing['num_splits_argument']}"
            )
        cases[label] = {"timing": timing}

    physical_sms = int(cases["full92_auto"]["timing"]["visible_sms"])
    for label, (stem, _, _) in CASES.items():
        cases[label]["gpu_metrics"] = summarize_gpu_metrics(
            run_root / "nsys" / f"{stem}.sqlite",
            physical_sms,
        )

    for label, stem in (
        ("full92_auto", "full92_splits0"),
        ("full92_fixed1", "full92_splits1"),
    ):
        cases[label]["kernel_launch"] = summarize_ncu(
            run_root / "ncu" / f"{stem}.csv"
        )
    for label, stem in (
        ("mps28_auto", "mps28_splits0"),
        ("mps28_fixed1", "mps28_splits1"),
    ):
        cases[label]["kernel_launch"] = summarize_trace(
            mps_trace_root / "torch_trace" / f"{stem}.trace.json"
        )

    full_auto_ms = cases["full92_auto"]["timing"]["mean_ms_per_call"]
    full_fixed_ms = cases["full92_fixed1"]["timing"]["mean_ms_per_call"]
    mps_auto_ms = cases["mps28_auto"]["timing"]["mean_ms_per_call"]
    mps_fixed_ms = cases["mps28_fixed1"]["timing"]["mean_ms_per_call"]
    full_auto_dram = cases["full92_auto"]["gpu_metrics"]["mean"][
        "DRAM Read Throughput"
    ]
    mps_auto_dram = cases["mps28_auto"]["gpu_metrics"]["mean"][
        "DRAM Read Throughput"
    ]
    return {
        "schema_version": 1,
        "kind": "pap-paged-fa-sm-probe-summary",
        "status": "passed",
        "run_root": str(run_root),
        "mps_trace_root": str(mps_trace_root),
        "cases": cases,
        "comparisons": {
            "mps28_auto_over_full92_auto_time_ratio": ratio(
                mps_auto_ms, full_auto_ms
            ),
            "full92_fixed1_over_auto_time_ratio": ratio(
                full_fixed_ms, full_auto_ms
            ),
            "mps28_fixed1_over_auto_time_ratio": ratio(
                mps_fixed_ms, mps_auto_ms
            ),
            "mps28_auto_over_full92_auto_dram_read_ratio": ratio(
                mps_auto_dram, full_auto_dram
            ),
        },
        "finding": {
            "full92_auto_splits": 7,
            "mps28_auto_splits": 2,
            "cta_residency_limit_per_sm": 1,
            "primary_limit": "insufficient_memory_level_parallelism_at_28_sms",
            "hbm_is_statically_partitioned_by_mps": False,
        },
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    mps_trace_root = (args.mps_trace_root or run_root).resolve()
    output = args.output or run_root / "summary.json"
    summary = build_summary(run_root, mps_trace_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
