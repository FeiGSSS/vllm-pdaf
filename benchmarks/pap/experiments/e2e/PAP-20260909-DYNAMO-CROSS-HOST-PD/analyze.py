#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare same-host and cross-host Dynamo 7P1D request evidence."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ANALYZER = ROOT / "benchmarks/pap/tooling/analyze_dynamo_ttft.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_dynamo_ttft", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def load_run(path: Path, placement: str, label: str, analyzer) -> dict[str, Any]:
    analysis = analyzer.analyze(path, 16)
    profile_rows = [
        json.loads(line)
        for line in (path / "aiperf/profile.jsonl").read_text().splitlines()
    ]
    profile_by_id = {row["metadata"]["x_request_id"]: row for row in profile_rows}
    if analysis["joined_requests"] != len(profile_rows):
        raise ValueError(f"request join incomplete for {path}")
    records = []
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in analysis["requests"]:
        client_id = request["client_request_id"]
        profile = profile_by_id[client_id]
        metadata = profile["metadata"]
        metrics = profile["metrics"]
        record = {
            "placement": placement,
            "run": label,
            "client_request_id": client_id,
            "conversation_id": metadata["conversation_id"],
            "turn_index": metadata["turn_index"],
            "worker_id": request["worker_id"],
            "input_tokens": request["input_tokens"],
            "cached_tokens": request["cached_tokens"] or 0,
            "ttft_ms": metrics["time_to_first_token"]["value"],
            "itl_ms": metrics["inter_token_latency"]["value"],
            "e2e_ms": metrics["request_latency"]["value"],
            "output_tokens": metrics["output_sequence_length"]["value"],
            "router_to_prefill_ms": request["router_to_prefill_ms"],
            "prefill_service_ms": request["prefill_service_ms"],
            "router_to_decode_ms": request["router_to_decode_ms"],
            "decode_to_first_token_ms": request["decode_to_first_token_ms"],
            "migrated": None,
        }
        records.append(record)
        conversations[record["conversation_id"]].append(record)
    for items in conversations.values():
        ordered = sorted(items, key=lambda item: item["turn_index"])
        if [item["turn_index"] for item in ordered] != [0, 1, 2]:
            raise ValueError(f"incomplete conversation in {path}")
        for previous, current in zip(ordered, ordered[1:]):
            current["migrated"] = previous["worker_id"] != current["worker_id"]

    start = min(row["metadata"]["request_start_ns"] for row in profile_rows)
    end = max(row["metadata"]["request_end_ns"] for row in profile_rows)
    duration_s = (end - start) / 1e9
    transfer = analysis["kv_transfer"]
    return {
        "placement": placement,
        "label": label,
        "path": str(path),
        "duration_s": duration_s,
        "records": records,
        "transfer": transfer,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = sum(row["input_tokens"] for row in records)
    cached_tokens = sum(row["cached_tokens"] for row in records)
    transitions = [row for row in records if row["migrated"] is not None]
    migrations = [row for row in transitions if row["migrated"]]
    return {
        "requests": len(records),
        "input_tokens": input_tokens,
        "estimated_cached_tokens": cached_tokens,
        "estimated_computed_tokens": input_tokens - cached_tokens,
        "estimated_computed_tokens_per_request": (input_tokens - cached_tokens)
        / len(records),
        "estimated_cache_read_pct": cached_tokens / input_tokens * 100,
        "transitions": len(transitions),
        "migrations": len(migrations),
        "migration_rate_pct": (
            len(migrations) / len(transitions) * 100 if transitions else None
        ),
        "ttft_ms": distribution([row["ttft_ms"] for row in records]),
        "itl_ms": distribution([row["itl_ms"] for row in records]),
        "e2e_ms": distribution([row["e2e_ms"] for row in records]),
        "prefill_service_ms": distribution(
            [row["prefill_service_ms"] for row in records]
        ),
        "decode_to_first_token_ms": distribution(
            [row["decode_to_first_token_ms"] for row in records]
        ),
    }


def placement_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for run in runs for record in run["records"]]
    durations = sum(run["duration_s"] for run in runs)
    output = summarize(records)
    output.update(
        {
            "runs": len(runs),
            "request_throughput_rps": len(records) / durations,
            "output_throughput_tps": sum(row["output_tokens"] for row in records)
            / durations,
            "by_turn": {
                str(turn): summarize(
                    [row for row in records if row["turn_index"] == turn]
                )
                for turn in (0, 1, 2)
            },
        }
    )
    transfers = sum(run["transfer"]["transfers"] for run in runs)
    total_mb = sum(
        run["transfer"]["transfers"] * run["transfer"]["weighted_mb_per_transfer"]
        for run in runs
    )
    total_ms = sum(
        run["transfer"]["transfers"] * run["transfer"]["weighted_transfer_ms"]
        for run in runs
    )
    output["kv_transfer"] = {
        "transfers": transfers,
        "total_mb": total_mb,
        "mb_per_request": total_mb / len(records),
        "wall_clock_volume_rate_mb_s": total_mb / durations,
        "weighted_transfer_ms": total_ms / transfers,
        "request_residence_throughput_mb_s": total_mb / total_ms * 1000,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-host-run", action="append", type=Path, required=True)
    parser.add_argument("--cross-host-run", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyzer = load_analyzer()
    runs = []
    for index, path in enumerate(args.same_host_run, 1):
        runs.append(load_run(path, "same_host", f"same_host_{index}", analyzer))
    for index, path in enumerate(args.cross_host_run, 1):
        runs.append(load_run(path, "cross_host", f"cross_host_{index}", analyzer))

    expected_outputs = None
    for run in runs:
        outputs = {
            (row["conversation_id"], row["turn_index"]): row["output_tokens"]
            for row in run["records"]
        }
        if expected_outputs is None:
            expected_outputs = outputs
        elif outputs != expected_outputs:
            raise ValueError(f"workload output lengths differ in {run['path']}")

    placements = {
        name: placement_summary([run for run in runs if run["placement"] == name])
        for name in ("same_host", "cross_host")
    }
    same = placements["same_host"]
    cross = placements["cross_host"]
    delta = {}
    for name, same_value, cross_value in (
        ("migration_rate", same["migration_rate_pct"], cross["migration_rate_pct"]),
        (
            "computed_tokens_per_request",
            same["estimated_computed_tokens_per_request"],
            cross["estimated_computed_tokens_per_request"],
        ),
        ("ttft_mean", same["ttft_ms"]["mean"], cross["ttft_ms"]["mean"]),
        ("itl_mean", same["itl_ms"]["mean"], cross["itl_ms"]["mean"]),
        ("e2e_mean", same["e2e_ms"]["mean"], cross["e2e_ms"]["mean"]),
        (
            "prefill_service_mean",
            same["prefill_service_ms"]["mean"],
            cross["prefill_service_ms"]["mean"],
        ),
        (
            "kv_transfer_residence_mean",
            same["kv_transfer"]["weighted_transfer_ms"],
            cross["kv_transfer"]["weighted_transfer_ms"],
        ),
        (
            "request_throughput",
            same["request_throughput_rps"],
            cross["request_throughput_rps"],
        ),
    ):
        delta[f"{name}_pct"] = (cross_value / same_value - 1) * 100

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_runs = [
        {key: value for key, value in run.items() if key != "records"} for run in runs
    ]
    summary = {"runs": serializable_runs, "placements": placements, "delta": delta}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = list(runs[0]["records"][0])
    with (args.output_dir / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerows(run["records"])


if __name__ == "__main__":
    main()
