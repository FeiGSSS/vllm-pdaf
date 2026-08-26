#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Export measured PAP Attention latency matrices without fitting a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key, value in serialized.items():
                if isinstance(value, list):
                    serialized[key] = ",".join(map(str, value))
            writer.writerow(serialized)


def flatten(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    expected = int(matrix["shape_count"]) * int(matrix["candidate_count"])
    if matrix.get("status") != "completed" or len(matrix["results"]) != expected:
        raise ValueError(
            f"Incomplete matrix: status={matrix.get('status')}, "
            f"rows={len(matrix['results'])}, expected={expected}"
        )
    rows = []
    for result in matrix["results"]:
        correctness = result["correctness"]
        if result.get("status") != "completed" or not correctness["allclose"]:
            raise ValueError(
                f"Invalid result for {result['shape']['shape_id']} / "
                f"{result['kernel']['config_id']}"
            )
        timing = result["paged_attention"]
        rows.append(
            {
                **result["shape"],
                **result["kernel"],
                "attention_median_ms": timing["median_ms"],
                "attention_mean_ms": timing["mean_ms"],
                "attention_min_ms": timing["min_ms"],
                "attention_max_ms": timing["max_ms"],
                "kv_append_median_ms": result["kv_append"]["median_ms"],
                "logical_kv_bytes": result["logical_kv_bytes"],
                "logical_kv_gbps": result["logical_kv_gbps"],
                "allclose": correctness["allclose"],
                "max_abs_error": correctness["max_abs_error"],
                "mean_abs_error": correctness["mean_abs_error"],
                "samples": result["samples"],
                "calls_per_sample": result["calls_per_sample"],
            }
        )
    return rows


def group_rows(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return grouped


def kernel_fields(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}{key}": row[key]
        for key in (
            "config_id",
            "implementation",
            "num_splits",
            "block_h",
            "num_warps",
            "num_stages",
            "block_n",
        )
    }


def shape_fields(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "config_id",
        "implementation",
        "num_splits",
        "block_h",
        "num_warps",
        "num_stages",
        "block_n",
        "attention_median_ms",
        "attention_mean_ms",
        "attention_min_ms",
        "attention_max_ms",
        "kv_append_median_ms",
        "logical_kv_bytes",
        "logical_kv_gbps",
        "allclose",
        "max_abs_error",
        "mean_abs_error",
        "samples",
        "calls_per_sample",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def best_by_shape(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for shape_rows in group_rows(rows, lambda row: row["shape_id"]).values():
        best = min(shape_rows, key=lambda row: row["attention_median_ms"])
        production = next(
            row for row in shape_rows if row["config_id"] == "production_auto"
        )
        pap = min(
            (row for row in shape_rows if row["implementation"] == "pap_grouped"),
            key=lambda row: row["attention_median_ms"],
        )
        vllm = min(
            (row for row in shape_rows if row["implementation"] == "vllm"),
            key=lambda row: row["attention_median_ms"],
        )
        output.append(
            {
                **shape_fields(best),
                **kernel_fields(best, "best_"),
                "best_attention_median_ms": best["attention_median_ms"],
                "production_auto_median_ms": production["attention_median_ms"],
                "production_slowdown_percent": (
                    production["attention_median_ms"] / best["attention_median_ms"]
                    - 1.0
                )
                * 100.0,
                "best_pap_config_id": pap["config_id"],
                "best_pap_median_ms": pap["attention_median_ms"],
                "best_vllm_config_id": vllm["config_id"],
                "best_vllm_median_ms": vllm["attention_median_ms"],
                "pap_vs_vllm_percent": (
                    vllm["attention_median_ms"] / pap["attention_median_ms"] - 1.0
                )
                * 100.0,
            }
        )
    return sorted(
        output,
        key=lambda row: (
            row["total_context_tokens"],
            row["batch_size"],
            row["distribution"],
        ),
    )


def fixed_total_table(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        row["total_context_tokens"]
        for row in best_rows
        if row["distribution"] == "equal"
    )
    return sorted(
        (
            row
            for row in best_rows
            if row["distribution"] == "equal"
            and counts[row["total_context_tokens"]] > 1
        ),
        key=lambda row: (row["total_context_tokens"], row["batch_size"]),
    )


def distribution_table(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_workload: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        by_workload[(row["batch_size"], row["total_context_tokens"])].append(row)
    output = []
    for (batch_size, total_context), workload_rows in by_workload.items():
        equal = next(
            (row for row in workload_rows if row["distribution"] == "equal"),
            None,
        )
        if equal is None or len(workload_rows) == 1:
            continue
        for row in workload_rows:
            output.append(
                {
                    "batch_size": batch_size,
                    "total_context_tokens": total_context,
                    "distribution": row["distribution"],
                    "context_lengths": row["context_lengths"],
                    "cv_context_tokens": row["cv_context_tokens"],
                    "gini_context_tokens": row["gini_context_tokens"],
                    "max_context_tokens": row["max_context_tokens"],
                    "best_config_id": row["best_config_id"],
                    "best_attention_median_ms": row["best_attention_median_ms"],
                    "equal_baseline_median_ms": equal["best_attention_median_ms"],
                    "distribution_penalty_percent": (
                        row["best_attention_median_ms"]
                        / equal["best_attention_median_ms"]
                        - 1.0
                    )
                    * 100.0,
                }
            )
    return sorted(
        output,
        key=lambda row: (
            row["total_context_tokens"],
            row["batch_size"],
            row["distribution"],
        ),
    )


def summarize_regrets(
    config_id: str,
    config_rows: list[dict[str, Any]],
    best_latency: dict[str, float],
) -> dict[str, Any]:
    regrets = [
        (row["attention_median_ms"] / best_latency[row["shape_id"]] - 1.0) * 100.0
        for row in config_rows
    ]
    return {
        **kernel_fields(config_rows[0]),
        "shape_count": len(config_rows),
        "win_count": sum(abs(value) < 1e-12 for value in regrets),
        "median_regret_percent": statistics.median(regrets),
        "p90_regret_percent": percentile(regrets, 0.90),
        "max_regret_percent": max(regrets),
        "mean_attention_median_ms": statistics.fmean(
            row["attention_median_ms"] for row in config_rows
        ),
    }


def config_summary(
    rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    best_latency = {
        row["shape_id"]: row["best_attention_median_ms"] for row in best_rows
    }
    output = [
        summarize_regrets(config_id, config_rows, best_latency)
        for config_id, config_rows in group_rows(
            rows, lambda row: row["config_id"]
        ).items()
    ]
    return sorted(output, key=lambda row: row["median_regret_percent"])


def parameter_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, Callable[[dict[str, Any]], bool]] = {
        "pap_num_splits": lambda row: (
            row["implementation"] == "pap_grouped"
            and row["block_h"] == 4
            and row["num_warps"] == 4
            and row["num_stages"] == 1
            and row["block_n"] == 32
        ),
        "pap_block_h": lambda row: (
            row["implementation"] == "pap_grouped"
            and row["num_splits"] == 8
            and row["num_warps"] == 4
            and row["num_stages"] == 1
            and row["block_n"] == 32
        ),
        "pap_num_warps": lambda row: (
            row["implementation"] == "pap_grouped"
            and row["num_splits"] == 8
            and row["block_h"] == 4
            and row["num_stages"] == 1
            and row["block_n"] == 32
        ),
        "pap_num_stages": lambda row: (
            row["implementation"] == "pap_grouped"
            and row["num_splits"] == 8
            and row["block_h"] == 4
            and row["num_warps"] == 4
            and row["block_n"] == 32
        ),
        "pap_block_n": lambda row: (
            row["implementation"] == "pap_grouped"
            and row["num_splits"] == 8
            and row["block_h"] == 4
            and row["num_warps"] == 4
            and row["num_stages"] == 1
        ),
        "vllm_num_splits": lambda row: row["implementation"] == "vllm",
    }
    parameter = {
        "pap_num_splits": "num_splits",
        "pap_block_h": "block_h",
        "pap_num_warps": "num_warps",
        "pap_num_stages": "num_stages",
        "pap_block_n": "block_n",
        "vllm_num_splits": "num_splits",
    }
    output = []
    for family, predicate in families.items():
        family_rows = [row for row in rows if predicate(row)]
        unique_rows = {
            (row["shape_id"], row["config_id"]): row for row in family_rows
        }.values()
        family_rows = list(unique_rows)
        by_shape = group_rows(family_rows, lambda row: row["shape_id"])
        family_best = {
            shape_id: min(shape_rows, key=lambda row: row["attention_median_ms"])[
                "attention_median_ms"
            ]
            for shape_id, shape_rows in by_shape.items()
        }
        by_value: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in family_rows:
            by_value[int(row[parameter[family]])].append(row)
        for value, value_rows in sorted(by_value.items()):
            regrets = [
                (row["attention_median_ms"] / family_best[row["shape_id"]] - 1.0)
                * 100.0
                for row in value_rows
            ]
            output.append(
                {
                    "family": family,
                    "parameter": parameter[family],
                    "value": value,
                    "shape_count": len(value_rows),
                    "family_win_count": sum(abs(item) < 1e-12 for item in regrets),
                    "median_regret_percent": statistics.median(regrets),
                    "p90_regret_percent": percentile(regrets, 0.90),
                    "max_regret_percent": max(regrets),
                }
            )
    return output


def matrix_summary(
    matrix: dict[str, Any],
    rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fixed_by_total: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in fixed_rows:
        fixed_by_total[int(row["total_context_tokens"])].append(row)
    fixed_spreads = []
    for total, total_rows in sorted(fixed_by_total.items()):
        fastest = min(total_rows, key=lambda row: row["best_attention_median_ms"])
        slowest = max(total_rows, key=lambda row: row["best_attention_median_ms"])
        fixed_spreads.append(
            {
                "total_context_tokens": total,
                "fastest_batch_size": fastest["batch_size"],
                "fastest_median_ms": fastest["best_attention_median_ms"],
                "slowest_batch_size": slowest["batch_size"],
                "slowest_median_ms": slowest["best_attention_median_ms"],
                "batch_spread_percent": (
                    slowest["best_attention_median_ms"]
                    / fastest["best_attention_median_ms"]
                    - 1.0
                )
                * 100.0,
            }
        )
    best_counts = Counter(row["best_config_id"] for row in best_rows)
    return {
        "schema_version": 1,
        "kind": "pap_attention_measured_matrix_summary",
        "matrix_status": matrix["status"],
        "shape_count": matrix["shape_count"],
        "candidate_count": matrix["candidate_count"],
        "row_count": len(rows),
        "completed_count": sum(row["allclose"] for row in rows),
        "allclose_count": sum(row["allclose"] for row in rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "ranges": {
            key: [min(row[key] for row in rows), max(row[key] for row in rows)]
            for key in (
                "batch_size",
                "total_context_tokens",
                "min_context_tokens",
                "max_context_tokens",
                "cv_context_tokens",
                "attention_median_ms",
            )
        },
        "best_config_frequency": dict(best_counts.most_common()),
        "fixed_total_batch_spreads": fixed_spreads,
        "largest_distribution_penalties": sorted(
            (row for row in distribution_rows if row["distribution"] != "equal"),
            key=lambda row: row["distribution_penalty_percent"],
            reverse=True,
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    matrix_bytes = args.matrix.read_bytes()
    matrix = json.loads(matrix_bytes)
    rows = flatten(matrix)
    best_rows = best_by_shape(rows)
    fixed_rows = fixed_total_table(best_rows)
    distribution_rows = distribution_table(best_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "latency_table.csv", rows)
    write_csv(args.output_dir / "best_by_shape.csv", best_rows)
    write_csv(args.output_dir / "fixed_total_equal.csv", fixed_rows)
    write_csv(args.output_dir / "distribution_comparison.csv", distribution_rows)
    write_csv(args.output_dir / "config_summary.csv", config_summary(rows, best_rows))
    write_csv(
        args.output_dir / "parameter_sensitivity.csv", parameter_sensitivity(rows)
    )
    summary = matrix_summary(matrix, rows, best_rows, fixed_rows, distribution_rows)
    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_environment_path = args.matrix.parent.parent / "run.env"
    if not run_environment_path.is_file():
        run_environment_path = args.matrix.parent.parent / "parallel_run.env"
    run_environment = {}
    if run_environment_path.is_file():
        run_environment = dict(
            line.split("=", 1)
            for line in run_environment_path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
    table_paths = sorted(args.output_dir.glob("*.csv"))
    manifest = {
        "schema_version": 1,
        "kind": "pap_attention_measured_matrix_tables",
        "source_matrix": str(args.matrix.resolve()),
        "source_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "source_bytes": len(matrix_bytes),
        "run_environment_file": (
            str(run_environment_path.resolve()) if run_environment else None
        ),
        "run_environment": run_environment,
        "tables": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in table_paths
        },
        "model_fitted": False,
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
