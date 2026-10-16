#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decompose PAP TTFT into Gateway phases and compare median and tail requests.

Joins the Gateway lines ``PAP proxy prefill IPC profile``, ``projection
admission profile``, ``projection stream profile`` (first chunk) and ``PAP
Dynamo placement`` to AIPerf records by request id (AIPerf ``x_request_id``).
"""

from __future__ import annotations

import contextlib
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import regex as re

EXPERIMENT_DIR = Path(__file__).resolve().parent
PHASES = [
    "register_ms",
    "tokenization_ms",
    "routing_ms",
    "migration_ms",
    "prefill_payload_ms",
    "prefill_ms",
    "readiness_ms",
    "projection_payload_ms",
    "admission_ms",
    "first_chunk_ms",
]
FIELD = re.compile(r"(\w+)=([^\s]+)")


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def parse_gateway(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for log in sorted((root / "service_logs").rglob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "pap_gateway:" not in line:
                continue
            for marker in (
                "prefill IPC profile",
                "projection admission profile",
                "projection stream profile",
                "Dynamo placement",
            ):
                if marker in line:
                    fields = dict(FIELD.findall(line.split(marker, 1)[1]))
                    request = fields.pop("request_id", None)
                    if request:
                        row = rows.setdefault(request, {})
                        for key, value in fields.items():
                            with contextlib.suppress(ValueError):
                                row[key] = float(value)
                    break
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_run(root: Path) -> dict[str, Any]:
    launch = read_env(root / "launch.env")
    gateway = parse_gateway(root)
    records = []
    for line in (root / "aiperf/profile.jsonl").read_text().splitlines():
        raw = json.loads(line)
        meta, metrics = raw["metadata"], raw["metrics"]
        row = {
            "run": root.name,
            "variant": launch["variant"],
            "request_id": meta["x_request_id"],
            "turn": int(meta["turn_index"]),
            "input_tokens": int(metrics["input_sequence_length"]["value"]),
            "cached_tokens": int(metrics["usage_prompt_cache_read_tokens"]["value"]),
            "ttft_ms": float(metrics["time_to_first_token"]["value"]),
            "itl_ms": float(metrics["inter_token_latency"]["value"]),
        }
        g = gateway.get(row["request_id"], {})
        row["joined"] = bool(g)
        row["selected_pa"] = int(g["selected_pa"]) if "selected_pa" in g else None
        row["gateway_ttft_ms"] = g.get("request_to_first_chunk_ms")
        for phase in PHASES:
            row[phase] = g.get(phase)
        records.append(row)
    return {"run": root.name, "variant": launch["variant"], "records": records}


def group_means(records: list[dict]) -> dict[str, float]:
    out = {
        "n": len(records),
        "ttft_ms": statistics.fmean(r["ttft_ms"] for r in records),
    }
    for phase in [*PHASES, "gateway_ttft_ms"]:
        values = [r[phase] for r in records if r[phase] is not None]
        out[phase] = statistics.fmean(values) if values else None
    return out


def main() -> None:
    runs = []
    for root in sorted((EXPERIMENT_DIR / "runs").glob("*_r*")):
        exit_code = root / "exit_code"
        if not exit_code.exists() or exit_code.read_text().strip() != "0":
            continue
        if read_env(root / "launch.env").get("source_dirty_files") != "0":
            continue
        runs.append(load_run(root))

    summary: dict[str, Any] = {"experiment": EXPERIMENT_DIR.name, "variants": {}}
    table = []
    for variant in sorted({run["variant"] for run in runs}):
        records = [
            r for run in runs if run["variant"] == variant for r in run["records"]
        ]
        entry: dict[str, Any] = {
            "runs": [run["run"] for run in runs if run["variant"] == variant],
            "ttft_ms_mean": statistics.fmean(r["ttft_ms"] for r in records),
            "ttft_ms_p50": percentile([r["ttft_ms"] for r in records], 0.5),
            "ttft_ms_p99": percentile([r["ttft_ms"] for r in records], 0.99),
        }
        joined = [r for r in records if r["joined"]]
        entry["joined"] = f"{len(joined)}/{len(records)}"
        if joined:
            p45 = percentile([r["ttft_ms"] for r in joined], 0.45)
            p55 = percentile([r["ttft_ms"] for r in joined], 0.55)
            p99 = percentile([r["ttft_ms"] for r in joined], 0.99)
            p95 = percentile([r["ttft_ms"] for r in joined], 0.95)
            groups = {
                "median (P45-P55)": [r for r in joined if p45 <= r["ttft_ms"] <= p55],
                "tail (>= P95)": [r for r in joined if r["ttft_ms"] >= p95],
                "tail (>= P99)": [r for r in joined if r["ttft_ms"] >= p99],
                "all": joined,
            }
            entry["groups"] = {name: group_means(rs) for name, rs in groups.items()}
            for name, means in entry["groups"].items():
                table.append({"variant": variant, "group": name} | means)
            pa_counts: dict[int, int] = {}
            pa_tokens: dict[int, int] = {}
            for r in joined:
                if r["selected_pa"] is None:
                    continue
                pa_counts[r["selected_pa"]] = pa_counts.get(r["selected_pa"], 0) + 1
                pa_tokens[r["selected_pa"]] = pa_tokens.get(r["selected_pa"], 0) + (
                    r["input_tokens"] - r["cached_tokens"]
                )
            if pa_counts:
                counts, tokens = list(pa_counts.values()), list(pa_tokens.values())
                entry["pa_requests"] = dict(sorted(pa_counts.items()))
                entry["pa_requests_cv"] = statistics.pstdev(counts) / statistics.fmean(
                    counts
                )
                entry["pa_prefill_tokens_cv"] = statistics.pstdev(
                    tokens
                ) / statistics.fmean(tokens)
                tail = groups["tail (>= P95)"]
                entry["tail_pa_requests"] = {
                    pa: sum(r["selected_pa"] == pa for r in tail)
                    for pa in sorted(pa_counts)
                }
        summary["variants"][variant] = entry

    if "profile-c40" in summary["variants"] and "noprofile-c40" in summary["variants"]:
        a = summary["variants"]["profile-c40"]["ttft_ms_mean"]
        b = summary["variants"]["noprofile-c40"]["ttft_ms_mean"]
        summary["profiling_overhead_ttft"] = a / b - 1

    results = EXPERIMENT_DIR / "results"
    results.mkdir(exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if table:
        with (results / "ttft_phases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    for variant, entry in summary["variants"].items():
        print(variant, {k: v for k, v in entry.items() if k not in ("groups", "runs")})
        for name, means in entry.get("groups", {}).items():
            cells = " ".join(
                f"{k[:-3]}={v:.0f}"
                for k, v in means.items()
                if k.endswith("_ms") and v is not None
            )
            print(f"  {name:18s} n={means['n']:4d} {cells}")
    print("profiling_overhead_ttft", summary.get("profiling_overhead_ttft"))


if __name__ == "__main__":
    main()
