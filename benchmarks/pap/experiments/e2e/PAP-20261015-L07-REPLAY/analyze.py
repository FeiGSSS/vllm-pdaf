#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate L07 replay runs and find each architecture's SLO capacity.

July's rule (PAP-20260729-RESEARCH-L07): a point is feasible for an SLO tier
only if every repetition has at least 95% of requests meeting both its TTFT and
ITL limits; its goodput is the lowest repetition's good requests per second;
an architecture's capacity is the highest goodput over feasible points.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPECTED_REQUESTS = 640
ATTAINMENT = 0.95
JULY_PD_STANDARD_GOODPUT = 15.305
TIERS = {
    "Strict": {"ttft_ms": 5000, "itl_ms": 50},
    "Standard": {"ttft_ms": 10000, "itl_ms": 75},
    "Relaxed": {"ttft_ms": 20000, "itl_ms": 100},
    "D2": {"ttft_ms": 10000, "itl_ms": 75, "e2e_ms": 20000},
}
METRICS = {
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}
ARCHITECTURES = {
    "pap7pa1p": "PAP 7PA1P",
    "pap6pa2p": "PAP 6PA2P",
    "pd6p2d": "PD 6P2D",
    "dp8": "DP8",
}
DYNAMO_AUDITS = {
    "6p2d": {
        "correctness_audit.env",
        "kv_transfer_audit.env",
        "vllm_cuda_graph_audit.env",
    },
    "dp8": {"correctness_audit.env", "vllm_cuda_graph_audit.env"},
}
CSV_COLUMNS = [
    "architecture",
    "variant",
    "concurrency",
    "run",
    "conversation_id",
    "turn",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "ttft_ms",
    "itl_ms",
    "e2e_ms",
    "start_ns",
    "end_ns",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def validate(root: Path) -> list[str]:
    """Return the reasons a run is invalid; empty means valid."""
    exit_code = root / "exit_code"
    if not exit_code.exists() or exit_code.read_text().strip() != "0":
        return ["nonzero or missing exit code"]
    problems = []
    profile = read_json(root / "aiperf/profile.json")
    if profile["request_count"]["avg"] != EXPECTED_REQUESTS:
        problems.append("request count")
    if profile["error_summary"] or profile["was_cancelled"]:
        problems.append("AIPerf errors or cancellation")
    if profile["osl_mismatch_count"]["avg"] != 0:
        problems.append("output length mismatch")
    launch = read_env(root / "launch.env")
    if launch.get("source_dirty_files") != "0":
        problems.append("dirty tracked worktree")
    audits = {p.name: p.read_text() for p in root.glob("*audit*.env")}
    if launch["runner"] == "pap":
        audits |= {p.name: p.read_text() for p in root.glob("*drain.env")}
        failed = [
            name
            for name, text in audits.items()
            if not name.startswith("mps_static_audit") and "STATUS=passed" not in text
        ]
        if failed:
            problems.append(f"audits failed {failed}")
        routing = read_json(root / "routing_audit.json")
        migration = routing["kv_migration"]
        if routing["status"] != "passed" or routing["route_count"] != EXPECTED_REQUESTS:
            problems.append("routing audit")
        if not (
            migration["enabled"]
            and migration["attempts"] == migration["exports"]
            and migration["misses"] == migration["failures"] == 0
            and migration["pending_releases"] == 0
        ):
            problems.append("KV migration audit")
    else:
        architecture = read_env(root / "requested.env")["DYNAMO_ARCHITECTURE"]
        missing = DYNAMO_AUDITS[architecture] - audits.keys()
        failed = [
            name
            for name, text in audits.items()
            if "STATUS=passed" not in text and "STATUS=not_applicable" not in text
        ]
        if missing or failed:
            problems.append(f"audits: missing {sorted(missing)}, failed {failed}")
    return problems


def load_run(root: Path) -> dict[str, Any]:
    launch = read_env(root / "launch.env")
    variant = launch["variant"]
    architecture, _, point = variant.partition("-c")
    records = []
    for line in (root / "aiperf/profile.jsonl").read_text().splitlines():
        raw = json.loads(line)
        meta, metrics = raw["metadata"], raw["metrics"]
        records.append(
            {
                "architecture": architecture,
                "variant": variant,
                "concurrency": int(point),
                "run": root.name,
                "conversation_id": meta["conversation_id"],
                "turn": int(meta["turn_index"]),
                "input_tokens": int(metrics["input_sequence_length"]["value"]),
                "output_tokens": int(metrics["output_sequence_length"]["value"]),
                "cached_tokens": int(
                    metrics["usage_prompt_cache_read_tokens"]["value"]
                ),
                **{k: float(metrics[v]["value"]) for k, v in METRICS.items()},
                "start_ns": int(meta["request_start_ns"]),
                "end_ns": int(meta["request_end_ns"]),
            }
        )
    duration_s = (
        max(r["end_ns"] for r in records) - min(r["start_ns"] for r in records)
    ) / 1e9
    return {
        "run": root.name,
        "architecture": architecture,
        "concurrency": int(point),
        "repetition": int(launch["repetition"]),
        "duration_s": duration_s,
        "records": records,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_metrics(run: dict[str, Any]) -> dict[str, float]:
    records = run["records"]
    out = {
        "throughput_rps": len(records) / run["duration_s"],
        "ttft_p95_ms": percentile([r["ttft_ms"] for r in records], 0.95),
        "itl_p95_ms": percentile([r["itl_ms"] for r in records], 0.95),
    }
    for tier, limits in TIERS.items():
        good = sum(all(r[k] <= v for k, v in limits.items()) for r in records)
        out[f"{tier}_attainment"] = good / len(records)
        out[f"{tier}_goodput_rps"] = good / run["duration_s"]
    turn0 = [r for r in records if r["turn"] == 0]
    out["turn0_cache_hit_rate"] = sum(r["cached_tokens"] for r in turn0) / sum(
        r["input_tokens"] for r in turn0
    )
    return out


def main() -> None:
    valid, invalid = [], []
    for root in sorted((EXPERIMENT_DIR / "runs").glob("*_r*")):
        if not (root / "launch.env").exists():
            continue
        problems = validate(root)
        if problems:
            invalid.append({"run": root.name, "reasons": problems})
        else:
            valid.append(load_run(root) | {"root": root})
    for run in valid:
        run["metrics"] = run_metrics(run)

    points: dict[tuple[str, int], list[dict]] = {}
    for run in valid:
        points.setdefault((run["architecture"], run["concurrency"]), []).append(run)
    point_rows = []
    for (architecture, concurrency), runs in sorted(points.items()):
        row: dict[str, Any] = {
            "architecture": architecture,
            "concurrency": concurrency,
            "runs": len(runs),
            "throughput_rps_min": min(r["metrics"]["throughput_rps"] for r in runs),
            "throughput_rps_mean": statistics.fmean(
                r["metrics"]["throughput_rps"] for r in runs
            ),
            "ttft_p95_ms_worst": max(r["metrics"]["ttft_p95_ms"] for r in runs),
            "itl_p95_ms_worst": max(r["metrics"]["itl_p95_ms"] for r in runs),
            "turn0_cache_hit_rate": statistics.fmean(
                r["metrics"]["turn0_cache_hit_rate"] for r in runs
            ),
        }
        for tier in TIERS:
            row[f"{tier}_attainment_min"] = min(
                r["metrics"][f"{tier}_attainment"] for r in runs
            )
            row[f"{tier}_feasible"] = row[f"{tier}_attainment_min"] >= ATTAINMENT
            row[f"{tier}_goodput_min"] = min(
                r["metrics"][f"{tier}_goodput_rps"] for r in runs
            )
        point_rows.append(row)

    capacity: dict[str, dict[str, Any]] = {}
    for tier in TIERS:
        capacity[tier] = {}
        for architecture in ARCHITECTURES:
            feasible = [
                row
                for row in point_rows
                if row["architecture"] == architecture and row[f"{tier}_feasible"]
            ]
            best = max(
                feasible, key=lambda row: row[f"{tier}_goodput_min"], default=None
            )
            capacity[tier][architecture] = (
                {
                    "concurrency": best["concurrency"],
                    "goodput": best[f"{tier}_goodput_min"],
                }
                if best
                else None
            )
        pap = [capacity[tier][a] for a in ("pap7pa1p", "pap6pa2p") if capacity[tier][a]]
        best_pap = max(pap, key=lambda item: item["goodput"], default=None)
        pd, dp = capacity[tier]["pd6p2d"], capacity[tier]["dp8"]
        capacity[tier]["pap_over_pd"] = (
            best_pap["goodput"] / pd["goodput"] if best_pap and pd else None
        )
        capacity[tier]["pap_over_dp"] = (
            best_pap["goodput"] / dp["goodput"] if best_pap and dp else None
        )

    pd_standard = capacity["Standard"]["pd6p2d"]
    summary = {
        "experiment": EXPERIMENT_DIR.name,
        "invalid_runs": invalid,
        "points": point_rows,
        "capacity": capacity,
        "pd_reproduction": {
            "july_standard_goodput": JULY_PD_STANDARD_GOODPUT,
            "current": pd_standard["goodput"] if pd_standard else None,
            "ratio": (
                pd_standard["goodput"] / JULY_PD_STANDARD_GOODPUT
                if pd_standard
                else None
            ),
        },
        "turn0_cache_hit_rate": statistics.fmean(
            r["metrics"]["turn0_cache_hit_rate"] for r in valid
        )
        if valid
        else None,
    }
    results = EXPERIMENT_DIR / "results"
    results.mkdir(exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (results / "requests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for run in valid:
            writer.writerows(run["records"])
    with (results / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)

    print(f"valid={len(valid)} invalid={len(invalid)}")
    for row in point_rows:
        flags = "".join(
            tier[0] if row[f"{tier}_feasible"] else "-"
            for tier in ("Strict", "Standard", "Relaxed")
        )
        print(
            f"{row['architecture']:9s} C{row['concurrency']:<3d} n={row['runs']} "
            f"rps_min={row['throughput_rps_min']:6.3f} "
            f"TTFTp95={row['ttft_p95_ms_worst']:7.0f} "
            f"ITLp95={row['itl_p95_ms_worst']:6.1f} "
            f"Std_good={row['Standard_goodput_min']:6.3f} feasible={flags} "
            f"turn0hit={row['turn0_cache_hit_rate']:.3f}"
        )
    for tier, entry in capacity.items():
        print(tier, json.dumps(entry))
    print("pd_reproduction", json.dumps(summary["pd_reproduction"]))


if __name__ == "__main__":
    main()
