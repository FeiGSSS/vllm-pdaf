#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate E2d-lite runs and find each cell × architecture capacity.

A point is feasible for an SLO when its mean attainment over runs is at least
95% (COMMON §2.4); its goodput is the mean of good requests per second over
runs. Latency CIs bootstrap requests within runs; goodput and throughput CIs
bootstrap runs.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
ATTAINMENT = 0.95
BOOTSTRAP_SAMPLES = 2000
METRICS = {
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}
SLOS = {
    "D1": {"ttft_ms": 5000, "itl_ms": 50, "e2e_ms": 10000},
    "D2": {"ttft_ms": 10000, "itl_ms": 75, "e2e_ms": 20000},
    "D3": {"ttft_ms": 15000, "itl_ms": 50, "e2e_ms": 20000},
    "D4": {"ttft_ms": 15000, "itl_ms": 75, "e2e_ms": 25000},
}
SLO_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
CELLS = ["l07o256", "n32ko256"]
ARCHS = ["pap7pa1p", "pd6p2d", "dp8"]
ARCHITECTURES = [f"{cell}-{arch}" for cell in CELLS for arch in ARCHS]
DYNAMO_AUDITS = {
    "6p2d": {
        "correctness_audit.env",
        "kv_transfer_audit.env",
        "vllm_cuda_graph_audit.env",
    },
    "4p4d": {
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate(root: Path) -> list[str]:
    """Return the reasons a run is invalid; empty means valid."""
    exit_code = root / "exit_code"
    if not exit_code.exists() or exit_code.read_text().strip() != "0":
        return ["nonzero or missing exit code"]
    problems = []
    requested = read_env(root / "requested.env")
    expected = int(
        requested.get("PAP_AIPERF_EXPECTED_REQUESTS")
        or requested["DYNAMO_AIPERF_EXPECTED_REQUESTS"]
    )
    profile = read_json(root / "aiperf/profile.json")
    if profile["request_count"]["avg"] != expected:
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
        if routing["status"] != "passed" or routing["route_count"] != expected:
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
    transfer = None
    breakdown = root / "ttft_breakdown.json"
    if breakdown.exists():
        kv = read_json(breakdown)["kv_transfer"]
        transfer = {
            "weighted_transfer_ms": kv["weighted_transfer_ms"],
            "weighted_mb_per_transfer": kv["weighted_mb_per_transfer"],
            "aggregate_gb_s": (kv["aggregate_throughput_mb_s"] or 0) / 1e3,
            "transfers": kv["transfers"],
        }
    return {
        "run": root.name,
        "architecture": architecture,
        "concurrency": int(point),
        "repetition": int(launch["repetition"]),
        "duration_s": duration_s,
        "records": records,
        "transfer": transfer,
    }


def good(records: list[dict], limits: dict[str, float]) -> int:
    return sum(all(r[k] <= v for k, v in limits.items()) for r in records)


def scaled(limits: dict[str, float], factor: float) -> dict[str, float]:
    return {k: v * factor for k, v in limits.items()}


def point_summary(runs: list[dict], rng: random.Random) -> dict[str, Any]:
    records = [r for run in runs for r in run["records"]]
    out: dict[str, Any] = {
        "runs": len(runs),
        "throughput_rps": statistics.fmean(
            len(run["records"]) / run["duration_s"] for run in runs
        ),
    }
    for key in ("ttft_ms", "itl_ms"):
        for q in (50, 95, 99):
            value = percentile([r[key] for r in records], q / 100)
            draws = sorted(
                percentile(
                    [
                        v
                        for run in runs
                        for v in rng.choices(
                            [r[key] for r in run["records"]], k=len(run["records"])
                        )
                    ],
                    q / 100,
                )
                for _ in range(BOOTSTRAP_SAMPLES // 10)
            )
            out[f"{key}_p{q}"] = value
            out[f"{key}_p{q}_ci"] = [percentile(draws, 0.025), percentile(draws, 0.975)]
    out["e2e_ms_mean"] = statistics.fmean(r["e2e_ms"] for r in records)
    for name, limits in SLOS.items():
        attainments = [
            good(run["records"], limits) / len(run["records"]) for run in runs
        ]
        goodputs = [good(run["records"], limits) / run["duration_s"] for run in runs]
        draws = sorted(
            statistics.fmean(rng.choices(goodputs, k=len(goodputs)))
            for _ in range(BOOTSTRAP_SAMPLES)
        )
        out[f"{name}_attainment"] = statistics.fmean(attainments)
        out[f"{name}_feasible"] = out[f"{name}_attainment"] >= ATTAINMENT
        out[f"{name}_goodput"] = statistics.fmean(goodputs)
        out[f"{name}_goodput_ci"] = [percentile(draws, 0.025), percentile(draws, 0.975)]
    for factor in SLO_SCALES:
        limits = scaled(SLOS["D2"], factor)
        attainment = statistics.fmean(
            good(run["records"], limits) / len(run["records"]) for run in runs
        )
        out[f"D2x{factor}_feasible"] = attainment >= ATTAINMENT
        out[f"D2x{factor}_goodput"] = statistics.fmean(
            good(run["records"], limits) / run["duration_s"] for run in runs
        )
    out["ttft_ms_by_turn"] = {
        turn: statistics.fmean(r["ttft_ms"] for r in records if r["turn"] == turn)
        for turn in sorted({r["turn"] for r in records})
    }
    transfers = [
        run["transfer"]
        for run in runs
        if run["transfer"] and run["transfer"]["weighted_transfer_ms"] is not None
    ]
    if transfers:
        out["kv_transfer_ms"] = statistics.fmean(
            t["weighted_transfer_ms"] for t in transfers
        )
        out["kv_mb_per_transfer"] = statistics.fmean(
            t["weighted_mb_per_transfer"] for t in transfers
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
            valid.append(load_run(root))

    rng = random.Random(0)
    points: dict[tuple[str, int], list[dict]] = {}
    for run in valid:
        points.setdefault((run["architecture"], run["concurrency"]), []).append(run)
    rows = []
    for (architecture, concurrency), runs in sorted(points.items()):
        rows.append(
            {"architecture": architecture, "concurrency": concurrency}
            | point_summary(runs, rng)
        )

    capacity: dict[str, dict[str, Any]] = {}
    for name in [*SLOS, *(f"D2x{f}" for f in SLO_SCALES)]:
        capacity[name] = {}
        for architecture in ARCHITECTURES:
            feasible = [
                row
                for row in rows
                if row["architecture"] == architecture and row[f"{name}_feasible"]
            ]
            best = max(feasible, key=lambda row: row[f"{name}_goodput"], default=None)
            capacity[name][architecture] = (
                {
                    "concurrency": best["concurrency"],
                    "goodput": best[f"{name}_goodput"],
                    "ci": best.get(f"{name}_goodput_ci"),
                    "last_feasible_concurrency": max(
                        r["concurrency"] for r in feasible
                    ),
                }
                if best
                else None
            )

    results = EXPERIMENT_DIR / "results"
    results.mkdir(exist_ok=True)
    summary = {
        "experiment": EXPERIMENT_DIR.name,
        "invalid_runs": invalid,
        "points": rows,
        "capacity": capacity,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (results / "requests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for run in valid:
            writer.writerows(run["records"])
    flat = [
        {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
        for row in rows
    ]
    with (results / "points.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in flat for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)

    print(f"valid={len(valid)} invalid={len(invalid)}")
    for row in rows:
        print(
            f"{row['architecture']:9s} C{row['concurrency']:<3d} n={row['runs']} "
            f"rps={row['throughput_rps']:6.3f} TTFT p50/p99={row['ttft_ms_p50']:6.0f}/"
            f"{row['ttft_ms_p99']:6.0f} ITL p50/p95/p99={row['itl_ms_p50']:5.1f}/"
            f"{row['itl_ms_p95']:5.1f}/{row['itl_ms_p99']:5.1f} "
            f"D2={row['D2_attainment']:.3f} good={row['D2_goodput']:.3f}"
            + (
                f" xfer={row['kv_transfer_ms']:.0f}ms"
                if "kv_transfer_ms" in row
                else ""
            )
        )
    for name in SLOS:
        print(
            name,
            json.dumps(
                {
                    a: (c and [c["concurrency"], round(c["goodput"], 3)])
                    for a, c in capacity[name].items()
                }
            ),
        )


if __name__ == "__main__":
    main()
