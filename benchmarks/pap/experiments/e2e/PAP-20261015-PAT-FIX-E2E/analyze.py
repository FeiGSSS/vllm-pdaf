#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate E0 runs and compare unfixed PAT, fixed PAT and Triton."""

from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPECTED_REQUESTS = 180
CONCURRENCY = 30
BOOTSTRAP_SAMPLES = 10_000
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
# Group name -> (variant, repetitions); Triton repetitions 1-3 ran in block A.
GROUPS = {
    "unfixed-auto": ("unfixed-auto", {1, 2, 3}),
    "fixed-auto": ("fixed-auto", {1, 2, 3}),
    "triton-A": ("triton", {1, 2, 3}),
    "triton-B": ("triton", {4, 5, 6}),
    "triton": ("triton", {1, 2, 3, 4, 5, 6}),
}
COMPARISONS = [
    ("fixed-auto", "unfixed-auto"),
    ("fixed-auto", "triton-B"),
    ("unfixed-auto", "triton-A"),
    ("triton-B", "triton-A"),
]
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
    problems = []
    exit_code = root / "exit_code"
    if not exit_code.exists() or exit_code.read_text().strip() != "0":
        return ["nonzero or missing exit code"]
    profile = read_json(root / "aiperf/profile.json")
    if profile["request_count"]["avg"] != EXPECTED_REQUESTS:
        problems.append("request count")
    if profile["error_summary"] or profile["was_cancelled"]:
        problems.append("AIPerf errors or cancellation")
    if profile["osl_mismatch_count"]["avg"] != 0:
        problems.append("output length mismatch")
    status_audits = [
        p
        for p in [*root.glob("*audit*.env"), *root.glob("*drain.env")]
        if not p.name.startswith("mps_static_audit")
    ]
    failed = [p.name for p in status_audits if "STATUS=passed" not in p.read_text()]
    if len(status_audits) < 16 or failed:
        problems.append(f"audits: {len(status_audits)} found, failed {failed}")
    if not all(
        "MPS_MODE=static" in text
        and "PREFILL_VISIBLE_SMS=80" in text
        and "ATTENTION_VISIBLE_SMS=12" in text
        for text in (p.read_text() for p in root.glob("mps_static_audit_pa_*.env"))
    ):
        problems.append("MPS static partition")
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
    for pattern in ("mps_static_audit_pa_*.env", "runtime_cuda_context_audit_pa_*.env"):
        audits = sorted(root.glob(pattern))
        if len(audits) != 7:
            problems.append(f"{pattern} count {len(audits)}")
    if read_env(root / "launch.env").get("source_dirty_files") != "0":
        problems.append("dirty tracked worktree")
    return problems


def load_run(root: Path) -> dict[str, Any]:
    launch = read_env(root / "launch.env")
    variant, repetition = launch["variant"], int(launch["repetition"])
    records = []
    for line in (root / "aiperf/profile.jsonl").read_text().splitlines():
        raw = json.loads(line)
        meta, metrics = raw["metadata"], raw["metrics"]
        records.append(
            {
                "architecture": "pap7pa1p",
                "variant": variant,
                "concurrency": CONCURRENCY,
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
    stats = [
        read_json(p) for p in sorted(root.glob("attention_fast_path_stats_[0-9].json"))
    ]
    return {
        "run": root.name,
        "variant": variant,
        "repetition": repetition,
        "pat_so_sha256": launch.get("pat_so_sha256"),
        "duration_s": duration_s,
        "records": records,
        "pat_rebuilds": [s.get("attention_kernel_pat_rebuilds", 0) for s in stats],
        "triton_selections": [
            s.get("attention_kernel_triton_selections", 0) for s in stats
        ],
        "migrations": read_json(root / "routing_audit.json")["kv_migration"][
            "attempts"
        ],
    }


def run_metrics(run: dict[str, Any]) -> dict[str, float]:
    records = run["records"]
    out = {"throughput_rps": len(records) / run["duration_s"]}
    for name, slo in SLOS.items():
        met = sum(all(r[k] <= v for k, v in slo.items()) for r in records)
        out[f"{name}_attainment"] = met / len(records)
        out[f"{name}_goodput_rps"] = met / run["duration_s"]
    return out


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    records = [r for run in runs for r in run["records"]]
    summary: dict[str, Any] = {
        "runs": [run["run"] for run in runs],
        "n_requests": len(records),
    }
    for key in METRICS:
        values = [r[key] for r in records]
        summary[key] = {
            "mean": statistics.fmean(values),
            **{f"p{q}": percentile(values, q / 100) for q in (50, 90, 95, 99)},
        }
    per_run = [run_metrics(run) for run in runs]
    for key in per_run[0]:
        summary[key] = statistics.fmean(m[key] for m in per_run)
    return summary


def request_mean(runs: list[dict], key: str) -> float:
    return statistics.fmean(r[key] for run in runs for r in run["records"])


def resampled_request_mean(runs: list[dict], key: str, rng: random.Random) -> float:
    total = count = 0.0
    for run in runs:
        values = [r[key] for r in run["records"]]
        total += sum(rng.choices(values, k=len(values)))
        count += len(values)
    return total / count


def bootstrap_ratio(a: list[dict], b: list[dict], rng: random.Random) -> dict[str, Any]:
    """Ratio-minus-one of group means with 95% percentile CIs.

    Latency metrics resample requests within each run; throughput and goodput
    resample runs, because each run contributes one value.
    """
    result: dict[str, Any] = {}
    for key in METRICS:
        draws = sorted(
            resampled_request_mean(a, key, rng) / resampled_request_mean(b, key, rng)
            - 1
            for _ in range(BOOTSTRAP_SAMPLES)
        )
        result[key] = {
            "ratio_minus_one": request_mean(a, key) / request_mean(b, key) - 1,
            "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        }
    ma, mb = [run_metrics(run) for run in a], [run_metrics(run) for run in b]
    for key in ("throughput_rps", "D2_goodput_rps"):
        va, vb = [m[key] for m in ma], [m[key] for m in mb]
        draws = sorted(
            statistics.fmean(rng.choices(va, k=len(va)))
            / statistics.fmean(rng.choices(vb, k=len(vb)))
            - 1
            for _ in range(BOOTSTRAP_SAMPLES)
        )
        result[key] = {
            "ratio_minus_one": statistics.fmean(va) / statistics.fmean(vb) - 1,
            "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
            "resampling": "runs",
        }
    return result


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

    groups = {
        name: [r for r in valid if r["variant"] == variant and r["repetition"] in reps]
        for name, (variant, reps) in GROUPS.items()
    }
    rng = random.Random(0)
    summary = {
        "experiment": EXPERIMENT_DIR.name,
        "invalid_runs": invalid,
        "runs": [
            {k: v for k, v in run.items() if k != "records"} | run_metrics(run)
            for run in valid
        ],
        "groups": {name: summarize(runs) for name, runs in groups.items() if runs},
        "comparisons": {
            f"{a}/{b}": bootstrap_ratio(groups[a], groups[b], rng)
            for a, b in COMPARISONS
            if groups[a] and groups[b]
        },
    }
    results = EXPERIMENT_DIR / "results"
    results.mkdir(exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (results / "requests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for run in valid:
            writer.writerows(run["records"])

    print(f"valid={len(valid)} invalid={len(invalid)}")
    for name, group in summary["groups"].items():
        print(
            f"{name:13s} n={len(group['runs'])} TTFT={group['ttft_ms']['mean']:.0f} "
            f"ITL={group['itl_ms']['mean']:.2f} rps={group['throughput_rps']:.3f} "
            f"D2={group['D2_attainment']:.3f}"
        )
    for name, comparison in summary["comparisons"].items():
        cells = [
            f"{k} {v['ratio_minus_one']:+.2%} [{v['ci95'][0]:+.2%},{v['ci95'][1]:+.2%}]"
            for k, v in comparison.items()
        ]
        print(f"{name}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
