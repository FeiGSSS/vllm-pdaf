#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare EngineCore-queued and Snapshot PAP KV exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

import regex as re

INPUT_SHA256 = "ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5"
METRICS = {
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}
PLACEMENT = re.compile(r"PAP Dynamo placement request_id=(\S+) selected_pa=(\d+)")
EXPORT = re.compile(r"export_ms=([0-9.]+)")
COPIED = re.compile(r"PAP KV migration copied request_id=")
VERIFIED = re.compile(r"PAP KV migration verified request_id=")
EXPECTED_CONFIG = {
    "PAP_ENABLE_PA_KV_MIGRATION": "1",
    "PAP_HF_OVERRIDES": "''",
    "MAX_MODEL_LEN": "32768",
    "PAP_AIPERF_SESSIONS": "60",
    "PAP_AIPERF_EXPECTED_REQUESTS": "180",
    "PAP_AIPERF_CONCURRENCY": "30",
    "PAP_BLOCK_SIZE": "16",
    "PAP_PREFILL_MAX_NUM_BATCHED_TOKENS": "2048",
    "PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS": "256",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_config(path: Path) -> dict[str, str]:
    config = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            config[key] = value
    return config


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metric(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def analyze_run(mode: str, label: str, root: Path) -> dict[str, Any]:
    config = read_config(root / "effective_config.env")
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"{label}: {key}={config.get(key)!r}")
    expected_export = "engine_core" if mode == "engine_core" else "snapshot"
    if config.get("PAP_KV_MIGRATION_EXPORT_MODE") != expected_export:
        raise ValueError(f"{label}: export mode mismatch")
    if mode == "verify" and config.get("PAP_KV_MIGRATION_VERIFY") != "1":
        raise ValueError(f"{label}: byte verification is disabled")
    if digest(root / "aiperf/inputs.json") != INPUT_SHA256:
        raise ValueError(f"{label}: AIPerf input mismatch")

    topology = read_json(root / "topology_manifest.json")
    if not (
        topology["topology"] == "7pa1p"
        and topology["routing_policy"] == "dynamo"
        and topology["pa_kv_migration"]
        and len(topology["pa_groups"]) == 7
        and len(topology["projections"]) == 1
    ):
        raise ValueError(f"{label}: topology mismatch")
    for name in ("routing_audit.json", "migration_source_audit.json"):
        audit = read_json(root / name)
        if audit["status"] != "passed" or audit.get("errors"):
            raise ValueError(f"{label}: {name} failed")
    if read_json(root / "cleanup_audit.json")["status"] != "passed":
        raise ValueError(f"{label}: cleanup failed")

    profile = read_json(root / "aiperf/profile.json")
    if profile["error_summary"] or profile["was_cancelled"]:
        raise ValueError(f"{label}: AIPerf failed")
    if profile["osl_mismatch_count"]["avg"] != 0:
        raise ValueError(f"{label}: output-length mismatch")
    records = [
        json.loads(line)
        for line in (root / "aiperf/profile.jsonl").read_text().splitlines()
        if line
    ]
    proxy = (root / "service_logs/proxy.log").read_text(errors="replace")
    placements = {
        match.group(1): int(match.group(2)) for match in PLACEMENT.finditer(proxy)
    }
    if len(records) != 180 or len(placements) != 180:
        raise ValueError(f"{label}: incomplete client or placement records")

    rows = []
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record["metadata"]
        values = record["metrics"]
        request_id = metadata["x_request_id"]
        row = {
            "key": [metadata["conversation_id"], int(metadata["turn_index"])],
            "conversation_id": metadata["conversation_id"],
            "turn": int(metadata["turn_index"]),
            "pa": placements[request_id],
            "prompt_tokens": int(values["usage_prompt_tokens"]["value"]),
            "cached_tokens": int(values["usage_prompt_cache_read_tokens"]["value"]),
            **{
                output: float(values[source]["value"])
                for output, source in METRICS.items()
            },
        }
        rows.append(row)
        conversations[row["conversation_id"]].append(row)
    if len(conversations) != 60:
        raise ValueError(f"{label}: expected 60 conversations")
    for conversation in conversations.values():
        conversation.sort(key=lambda row: row["turn"])
        if [row["turn"] for row in conversation] != [0, 1, 2]:
            raise ValueError(f"{label}: invalid turn sequence")
        for previous, current in zip(conversation, conversation[1:]):
            current["migrated"] = current["pa"] != previous["pa"]
    for row in rows:
        row.setdefault("migrated", False)
        row["computed_tokens"] = row["prompt_tokens"] - row["cached_tokens"]

    exports = [float(value) for value in EXPORT.findall(proxy)]
    migrations = sum(row["migrated"] for row in rows)
    prefill_logs = "\n".join(
        path.read_text(errors="replace")
        for path in sorted((root / "service_logs").glob("prefill_*.log"))
    )
    copies = len(COPIED.findall(prefill_logs))
    verifications = len(VERIFIED.findall(prefill_logs))
    if len(exports) != migrations or copies != migrations:
        raise ValueError(f"{label}: migration boundaries disagree")
    if mode == "verify" and verifications != migrations:
        raise ValueError(f"{label}: byte verification count mismatch")
    if "PAP KV migration verification mismatch" in prefill_logs:
        raise ValueError(f"{label}: KV byte mismatch")

    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    cached_tokens = sum(row["cached_tokens"] for row in rows)
    result = {
        "mode": mode,
        "label": label,
        "requests": len(rows),
        "migrations": migrations,
        "copies": copies,
        "verifications": verifications,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "cache_read_rate": cached_tokens / prompt_tokens,
        "benchmark_duration_s": float(profile["benchmark_duration"]["avg"]),
        "export_ms": metric(exports),
        **{
            name: metric([row[name] for row in rows])
            for name in ("ttft_ms", "itl_ms", "e2e_ms")
        },
        "rows": rows,
        "source_patch_sha256": digest(root / "tracked_worktree.patch"),
    }
    snapshot_paths = sorted(root.glob("prefill_kv_snapshot_final_*.json"))
    if snapshot_paths:
        snapshots = [read_json(path) for path in snapshot_paths]
        requests = sum(item["requests"] for item in snapshots)
        if any(item["failures"] or item["active_exports"] for item in snapshots):
            raise ValueError(f"{label}: unhealthy Snapshot service")
        result["snapshot_service"] = {
            "requests": requests,
            "mean_ms": sum(
                item["requests"] * item["mean_service_ms"] for item in snapshots
            )
            / requests,
            "max_ms": max(item["max_service_ms"] for item in snapshots),
        }
    return result


def combine(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for run in runs for row in run["rows"]]
    exports = [value for run in runs for value in run["export_values"]]
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    cached_tokens = sum(row["cached_tokens"] for row in rows)
    migrated = [row for row in rows if row["migrated"]]
    later_stayed = [row for row in rows if row["turn"] > 0 and not row["migrated"]]
    return {
        "runs": [run["label"] for run in runs],
        "requests": len(rows),
        "migrations": len(migrated),
        "migration_rate": len(migrated) / len(rows),
        "cache_read_rate": cached_tokens / prompt_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "request_throughput_rps": len(rows)
        / sum(run["benchmark_duration_s"] for run in runs),
        "export_ms": metric(exports),
        **{
            name: metric([row[name] for row in rows])
            for name in ("ttft_ms", "itl_ms", "e2e_ms")
        },
        "migrated_ttft_ms": metric([row["ttft_ms"] for row in migrated]),
        "later_stayed_ttft_ms": metric([row["ttft_ms"] for row in later_stayed]),
    }


def paired_groups(
    engine_runs: list[dict[str, Any]], snapshot_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for engine, snapshot in zip(engine_runs, snapshot_runs):
        engine_rows = {tuple(row["key"]): row for row in engine["rows"]}
        snapshot_rows = {tuple(row["key"]): row for row in snapshot["rows"]}
        if engine_rows.keys() != snapshot_rows.keys():
            raise ValueError("paired input identities differ")
        for key, engine_row in engine_rows.items():
            snapshot_row = snapshot_rows[key]
            if engine_row["migrated"] and snapshot_row["migrated"]:
                group = "both_migrate"
            elif not engine_row["migrated"] and not snapshot_row["migrated"]:
                group = "neither_migrates"
            elif engine_row["migrated"]:
                group = "engine_core_only"
            else:
                group = "snapshot_only"
            groups[group].append(snapshot_row["ttft_ms"] - engine_row["ttft_ms"])
    return {
        name: {
            "requests": len(values),
            "mean_ttft_delta_ms": fmean(values),
            "median_ttft_delta_ms": median(values),
        }
        for name, values in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = []
    for spec in args.run:
        mode_label, separator, path = spec.partition("=")
        mode, label_separator, label = mode_label.partition(":")
        if (
            not separator
            or not label_separator
            or mode
            not in {
                "engine_core",
                "snapshot",
                "verify",
            }
        ):
            parser.error(f"invalid --run: {spec}")
        run = analyze_run(mode, label, Path(path))
        run["export_values"] = [
            float(value)
            for value in EXPORT.findall(
                (Path(path) / "service_logs/proxy.log").read_text(errors="replace")
            )
        ]
        runs.append(run)
    engine_runs = [run for run in runs if run["mode"] == "engine_core"]
    snapshot_runs = [run for run in runs if run["mode"] == "snapshot"]
    verify_runs = [run for run in runs if run["mode"] == "verify"]
    if len(engine_runs) != 2 or len(snapshot_runs) != 2 or len(verify_runs) != 1:
        raise ValueError("expected two runs per performance mode and one verify run")
    engine_patches = {run["source_patch_sha256"] for run in engine_runs}
    snapshot_patches = {run["source_patch_sha256"] for run in snapshot_runs}
    if len(engine_patches | snapshot_patches) != 1:
        raise ValueError("performance runs used different source patches")
    engine = combine(engine_runs)
    snapshot = combine(snapshot_runs)
    summary = {
        "workload": {
            "model": "Qwen3-8B FP16",
            "topology": "7PA1P on 8 NVIDIA L20 GPUs",
            "context": "native 32K",
            "sessions": 60,
            "turns": 3,
            "concurrency": 30,
            "inputs_sha256": INPUT_SHA256,
        },
        "engine_core": engine,
        "snapshot": snapshot,
        "change_percent": {
            name: (snapshot[name]["mean"] / engine[name]["mean"] - 1) * 100
            for name in ("export_ms", "ttft_ms", "itl_ms", "e2e_ms")
        }
        | {
            "request_throughput_rps": (
                snapshot["request_throughput_rps"] / engine["request_throughput_rps"]
                - 1
            )
            * 100
        },
        "paired_request_groups": paired_groups(engine_runs, snapshot_runs),
        "byte_verification": {
            "run": verify_runs[0]["label"],
            "migrations": verify_runs[0]["migrations"],
            "verified": verify_runs[0]["verifications"],
            "mismatches": 0,
        },
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key
                not in {
                    "rows",
                    "export_values",
                }
            }
            for run in runs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
