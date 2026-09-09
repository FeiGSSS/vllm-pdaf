#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare same-host and cross-host PAP with Snapshot KV migration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import regex as re

INPUT_SHA256 = "ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5"
BYTES_PER_TOKEN = 2 * 8 * 128 * 2 * 36
PLACEMENT = re.compile(r"PAP Dynamo placement request_id=(\S+) selected_pa=(\d+)")
REQUEST = re.compile(
    r"request_id=(\S+).* pa_index=(\d+).* prefill_ms=(\d+).* "
    r"migration_prefix_tokens=(\d+)"
)
PREPARED = re.compile(
    r"PAP KV migration prepared request_id=(\S+).* prefix_tokens=(\d+)"
)
EXPORT = re.compile(
    r"PAP KV migration export timing request_id=\S+ transfer_id=\S+ "
    r"export_ms=([0-9.]+)"
)
COPIED = re.compile(r"PAP KV migration copied request_id=(\S+).* tokens=(\d+)")
COPY_TIME = re.compile(
    r"PAP KV migration copy timing request_id=(\S+) copy_ms=([0-9.]+)"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(path: Path) -> dict[str, str]:
    config = {}
    for line in path.read_text().splitlines():
        line = line.removeprefix("export ")
        words = shlex.split(line)
        if len(words) != 1 or "=" not in words[0]:
            raise ValueError(f"invalid effective config line: {line!r}")
        key, value = words[0].split("=", 1)
        config[key] = value
    return config


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


def validate_run(side: str, label: str, root: Path) -> None:
    config = read_config(root / "effective_config.env")
    common = {
        "PAP_AIPERF_SESSIONS": "60",
        "PAP_AIPERF_EXPECTED_REQUESTS": "180",
        "PAP_AIPERF_CONCURRENCY": "30",
    }
    expected = (
        {
            **common,
            "PAP_ENABLE_PA_KV_MIGRATION": "1",
            "PAP_KV_MIGRATION_EXPORT_MODE": "snapshot",
            "PAP_PREFILL_IPC_PROFILE": "0",
        }
        if side == "same"
        else {
            **common,
            "PAP_CROSS_HOST_ENABLE_PA_KV_MIGRATION": "1",
            "PAP_CROSS_HOST_ROUTING_TRACE": "0",
            "PYTHONHASHSEED": "123",
        }
    )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{label}: {key}={config.get(key)!r}")
    if digest(root / "aiperf/inputs.json") != INPUT_SHA256:
        raise ValueError(f"{label}: AIPerf inputs differ")
    profile = read_json(root / "aiperf/profile.json")
    if (
        profile["request_count"]["avg"] != 180
        or profile["error_summary"]
        or profile["was_cancelled"]
        or profile["osl_mismatch_count"]["avg"] != 0
    ):
        raise ValueError(f"{label}: AIPerf correctness failed")
    if read_json(root / "cleanup_audit.json")["status"] != "passed":
        raise ValueError(f"{label}: cleanup failed")
    if side == "same":
        topology = read_json(root / "topology_manifest.json")
        if not (
            topology["topology"] == "7pa1p"
            and topology["routing_policy"] == "dynamo"
            and topology["pa_kv_migration"]
            and len(topology["pa_groups"]) == 7
            and len(topology["projections"]) == 1
        ):
            raise ValueError(f"{label}: same-host topology mismatch")
        for name in ("routing_audit.json", "migration_source_audit.json"):
            audit = read_json(root / name)
            if audit["status"] != "passed" or audit.get("errors"):
                raise ValueError(f"{label}: {name} failed")
    else:
        for name in (
            "performance_summary.json",
            "source_audit.json",
            "block_hash_audit.json",
            "startup_topology_audit.json",
        ):
            audit = read_json(root / name)
            if audit["status"] != "passed" or audit.get("errors"):
                raise ValueError(f"{label}: {name} failed")


def analyze_run(side: str, label: str, root: Path) -> dict[str, Any]:
    validate_run(side, label, root)
    gateway_log = root / (
        "service_logs/proxy.log" if side == "same" else "service_logs/gateway.log"
    )
    gateway_text = gateway_log.read_text(errors="replace")
    placements = {
        match.group(1): int(match.group(2))
        for match in PLACEMENT.finditer(gateway_text)
    }
    request_boundaries = {
        match.group(1): {
            "pa": int(match.group(2)),
            "prefill_ms": int(match.group(3)),
            "migration_prefix_tokens": int(match.group(4)),
        }
        for match in REQUEST.finditer(gateway_text)
    }
    prepared = [int(match.group(2)) for match in PREPARED.finditer(gateway_text)]
    export_ms = [float(value) for value in EXPORT.findall(gateway_text)]
    prefill_dir = root / ("service_logs" if side == "same" else "remote_logs")
    prefill_text = "\n".join(
        path.read_text(errors="replace")
        for path in sorted(prefill_dir.glob("prefill_*.log"))
    )
    copied_tokens = {
        match.group(1): int(match.group(2)) for match in COPIED.finditer(prefill_text)
    }
    copy_ms = {
        match.group(1): float(match.group(2))
        for match in COPY_TIME.finditer(prefill_text)
    }
    if copied_tokens.keys() != copy_ms.keys():
        raise ValueError(f"{label}: Copy and timing request IDs differ")

    records = [
        json.loads(line)
        for line in (root / "aiperf/profile.jsonl").read_text().splitlines()
        if line
    ]
    if len(records) != len(placements) or len(records) != len(request_boundaries):
        raise ValueError(f"{label}: request boundary counts differ")
    rows = []
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        metadata = record["metadata"]
        values = record["metrics"]
        request_id = metadata["x_request_id"]
        boundary = request_boundaries[request_id]
        if boundary["pa"] != placements[request_id]:
            raise ValueError(f"{label}: placement mismatch for {request_id}")
        row = {
            "side": side,
            "run": label,
            "request_id": request_id,
            "conversation_id": metadata["conversation_id"],
            "turn": int(metadata["turn_index"]),
            "pa": placements[request_id],
            "migration_prefix_tokens": boundary["migration_prefix_tokens"],
            "prompt_tokens": int(values["usage_prompt_tokens"]["value"]),
            "cached_tokens": int(values["usage_prompt_cache_read_tokens"]["value"]),
            "prefill_ms": boundary["prefill_ms"],
            "ttft_ms": float(values["time_to_first_token"]["value"]),
            "itl_ms": float(values["inter_token_latency"]["value"]),
            "e2e_ms": float(values["request_latency"]["value"]),
        }
        row["computed_tokens"] = row["prompt_tokens"] - row["cached_tokens"]
        rows.append(row)
        conversations[row["conversation_id"]].append(row)
    if len(rows) != 180 or len(conversations) != 60:
        raise ValueError(f"{label}: workload cardinality mismatch")
    for conversation in conversations.values():
        conversation.sort(key=lambda row: row["turn"])
        if [row["turn"] for row in conversation] != [0, 1, 2]:
            raise ValueError(f"{label}: invalid conversation turn sequence")
        for previous, current in zip(conversation, conversation[1:]):
            current["migrated"] = current["pa"] != previous["pa"]
    for row in rows:
        row.setdefault("migrated", False)

    migrations = [row for row in rows if row["migrated"]]
    if not (len(migrations) == len(prepared) == len(export_ms) == len(copied_tokens)):
        raise ValueError(f"{label}: migration boundary counts differ")
    if any(row["migration_prefix_tokens"] <= 0 for row in migrations):
        raise ValueError(f"{label}: migrated request has no copied prefix")
    if side == "cross":
        migration = read_json(root / "topology_stats.json")["kv_migration"]
        if not (
            migration["enabled"]
            and migration["attempts"] == len(migrations)
            and migration["exports"] == len(migrations)
            and migration["misses"] == 0
            and migration["failures"] == 0
            and migration["pending_releases"] == 0
        ):
            raise ValueError(f"{label}: migration coordinator audit failed")

    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    cached_tokens = sum(row["cached_tokens"] for row in rows)
    copied_bytes = {
        request_id: tokens * BYTES_PER_TOKEN
        for request_id, tokens in copied_tokens.items()
    }
    bandwidth = [
        copied_bytes[request_id] / (copy_ms[request_id] / 1000) / 1e9
        for request_id in copied_tokens
    ]
    profile = read_json(root / "aiperf/profile.json")
    snapshots = [
        read_json(path)
        for path in sorted(root.glob("prefill_kv_snapshot_final_*.json"))
    ]
    if len(snapshots) != 7 or any(
        not item["ready"] or item["failures"] or item["active_exports"]
        for item in snapshots
    ):
        raise ValueError(f"{label}: Snapshot service audit failed")
    return {
        "side": side,
        "run": label,
        "root": root.name,
        "requests": len(rows),
        "migrations": len(migrations),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "cache_read_rate": cached_tokens / prompt_tokens,
        "benchmark_duration_s": float(profile["benchmark_duration"]["avg"]),
        "copied_prefix_tokens": sum(prepared),
        "copied_tokens": sum(copied_tokens.values()),
        "export_ms": metric(export_ms),
        "copy_ms": metric(list(copy_ms.values())),
        "copy_bandwidth_gbps": metric(bandwidth),
        "copy_bandwidth_gbps_aggregate": sum(copied_bytes.values())
        / (sum(copy_ms.values()) / 1000)
        / 1e9,
        **{
            name: metric([float(row[name]) for row in rows])
            for name in ("prefill_ms", "ttft_ms", "itl_ms", "e2e_ms")
        },
        "rows": rows,
    }


def combine(side: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for run in runs for row in run["rows"]]
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    cached_tokens = sum(row["cached_tokens"] for row in rows)
    result = {
        "runs": [run["run"] for run in runs],
        "requests": len(rows),
        "migrations": sum(run["migrations"] for run in runs),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "computed_tokens_per_run": (prompt_tokens - cached_tokens) / len(runs),
        "cache_read_rate": cached_tokens / prompt_tokens,
        "request_throughput_rps": len(rows)
        / sum(run["benchmark_duration_s"] for run in runs),
        **{
            name: metric([float(row[name]) for row in rows])
            for name in ("prefill_ms", "ttft_ms", "itl_ms", "e2e_ms")
        },
        "by_turn": {},
    }
    migrations = result["migrations"]
    result["copied_prefix_tokens"] = sum(run["copied_prefix_tokens"] for run in runs)
    result["copied_tokens"] = sum(run["copied_tokens"] for run in runs)
    result["export_ms_mean"] = (
        sum(run["export_ms"]["mean"] * run["migrations"] for run in runs) / migrations
    )
    result["copy_ms_mean"] = (
        sum(run["copy_ms"]["mean"] * run["migrations"] for run in runs) / migrations
    )
    copied_bytes = sum(run["copied_tokens"] for run in runs) * BYTES_PER_TOKEN
    copy_time_ms = sum(run["copy_ms"]["mean"] * run["migrations"] for run in runs)
    result["copy_bandwidth_gbps_aggregate"] = copied_bytes / (copy_time_ms / 1000) / 1e9
    for turn in range(3):
        selected = [row for row in rows if row["turn"] == turn]
        turn_prompt = sum(row["prompt_tokens"] for row in selected)
        turn_cached = sum(row["cached_tokens"] for row in selected)
        result["by_turn"][str(turn)] = {
            "requests": len(selected),
            "migrations": sum(row["migrated"] for row in selected),
            "cache_read_rate": turn_cached / turn_prompt,
            "computed_tokens": turn_prompt - turn_cached,
            "prefill_ms": metric([float(row["prefill_ms"]) for row in selected]),
            "ttft_ms": metric([float(row["ttft_ms"]) for row in selected]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="SIDE:LABEL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests-output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        identity, separator, path = spec.partition("=")
        side, side_separator, label = identity.partition(":")
        if not separator or not side_separator or side not in {"same", "cross"}:
            parser.error(f"invalid --run: {spec}")
        runs.append(analyze_run(side, label, Path(path)))
    grouped = {
        side: combine(side, [run for run in runs if run["side"] == side])
        for side in ("same", "cross")
    }
    if any(len(grouped[side]["runs"]) != 2 for side in grouped):
        raise ValueError("expected two same-host and two cross-host runs")
    same = grouped["same"]
    cross = grouped["cross"]
    comparison = {
        "ttft_mean_ms_delta": cross["ttft_ms"]["mean"] - same["ttft_ms"]["mean"],
        "ttft_mean_relative": cross["ttft_ms"]["mean"] / same["ttft_ms"]["mean"] - 1,
        "prefill_mean_ms_delta": cross["prefill_ms"]["mean"]
        - same["prefill_ms"]["mean"],
        "cache_read_rate_delta": cross["cache_read_rate"] - same["cache_read_rate"],
        "computed_tokens_per_run_delta": cross["computed_tokens_per_run"]
        - same["computed_tokens_per_run"],
        "itl_mean_ms_delta": cross["itl_ms"]["mean"] - same["itl_ms"]["mean"],
        "e2e_mean_ms_delta": cross["e2e_ms"]["mean"] - same["e2e_ms"]["mean"],
        "request_throughput_relative": cross["request_throughput_rps"]
        / same["request_throughput_rps"]
        - 1,
    }
    serializable_runs = [{k: v for k, v in run.items() if k != "rows"} for run in runs]
    args.output.write_text(
        json.dumps(
            {"runs": serializable_runs, "combined": grouped, "comparison": comparison},
            indent=2,
        )
        + "\n"
    )
    fields = list(runs[0]["rows"][0])
    with args.requests_output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            writer.writerows(run["rows"])


if __name__ == "__main__":
    main()
