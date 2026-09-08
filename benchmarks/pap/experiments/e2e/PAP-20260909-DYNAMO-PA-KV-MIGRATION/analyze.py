# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare same-host Dynamo PA routing with and without KV migration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import regex as re

DYNAMO_PATTERN = re.compile(r"PAP Dynamo placement request_id=(\S+) selected_pa=(\d+)")
PREFILL_PATTERN = re.compile(r"request_id=(\S+).*prefill_ms=(\d+) ")
PREPARED_PATTERN = re.compile(
    r"PAP KV migration prepared request_id=(\S+) conversation_id=(\S+) "
    r"source_pa=(\d+) target_pa=(\d+) prefix_tokens=(\d+) blocks=(\d+)"
)
EXPORT_TIMING_PATTERN = re.compile(
    r"PAP KV migration export timing request_id=(\S+) transfer_id=\S+ "
    r"export_ms=([0-9.]+)"
)
RELEASE_TIMING_PATTERN = re.compile(
    r"PAP KV migration release timing request_id=(\S+) transfer_id=\S+ "
    r"release_ms=([0-9.]+)"
)
COPY_PATTERN = re.compile(
    r"PAP KV migration copied request_id=(\S+) source_pa=(\d+) blocks=(\d+) "
    r"tokens=(\d+) ranges=(\d+)"
)
COPY_TIMING_PATTERN = re.compile(
    r"PAP KV migration copy timing request_id=(\S+) copy_ms=([0-9.]+)"
)
GATEWAY_ID_PATTERN = re.compile(
    r"chatcmpl-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12})-"
)
METRICS = {
    "ttft_ms": "time_to_first_token",
    "itl_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}
KV_BYTES_PER_TOKEN = 36 * 8 * 256 * 2
CONFIG_EXCLUSIONS = {
    "PAP_AIPERF_OUTPUT_DIR",
    "PAP_ENABLE_PA_KV_MIGRATION",
    "RUN_LOG_DIR",
    "RUN_ROOT",
}
METADATA_EXCLUSIONS = {
    "result_root",
    "pa_kv_migration",
    "run_id",
    "started_at",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_effective_config(path: Path) -> dict[str, str]:
    config = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key not in CONFIG_EXCLUSIONS:
            config[key] = value
    return config


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(row["prompt_tokens"] for row in rows)
    cached_tokens = sum(row["cached_tokens"] for row in rows)
    return {
        "requests": len(rows),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "cache_read_rate": cached_tokens / prompt_tokens,
        "prefill_ms": metric_summary([row["prefill_ms"] for row in rows]),
        **{metric: metric_summary([row[metric] for row in rows]) for metric in METRICS},
    }


def optional_metric_summary(values: list[float]) -> dict[str, float] | None:
    return metric_summary(values) if values else None


def gateway_request_id(worker_request_id: str) -> str:
    match = GATEWAY_ID_PATTERN.search(worker_request_id)
    if match is None:
        raise ValueError(f"cannot recover Gateway request ID from {worker_request_id}")
    return match.group(1)


def analyze_run(
    label: str, mode: str, root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = read_json(root / "aiperf/profile.json")
    routing_audit = read_json(root / "routing_audit.json")
    cleanup_audit = read_json(root / "cleanup_audit.json")
    if profile["error_summary"] or profile["was_cancelled"]:
        raise ValueError(f"{label}: AIPerf reported errors or cancellation")
    if profile["osl_mismatch_count"]["avg"] != 0:
        raise ValueError(f"{label}: output length mismatch")
    if routing_audit["status"] != "passed" or cleanup_audit["status"] != "passed":
        raise ValueError(f"{label}: routing or cleanup audit failed")

    manifest = read_json(root / "topology_manifest.json")
    if (
        manifest["topology"] != "7pa1p"
        or len(manifest["pa_groups"]) != 7
        or len(manifest["projections"]) != 1
        or manifest["routing_policy"] != "dynamo"
        or bool(manifest["pa_kv_migration"]) != (mode == "migration")
    ):
        raise ValueError(f"{label}: topology manifest does not match the experiment")

    proxy_text = (root / "service_logs/proxy.log").read_text(errors="replace")
    if mode not in {"control", "migration"}:
        raise ValueError(f"unsupported mode: {mode}")
    placements = {
        match.group(1): int(match.group(2))
        for match in DYNAMO_PATTERN.finditer(proxy_text)
    }
    prefill_ms = {
        match.group(1): int(match.group(2))
        for match in PREFILL_PATTERN.finditer(proxy_text)
    }
    prepared = {
        match.group(1): {
            "source_pa": int(match.group(3)),
            "target_pa": int(match.group(4)),
            "exported_tokens": int(match.group(5)),
            "exported_blocks": int(match.group(6)),
        }
        for match in PREPARED_PATTERN.finditer(proxy_text)
    }
    export_ms = {
        match.group(1): float(match.group(2))
        for match in EXPORT_TIMING_PATTERN.finditer(proxy_text)
    }
    release_ms = {
        match.group(1): float(match.group(2))
        for match in RELEASE_TIMING_PATTERN.finditer(proxy_text)
    }
    prefill_logs = "\n".join(
        path.read_text(errors="replace")
        for path in sorted((root / "service_logs").glob("prefill_*.log"))
    )
    copied = {}
    for match in COPY_PATTERN.finditer(prefill_logs):
        request_id = gateway_request_id(match.group(1))
        copied[request_id] = {
            "copied_blocks": int(match.group(3)),
            "copied_tokens": int(match.group(4)),
            "copy_ranges": int(match.group(5)),
        }
    copy_ms = {
        gateway_request_id(match.group(1)): float(match.group(2))
        for match in COPY_TIMING_PATTERN.finditer(prefill_logs)
    }
    client_records = read_jsonl(root / "aiperf/profile.jsonl")
    if len(client_records) != 180 or len(placements) != 180 or len(prefill_ms) != 180:
        raise ValueError(
            f"{label}: expected 180 client/placement/Prefill records, got "
            f"{len(client_records)}/{len(placements)}/{len(prefill_ms)}"
        )

    rows = []
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for client in client_records:
        metadata = client["metadata"]
        request_id = metadata["x_request_id"]
        metrics = client["metrics"]
        row = {
            "run": label,
            "mode": mode,
            "request_id": request_id,
            "conversation_id": metadata["x_correlation_id"],
            "turn": int(metadata["turn_index"]),
            "pa": placements[request_id],
            "prompt_tokens": int(metrics["usage_prompt_tokens"]["value"]),
            "cached_tokens": int(metrics["usage_prompt_cache_read_tokens"]["value"]),
            "output_tokens": int(metrics["output_sequence_length"]["value"]),
            "prefill_ms": prefill_ms[request_id],
            "migration_exported_tokens": prepared.get(request_id, {}).get(
                "exported_tokens", 0
            ),
            "migration_copied_tokens": copied.get(request_id, {}).get(
                "copied_tokens", 0
            ),
            "migration_copy_ranges": copied.get(request_id, {}).get("copy_ranges", 0),
            "migration_export_ms": export_ms.get(request_id, 0.0),
            "migration_copy_ms": copy_ms.get(request_id, 0.0),
            "migration_release_ms": release_ms.get(request_id, 0.0),
            **{
                output: float(metrics[source]["value"])
                for output, source in METRICS.items()
            },
        }
        row["computed_tokens"] = row["prompt_tokens"] - row["cached_tokens"]
        row["migration_copied_bytes"] = (
            row["migration_copied_tokens"] * KV_BYTES_PER_TOKEN
        )
        row["migration_copy_bandwidth_gbps"] = (
            row["migration_copied_bytes"] / row["migration_copy_ms"] / 1_000_000
            if row["migration_copy_ms"]
            else 0.0
        )
        rows.append(row)
        conversations[row["conversation_id"]].append(row)
    if len(conversations) != 60:
        raise ValueError(f"{label}: expected 60 conversations")

    migrations = 0
    for conversation_rows in conversations.values():
        conversation_rows.sort(key=lambda row: row["turn"])
        if [row["turn"] for row in conversation_rows] != [0, 1, 2]:
            raise ValueError(f"{label}: invalid turn sequence")
        for previous, current in zip(conversation_rows, conversation_rows[1:]):
            current["previous_pa"] = previous["pa"]
            current["migrated"] = current["pa"] != previous["pa"]
            migrations += current["migrated"]
    for row in rows:
        row.setdefault("previous_pa", "")
        row.setdefault("migrated", False)

    migrated_ids = {row["request_id"] for row in rows if row["migrated"]}
    if mode == "migration":
        if not (
            set(prepared)
            == set(copied)
            == set(copy_ms)
            == set(export_ms)
            == set(release_ms)
            == migrated_ids
        ):
            raise ValueError(f"{label}: migration boundaries do not match PA moves")
    elif any((prepared, copied, copy_ms, export_ms, release_ms)):
        raise ValueError(f"{label}: control run contains KV migration activity")
    for request_id, item in prepared.items():
        row = next(row for row in rows if row["request_id"] == request_id)
        if (
            item["source_pa"] != row["previous_pa"]
            or item["target_pa"] != row["pa"]
            or copied[request_id]["copied_tokens"] > item["exported_tokens"]
        ):
            raise ValueError(f"{label}: invalid migration mapping for {request_id}")

    route_counts = Counter(row["pa"] for row in rows)
    summary = {
        "run": label,
        "mode": mode,
        "root": str(root.resolve()),
        "requests": len(rows),
        "conversations": len(conversations),
        "transitions": 120,
        "migrations": migrations,
        "migration_rate": migrations / 120,
        "route_counts": dict(sorted(route_counts.items())),
        "route_count_range": max(route_counts.values()) - min(route_counts.values()),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "cached_tokens": sum(row["cached_tokens"] for row in rows),
        "computed_tokens": sum(row["computed_tokens"] for row in rows),
        "cache_read_rate": sum(row["cached_tokens"] for row in rows)
        / sum(row["prompt_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "benchmark_duration_s": float(profile["benchmark_duration"]["avg"]),
        "request_throughput": float(profile["request_throughput"]["avg"]),
        "output_token_throughput": float(profile["output_token_throughput"]["avg"]),
        "prefill_ms": metric_summary([row["prefill_ms"] for row in rows]),
        **{metric: metric_summary([row[metric] for row in rows]) for metric in METRICS},
        "by_turn": {},
        "migration": {
            **routing_audit["kv_migration"],
            "copied_tokens": sum(row["migration_copied_tokens"] for row in rows),
            "copied_bytes": sum(row["migration_copied_bytes"] for row in rows),
            "export_ms": optional_metric_summary(list(export_ms.values())),
            "copy_ms": optional_metric_summary(list(copy_ms.values())),
            "release_ms": optional_metric_summary(list(release_ms.values())),
            "copy_bandwidth_gbps": optional_metric_summary(
                [
                    row["migration_copy_bandwidth_gbps"]
                    for row in rows
                    if row["migration_copy_bandwidth_gbps"]
                ]
            ),
        },
        "source_fingerprint": read_json(root / "source_fingerprint.json"),
        "dataset_selection_sha256": sha256(root / "dataset_selection.json"),
        "effective_config": read_effective_config(root / "effective_config.env"),
        "run_metadata": {
            key: value
            for key, value in read_json(root / "run_metadata.json").items()
            if key not in METADATA_EXCLUSIONS
        },
        "topology_manifest": {
            key: value for key, value in manifest.items() if key != "pa_kv_migration"
        },
    }
    for turn in range(3):
        selected = [row for row in rows if row["turn"] == turn]
        prompt_tokens = sum(row["prompt_tokens"] for row in selected)
        cached_tokens = sum(row["cached_tokens"] for row in selected)
        summary["by_turn"][str(turn)] = {
            "requests": len(selected),
            "migrations": sum(row["migrated"] for row in selected),
            "cache_read_rate": cached_tokens / prompt_tokens,
            "computed_tokens": prompt_tokens - cached_tokens,
            "prefill_ms": metric_summary([row["prefill_ms"] for row in selected]),
            **{
                metric: metric_summary([row[metric] for row in selected])
                for metric in METRICS
            },
        }
    return summary, rows


def audit_comparability(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if Counter(run["mode"] for run in runs) != {"control": 2, "migration": 2}:
        raise ValueError("experiment requires two accepted runs per migration mode")
    fields = (
        "source_fingerprint",
        "dataset_selection_sha256",
        "effective_config",
        "run_metadata",
        "topology_manifest",
    )
    reference = runs[0]
    mismatches = [
        field
        for field in fields
        if any(run[field] != reference[field] for run in runs[1:])
    ]
    if mismatches:
        raise ValueError(
            "runs are not a one-variable routing-policy comparison: "
            + ", ".join(mismatches)
        )
    return {
        "status": "passed",
        "runs_per_mode": 2,
        "only_intentional_variable": "PAP_ENABLE_PA_KV_MIGRATION",
        "matched_fields": list(fields),
        "source_fingerprint": reference["source_fingerprint"],
        "dataset_selection_sha256": reference["dataset_selection_sha256"],
    }


def combine(mode: str, runs: list[dict[str, Any]], rows: list[dict[str, Any]]):
    selected_runs = [run for run in runs if run["mode"] == mode]
    selected_rows = [row for row in rows if row["mode"] == mode]
    duration = sum(run["benchmark_duration_s"] for run in selected_runs)
    output_tokens = sum(row["output_tokens"] for row in selected_rows)
    prompt_tokens = sum(row["prompt_tokens"] for row in selected_rows)
    cached_tokens = sum(row["cached_tokens"] for row in selected_rows)
    route_counts = Counter(row["pa"] for row in selected_rows)
    transitions = [row for row in selected_rows if row["turn"] > 0]
    transition_groups = {}
    for migrated in (False, True):
        group = [row for row in transitions if row["migrated"] is migrated]
        if group:
            transition_groups["migrated" if migrated else "stayed"] = rows_summary(
                group
            )
    return {
        "mode": mode,
        "runs": [run["run"] for run in selected_runs],
        "requests": len(selected_rows),
        "transitions": sum(run["transitions"] for run in selected_runs),
        "migrations": sum(run["migrations"] for run in selected_runs),
        "migration_rate": sum(run["migrations"] for run in selected_runs)
        / sum(run["transitions"] for run in selected_runs),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "cache_read_rate": cached_tokens / prompt_tokens,
        "route_counts": dict(sorted(route_counts.items())),
        "route_count_range": max(route_counts.values()) - min(route_counts.values()),
        "benchmark_duration_s": duration,
        "request_throughput": len(selected_rows) / duration,
        "output_token_throughput": output_tokens / duration,
        "prefill_ms": metric_summary([row["prefill_ms"] for row in selected_rows]),
        **{
            metric: metric_summary([row[metric] for row in selected_rows])
            for metric in METRICS
        },
        "by_turn": {
            str(turn): {
                **rows_summary([row for row in selected_rows if row["turn"] == turn]),
                "migrations": sum(
                    row["migrated"] for row in selected_rows if row["turn"] == turn
                ),
            }
            for turn in range(3)
        },
        "by_transition": transition_groups,
        "migration": {
            "exports": sum(run["migration"]["exports"] for run in selected_runs),
            "misses": sum(run["migration"]["misses"] for run in selected_runs),
            "exported_tokens": sum(
                run["migration"]["exported_prefix_tokens"] for run in selected_runs
            ),
            "copied_tokens": sum(
                row["migration_copied_tokens"] for row in selected_rows
            ),
            "copied_bytes": sum(row["migration_copied_bytes"] for row in selected_rows),
            "export_ms": optional_metric_summary(
                [
                    row["migration_export_ms"]
                    for row in selected_rows
                    if row["migration_export_ms"]
                ]
            ),
            "copy_ms": optional_metric_summary(
                [
                    row["migration_copy_ms"]
                    for row in selected_rows
                    if row["migration_copy_ms"]
                ]
            ),
            "release_ms": optional_metric_summary(
                [
                    row["migration_release_ms"]
                    for row in selected_rows
                    if row["migration_release_ms"]
                ]
            ),
            "copy_bandwidth_gbps": optional_metric_summary(
                [
                    row["migration_copy_bandwidth_gbps"]
                    for row in selected_rows
                    if row["migration_copy_bandwidth_gbps"]
                ]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", required=True, metavar="MODE:LABEL=PATH"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    summaries = []
    rows = []
    for spec in args.run:
        mode_label, separator, path = spec.partition("=")
        mode, label_separator, label = mode_label.partition(":")
        if not separator or not label_separator or not mode or not label or not path:
            parser.error(f"invalid --run value: {spec}")
        summary, request_rows = analyze_run(label, mode, Path(path))
        summaries.append(summary)
        rows.extend(request_rows)
    comparability = audit_comparability(summaries)
    modes = {mode: combine(mode, summaries, rows) for mode in ("control", "migration")}
    control = modes["control"]
    migration = modes["migration"]
    comparison = {
        "migration_rate_percentage_points": (
            migration["migration_rate"] - control["migration_rate"]
        )
        * 100,
        "cache_read_rate_percentage_points": (
            migration["cache_read_rate"] - control["cache_read_rate"]
        )
        * 100,
        "computed_tokens_percent": (
            migration["computed_tokens"] / control["computed_tokens"] - 1
        )
        * 100,
        **{
            f"{metric}_mean_percent": (
                migration[metric]["mean"] / control[metric]["mean"] - 1
            )
            * 100
            for metric in ("prefill_ms", *METRICS)
        },
        "request_throughput_percent": (
            migration["request_throughput"] / control["request_throughput"] - 1
        )
        * 100,
        "output_token_throughput_percent": (
            migration["output_token_throughput"] / control["output_token_throughput"]
            - 1
        )
        * 100,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "comparability": comparability,
                "runs": summaries,
                "modes": modes,
                "comparison": comparison,
            },
            indent=2,
        )
        + "\n"
    )
    fields = [
        "run",
        "mode",
        "request_id",
        "conversation_id",
        "turn",
        "previous_pa",
        "pa",
        "migrated",
        "prompt_tokens",
        "cached_tokens",
        "computed_tokens",
        "output_tokens",
        "prefill_ms",
        "migration_exported_tokens",
        "migration_copied_tokens",
        "migration_copy_ranges",
        "migration_export_ms",
        "migration_copy_ms",
        "migration_release_ms",
        "migration_copied_bytes",
        "migration_copy_bandwidth_gbps",
        "ttft_ms",
        "itl_ms",
        "e2e_ms",
    ]
    with (args.output_dir / "requests.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
