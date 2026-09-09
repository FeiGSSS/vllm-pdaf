# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join Dynamo routing traces with AIPerf request records."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_by_worker(record: dict[str, Any]) -> dict[int, dict[str, int]]:
    groups = record["loads_before"]
    if len(groups) != 1:
        raise ValueError("expected exactly one Dynamo routing group")
    return {int(row["worker_id"]): row for row in groups[0]["loads"]}


def overlap_by_worker(record: dict[str, Any]) -> dict[int, int]:
    block_size = int(record["overlap_scores"]["block_size"])
    return {
        int(row["worker_id"]): int(row["device_blocks"]) * block_size
        for row in record["overlap_scores"]["workers"]
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def read_effective_config(path: Path) -> dict[str, str]:
    config = {}
    for line in path.read_text().splitlines():
        words = shlex.split(line.removeprefix("export "))
        if len(words) != 1 or "=" not in words[0]:
            raise ValueError(f"invalid effective config line: {line!r}")
        key, value = words[0].split("=", 1)
        config[key] = value
    return config


def analyze_run(label: str, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = json.loads((root / "aiperf/profile.json").read_text())
    performance = json.loads((root / "performance_summary.json").read_text())
    topology = json.loads((root / "topology_stats.json").read_text())
    source_audit = json.loads((root / "source_audit.json").read_text())
    cleanup_audit = json.loads((root / "cleanup_audit.json").read_text())
    config = read_effective_config(root / "effective_config.env")
    if profile["request_count"]["avg"] != 180:
        raise ValueError(f"{label}: AIPerf request count is not 180")
    if profile["error_summary"] or profile["was_cancelled"]:
        raise ValueError(f"{label}: AIPerf contains errors or cancellation")
    if performance["status"] != "passed" or performance["errors"]:
        raise ValueError(f"{label}: performance audit failed")
    if source_audit["status"] != "passed":
        raise ValueError(f"{label}: source audit failed")
    if cleanup_audit["status"] != "passed":
        raise ValueError(f"{label}: cleanup audit failed")
    if topology["lifecycle"]["completed"] != 180:
        raise ValueError(f"{label}: lifecycle request count is not 180")
    if topology["lifecycle"]["failed"] != 0:
        raise ValueError(f"{label}: lifecycle contains failures")
    expected_config = {
        "PAP_AIPERF_SESSIONS": "60",
        "PAP_AIPERF_EXPECTED_REQUESTS": "180",
        "PAP_AIPERF_CONCURRENCY": "30",
        "PAP_CROSS_HOST_ROUTING_TRACE": "1",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{label}: effective {key}={config.get(key)!r}, expected {expected!r}"
            )
    block_hash_path = root / "block_hash_audit.json"
    block_hash_audit = (
        json.loads(block_hash_path.read_text()) if block_hash_path.exists() else None
    )
    if block_hash_audit is not None and block_hash_audit["status"] != "passed":
        raise ValueError(f"{label}: block hash audit failed")
    trace_records = read_jsonl(root / "routing_trace.jsonl")
    selections = {
        row["request_id"]: row for row in trace_records if row["event"] == "selection"
    }
    prefill_results = {
        row["request_id"]: row
        for row in trace_records
        if row["event"] == "prefill_result"
    }
    aiperf_records = read_jsonl(root / "aiperf/profile.jsonl")
    if len(selections) != 180 or len(prefill_results) != 180:
        raise ValueError(
            f"{label}: expected 180 selection and Prefill records, got "
            f"{len(selections)} and {len(prefill_results)}"
        )
    if len(aiperf_records) != 180:
        raise ValueError(f"{label}: expected 180 AIPerf records")

    joined: list[dict[str, Any]] = []
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for client in aiperf_records:
        metadata = client["metadata"]
        request_id = metadata["x_request_id"]
        selection = selections[request_id]
        prefill = prefill_results[request_id]
        conversation_id = metadata["x_correlation_id"]
        if selection["conversation_id"] != conversation_id:
            raise ValueError(f"{label}: conversation ID mismatch for {request_id}")
        if prefill["conversation_id"] != conversation_id:
            raise ValueError(f"{label}: Prefill conversation mismatch for {request_id}")
        overlaps = overlap_by_worker(selection)
        loads = load_by_worker(selection)
        selected_worker = int(selection["selected"]["worker_id"])
        selected_load = loads[selected_worker]
        row = {
            "run": label,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "turn": int(metadata["turn_index"]),
            "request_end_ns": int(metadata["request_end_ns"]),
            "selection_timestamp_ns": int(selection["timestamp_ns"]),
            "selected_worker": selected_worker,
            "block_size": int(selection["overlap_scores"]["block_size"]),
            "prompt_tokens": int(prefill["prompt_tokens"]),
            "cached_tokens": int(prefill["cached_tokens"]),
            "computed_tokens": int(prefill["prompt_tokens"])
            - int(prefill["cached_tokens"]),
            "selected_overlap_tokens": overlaps[selected_worker],
            "max_overlap_tokens": max(overlaps.values()),
            "selected_active_requests": int(selected_load["active_requests"]),
            "selected_prefill_load": int(selected_load["potential_prefill_tokens"]),
            "selected_decode_blocks": int(selected_load["potential_decode_blocks"]),
            "prefill_ms": int(prefill["prefill_ms"]),
            "ttft_ms": float(client["metrics"]["time_to_first_token"]["value"]),
            "probe_ms": (
                int(selection["probe_done_ns"]) - int(selection["probe_start_ns"])
            )
            / 1_000_000,
            "selection_ms": (
                int(selection["selection_done_ns"])
                - int(selection["selection_start_ns"])
            )
            / 1_000_000,
            "overlaps": overlaps,
            "loads": loads,
        }
        joined.append(row)
        by_conversation[conversation_id].append(row)

    if len(by_conversation) != 60:
        raise ValueError(f"{label}: expected 60 conversations")
    transitions = []
    for rows in by_conversation.values():
        rows.sort(key=lambda row: row["turn"])
        if [row["turn"] for row in rows] != [0, 1, 2]:
            raise ValueError(f"{label}: conversation does not contain turns 0, 1, 2")
        for previous, current in zip(rows, rows[1:]):
            previous_worker = int(previous["selected_worker"])
            current["previous_worker"] = previous_worker
            current["previous_overlap_tokens"] = current["overlaps"][previous_worker]
            current["previous_prompt_floor_tokens"] = (
                int(previous["prompt_tokens"])
                // int(current["block_size"])
                * int(current["block_size"])
            )
            current["previous_prefix_visible"] = (
                current["previous_overlap_tokens"]
                >= current["previous_prompt_floor_tokens"]
            )
            current["inter_turn_gap_ms"] = (
                int(current["selection_timestamp_ns"]) - int(previous["request_end_ns"])
            ) / 1_000_000
            current["previous_active_requests"] = current["loads"][previous_worker][
                "active_requests"
            ]
            current["previous_prefill_load"] = current["loads"][previous_worker][
                "potential_prefill_tokens"
            ]
            current["previous_decode_blocks"] = current["loads"][previous_worker][
                "potential_decode_blocks"
            ]
            current["migrated"] = current["selected_worker"] != previous_worker
            current["actual_previous_prefix_hit"] = (
                None
                if current["migrated"]
                else current["cached_tokens"] >= current["previous_prompt_floor_tokens"]
            )
            transitions.append(current)
    for row in joined:
        row.setdefault("previous_worker", "")
        row.setdefault("previous_overlap_tokens", "")
        row.setdefault("previous_prompt_floor_tokens", "")
        row.setdefault("previous_prefix_visible", "")
        row.setdefault("inter_turn_gap_ms", "")
        row.setdefault("previous_active_requests", "")
        row.setdefault("previous_prefill_load", "")
        row.setdefault("previous_decode_blocks", "")
        row.setdefault("migrated", False)
        row.setdefault("actual_previous_prefix_hit", "")

    migrations = [row for row in transitions if row["migrated"]]
    load_overrides = [
        row
        for row in migrations
        if row["previous_overlap_tokens"] > row["selected_overlap_tokens"]
    ]
    locality_improvements = [
        row
        for row in migrations
        if row["selected_overlap_tokens"] > row["previous_overlap_tokens"]
    ]
    equal_locality = [
        row
        for row in migrations
        if row["selected_overlap_tokens"] == row["previous_overlap_tokens"]
    ]
    prompt_tokens = sum(row["prompt_tokens"] for row in joined)
    cached_tokens = sum(row["cached_tokens"] for row in joined)
    summary = {
        "run": label,
        "root": root.name,
        "requests": len(joined),
        "conversations": len(by_conversation),
        "transitions": len(transitions),
        "migrations": len(migrations),
        "migration_rate": len(migrations) / len(transitions),
        "same_pa_transitions": len(transitions) - len(migrations),
        "previous_prefix_visibility": {
            "visible_transitions": sum(
                row["previous_prefix_visible"] for row in transitions
            ),
            "missing_transitions": sum(
                not row["previous_prefix_visible"] for row in transitions
            ),
            "visibility_rate": mean(
                [float(row["previous_prefix_visible"]) for row in transitions]
            ),
            "visible_migrations": sum(
                row["previous_prefix_visible"] for row in migrations
            ),
            "missing_migrations": sum(
                not row["previous_prefix_visible"] for row in migrations
            ),
            "visible_inter_turn_gap_ms_mean": mean(
                [
                    row["inter_turn_gap_ms"]
                    for row in transitions
                    if row["previous_prefix_visible"]
                ]
            ),
            "missing_inter_turn_gap_ms_mean": mean(
                [
                    row["inter_turn_gap_ms"]
                    for row in transitions
                    if not row["previous_prefix_visible"]
                ]
            ),
            "router_missing_on_same_pa": sum(
                not row["migrated"] and not row["previous_prefix_visible"]
                for row in transitions
            ),
            "actual_full_hit_despite_router_missing": sum(
                not row["migrated"]
                and not row["previous_prefix_visible"]
                and row["actual_previous_prefix_hit"] is True
                for row in transitions
            ),
        },
        "migration_reasons": {
            "previous_pa_had_more_overlap": len(load_overrides),
            "selected_pa_had_more_overlap": len(locality_improvements),
            "equal_overlap": len(equal_locality),
        },
        "migration_load_delta_selected_minus_previous": {
            "active_requests_mean": mean(
                [
                    row["selected_active_requests"] - row["previous_active_requests"]
                    for row in migrations
                ]
            )
            if migrations
            else 0.0,
            "prefill_tokens_mean": mean(
                [
                    row["selected_prefill_load"] - row["previous_prefill_load"]
                    for row in migrations
                ]
            )
            if migrations
            else 0.0,
            "decode_blocks_mean": mean(
                [
                    row["selected_decode_blocks"] - row["previous_decode_blocks"]
                    for row in migrations
                ]
            )
            if migrations
            else 0.0,
        },
        "selected_max_overlap_requests": sum(
            row["selected_overlap_tokens"] == row["max_overlap_tokens"]
            for row in joined
        ),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cache_read_rate": cached_tokens / prompt_tokens,
        "computed_tokens": prompt_tokens - cached_tokens,
        "request_throughput": profile["request_throughput"]["avg"],
        "output_token_throughput": profile["output_token_throughput"]["avg"],
        "itl_ms": {
            "mean": profile["inter_token_latency"]["avg"],
            "p99": profile["inter_token_latency"]["p99"],
        },
        "e2e_ms": {
            "mean": profile["request_latency"]["avg"],
            "p99": profile["request_latency"]["p99"],
        },
        "effective_config": {
            "python_hash_seed": config.get("PYTHONHASHSEED"),
            "kv_fence": config.get("PAP_CROSS_HOST_ENABLE_KV_FENCE", "0"),
        },
        "kv_fence": topology.get("kv_fence"),
        "block_hash_audit": block_hash_audit,
        "source_audit": source_audit,
        "cleanup_audit": cleanup_audit,
        "prefill_ms": {
            "mean": mean([row["prefill_ms"] for row in joined]),
            "p99": percentile([row["prefill_ms"] for row in joined], 0.99),
        },
        "ttft_ms": {
            "mean": mean([row["ttft_ms"] for row in joined]),
            "p99": percentile([row["ttft_ms"] for row in joined], 0.99),
        },
        "trace_overhead_ms": {
            "overlap_and_load_probe_mean": mean([row["probe_ms"] for row in joined]),
            "overlap_and_load_probe_p99": percentile(
                [row["probe_ms"] for row in joined], 0.99
            ),
            "selection_mean": mean([row["selection_ms"] for row in joined]),
            "selection_p99": percentile([row["selection_ms"] for row in joined], 0.99),
        },
        "by_turn": {},
    }
    for turn in range(3):
        rows = [row for row in joined if row["turn"] == turn]
        summary["by_turn"][str(turn)] = {
            "requests": len(rows),
            "migrations": sum(row["migrated"] for row in rows),
            "cache_read_rate": sum(row["cached_tokens"] for row in rows)
            / sum(row["prompt_tokens"] for row in rows),
            "computed_tokens": sum(row["computed_tokens"] for row in rows),
            "prefill_ms_mean": mean([row["prefill_ms"] for row in rows]),
            "ttft_ms_mean": mean([row["ttft_ms"] for row in rows]),
        }
    return summary, joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    rows = []
    for spec in args.run:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            parser.error(f"invalid --run value: {spec}")
        summary, run_rows = analyze_run(label, Path(path))
        summaries.append(summary)
        rows.extend(run_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_label = {row["run"]: row for row in summaries}

    def compare(before: str, after: str) -> dict[str, Any]:
        baseline = by_label[before]
        current = by_label[after]
        return {
            "before": before,
            "after": after,
            "migration_count_delta": current["migrations"] - baseline["migrations"],
            "migration_rate_delta": current["migration_rate"]
            - baseline["migration_rate"],
            "visibility_rate_delta": current["previous_prefix_visibility"][
                "visibility_rate"
            ]
            - baseline["previous_prefix_visibility"]["visibility_rate"],
            "cache_read_rate_delta": current["cache_read_rate"]
            - baseline["cache_read_rate"],
            "computed_tokens_delta": current["computed_tokens"]
            - baseline["computed_tokens"],
            "ttft_mean_ms_delta": current["ttft_ms"]["mean"]
            - baseline["ttft_ms"]["mean"],
            "ttft_mean_relative": current["ttft_ms"]["mean"]
            / baseline["ttft_ms"]["mean"]
            - 1,
            "request_throughput_delta": current["request_throughput"]
            - baseline["request_throughput"],
            "request_throughput_relative": current["request_throughput"]
            / baseline["request_throughput"]
            - 1,
        }

    comparisons = {}
    if "old_replay" in by_label and "hash_only" in by_label:
        comparisons["hash_fix_vs_old"] = compare("old_replay", "hash_only")
    if "hash_only" in by_label and "hash_fence" in by_label:
        comparisons["fence_vs_hash_only"] = compare("hash_only", "hash_fence")
    (args.output_dir / "summary.json").write_text(
        json.dumps({"runs": summaries, "comparisons": comparisons}, indent=2) + "\n"
    )
    fields = [
        "run",
        "request_id",
        "conversation_id",
        "turn",
        "request_end_ns",
        "selection_timestamp_ns",
        "previous_worker",
        "selected_worker",
        "block_size",
        "migrated",
        "actual_previous_prefix_hit",
        "prompt_tokens",
        "cached_tokens",
        "computed_tokens",
        "previous_overlap_tokens",
        "previous_prompt_floor_tokens",
        "previous_prefix_visible",
        "inter_turn_gap_ms",
        "selected_overlap_tokens",
        "max_overlap_tokens",
        "previous_active_requests",
        "selected_active_requests",
        "previous_prefill_load",
        "selected_prefill_load",
        "previous_decode_blocks",
        "selected_decode_blocks",
        "prefill_ms",
        "ttft_ms",
        "probe_ms",
        "selection_ms",
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
