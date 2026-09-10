#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare aligned single-host DP, PD, and PAP KV-migration runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

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
DISPLAY_NAMES = {
    "dp8": "DP8",
    "pd7p1d": "PD 7P1D",
    "pd6p2d": "PD 6P2D",
    "pd4p4d": "PD 4P4D",
    "pd2p6d": "PD 2P6D",
    "pap7pa1p_kv_migration": "PAP 7PA1P + KV migration",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_config(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            output[key] = value
    return output


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(architecture: str, config: dict[str, str], label: str) -> None:
    if architecture.startswith("pap"):
        observed = {
            "model": config.get("MODEL_PATH"),
            "max_model_len": config.get("MAX_MODEL_LEN"),
            "hf_overrides": config.get("PAP_HF_OVERRIDES"),
            "sessions": config.get("PAP_AIPERF_SESSIONS"),
            "concurrency": config.get("PAP_AIPERF_CONCURRENCY"),
            "expected_requests": config.get("PAP_AIPERF_EXPECTED_REQUESTS"),
            "prefill_budget": config.get("PAP_PREFILL_MAX_NUM_BATCHED_TOKENS"),
            "decode_budget": config.get("PAP_PROJECTION_MAX_NUM_BATCHED_TOKENS"),
            "block_size": config.get("PAP_BLOCK_SIZE"),
            "migration": config.get("PAP_ENABLE_PA_KV_MIGRATION"),
        }
    else:
        observed = {
            "model": config.get("MODEL_PATH"),
            "max_model_len": config.get("MAX_MODEL_LEN"),
            "hf_overrides": config.get("HF_OVERRIDES"),
            "sessions": config.get("AIPERF_SESSIONS"),
            "concurrency": config.get("AIPERF_CONCURRENCY"),
            "expected_requests": config.get("AIPERF_EXPECTED_REQUESTS"),
            "prefill_budget": config.get("PREFILL_MAX_NUM_BATCHED_TOKENS"),
            "decode_budget": config.get("DECODE_MAX_NUM_BATCHED_TOKENS"),
            "block_size": config.get("BLOCK_SIZE"),
            "migration": None,
        }
    expected = {
        "model": "/data/ssd1/llm-models/Qwen3-8B",
        "max_model_len": "32768",
        "hf_overrides": "''",
        "sessions": "60",
        "concurrency": "30",
        "expected_requests": "180",
        "prefill_budget": "2048",
        "decode_budget": "256",
        "block_size": "16",
        "migration": "1" if architecture.startswith("pap") else None,
    }
    require(observed == expected, f"{label}: effective configuration mismatch")


def validate_audits(architecture: str, root: Path, label: str) -> None:
    if architecture.startswith("pap"):
        cleanup = read_json(root / "cleanup_audit.json")
        routing = read_json(root / "routing_audit.json")
        topology = read_json(root / "topology_manifest.json")
        require(cleanup["status"] == "passed", f"{label}: cleanup audit")
        require(routing["status"] == "passed", f"{label}: routing audit")
        require(routing["route_count"] == 180, f"{label}: route count")
        require(
            routing["kv_migration"]["enabled"]
            and routing["kv_migration"]["attempts"]
            == routing["kv_migration"]["exports"]
            and routing["kv_migration"]["misses"] == 0
            and routing["kv_migration"]["failures"] == 0
            and routing["kv_migration"]["pending_releases"] == 0,
            f"{label}: KV migration audit",
        )
        require(
            topology["topology"] == "7pa1p"
            and len(topology["pa_groups"]) == 7
            and len(topology["projections"]) == 1,
            f"{label}: PAP topology",
        )
        mps_audits = sorted(root.glob("mps_static_audit_pa_*.env"))
        require(len(mps_audits) == 7, f"{label}: MPS audit count")
        require(
            all(
                "MPS_MODE=static" in path.read_text()
                and "PREFILL_VISIBLE_SMS=80" in path.read_text()
                and "ATTENTION_VISIBLE_SMS=12" in path.read_text()
                for path in mps_audits
            ),
            f"{label}: MPS topology",
        )
        runtime_audits = sorted(root.glob("runtime_cuda_context_audit_pa_*.env"))
        require(len(runtime_audits) == 7, f"{label}: runtime audit count")
        require(
            all("STATUS=passed" in path.read_text() for path in runtime_audits),
            f"{label}: runtime audit status",
        )
    else:
        startup = read_json(root / "startup_audit.json")
        require(startup["status"] == "passed", f"{label}: startup audit")
        completion = read_json(root / "completion_audit.json")
        require(completion["status"] == "passed", f"{label}: completion audit")
        cleanup_name = (
            "cleanup_audit.json"
            if (root / "cleanup_audit.json").exists()
            else "cleanup_wait.json"
        )
        cleanup = read_json(root / cleanup_name)
        require(cleanup["status"] == "passed", f"{label}: cleanup audit")


def load_run(spec: str) -> dict[str, Any]:
    architecture_label, separator, raw_path = spec.partition("=")
    architecture, label_separator, label = architecture_label.partition(":")
    require(
        bool(separator and label_separator and architecture and label and raw_path),
        f"invalid run specification: {spec}",
    )
    require(architecture in DISPLAY_NAMES, f"unsupported architecture: {architecture}")
    root = Path(raw_path)
    profile = read_json(root / "aiperf/profile.json")
    require(profile["request_count"]["avg"] == 180, f"{label}: request count")
    require(not profile["error_summary"], f"{label}: AIPerf errors")
    require(not profile["was_cancelled"], f"{label}: cancelled")
    require(profile["osl_mismatch_count"]["avg"] == 0, f"{label}: output mismatch")
    config = read_config(root / "effective_config.env")
    validate_config(architecture, config, label)
    validate_audits(architecture, root, label)
    inputs_sha = sha256(root / "aiperf/inputs.json")
    require(
        inputs_sha
        == "ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5",
        f"{label}: input bytes mismatch",
    )

    records = []
    for line in (root / "aiperf/profile.jsonl").read_text().splitlines():
        raw = json.loads(line)
        metadata = raw["metadata"]
        metrics = raw["metrics"]
        record = {
            "architecture": architecture,
            "run": label,
            "conversation_id": metadata["conversation_id"],
            "turn": int(metadata["turn_index"]),
            "input_tokens": int(metrics["input_sequence_length"]["value"]),
            "output_tokens": int(metrics["output_sequence_length"]["value"]),
            "cached_tokens": int(metrics["usage_prompt_cache_read_tokens"]["value"]),
            **{
                output: float(metrics[source]["value"])
                for output, source in METRICS.items()
            },
        }
        records.append(record)
    require(len(records) == 180, f"{label}: request records")
    identities = {(row["conversation_id"], row["turn"]) for row in records}
    require(len(identities) == 180, f"{label}: request identity")
    return {
        "architecture": architecture,
        "label": label,
        "root": str(root),
        "duration_s": float(profile["benchmark_duration"]["avg"]),
        "records": records,
        "migration_attempts": (
            read_json(root / "routing_audit.json")["kv_migration"]["attempts"]
            if architecture.startswith("pap")
            else None
        ),
    }


def summarize(architecture: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    records = [row for run in runs for row in run["records"]]
    duration = sum(run["duration_s"] for run in runs)
    output_tokens = sum(row["output_tokens"] for row in records)
    summary = {
        "display_name": DISPLAY_NAMES[architecture],
        "runs": [run["label"] for run in runs],
        "requests": len(records),
        "duration_s": duration,
        "request_throughput_rps": len(records) / duration,
        "output_throughput_tps": output_tokens / duration,
        **{
            metric: distribution([row[metric] for row in records]) for metric in METRICS
        },
        "per_run_mean_range": {
            metric: [
                min(
                    statistics.mean(row[metric] for row in run["records"])
                    for run in runs
                ),
                max(
                    statistics.mean(row[metric] for row in run["records"])
                    for run in runs
                ),
            ]
            for metric in METRICS
        },
        "by_turn": {
            str(turn): {
                metric: distribution(
                    [row[metric] for row in records if row["turn"] == turn]
                )
                for metric in METRICS
            }
            for turn in range(3)
        },
        "goodput": {},
    }
    for name, slo in SLOS.items():
        good = sum(
            all(row[metric] <= threshold for metric, threshold in slo.items())
            for row in records
        )
        summary["goodput"][name] = {
            "good_requests": good,
            "attainment_pct": good / len(records) * 100,
            "goodput_rps": good / duration,
            "feasible_at_95pct": good / len(records) >= 0.95,
        }
    if architecture.startswith("pap"):
        prompt_tokens = sum(row["input_tokens"] for row in records)
        cached_tokens = sum(row["cached_tokens"] for row in records)
        summary["pap"] = {
            "migration_attempts": sum(run["migration_attempts"] for run in runs),
            "cache_read_pct": cached_tokens / prompt_tokens * 100,
            "computed_tokens": prompt_tokens - cached_tokens,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(spec) for spec in args.run]
    architecture_order = {name: index for index, name in enumerate(DISPLAY_NAMES)}
    runs.sort(key=lambda run: (architecture_order[run["architecture"]], run["label"]))
    architectures = sorted({run["architecture"] for run in runs})
    require(set(architectures) == set(DISPLAY_NAMES), "architecture set mismatch")
    require(
        all(
            sum(run["architecture"] == name for run in runs) == 2
            for name in architectures
        ),
        "two runs required per architecture",
    )

    expected_outputs = None
    for run in runs:
        outputs = {
            (row["conversation_id"], row["turn"]): row["output_tokens"]
            for row in run["records"]
        }
        if expected_outputs is None:
            expected_outputs = outputs
        else:
            require(
                outputs == expected_outputs, f"{run['label']}: output lengths differ"
            )

    summaries = {
        architecture: summarize(
            architecture,
            [run for run in runs if run["architecture"] == architecture],
        )
        for architecture in DISPLAY_NAMES
    }
    pap = summaries["pap7pa1p_kv_migration"]
    relative_to_pap = {}
    for architecture, summary in summaries.items():
        if architecture == "pap7pa1p_kv_migration":
            continue
        relative_to_pap[architecture] = {
            "pap_ttft_change_pct": pap["ttft_ms"]["mean"]
            / summary["ttft_ms"]["mean"]
            * 100
            - 100,
            "pap_itl_change_pct": pap["itl_ms"]["mean"]
            / summary["itl_ms"]["mean"]
            * 100
            - 100,
            "pap_e2e_change_pct": pap["e2e_ms"]["mean"]
            / summary["e2e_ms"]["mean"]
            * 100
            - 100,
            "pap_throughput_change_pct": pap["request_throughput_rps"]
            / summary["request_throughput_rps"]
            * 100
            - 100,
        }

    pareto = []
    for architecture, candidate in summaries.items():
        dominated = any(
            other_name != architecture
            and other["ttft_ms"]["mean"] <= candidate["ttft_ms"]["mean"]
            and other["itl_ms"]["mean"] <= candidate["itl_ms"]["mean"]
            and other["request_throughput_rps"] >= candidate["request_throughput_rps"]
            and (
                other["ttft_ms"]["mean"] < candidate["ttft_ms"]["mean"]
                or other["itl_ms"]["mean"] < candidate["itl_ms"]["mean"]
                or other["request_throughput_rps"] > candidate["request_throughput_rps"]
            )
            for other_name, other in summaries.items()
        )
        if not dominated:
            pareto.append(architecture)

    output = {
        "comparability": {
            "status": "passed",
            "runs_per_architecture": 2,
            "requests_per_architecture": 360,
            "input_sha256": "ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5",
            "hf_overrides": "",
            "sessions": 60,
            "turns": 3,
            "concurrency": 30,
            "model": "Qwen3-8B FP16",
            "max_model_len": 32768,
        },
        "architectures": summaries,
        "relative_to_pap": relative_to_pap,
        "pareto_frontier": pareto,
        "slos": SLOS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    fields = list(runs[0]["records"][0])
    with (args.output_dir / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerows(run["records"])


if __name__ == "__main__":
    main()
