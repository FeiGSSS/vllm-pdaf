#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audit retained PAP evidence on CPU; never launch serving or modify raw files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import regex as re
import torch

ROOT = Path(__file__).resolve().parents[5]
RUNS = ROOT / "benchmarks/pap/experiments/e2e/PAP-20260905-REFACTOR-VALIDATION/runs"
CURRENT = "20260905_190715_3493447"
REFERENCE = "20260905_135804_3149938"


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.double().flatten()
    return {
        "mean": values.mean().item(),
        "p50": values.quantile(0.5).item(),
        "p99": values.quantile(0.99).item(),
        "max": values.max().item(),
    }


def trace_audit() -> dict:
    capture = RUNS / REFERENCE / "coding-half-trace/trace_capture"
    manifest = json.loads((capture / "capture.json").read_text())
    for name, expected in manifest["raw_file_sha256"].items():
        actual = hashlib.sha256((capture / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Raw trace hash mismatch: {name}")
    spec = importlib.util.spec_from_file_location(
        "merge_saved_trace",
        ROOT / "benchmarks/pap/tooling/merge_detailed_projection_trace.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.merge(capture / "projection.pt")
    saved = torch.load(capture / "merged.pt", map_location="cpu", weights_only=False)
    tensors = [key for key, value in payload.items() if torch.is_tensor(value)]
    if not all(torch.equal(payload[key], saved[key]) for key in tensors):
        raise ValueError("Rejoining the raw trace changed retained tensor fields")
    if payload["request_ids"] != saved["request_ids"]:
        raise ValueError("Rejoining the raw trace changed request IDs")
    if not payload["step_id"].diff().eq(1).all():
        raise ValueError("Cycle reconstruction requires consecutive global steps")

    # D is dispatch-done and G is gather-done on the same Projection GPU.
    # The first saved row lacks D[0, 0]; reconstruct rows 1 onward only.
    next_dispatch = payload["projection_next_dispatch_done_ns"]
    dispatch = torch.cat((next_dispatch[:-1, -1:], next_dispatch[1:, :-1]), 1)
    gather = payload["projection_gather_done_ns"][1:]
    wait_ns = gather - dispatch
    projection_ns = next_dispatch[1:] - gather
    cycle_ns = next_dispatch[1:, -1] - dispatch[:, 0]
    if not wait_ns.gt(0).all() or not projection_ns.gt(0).all():
        raise ValueError("Nonpositive adjacent-boundary interval")
    if not torch.equal((wait_ns + projection_ns).sum(1), cycle_ns):
        raise ValueError("Adjacent-boundary decomposition does not telescope")

    pa = payload["latency_ns"].double() / 1e6
    attention = payload["attention_kernel_latency_ns"].double() / 1e6
    projection = payload["projection_latency_ns"].double() / 1e6
    proxy = pa.max(2).values.sum(1) + projection.sum(1)
    cycle = cycle_ns.double() / 1e6
    attention_mean = attention.mean(1)
    features = [
        "request_count",
        "logical_context_tokens",
        "unique_context_tokens",
        "shared_context_tokens",
    ]
    requests = payload["request_ids"]
    single_request = payload["request_count"].eq(1)
    aliased_active_blocks = single_request & payload["unique_block_count"].lt(
        payload["request_block_counts"][:, :, 0]
    )
    alias_examples = []
    for step, peer in aliased_active_blocks.nonzero()[:3].tolist():
        alias_examples.append(
            {
                "global_step": payload["step_id"][step].item(),
                "pa": peer,
                "local_epoch": payload["peer_epoch"][step, peer].item(),
                "request_ids": requests[step][peer],
                "seq_len": payload["seq_lens"][step, peer, 0].item(),
                "prefix_len": payload["prefix_lens"][step, peer, 0].item(),
                "active_block_count": payload["request_block_counts"][
                    step, peer, 0
                ].item(),
                "unique_active_blocks": payload["unique_block_count"][
                    step, peer
                ].item(),
                "lease_vector_length": payload["request_leased_block_counts"][
                    step, peer, 0
                ].item(),
                "unique_leased_blocks": payload["unique_leased_block_count"][
                    step, peer
                ].item(),
            }
        )
    return {
        "numerical_correctness_admissibility": {
            "status": "invalid_aliasing_observed"
            if alias_examples
            else "not_established",
            "single_request_pa_step_cells": single_request.sum().item(),
            "aliased_active_block_cells": aliased_active_blocks.sum().item(),
            "examples": alias_examples,
        },
        "source": str(capture.relative_to(ROOT)),
        "raw_hashes_verified": len(manifest["raw_file_sha256"]),
        "rejoined_tensor_fields": len(tensors),
        "step_range": [
            payload["step_id"].min().item(),
            payload["step_id"].max().item(),
        ],
        "exact_cycles": cycle.numel(),
        "exact_cycle_ms": summarize(cycle),
        "dispatch_to_gather_ms": summarize(wait_ns.sum(1).double() / 1e6),
        "gather_to_next_dispatch_ms": summarize(projection_ns.sum(1).double() / 1e6),
        "legacy_proxy_same_cycles_ms": summarize(proxy[1:]),
        "legacy_proxy_minus_exact_ms": summarize(proxy[1:] - cycle),
        "attention_mean_sum_ms": summarize(attention.mean(2).sum(1)),
        "attention_max_sum_ms": summarize(attention.max(2).values.sum(1)),
        "attention_skew_ms": summarize(
            (attention.max(2).values - attention.mean(2)).sum(1)
        ),
        "pa_skew_ms": summarize((pa.max(2).values - pa.mean(2)).sum(1)),
        "layer0_pa_max_ms": summarize(pa[:, 0].max(1).values),
        "layer1_35_pa_max_ms": summarize(pa[:, 1:].max(2).values.mean(1)),
        "projection_last_ms": summarize(projection[:, -1]),
        "projection_regular_ms": summarize(projection[:, :-1].mean(1)),
        "pa_attention_mean_ms": attention.mean((0, 1)).tolist(),
        "pa_load_means": {
            name: payload[name].double().mean(0).tolist() for name in features
        },
        "load_attention_pearson_descriptive_only": {
            name: torch.corrcoef(
                torch.stack(
                    (payload[name].double().flatten(), attention_mean.flatten())
                )
            )[0, 1].item()
            for name in features
        },
        "pat_cell_fraction": payload["attention_is_pat"].double().mean().item(),
        "distinct_requests": len(
            {r for step in requests for peer in step for r in peer}
        ),
        "active_requests_per_step": summarize(payload["request_count"].sum(1)),
        "unchanged_membership_transitions": sum(
            requests[index] == requests[index - 1] for index in range(1, len(requests))
        ),
    }


def records_for(run: str, case: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (RUNS / run / case / "aiperf/profile.jsonl")
        .read_text()
        .splitlines()
    ]


def e2e_audit(run: str, case: str) -> dict:
    path = RUNS / run / case
    records = records_for(run, case)
    profile = json.loads((path / "aiperf/profile.json").read_text())
    samples = []
    for logfile in sorted((path / "service_logs").glob("prefill_*.log")):
        samples.extend(
            tuple(map(float, match))
            for match in re.findall(
                r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
                r"GPU KV cache usage: ([\d.]+)%",
                logfile.read_text(),
            )
        )
    fields = [
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
        "input_sequence_length",
        "usage_prompt_cache_read_tokens",
    ]
    return {
        "run": run,
        "case": case,
        "records": len(records),
        "errors": sum(bool(row.get("error")) for row in records),
        "cancelled_success_records": sum(
            bool(row["metadata"].get("was_cancelled")) for row in records
        ),
        "output_tokens": sum(
            row["metrics"]["output_sequence_length"]["value"] for row in records
        ),
        "mean_metrics": {
            name: sum(row["metrics"][name]["value"] for row in records) / len(records)
            for name in fields
        },
        "profile_metrics": {
            name: profile[name]["avg"]
            for name in ("benchmark_duration", "output_token_throughput")
        },
        "scheduler_samples": len(samples),
        "max_running_waiting_kv_percent": list(map(max, zip(*samples))),
        "samples_with_waiting": sum(row[1] > 0 for row in samples),
        "samples_kv_at_least_90_percent": sum(row[2] >= 90 for row in samples),
        "samples_with_both": sum(row[1] > 0 and row[2] >= 90 for row in samples),
        "prefill_phase_records": (path / "service_logs/proxy.log")
        .read_text()
        .count("prefill IPC profile request_id="),
    }


def provenance_audit(run: str) -> dict:
    path = RUNS / run / "provenance"
    archive = path / "source.tar.gz"
    matches, different = [], []
    with tarfile.open(archive) as source:
        for member in source:
            if not member.isfile() or not member.name.startswith("vllm/pap/"):
                continue
            local = ROOT / member.name
            equal = (
                local.exists()
                and source.extractfile(member).read() == local.read_bytes()
            )
            (matches if equal else different).append(member.name)
    return {
        "run": run,
        "recorded_git_commit": (path / "git_commit.txt").read_text().strip(),
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archived_pap_files_matching_current": len(matches),
        "archived_pap_files_different_from_current": different,
    }


def main() -> None:
    torch.set_num_threads(1)
    current = records_for(CURRENT, "coding-half")
    reference = records_for(REFERENCE, "coding-half")

    def key(row):
        return row["metadata"]["conversation_id"], row["metadata"]["turn_index"]

    current, reference = (
        {key(row): row for row in rows} for rows in (current, reference)
    )
    if current.keys() != reference.keys():
        raise ValueError("The cleanup comparison does not have matching turns")
    paired_differences = {}
    for field in (
        "input_sequence_length",
        "output_sequence_length",
        "usage_prompt_cache_read_tokens",
    ):
        differences = [
            current[item]["metrics"][field]["value"]
            - reference[item]["metrics"][field]["value"]
            for item in current
        ]
        paired_differences[field] = {
            "different_turns": sum(value != 0 for value in differences),
            "min_delta": min(differences),
            "max_delta": max(differences),
        }
    result = {
        "provenance": [provenance_audit(run) for run in (CURRENT, REFERENCE)],
        "trace": trace_audit(),
        "e2e": [
            e2e_audit(run, case)
            for run, case in [
                (CURRENT, "coding-half"),
                (REFERENCE, "coding-half"),
                (REFERENCE, "coding"),
                (REFERENCE, "coding-full"),
                (REFERENCE, "coding-half-trace"),
            ]
        ],
        "cleanup_paired_metric_differences": paired_differences,
    }
    print(json.dumps(result, indent=2))
    if (
        result["trace"]["numerical_correctness_admissibility"]["status"]
        == "invalid_aliasing_observed"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
