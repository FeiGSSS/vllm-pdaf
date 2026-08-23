#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate PAP token-latency changes from changing Decode population."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import regex as re


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _summary(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "p50": _percentile(numeric, 0.50),
        "p95": _percentile(numeric, 0.95),
        "min": min(numeric),
        "max": max(numeric),
    }


def _metric(record: dict[str, Any], name: str) -> Any:
    return record["metrics"][name]["value"]


def load_profile(path: Path) -> list[dict[str, Any]]:
    """Load successful AIPerf request records from one profile JSONL."""
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["metadata"].get("was_cancelled", False):
                continue
            records.append(record)
    if not records:
        raise ValueError(f"profile contains no successful records: {path}")
    return records


_ROUTING_PATTERN = re.compile(r"\brequest_id=([^\s]+).*?\bpa_index=(\d+)\b")


def load_pa_assignments(path: Path) -> dict[str, int]:
    """Load each request's final Decode PA assignment from a gateway log."""
    assignments: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            match = _ROUTING_PATTERN.search(line)
            if match is None:
                continue
            request_id = match.group(1)
            pa_index = int(match.group(2))
            previous = assignments.setdefault(request_id, pa_index)
            if previous != pa_index:
                raise ValueError(
                    f"conflicting PA assignments for {request_id}: "
                    f"{previous} and {pa_index}"
                )
    if not assignments:
        raise ValueError(f"gateway log contains no PA assignments: {path}")
    return assignments


def _request_timeline(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    request_start_ns = int(metadata["request_start_ns"])
    first_token_ns = request_start_ns + round(
        float(_metric(record, "time_to_first_token")) * 1_000_000
    )
    intervals_ms = [float(value) for value in _metric(record, "inter_chunk_latency")]
    output_tokens = int(_metric(record, "output_sequence_length"))
    token_times_ns = [first_token_ns]
    for interval_ms in intervals_ms:
        token_times_ns.append(token_times_ns[-1] + round(interval_ms * 1_000_000))
    return {
        "request_id": metadata.get("x_request_id"),
        "input_tokens": int(_metric(record, "input_sequence_length")),
        "output_tokens": output_tokens,
        "token_times_ns": token_times_ns,
        "intervals_ms": intervals_ms,
    }


def build_interval_samples(
    records: list[dict[str, Any]],
    *,
    pa_by_request_id: dict[str, int] | None = None,
    pa_count: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Reconstruct token intervals and the concurrent Decode load at each one."""
    timelines = [_request_timeline(record) for record in records]
    if pa_by_request_id is not None:
        missing = sorted(
            str(timeline["request_id"])
            for timeline in timelines
            if timeline["request_id"] not in pa_by_request_id
        )
        if missing:
            raise ValueError(
                f"missing PA assignments for {len(missing)} requests: {missing[:3]}"
            )
        invalid = sorted(
            {
                pa_by_request_id[timeline["request_id"]]
                for timeline in timelines
                if not 0 <= pa_by_request_id[timeline["request_id"]] < pa_count
            }
        )
        if invalid:
            raise ValueError(f"PA assignments outside [0, {pa_count}): {invalid}")
    start_ns = min(timeline["token_times_ns"][0] for timeline in timelines)
    end_ns = max(timeline["token_times_ns"][-1] for timeline in timelines)
    decode_residency_ns = sum(
        timeline["token_times_ns"][-1] - timeline["token_times_ns"][0]
        for timeline in timelines
    )
    samples = []
    for timeline in timelines:
        token_times_ns = timeline["token_times_ns"]
        for token_index, interval_ms in enumerate(timeline["intervals_ms"]):
            interval_start_ns = token_times_ns[token_index]
            interval_end_ns = token_times_ns[token_index + 1]
            midpoint_ns = (interval_start_ns + interval_end_ns) // 2
            active_count = 0
            active_context_tokens = 0
            pa_context_tokens = [0] * pa_count
            pa_active_requests = [0] * pa_count
            for other in timelines:
                other_times = other["token_times_ns"]
                if other_times[0] <= midpoint_ns <= other_times[-1]:
                    observed_chunks = bisect.bisect_right(other_times, midpoint_ns)
                    generated_tokens = round(
                        other["output_tokens"] * observed_chunks / len(other_times)
                    )
                    active_count += 1
                    current_context = other["input_tokens"] + generated_tokens
                    active_context_tokens += current_context
                    if pa_by_request_id is not None:
                        pa_index = pa_by_request_id[other["request_id"]]
                        pa_context_tokens[pa_index] += current_context
                        pa_active_requests[pa_index] += 1
            pa_fields = {}
            if pa_by_request_id is not None:
                mean_context = active_context_tokens / pa_count
                max_context = max(pa_context_tokens)
                squared_deviation = statistics.fmean(
                    (value - mean_context) ** 2 for value in pa_context_tokens
                )
                pa_fields = {
                    "pa_context_tokens": pa_context_tokens,
                    "pa_active_requests": pa_active_requests,
                    "pa_context_mean_tokens": mean_context,
                    "pa_context_max_tokens": max_context,
                    "pa_context_max_minus_mean_tokens": (max_context - mean_context),
                    "pa_context_peak_over_mean": max_context / mean_context,
                    "pa_context_coefficient_of_variation": (
                        math.sqrt(squared_deviation) / mean_context
                    ),
                }
            samples.append(
                {
                    "request_id": timeline["request_id"],
                    "token_index": token_index + 1,
                    "interval_ms": interval_ms,
                    "midpoint_ns": midpoint_ns,
                    "active_decode_requests": active_count,
                    "active_context_tokens": active_context_tokens,
                    **pa_fields,
                }
            )
    duration_ns = max(1, end_ns - start_ns)
    return samples, {
        "decode_window_s": duration_ns / 1_000_000_000,
        "decode_residency_s": decode_residency_ns / 1_000_000_000,
        "effective_decode_concurrency": decode_residency_ns / duration_ns,
    }


def summarize_arm(
    records: list[dict[str, Any]],
    *,
    context_bucket_tokens: int,
    pa_by_request_id: dict[str, int] | None = None,
    pa_count: int = 7,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize one run and retain positive token intervals for matching."""
    samples, occupancy = build_interval_samples(
        records,
        pa_by_request_id=pa_by_request_id,
        pa_count=pa_count,
    )
    positive = [sample for sample in samples if sample["interval_ms"] > 0]
    by_concurrency: dict[int, list[float]] = defaultdict(list)
    by_load_bin: dict[tuple[int, int], list[float]] = defaultdict(list)
    for sample in positive:
        concurrency = int(sample["active_decode_requests"])
        context_bucket = int(sample["active_context_tokens"]) // context_bucket_tokens
        by_concurrency[concurrency].append(float(sample["interval_ms"]))
        by_load_bin[(concurrency, context_bucket)].append(float(sample["interval_ms"]))
    summary = {
        "request_count": len(records),
        "token_interval_count": len(samples),
        "positive_token_interval_count": len(positive),
        "positive_token_interval_ms": _summary(
            [sample["interval_ms"] for sample in positive]
        ),
        **occupancy,
        "by_decode_concurrency": {
            str(key): _summary(values) for key, values in sorted(by_concurrency.items())
        },
        "by_decode_load_bin": {
            f"c{key[0]}:ctx{key[1]}": _summary(values)
            for key, values in sorted(by_load_bin.items())
        },
    }
    if pa_by_request_id is not None:
        summary["pa_count"] = pa_count
        summary["pa_context_load"] = {
            name: _summary([sample[name] for sample in positive])
            for name in (
                "pa_context_max_tokens",
                "pa_context_mean_tokens",
                "pa_context_max_minus_mean_tokens",
                "pa_context_peak_over_mean",
                "pa_context_coefficient_of_variation",
            )
        }
    return summary, positive


def _matched_metric_comparison(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    *,
    metric: str,
    context_bucket_tokens: int,
    minimum_bin_samples: int,
) -> dict[str, Any]:
    """Compare a sample metric at matched global Decode load."""
    grouped: dict[str, dict[tuple[int, int], list[float]]] = {
        "baseline": defaultdict(list),
        "treatment": defaultdict(list),
    }
    for arm, samples in (("baseline", baseline), ("treatment", treatment)):
        for sample in samples:
            if metric not in sample:
                continue
            key = (
                int(sample["active_decode_requests"]),
                int(sample["active_context_tokens"]) // context_bucket_tokens,
            )
            grouped[arm][key].append(float(sample[metric]))
    common_keys = [
        key
        for key in grouped["baseline"].keys() & grouped["treatment"].keys()
        if len(grouped["baseline"][key]) >= minimum_bin_samples
        and len(grouped["treatment"][key]) >= minimum_bin_samples
    ]
    matched_weight = sum(
        min(len(grouped["baseline"][key]), len(grouped["treatment"][key]))
        for key in common_keys
    )
    if matched_weight == 0:
        return {
            "metric": metric,
            "matched_bins": 0,
            "matched_weight": 0,
            "minimum_bin_samples": minimum_bin_samples,
        }
    means = {"baseline": 0.0, "treatment": 0.0}
    for key in common_keys:
        weight = min(len(grouped["baseline"][key]), len(grouped["treatment"][key]))
        for arm in means:
            means[arm] += weight * statistics.fmean(grouped[arm][key])
    baseline_mean = means["baseline"] / matched_weight
    treatment_mean = means["treatment"] / matched_weight
    return {
        "metric": metric,
        "matched_bins": len(common_keys),
        "matched_weight": matched_weight,
        "minimum_bin_samples": minimum_bin_samples,
        "baseline_standardized_mean": baseline_mean,
        "treatment_standardized_mean": treatment_mean,
        "treatment_minus_baseline": treatment_mean - baseline_mean,
        "treatment_change_fraction": treatment_mean / baseline_mean - 1.0,
    }


def _stratified_comparison(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    *,
    context_bucket_tokens: int,
    minimum_bin_samples: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[tuple[int, ...], list[float]]] = {
        "baseline_concurrency": defaultdict(list),
        "treatment_concurrency": defaultdict(list),
        "baseline_load": defaultdict(list),
        "treatment_load": defaultdict(list),
    }
    for arm, samples in (("baseline", baseline), ("treatment", treatment)):
        for sample in samples:
            concurrency = int(sample["active_decode_requests"])
            context_bucket = (
                int(sample["active_context_tokens"]) // context_bucket_tokens
            )
            latency = float(sample["interval_ms"])
            grouped[f"{arm}_concurrency"][(concurrency,)].append(latency)
            grouped[f"{arm}_load"][(concurrency, context_bucket)].append(latency)

    def compare(kind: str) -> dict[str, Any]:
        baseline_groups = grouped[f"baseline_{kind}"]
        treatment_groups = grouped[f"treatment_{kind}"]
        common_keys = [
            key
            for key in baseline_groups.keys() & treatment_groups.keys()
            if len(baseline_groups[key]) >= minimum_bin_samples
            and len(treatment_groups[key]) >= minimum_bin_samples
        ]
        matched_weight = sum(
            min(len(baseline_groups[key]), len(treatment_groups[key]))
            for key in common_keys
        )
        if matched_weight == 0:
            return {
                "matched_bins": 0,
                "matched_weight": 0,
                "minimum_bin_samples": minimum_bin_samples,
            }
        baseline_mean = 0.0
        treatment_mean = 0.0
        bins = []
        for key in sorted(common_keys):
            weight = min(len(baseline_groups[key]), len(treatment_groups[key]))
            baseline_bin_mean = statistics.fmean(baseline_groups[key])
            treatment_bin_mean = statistics.fmean(treatment_groups[key])
            baseline_mean += weight * baseline_bin_mean
            treatment_mean += weight * treatment_bin_mean
            bins.append(
                {
                    "key": list(key),
                    "weight": weight,
                    "baseline_count": len(baseline_groups[key]),
                    "treatment_count": len(treatment_groups[key]),
                    "baseline_mean_ms": baseline_bin_mean,
                    "treatment_mean_ms": treatment_bin_mean,
                    "treatment_minus_baseline_ms": (
                        treatment_bin_mean - baseline_bin_mean
                    ),
                }
            )
        baseline_mean /= matched_weight
        treatment_mean /= matched_weight
        return {
            "matched_bins": len(common_keys),
            "matched_weight": matched_weight,
            "minimum_bin_samples": minimum_bin_samples,
            "baseline_standardized_mean_ms": baseline_mean,
            "treatment_standardized_mean_ms": treatment_mean,
            "treatment_minus_baseline_ms": treatment_mean - baseline_mean,
            "treatment_change_fraction": treatment_mean / baseline_mean - 1.0,
            "bins": bins,
        }

    return {
        "concurrency_matched": compare("concurrency"),
        "concurrency_and_context_matched": compare("load"),
    }


def analyze_profiles(
    baseline_records: list[dict[str, Any]],
    treatment_records: list[dict[str, Any]],
    *,
    context_bucket_tokens: int = 16_384,
    minimum_bin_samples: int = 100,
    baseline_pa_by_request_id: dict[str, int] | None = None,
    treatment_pa_by_request_id: dict[str, int] | None = None,
    pa_count: int = 7,
) -> dict[str, Any]:
    """Compare raw and Decode-load-stratified token intervals."""
    baseline_summary, baseline_samples = summarize_arm(
        baseline_records,
        context_bucket_tokens=context_bucket_tokens,
        pa_by_request_id=baseline_pa_by_request_id,
        pa_count=pa_count,
    )
    treatment_summary, treatment_samples = summarize_arm(
        treatment_records,
        context_bucket_tokens=context_bucket_tokens,
        pa_by_request_id=treatment_pa_by_request_id,
        pa_count=pa_count,
    )
    comparison = _stratified_comparison(
        baseline_samples,
        treatment_samples,
        context_bucket_tokens=context_bucket_tokens,
        minimum_bin_samples=minimum_bin_samples,
    )
    if baseline_pa_by_request_id is not None and treatment_pa_by_request_id is not None:
        comparison["pa_context_imbalance_load_matched"] = {
            metric: _matched_metric_comparison(
                baseline_samples,
                treatment_samples,
                metric=metric,
                context_bucket_tokens=context_bucket_tokens,
                minimum_bin_samples=minimum_bin_samples,
            )
            for metric in (
                "pa_context_max_minus_mean_tokens",
                "pa_context_peak_over_mean",
                "pa_context_coefficient_of_variation",
            )
        }
    return {
        "schema_version": 2,
        "method": (
            "Reconstruct output-chunk timestamps from AIPerf request_start_ns, "
            "TTFT, and inter_chunk_latency. At each positive stream interval "
            "midpoint, count active Decode requests and sum their current "
            "contexts; multi-token chunks distribute their tokens in proportion "
            "to observed chunk progress. Compare arms with shared-bin "
            "minimum-count weights."
        ),
        "context_bucket_tokens": context_bucket_tokens,
        "baseline": baseline_summary,
        "treatment": treatment_summary,
        "comparison": comparison,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--context-bucket-tokens", type=int, default=16_384)
    parser.add_argument("--minimum-bin-samples", type=int, default=100)
    parser.add_argument("--baseline-proxy-log", type=Path)
    parser.add_argument("--treatment-proxy-log", type=Path)
    parser.add_argument("--pa-count", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.context_bucket_tokens <= 0:
        parser.error("--context-bucket-tokens must be positive")
    if args.minimum_bin_samples <= 0:
        parser.error("--minimum-bin-samples must be positive")
    if args.pa_count <= 0:
        parser.error("--pa-count must be positive")
    if (args.baseline_proxy_log is None) != (args.treatment_proxy_log is None):
        parser.error("both proxy logs must be provided together")
    return args


def main() -> None:
    args = _parse_args()
    result = analyze_profiles(
        load_profile(args.baseline),
        load_profile(args.treatment),
        context_bucket_tokens=args.context_bucket_tokens,
        minimum_bin_samples=args.minimum_bin_samples,
        baseline_pa_by_request_id=(
            load_pa_assignments(args.baseline_proxy_log)
            if args.baseline_proxy_log is not None
            else None
        ),
        treatment_pa_by_request_id=(
            load_pa_assignments(args.treatment_proxy_log)
            if args.treatment_proxy_log is not None
            else None
        ),
        pa_count=args.pa_count,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
