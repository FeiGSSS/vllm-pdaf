# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Break Dynamo TTFT into routing, Prefill, handoff, and Decode segments."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import regex as re

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[^ ]+)Z")
UUID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def _timestamp(line: str) -> float | None:
    match = TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.fromisoformat(match.group(1) + "+00:00").timestamp()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p50": median(values) if values else None,
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _read_lines(path: Path) -> list[str]:
    return [ANSI_RE.sub("", line) for line in path.read_text().splitlines()]


def _parse_worker_logs(
    paths: list[Path],
) -> tuple[dict[int, dict[str, Any]], dict[str, float], dict[str, float]]:
    workers: dict[int, dict[str, Any]] = {}
    received: dict[str, float] = {}
    completed: dict[str, float] = {}
    for index, path in enumerate(paths):
        instance_id = None
        final_hit_rate = None
        max_kv_usage = 0.0
        max_waiting = 0
        lines = _read_lines(path)
        for line in lines:
            timestamp = _timestamp(line)
            if "request received" in line and timestamp is not None:
                request_ids = UUID_RE.findall(line)
                if request_ids:
                    received.setdefault(request_ids[0], timestamp)
                match = re.search(r"instance_id=(\d+)", line)
                if match is not None:
                    instance_id = int(match.group(1))
            elif "request completed" in line and timestamp is not None:
                request_ids = UUID_RE.findall(line)
                if request_ids:
                    completed.setdefault(request_ids[0], timestamp)
            if "Prefix cache hit rate:" in line:
                match = re.search(r"Waiting: (\d+) reqs", line)
                if match is not None:
                    max_waiting = max(max_waiting, int(match.group(1)))
                match = re.search(r"GPU KV cache usage: ([0-9.]+)%", line)
                if match is not None:
                    max_kv_usage = max(max_kv_usage, float(match.group(1)))
                match = re.search(r"Prefix cache hit rate: ([0-9.]+)%", line)
                if match is not None:
                    final_hit_rate = float(match.group(1))
        if instance_id is None:
            continue
        workers[instance_id] = {
            "index": index,
            "log": str(path),
            "final_prefix_cache_hit_pct": final_hit_rate,
            "max_kv_cache_usage_pct": max_kv_usage,
            "max_waiting_requests": max_waiting,
        }
    return workers, received, completed


def _parse_frontend(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    for line in _read_lines(path):
        timestamp = _timestamp(line)
        if "Selected worker" in line and "worker_type=prefill" in line:
            request_ids = re.findall(r'request_id="([0-9a-f-]+)"', line)
            worker = re.search(r"worker_id=(\d+)", line)
            cached = re.search(r"effective_cached_blocks=([0-9.]+)", line)
            logit = re.search(r"logit=([0-9.]+)", line)
            if request_ids and worker is not None and timestamp is not None:
                selected[request_ids[0]] = {
                    "timestamp": timestamp,
                    "worker_id": int(worker.group(1)),
                    "cached_blocks": float(cached.group(1)) if cached else None,
                    "logit": float(logit.group(1)) if logit else None,
                }
        elif "request completed" in line and "prefill_worker_id=" in line:
            request_ids = UUID_RE.findall(line)
            request_id = next((item for item in request_ids if item in selected), None)
            worker = re.search(r"prefill_worker_id=(\d+)", line)
            input_tokens = re.search(r"input_tokens=(\d+)", line)
            ttft = re.search(r'ttft_ms="?([0-9.]+)', line)
            if request_id is not None and worker is not None and input_tokens and ttft:
                completions[request_id] = {
                    "timestamp": timestamp,
                    "worker_id": int(worker.group(1)),
                    "input_tokens": int(input_tokens.group(1)),
                    "ttft_ms": float(ttft.group(1)),
                }
    return selected, completions


def _parse_transfer_metrics(paths: list[Path]) -> dict[str, float | int | None]:
    rows = []
    pattern = re.compile(
        r"Num successful transfers=(\d+).*?"
        r"Avg xfer time \(ms\)=([0-9.]+).*?"
        r"Avg MB per transfer=([0-9.]+).*?"
        r"Throughput \(MB/s\)=([0-9.]+).*?"
        r"Avg number of descriptors=([0-9.]+)"
    )
    for path in paths:
        for line in _read_lines(path):
            match = pattern.search(line)
            if match is None:
                continue
            rows.append(tuple(float(value) for value in match.groups()))
    transfers = sum(int(row[0]) for row in rows)
    total_mb = sum(row[0] * row[2] for row in rows)
    total_ms = sum(row[0] * row[1] for row in rows)
    return {
        "metric_windows": len(rows),
        "transfers": transfers,
        "aggregate_throughput_mb_s": (total_mb / total_ms * 1000 if total_ms else None),
        "weighted_transfer_ms": (total_ms / transfers if transfers else None),
        "weighted_mb_per_transfer": (total_mb / transfers if transfers else None),
        "weighted_descriptors": (
            sum(row[0] * row[4] for row in rows) / transfers if transfers else None
        ),
        "throughput_mb_s": _summary([row[3] for row in rows]),
    }


def analyze(run_root: Path, block_size: int) -> dict[str, Any]:
    logs = run_root / "service_logs"
    decode_logs = sorted(logs.glob("decode_*.log"))
    transfer_metrics = _parse_transfer_metrics(decode_logs)
    frontend_log = logs / "frontend.log"
    if not frontend_log.exists():
        return {
            "run_root": str(run_root),
            "block_size": block_size,
            "kv_transfer": transfer_metrics,
        }
    prefill_workers, prefill_received, prefill_completed = _parse_worker_logs(
        sorted(logs.glob("prefill_*.log"))
    )
    _, decode_received, _ = _parse_worker_logs(decode_logs)
    selected, completions = _parse_frontend(frontend_log)

    rows: list[dict[str, Any]] = []
    for request_id, completion in completions.items():
        selection = selected[request_id]
        selected_at = selection["timestamp"]
        prefill_start = prefill_received.get(request_id)
        prefill_end = prefill_completed.get(request_id)
        decode_start = decode_received.get(request_id)
        cached_blocks = selection["cached_blocks"]
        cached_tokens = (
            min(completion["input_tokens"], cached_blocks * block_size)
            if cached_blocks is not None
            else None
        )
        rows.append(
            {
                "request_id": request_id,
                "worker_id": completion["worker_id"],
                "input_tokens": completion["input_tokens"],
                "cached_tokens": cached_tokens,
                "ttft_ms": completion["ttft_ms"],
                "router_to_prefill_ms": (
                    (prefill_start - selected_at) * 1000
                    if prefill_start is not None
                    else None
                ),
                "prefill_service_ms": (
                    (prefill_end - prefill_start) * 1000
                    if prefill_start is not None and prefill_end is not None
                    else None
                ),
                "router_to_decode_ms": (
                    (decode_start - selected_at) * 1000
                    if decode_start is not None
                    else None
                ),
                "decode_to_first_token_ms": (
                    completion["ttft_ms"] - (decode_start - selected_at) * 1000
                    if decode_start is not None
                    else None
                ),
            }
        )

    by_worker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_worker[row["worker_id"]].append(row)

    worker_output = []
    for worker_id, worker_rows in sorted(
        by_worker.items(), key=lambda item: prefill_workers[item[0]]["index"]
    ):
        input_tokens = sum(row["input_tokens"] for row in worker_rows)
        cached_tokens = sum(row["cached_tokens"] or 0 for row in worker_rows)
        output = dict(prefill_workers[worker_id])
        output.update(
            {
                "worker_id": worker_id,
                "requests": len(worker_rows),
                "input_tokens": input_tokens,
                "estimated_cached_tokens": cached_tokens,
                "estimated_cache_read_pct": (
                    cached_tokens / input_tokens * 100 if input_tokens else None
                ),
                "ttft_ms": _summary([row["ttft_ms"] for row in worker_rows]),
                "router_to_prefill_ms": _summary(
                    [
                        row["router_to_prefill_ms"]
                        for row in worker_rows
                        if row["router_to_prefill_ms"] is not None
                    ]
                ),
                "prefill_service_ms": _summary(
                    [
                        row["prefill_service_ms"]
                        for row in worker_rows
                        if row["prefill_service_ms"] is not None
                    ]
                ),
                "router_to_decode_ms": _summary(
                    [
                        row["router_to_decode_ms"]
                        for row in worker_rows
                        if row["router_to_decode_ms"] is not None
                    ]
                ),
                "decode_to_first_token_ms": _summary(
                    [
                        row["decode_to_first_token_ms"]
                        for row in worker_rows
                        if row["decode_to_first_token_ms"] is not None
                    ]
                ),
            }
        )
        worker_output.append(output)

    return {
        "run_root": str(run_root),
        "block_size": block_size,
        "selected_requests": len(selected),
        "completed_requests": len(completions),
        "joined_requests": len(rows),
        "kv_transfer": transfer_metrics,
        "overall": {
            "ttft_ms": _summary([row["ttft_ms"] for row in rows]),
            "router_to_prefill_ms": _summary(
                [
                    row["router_to_prefill_ms"]
                    for row in rows
                    if row["router_to_prefill_ms"] is not None
                ]
            ),
            "prefill_service_ms": _summary(
                [
                    row["prefill_service_ms"]
                    for row in rows
                    if row["prefill_service_ms"] is not None
                ]
            ),
            "router_to_decode_ms": _summary(
                [
                    row["router_to_decode_ms"]
                    for row in rows
                    if row["router_to_decode_ms"] is not None
                ]
            ),
            "decode_to_first_token_ms": _summary(
                [
                    row["decode_to_first_token_ms"]
                    for row in rows
                    if row["decode_to_first_token_ms"] is not None
                ]
            ),
        },
        "workers": worker_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_root, args.block_size)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
