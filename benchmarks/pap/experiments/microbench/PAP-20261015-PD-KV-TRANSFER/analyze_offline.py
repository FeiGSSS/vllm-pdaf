#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bound PD P->D KV handoff as a share of TTFT from archived Dynamo runs.

Usage: analyze_offline.py <arch>:<run>=<extracted run root> ...

Each run root must already contain ``ttft_breakdown.json`` from
``benchmarks/pap/tooling/analyze_dynamo_ttft.py``. Two per-request bounds:

- ``d2ft``: Decode receipt to first token. It contains the KV read, Decode
  queueing and the first Decode step, so it upper-bounds the handoff.
- ``xfer_est``: full-prompt KV bytes divided by the run's measured aggregate
  NIXL throughput. Decode reads only blocks it lacks, so this overestimates the
  transfer (measured MB per transfer is 0.6-0.7 of the full-prompt size).
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

KV_BYTES_PER_TOKEN = 147_456
BUCKETS = [(0, 16384), (16384, 20480), (20480, 24576), (24576, 28672), (28672, 32769)]
RESULTS = Path(__file__).resolve().parent / "results"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bucket_name(lo: int, hi: int) -> str:
    return f"<{hi // 1024}K" if lo == 0 else f"{lo // 1024}-{hi // 1024}K"


def main() -> None:
    rows, runs = [], []
    for spec in sys.argv[1:]:
        label, _, root = spec.partition("=")
        arch, _, run = label.partition(":")
        data = json.loads((Path(root) / "ttft_breakdown.json").read_text())
        transfer = data["kv_transfer"]
        rate_bytes_per_ms = transfer["aggregate_throughput_mb_s"] * 1e6 / 1e3
        runs.append(
            {
                "architecture": arch,
                "run": run,
                "transfers": transfer["transfers"],
                "weighted_transfer_ms": transfer["weighted_transfer_ms"],
                "weighted_mb_per_transfer": transfer["weighted_mb_per_transfer"],
                "aggregate_gb_s": transfer["aggregate_throughput_mb_s"] / 1e3,
                "ttft_mean_ms": data["overall"]["ttft_ms"]["mean"],
            }
        )
        for request in data["requests"]:
            tokens = request["input_tokens"]
            rows.append(
                {
                    "architecture": arch,
                    "run": run,
                    "turn": request["turn_index"],
                    "input_tokens": tokens,
                    "cached_tokens": request["cached_tokens"],
                    "ttft_ms": request["ttft_ms"],
                    "d2ft_ms": request["decode_to_first_token_ms"],
                    "xfer_est_ms": tokens * KV_BYTES_PER_TOKEN / rate_bytes_per_ms,
                }
            )

    table = []
    for arch in dict.fromkeys(r["architecture"] for r in rows):
        for lo, hi in BUCKETS:
            group = [
                r
                for r in rows
                if r["architecture"] == arch and lo <= r["input_tokens"] < hi
            ]
            if not group:
                continue
            entry = {
                "architecture": arch,
                "context": bucket_name(lo, hi),
                "n": len(group),
            }
            for key in ("d2ft", "xfer_est"):
                shares = [r[f"{key}_ms"] / r["ttft_ms"] for r in group]
                entry[f"{key}_ms_mean"] = statistics.fmean(
                    r[f"{key}_ms"] for r in group
                )
                entry[f"{key}_share_mean"] = statistics.fmean(shares)
                entry[f"{key}_share_p99"] = percentile(shares, 0.99)
            entry["ttft_ms_mean"] = statistics.fmean(r["ttft_ms"] for r in group)
            table.append(entry)

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "offline_requests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS / "offline_handoff_by_context.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (RESULTS / "offline_runs.json").write_text(json.dumps(runs, indent=2) + "\n")

    print(
        "arch     context   n  TTFT_ms  d2ft_ms d2ft%  d2ft%p99"
        "  xfer_ms xfer%  xfer%p99"
    )
    for e in table:
        print(
            f"{e['architecture']:8s} {e['context']:8s} {e['n']:3d} "
            f"{e['ttft_ms_mean']:8.0f} {e['d2ft_ms_mean']:8.0f} "
            f"{e['d2ft_share_mean']:5.1%} {e['d2ft_share_p99']:8.1%} "
            f"{e['xfer_est_ms_mean']:8.0f} {e['xfer_est_share_mean']:5.1%} "
            f"{e['xfer_est_share_p99']:8.1%}"
        )


if __name__ == "__main__":
    main()
