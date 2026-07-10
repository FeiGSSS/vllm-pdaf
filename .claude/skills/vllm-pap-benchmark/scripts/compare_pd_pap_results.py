#!/usr/bin/env python3
"""Compare canonical PD and PAP benchmark JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PD = Path(
    "/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/"
    "20260701_171300/1P1D_i128_o32_q16.json"
)
DEFAULT_PAP = Path(
    "/home/fei/research/PD/vllm-pap/test/baseline/pap/results/runs/"
    "20260710_e904_nixl_rep1_clean/1PA1P_i128_o32_q16.json"
)

METRICS = [
    ("completed", "successful"),
    ("failed", "failed"),
    ("duration", "duration_s"),
    ("request_throughput", "req_s"),
    ("output_throughput", "output_tok_s"),
    ("total_token_throughput", "total_tok_s"),
    ("mean_ttft_ms", "mean_ttft_ms"),
    ("median_ttft_ms", "median_ttft_ms"),
    ("p99_ttft_ms", "p99_ttft_ms"),
    ("mean_tpot_ms", "mean_tpot_ms"),
    ("median_tpot_ms", "median_tpot_ms"),
    ("p99_tpot_ms", "p99_tpot_ms"),
    ("mean_itl_ms", "mean_itl_ms"),
    ("p99_itl_ms", "p99_itl_ms"),
    ("max_concurrent_requests", "peak_concurrency"),
]


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_ratio(pd_value: object, pap_value: object) -> str:
    if (
        isinstance(pd_value, (int, float))
        and isinstance(pap_value, (int, float))
        and pd_value != 0
    ):
        return f"{pap_value / pd_value:.3f}x"
    return "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd", type=Path, default=DEFAULT_PD)
    parser.add_argument("--pap", type=Path, default=DEFAULT_PAP)
    args = parser.parse_args()

    pd_result = load_json(args.pd)
    pap_result = load_json(args.pap)

    print(f"PD: `{args.pd}`")
    print(f"PAP: `{args.pap}`")
    print()
    print("| metric | PD | PAP | PAP/PD |")
    print("|---|---:|---:|---:|")
    for key, label in METRICS:
        pd_value = pd_result.get(key)
        pap_value = pap_result.get(key)
        ratio = format_ratio(pd_value, pap_value)
        print(
            f"| {label} | {format_value(pd_value)} | "
            f"{format_value(pap_value)} | {ratio} |"
        )

    pd_peak = pd_result.get("max_concurrent_requests")
    pap_peak = pap_result.get("max_concurrent_requests")
    if (
        isinstance(pd_peak, (int, float))
        and isinstance(pap_peak, (int, float))
        and pap_peak > pd_peak * 1.5
    ):
        print()
        print(
            "Note: PAP peak concurrency is much higher than PD, so TTFT includes "
            "queue buildup and should not be attributed to one operation alone."
        )


if __name__ == "__main__":
    main()
