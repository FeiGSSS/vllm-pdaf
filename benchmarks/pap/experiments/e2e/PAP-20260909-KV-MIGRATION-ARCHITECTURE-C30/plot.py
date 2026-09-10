#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plot the aligned single-host architecture tradeoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())

    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    for name, item in data["architectures"].items():
        pap = name == "pap7pa1p_kv_migration"
        x = item["ttft_ms"]["mean"] / 1000
        y = item["itl_ms"]["mean"]
        throughput = item["request_throughput_rps"]
        axis.scatter(
            x,
            y,
            s=throughput * 180,
            marker="*" if pap else "o",
            color="#d62728" if pap else "#4c78a8",
            edgecolor="black",
            linewidth=0.7,
            zorder=3 if pap else 2,
        )
        axis.annotate(
            item["display_name"],
            (x, y),
            xytext=(7, 6 if name != "pd6p2d" else -15),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set_xscale("log")
    axis.set_xlabel("Mean TTFT (seconds, log scale; lower is better)")
    axis.set_ylabel("Mean request-level ITL (ms; lower is better)")
    axis.set_title("Qwen3-8B FP16, 8×L20, native 32K, 60×3 sessions, C30")
    axis.grid(True, which="both", alpha=0.25)
    axis.text(
        0.99,
        0.02,
        "Marker area ∝ request throughput",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
