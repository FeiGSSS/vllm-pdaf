# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot the measured configuration tradeoffs, excluding diagnostic runs."""

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
groups = {
    "80/12 SM\n2048 tokens\n2 runs": ("baseline_r1", "baseline_r2"),
    "76/16 SM\n2048 tokens\n1 run": ("attention16_r1",),
    "72/20 SM\n2048 tokens\n2 runs": ("attention20_r1", "attention20_r2"),
    "80/12 SM\n4096 tokens\n1 run": ("budget4096_r1",),
}
values = []
for names in groups.values():
    rows = [
        json.loads((root / "runs" / name / "analysis.json").read_text())
        for name in names
    ]
    duration = sum(row["duration_s"] for row in rows)
    good = sum(row["slo"]["D3"]["good"] for row in rows)
    values.append(
        (
            mean(row["time_to_first_token"]["mean"] for row in rows) / 1000,
            mean(row["inter_token_latency"]["mean"] for row in rows),
            180 * len(rows) / duration,
            good / duration,
            good / (180 * len(rows)),
        )
    )

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
titles = (
    "Mean TTFT (s), lower is better",
    "Mean request ITL (ms), lower is better",
    "Request throughput (req/s)",
    "D3 goodput (req/s); labels show attainment",
)
colors = ["#477694", "#75a594", "#8062a0", "#cb9767"]
for index, axis in enumerate(axes.flat):
    data = [value[index] for value in values]
    bars = axis.bar(range(4), data, color=colors, width=0.65)
    axis.set_xticks(range(4), list(groups), fontsize=9)
    axis.set_title(titles[index], fontsize=11)
    axis.set_ylim(0, max(data) * 1.24)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    for i, bar in enumerate(bars):
        label = f"{data[i]:.2f}"
        if index == 3:
            label += f"\n{values[i][4] * 100:.1f}%"
            if values[i][4] < 0.95:
                bar.set_hatch("//")
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
fig.suptitle("PAP 7PA1P / Qwen3-8B FP16 / 8 x L20 / concurrency 30", fontsize=14)
fig.supxlabel(
    "D3: TTFT <= 15 s, request ITL <= 50 ms, E2E <= 20 s. "
    "Hatching: below 95% attainment. Pilot results, 1-2 runs/config.",
    fontsize=9,
)
fig.savefig(root / "tradeoffs.png", dpi=170)
fig.savefig(root / "tradeoffs.svg")
