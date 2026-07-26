# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Projection fan-in skew distributions from two PAP trace runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.pap.tooling.trace_summary import (
    summarize_pap_trace_logs,
    summary_to_jsonable,
)


def _samples(log_dir: Path) -> tuple[dict[str, Any], dict[str, list[float]]]:
    summary = summarize_pap_trace_logs(log_dir, include_samples=True)
    fanin_samples = summary.pop("projection_fanin_samples")
    correlation_samples = summary.pop("projection_attention_correlation_samples")
    assert isinstance(fanin_samples, dict)
    assert isinstance(correlation_samples, dict)
    if fanin_samples.get("spread_ms"):
        samples = {
            "peers": fanin_samples["peers"],
            "first_ready_ms": fanin_samples["first_ready_ms"],
            "last_ready_ms": fanin_samples["last_ready_ms"],
            "pa_completion_skew_ms": fanin_samples["spread_ms"],
            "pa_completion_skew_over_fastest_pct": fanin_samples[
                "spread_over_fastest_pct"
            ],
        }
    else:
        samples = correlation_samples
    assert isinstance(samples, dict)
    return summary_to_jsonable(summary), samples


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100.0)
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 90),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def _stats_by_peer_count(
    samples: dict[str, list[float]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    peers = samples.get("peers")
    if not peers:
        return {}
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for peer_count in sorted({int(value) for value in peers}):
        indices = [
            index
            for index, value in enumerate(peers)
            if int(value) == peer_count
        ]
        result[str(peer_count)] = {
            metric: _stats([values[index] for index in indices])
            for metric, values in samples.items()
            if metric != "peers"
        }
    return result


def _plot_histogram(
    baseline: list[float],
    comparison: list[float],
    *,
    baseline_label: str,
    comparison_label: str,
    xlabel: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    upper = _percentile(baseline + comparison, 95)
    baseline_body = [value for value in baseline if value <= upper]
    comparison_body = [value for value in comparison if value <= upper]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.hist(
        baseline_body,
        bins=80,
        range=(0, upper),
        density=True,
        alpha=0.55,
        label=baseline_label,
    )
    axis.hist(
        comparison_body,
        bins=80,
        range=(0, upper),
        density=True,
        alpha=0.55,
        label=comparison_label,
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Density")
    axis.set_yscale("log")
    axis.set_title("Central 95% (full tail retained in summary JSON)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_skew_by_peer_count(
    baseline: dict[str, dict[str, dict[str, float | int]]],
    comparison: dict[str, dict[str, dict[str, float | int]]],
    *,
    baseline_label: str,
    comparison_label: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for grouped, label, color in (
        (baseline, baseline_label, "tab:blue"),
        (comparison, comparison_label, "tab:orange"),
    ):
        peer_counts = sorted(int(value) for value in grouped)
        medians = [
            grouped[str(peers)]["pa_completion_skew_ms"]["median"]
            for peers in peer_counts
        ]
        p90s = [
            grouped[str(peers)]["pa_completion_skew_ms"]["p90"]
            for peers in peer_counts
        ]
        axis.plot(
            peer_counts,
            medians,
            marker="o",
            color=color,
            label=f"{label} median",
        )
        axis.plot(
            peer_counts,
            p90s,
            marker="^",
            linestyle="--",
            color=color,
            label=f"{label} P90",
        )
    axis.set_xlabel("Participating PA count")
    axis.set_ylabel("Last PA return - first PA return (ms)")
    axis.set_title("Fan-in skew conditioned on barrier width")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_logs", type=Path)
    parser.add_argument("comparison_logs", type=Path)
    parser.add_argument(
        "--baseline-label",
        default="conversation_affinity",
    )
    parser.add_argument(
        "--comparison-label",
        default="attention_load",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    baseline_summary, baseline_samples = _samples(args.baseline_logs)
    comparison_summary, comparison_samples = _samples(args.comparison_logs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        (
            "first_ready_ms",
            "Projection wait until first PA return (ms)",
            "fanin_first_ready_ms_histogram.png",
        ),
        (
            "last_ready_ms",
            "Projection wait until last PA return (ms)",
            "fanin_last_ready_ms_histogram.png",
        ),
        (
            "pa_completion_skew_ms",
            "Projection fan-in spread: last PA return - first PA return (ms)",
            "fanin_skew_ms_histogram.png",
        ),
        (
            "pa_completion_skew_over_fastest_pct",
            "Fan-in spread / fastest PA return latency (%)",
            "fanin_skew_relative_histogram.png",
        ),
    )
    for metric, xlabel, filename in metrics:
        baseline = baseline_samples.get(metric, [])
        comparison = comparison_samples.get(metric, [])
        if not baseline or not comparison:
            raise ValueError(f"missing fan-in samples for {metric}")
        _plot_histogram(
            baseline,
            comparison,
            baseline_label=args.baseline_label,
            comparison_label=args.comparison_label,
            xlabel=xlabel,
            output=args.output_dir / filename,
        )

    baseline_by_peers = _stats_by_peer_count(baseline_samples)
    comparison_by_peers = _stats_by_peer_count(comparison_samples)
    _plot_skew_by_peer_count(
        baseline_by_peers,
        comparison_by_peers,
        baseline_label=args.baseline_label,
        comparison_label=args.comparison_label,
        output=args.output_dir / "fanin_skew_by_peer_count.png",
    )

    payload = {
        "relative_metric_definition": (
            "(last_projection_gpu_ready - first_projection_gpu_ready) / "
            "first_projection_gpu_ready_latency * 100"
        ),
        "measurement": (
            "Projection-side CUDA events immediately after each PA output-ready wait"
        ),
        args.baseline_label: baseline_summary["projection_fanin"],
        args.comparison_label: comparison_summary["projection_fanin"],
        "conditioned_on_peer_count": {
            args.baseline_label: baseline_by_peers,
            args.comparison_label: comparison_by_peers,
        },
    }
    (args.output_dir / "fanin_skew_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
