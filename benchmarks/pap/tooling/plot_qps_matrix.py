# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize and plot the standardized PAP architecture QPS matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ARCHITECTURES = (
    "dp8",
    "2p6d",
    "4p4d",
    "6p2d",
    "pap_7pa1p_2k",
    "pap_7pa1p_32k",
    "pap_6pa2p_2k",
    "pap_6pa2p_32k",
)
ARCHITECTURE_LABELS = {
    "dp8": "DP8",
    "2p6d": "2P6D",
    "4p4d": "4P4D",
    "6p2d": "6P2D",
    "pap_7pa1p_2k": "PAP 7PA1P-2K",
    "pap_7pa1p_32k": "PAP 7PA1P-32K",
    "pap_6pa2p_2k": "PAP 6PA2P-2K",
    "pap_6pa2p_32k": "PAP 6PA2P-32K",
}
QPS_POINTS = (0.6, 0.9, 1.2, 1.5, 1.8)
SUMMARY_FIELDS = (
    "architecture",
    "configured_qps",
    "actual_qps",
    "request_count",
    "output_tokens_per_second",
    "ttft_mean_ms",
    "ttft_p99_ms",
    "tbt_mean_ms",
    "tbt_p99_ms",
    "e2e_mean_ms",
    "e2e_p99_ms",
    "status",
    "kv_transfer_aggregate_mb_s",
    "profile",
)


def _qps_tag(qps: float) -> str:
    return f"{qps:.1f}".replace(".", "p")


def _profile_is_complete(profile: dict[str, Any], qps: float) -> bool:
    phases = profile.get("input_config", {}).get("phases", [])
    matching_phase = any(
        phase.get("name") == "profiling"
        and phase.get("type") == "poisson"
        and phase.get("sessions") == 60
        and phase.get("concurrency") == 60
        and phase.get("rate") == qps
        for phase in phases
    )
    return (
        profile.get("request_count", {}).get("avg") == 180
        and not profile.get("error_summary")
        and not profile.get("was_cancelled")
        and matching_phase
    )


def _load_point(
    matrix_root: Path, architecture: str, qps: float
) -> dict[str, Any] | None:
    point_root = matrix_root / architecture / f"qps_{_qps_tag(qps)}"
    for attempt in sorted(point_root.glob("attempt_*"), reverse=True):
        profile_path = attempt / "aiperf" / "profile.json"
        if not profile_path.is_file():
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if not _profile_is_complete(profile, qps):
            continue
        correctness = attempt / "correctness_audit.env"
        if not correctness.is_file() or "STATUS=passed" not in (
            correctness.read_text(encoding="utf-8").splitlines()
        ):
            continue
        if architecture.startswith("pap_"):
            graph_audit = attempt / "pap_whole_step_graph_audit.env"
        else:
            graph_audit = attempt / "vllm_cuda_graph_audit.env"
        if not graph_audit.is_file() or "STATUS=passed" not in (
            graph_audit.read_text(encoding="utf-8").splitlines()
        ):
            continue
        status = "passed"
        kv_throughput = None
        if architecture in {"2p6d", "4p4d", "6p2d"}:
            kv_audit = attempt / "kv_transfer_audit.env"
            if not kv_audit.is_file():
                continue
            if "STATUS=failed" in kv_audit.read_text(encoding="utf-8").splitlines():
                status = "kv_transfer_below_floor"
            analysis_path = attempt / "dynamo_ttft_analysis.json"
            if analysis_path.is_file():
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                kv_throughput = analysis.get("kv_transfer", {}).get(
                    "aggregate_throughput_mb_s"
                )
        return {
            "architecture": architecture,
            "configured_qps": qps,
            "actual_qps": profile["request_throughput"]["avg"],
            "request_count": int(profile["request_count"]["avg"]),
            "output_tokens_per_second": profile["output_token_throughput"]["avg"],
            "ttft_mean_ms": profile["time_to_first_token"]["avg"],
            "ttft_p99_ms": profile["time_to_first_token"]["p99"],
            "tbt_mean_ms": profile["inter_token_latency"]["avg"],
            "tbt_p99_ms": profile["inter_token_latency"]["p99"],
            "e2e_mean_ms": profile["request_latency"]["avg"],
            "e2e_p99_ms": profile["request_latency"]["p99"],
            "status": status,
            "kv_transfer_aggregate_mb_s": kv_throughput,
            "profile": str(profile_path.relative_to(matrix_root)),
        }
    return None


def load_rows(matrix_root: Path) -> list[dict[str, Any]]:
    rows = []
    for architecture in ARCHITECTURES:
        for qps in QPS_POINTS:
            row = _load_point(matrix_root, architecture, qps)
            if row is not None:
                rows.append(row)
    return rows


def write_summaries(matrix_root: Path, rows: list[dict[str, Any]]) -> None:
    summary_tsv = matrix_root / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (matrix_root / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )


def plot_rows(matrix_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hasattr(plt.style, "core"):
        plt.style.core = plt.style
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex", "grid"])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=True)
    panels = (
        ("e2e_p99_ms", 1e-3, "P99 E2E latency (s)"),
        ("tbt_mean_ms", 1.0, "Mean TBT (ms)"),
        ("ttft_mean_ms", 1e-3, "Mean TTFT (s)"),
    )
    markers = ("o", "s", "^", "D", "P", "h", "v", "*")
    for architecture, marker in zip(ARCHITECTURES, markers, strict=True):
        architecture_rows = {
            row["configured_qps"]: row
            for row in rows
            if row["architecture"] == architecture
        }
        for axis, (field, scale, ylabel) in zip(axes, panels, strict=True):
            x_values = [qps for qps in QPS_POINTS if qps in architecture_rows]
            y_values = [architecture_rows[qps][field] * scale for qps in x_values]
            axis.plot(
                x_values,
                y_values,
                marker=marker,
                linewidth=1.5,
                markersize=4.5,
                label=ARCHITECTURE_LABELS[architecture],
            )
            warning_x = [
                qps
                for qps in x_values
                if architecture_rows[qps]["status"] == "kv_transfer_below_floor"
            ]
            warning_y = [architecture_rows[qps][field] * scale for qps in warning_x]
            axis.scatter(
                warning_x,
                warning_y,
                marker="x",
                color="black",
                s=32,
                linewidths=1.1,
                zorder=5,
            )
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Configured request rate (req/s)")
            axis.set_xticks(QPS_POINTS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    figures = matrix_root / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "qps_latency.png", dpi=240)
    fig.savefig(figures / "qps_latency.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_root = args.matrix_root.resolve()
    rows = load_rows(matrix_root)
    write_summaries(matrix_root, rows)
    plot_rows(matrix_root, rows)
    total_points = len(ARCHITECTURES) * len(QPS_POINTS)
    print(f"Summarized {len(rows)}/{total_points} points in {matrix_root}")


if __name__ == "__main__":
    main()
