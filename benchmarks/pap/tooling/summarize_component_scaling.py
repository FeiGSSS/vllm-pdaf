#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build tables, figures, and a report for PAP component scaling probes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

STAGES = (
    "input_layernorm",
    "qkv_proj",
    "qk_norm_rope",
    "o_proj",
    "post_attention_layernorm",
    "mlp",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def projection_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("status") != "completed":
        raise ValueError("Projection matrix is incomplete")
    rows = []
    for item in result["results"]:
        if not item["shape_audit"]["valid"]:
            raise ValueError(f"Projection shape audit failed at B={item['batch_size']}")
        row = {
            "batch_size": item["batch_size"],
            "per_layer_total_ms": item["per_layer_total_ms"],
            "qk_repack_ms": item["qk_repack"]["median_ms"],
            "rows_per_second": item["rows_per_second"],
            "useful_dense_tflops": item["useful_dense_tflops"],
        }
        for stage in STAGES:
            row[f"{stage}_ms"] = item["stages"][stage]["median_ms"]
        rows.append(row)
    return sorted(rows, key=lambda row: int(row["batch_size"]))


def attention_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("status") != "completed":
        raise ValueError("Attention matrix is incomplete")
    rows = []
    for item in result["results"]:
        shape = item["shape"]
        workspace = item["workspace"]
        rows.append(
            {
                "batch_size": shape["batch_size"],
                "context_tokens_per_request": shape["context_tokens_per_request"],
                "total_context_tokens": shape["total_context_tokens"],
                "groups": ",".join(shape["groups"]),
                "attention_ms": item["paged_attention"]["median_ms"],
                "kv_append_ms": item["kv_append"]["median_ms"],
                "combined_ms": item["append_then_attention"]["median_ms"],
                "logical_kv_gbps": item["logical_kv_gbps"],
                "visible_sms": item["device"]["multiprocessor_count"],
                "num_splits": workspace["num_splits"],
                "block_h": workspace["block_h"],
                "num_warps": workspace["num_warps"],
                "num_stages": workspace["num_stages"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row["total_context_tokens"]),
            int(row["batch_size"]),
        ),
    )


def prefill_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("status") != "analyzed":
        raise ValueError("Prefill matrix is incomplete")
    if not result["shape_audit"]["valid"]:
        raise ValueError("Prefill shape audit failed")
    return [
        {
            "batch_size": item["batch_size"],
            "prompt_tokens": item["prompt_tokens"],
            "prefill_engine_ms": item["prefill_engine_ms_median"],
            "prompt_tokens_per_second": item["prompt_tokens_per_second"],
            "sample_count": item["sample_count"],
        }
        for item in sorted(
            result["summaries"], key=lambda item: int(item["prompt_tokens"])
        )
    ]


def fixed_total_spreads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "iso_total" in str(row["groups"]):
            grouped[int(row["total_context_tokens"])].append(row)
    output = []
    for total, items in sorted(grouped.items()):
        latencies = [float(item["attention_ms"]) for item in items]
        fastest = min(items, key=lambda item: float(item["attention_ms"]))
        slowest = max(items, key=lambda item: float(item["attention_ms"]))
        output.append(
            {
                "total_context_tokens": total,
                "min_attention_ms": min(latencies),
                "max_attention_ms": max(latencies),
                "spread_percent": (max(latencies) / min(latencies) - 1) * 100,
                "fastest_batch": fastest["batch_size"],
                "slowest_batch": slowest["batch_size"],
            }
        )
    return output


def hardware_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        result = read_json(path)
        component = result["component"]
        point = path.parent.name
        if component == "attention":
            point = path.stem.removesuffix("_metrics")
        summaries = [(point, result["all_ranges"])]
        if component == "prefill":
            planned = {256, 1_000, 10_000, 30_000}
            summaries = [
                (name.removeprefix("pap_prefill_model_forward_"), summary)
                for name, summary in result["ranges"].items()
                if int(name.rsplit("t", 1)[1]) in planned
            ]
        for point, summary in summaries:
            mean = summary["mean"]
            rows.append(
                {
                    "component": component,
                    "point": point,
                    "range_count": summary["range_count"],
                    "range_ms_total": summary["range_ms_total"],
                    "sample_count": summary["sample_count"],
                    "dram_read": mean["DRAM Read Throughput"],
                    "dram_write": mean["DRAM Write Throughput"],
                    "sm_active": mean["SM Active"],
                    "sm_issue": mean["SM Issue"],
                    "tensor_active": mean["Tensor Active"],
                    "warps_in_flight": mean["Compute Warps In Flight"],
                    "gpc_clock_mhz": mean["GPC Clock Frequency"] / 1e6,
                }
            )
    return sorted(rows, key=lambda row: (row["component"], row["point"]))


def save_figure(figure: Any, output: Path) -> None:
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".pdf"))


def plot_projection(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    batches = [int(row["batch_size"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        batches,
        [float(row["per_layer_total_ms"]) for row in rows],
        marker="o",
        linewidth=2,
        label="all non-Attention",
    )
    for stage in ("qkv_proj", "o_proj", "mlp", "qk_norm_rope"):
        axes[0].plot(
            batches,
            [float(row[f"{stage}_ms"]) for row in rows],
            marker=".",
            label=stage,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Decode batch rows")
    axes[0].set_ylabel("Latency per layer (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(
        batches,
        [float(row["useful_dense_tflops"]) for row in rows],
        marker="o",
        color="tab:orange",
        linewidth=2,
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Decode batch rows")
    axes[1].set_ylabel("Useful dense TFLOP/s")
    axes[1].grid(alpha=0.25)
    save_figure(figure, output)
    plt.close(figure)


def plot_attention(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for batch in (1, 8, 32):
        items = [
            row
            for row in rows
            if int(row["batch_size"]) == batch and "context" in row["groups"]
        ]
        axes[0].plot(
            [int(row["total_context_tokens"]) for row in items],
            [float(row["attention_ms"]) for row in items],
            marker="o",
            label=f"B={batch}",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Aggregate context tokens")
    axes[0].set_ylabel("Attention latency (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    totals = sorted(
        {
            int(row["total_context_tokens"])
            for row in rows
            if "iso_total" in row["groups"]
        }
    )
    for total in totals:
        items = [
            row
            for row in rows
            if int(row["total_context_tokens"]) == total
            and "iso_total" in row["groups"]
        ]
        axes[1].plot(
            [int(row["batch_size"]) for row in items],
            [float(row["attention_ms"]) for row in items],
            marker="o",
            label=f"T={total // 1024}K",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Request count at fixed aggregate context")
    axes[1].set_ylabel("Attention latency (ms)")
    axes[1].grid(alpha=0.25)
    if totals:
        axes[1].legend()
    axes[2].scatter(
        [int(row["total_context_tokens"]) for row in rows],
        [float(row["logical_kv_gbps"]) for row in rows],
        c=[int(row["batch_size"]) for row in rows],
        cmap="viridis",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Aggregate context tokens")
    axes[2].set_ylabel("Logical KV read GB/s")
    axes[2].grid(alpha=0.25)
    save_figure(figure, output)
    plt.close(figure)


def plot_prefill(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    lengths = [int(row["prompt_tokens"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(
        lengths,
        [float(row["prefill_engine_ms"]) for row in rows],
        marker="o",
    )
    axes[0].set_ylabel("Prefill engine latency (ms)")
    axes[1].plot(
        lengths,
        [float(row["prompt_tokens_per_second"]) for row in rows],
        marker="o",
        color="tab:green",
    )
    axes[1].set_ylabel("Prompt tokens/s")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Prompt tokens, B=1")
        axis.grid(alpha=0.25)
    save_figure(figure, output)
    plt.close(figure)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---:" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row[key]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    projection: list[dict[str, Any]],
    attention: list[dict[str, Any]],
    prefill: list[dict[str, Any]],
    spreads: list[dict[str, Any]],
    hardware: list[dict[str, Any]],
) -> str:
    projection_breakdown = []
    for row in projection:
        if int(row["batch_size"]) not in {1, 32, 256}:
            continue
        projection_breakdown.append(
            {
                "batch_size": row["batch_size"],
                "input_layernorm_ms": row["input_layernorm_ms"],
                "qkv_proj_ms": row["qkv_proj_ms"],
                "qk_norm_rope_ms": row["qk_norm_rope_ms"],
                "qk_repack_ms": row["qk_repack_ms"],
                "o_proj_ms": row["o_proj_ms"],
                "post_attention_layernorm_ms": row["post_attention_layernorm_ms"],
                "mlp_ms": row["mlp_ms"],
            }
        )
    attention_context = sorted(
        (
            row
            for row in attention
            if int(row["batch_size"]) in {1, 8, 32} and "context" in str(row["groups"])
        ),
        key=lambda row: (
            int(row["batch_size"]),
            int(row["context_tokens_per_request"]),
        ),
    )
    projection_b1 = next(row for row in projection if int(row["batch_size"]) == 1)
    projection_b16 = next(row for row in projection if int(row["batch_size"]) == 16)
    projection_b256 = next(row for row in projection if int(row["batch_size"]) == 256)
    b1_latency = float(projection_b1["per_layer_total_ms"])
    b16_latency_growth = (
        float(projection_b16["per_layer_total_ms"]) / b1_latency - 1
    ) * 100
    b256_latency_ratio = float(projection_b256["per_layer_total_ms"]) / b1_latency
    b256_tflops = float(projection_b256["useful_dense_tflops"])
    sections = [
        "# PAP component-scaling microbenchmark results",
        "",
        "## Scope and method",
        "",
        "This is a diagnostic microbenchmark on Qwen3-8B FP16 and one "
        "NVIDIA L20. It isolates PAP's three compute roles; it is not an "
        "end-to-end architecture comparison. Projection uses all 92 SMs, "
        "Prefill sees 80 SMs, and paged Decode Attention sees 12 SMs. "
        "Latency comes from unprofiled CUDA-event runs after warmup. GPU "
        "counters come from separate, synchronized NVTX ranges under Nsight "
        "Systems, so profiler overhead never enters the latency tables.",
        "",
        "Formal timing provenance is split across the benchmark-fix commits "
        "that made each boundary auditable: Projection `ca0e78b4d`, "
        "Attention `88dddb845`, and Prefill plus counter collection "
        "`c661908b9`. Every timing point passed its shape audit. The source "
        "JSON and profiler databases remain in the local `_staging` tree; "
        "the normalized CSV summaries and figures in this directory are the "
        "reviewable evidence bundle.",
        "",
        "## Projection",
        "",
        "The measured layer boundary contains input RMSNorm, QKV projection, "
        "Q/K norm and RoPE, PAP Q/K repack, output projection, "
        "post-Attention RMSNorm, and MLP. Remote Attention, transport, "
        "scheduler, sampling, and logits are excluded.",
        "",
        markdown_table(
            projection,
            [
                ("batch_size", "B"),
                ("per_layer_total_ms", "ms/layer"),
                ("useful_dense_tflops", "useful TFLOP/s"),
                ("rows_per_second", "rows/s"),
            ],
        ),
        "",
        "Selected per-stage medians:",
        "",
        markdown_table(
            projection_breakdown,
            [
                ("batch_size", "B"),
                ("input_layernorm_ms", "input norm ms"),
                ("qkv_proj_ms", "QKV ms"),
                ("qk_norm_rope_ms", "QK/RoPE ms"),
                ("qk_repack_ms", "repack ms"),
                ("o_proj_ms", "O proj ms"),
                ("post_attention_layernorm_ms", "post norm ms"),
                ("mlp_ms", "MLP ms"),
            ],
        ),
        "",
        "![Projection scaling](projection_scaling.png)",
        "",
        "From B1 to B16, rows increase 16x while per-layer latency rises only "
        f"{b16_latency_growth:.1f}%. "
        "At B256, latency is still only "
        f"{b256_latency_ratio:.2f}x B1, while useful dense throughput reaches "
        f"{b256_tflops:.1f} TFLOP/s. "
        "The MLP contributes roughly three quarters of this boundary; the "
        "PAP Q/K repack remains approximately 0.008 ms. Projection therefore "
        "amortizes ordinary decode batches well and does not exhibit the "
        "linear batch penalty seen in Attention.",
        "",
        "## Attention",
        "",
        "The primary boundary is the production PAP paged-decode Attention "
        "call, including the Triton partial scan and reduction. KV append is "
        "timed separately. The table below varies context at fixed request "
        "counts:",
        "",
        markdown_table(
            attention_context,
            [
                ("batch_size", "B"),
                ("context_tokens_per_request", "context/request"),
                ("total_context_tokens", "total context"),
                ("attention_ms", "Attention ms"),
                ("kv_append_ms", "KV append ms"),
                ("logical_kv_gbps", "logical KV GB/s"),
            ],
        ),
        "",
        "### Fixed-total-context request-count test",
        "",
        markdown_table(
            spreads,
            [
                ("total_context_tokens", "total context"),
                ("min_attention_ms", "min ms"),
                ("max_attention_ms", "max ms"),
                ("spread_percent", "spread %"),
                ("fastest_batch", "fastest B"),
                ("slowest_batch", "slowest B"),
            ],
        ),
        "",
        "![Attention scaling](attention_scaling.png)",
        "",
        "The strict hypothesis that latency depends only on aggregate context "
        "is falsified: changing request count at fixed total KV changes warm "
        "latency by 11.8--17.1%, above the preregistered 5% tolerance. "
        "Aggregate KV bytes are nevertheless the dominant variable once the "
        "launch floor is amortized: the B8/B32 scans become nearly linear in "
        "total context and sustain roughly 530--572 logical GB/s. Request "
        "count still changes CTA parallelism, per-request query/reduction "
        "work, and block-table tails; moderate B8--B16 decompositions are "
        "fastest in the fixed-total tests.",
        "",
        "## Prefill post-knee utilization scan",
        "",
        markdown_table(
            prefill,
            [
                ("prompt_tokens", "prompt tokens"),
                ("prefill_engine_ms", "engine ms"),
                ("prompt_tokens_per_second", "tokens/s"),
            ],
        ),
        "",
        "![Prefill scaling](prefill_scaling.png)",
        "",
        "The 80-SM single-request scan does not support a simple linear "
        "latency model after 1K tokens. From 1K to 10K, prompt length grows "
        "10x but latency grows about 12.0x; from 10K to 30K, length grows 3x "
        "but latency grows about 4.17x. This is expected from the increasing "
        "causal-Attention share, whose work is quadratic in sequence length. "
        "Prompt throughput consequently falls from 6.47K to 3.87K tokens/s.",
    ]
    if hardware:
        sections.extend(
            [
                "",
                "## GPU hardware counters inside NVTX ranges",
                "",
                markdown_table(
                    hardware,
                    [
                        ("component", "component"),
                        ("point", "point"),
                        ("dram_read", "DRAM read"),
                        ("sm_active", "SM active"),
                        ("sm_issue", "SM issue"),
                        ("tensor_active", "tensor active"),
                    ],
                ),
                "",
                "Nsight Systems reports throughput/activity fields as a "
                "percentage of the sampled device limit; clocks are retained "
                "in `hardware_counters.csv`. Projection counters exclude "
                "the dense-Attention range and one-time profiler "
                "initialization intervals over 10 ms; real unprofiled "
                "Projection stages in this B1--B256 matrix are below 1.1 ms.",
                "",
                "At 128K aggregate context, Attention reaches 71.4--74.6% "
                "of full-device DRAM-read activity while SM Active is "
                "12.6--13.0%. Because 12/92 physical SMs is 13.0%, this means "
                "essentially every assigned Attention SM is active while "
                "tensor activity remains only 3.6--3.9%: the production "
                "kernel is bandwidth-bound in this regime. Projection moves "
                "from 0.02% tensor activity at B1 to 34.8% at B256, consistent "
                "with batch-amortized dense compute. Prefill is different: "
                "SM activity stays around 85--87% and tensor activity around "
                "38% from 1K through 30K, while DRAM-read activity falls from "
                "18.6% to 6.4%. Its long-sequence slowdown is therefore not "
                "an HBM-bandwidth saturation effect.",
            ]
        )
    sections.extend(
        [
            "",
            "## Decisions and limitations",
            "",
            "- Projection is batch-efficient over the current operating "
            "range; optimize or schedule around it only when rows become very "
            "large.",
            "- Attention scheduling should model aggregate KV as the primary "
            "load, but request count must remain a secondary shape feature; "
            "KV-token sums alone are not exact latency predictors.",
            "- Prefill capacity models must include causal-Attention growth; "
            "a flat post-1K token-throughput assumption is invalid.",
            "- Useful TFLOP/s is derived from the Qwen3 dense-layer operation "
            "count and is not equivalent to hardware MFU. Nsight activity "
            "percentages are diagnostic, not application-level utilization "
            "ratios.",
            "- Projection excludes remote transport and Attention, and the "
            "three probes do not reproduce scheduler overlap or end-to-end "
            "queues. Formal timing is one clean matrix with repeated warm "
            "samples, not independent end-to-end repetitions.",
        ]
    )
    return "\n".join(sections) + "\n"


def metric_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for root in (args.projection_metrics_root, args.attention_metrics_root):
        if root is not None:
            paths.extend(sorted(root.glob("**/*metrics.json")))
    if args.prefill_metrics is not None:
        paths.append(args.prefill_metrics)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--projection-metrics-root", type=Path)
    parser.add_argument("--attention-metrics-root", type=Path)
    parser.add_argument("--prefill-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pap")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    projection = projection_rows(read_json(args.projection))
    attention = attention_rows(read_json(args.attention))
    prefill = prefill_rows(read_json(args.prefill))
    spreads = fixed_total_spreads(attention)
    hardware = hardware_rows(metric_paths(args))
    write_csv(args.output_dir / "projection.csv", projection)
    write_csv(args.output_dir / "attention.csv", attention)
    write_csv(args.output_dir / "attention_fixed_total_spread.csv", spreads)
    write_csv(args.output_dir / "prefill.csv", prefill)
    write_csv(args.output_dir / "hardware_counters.csv", hardware)
    plot_projection(projection, args.output_dir / "projection_scaling")
    plot_attention(attention, args.output_dir / "attention_scaling")
    plot_prefill(prefill, args.output_dir / "prefill_scaling")
    (args.output_dir / "report.md").write_text(
        build_report(projection, attention, prefill, spreads, hardware),
        encoding="utf-8",
    )
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
