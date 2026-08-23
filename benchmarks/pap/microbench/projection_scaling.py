#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure Qwen3 Projection-side decode scaling with real vLLM kernels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
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


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique")
    return values


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def iter_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_path in sorted(path.glob("samples_pid*_rank*.jsonl")):
        for line in sample_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line))
    return samples


def sample_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize empty samples")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def model_dense_flops_per_row(config_path: Path) -> int:
    """Return useful dense linear FLOPs for one row and one layer."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden_size // num_heads)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = 2 * hidden_size * (q_size + 2 * kv_size)
    output = 2 * q_size * hidden_size
    gate_up = 2 * hidden_size * (2 * intermediate_size)
    down = 2 * intermediate_size * hidden_size
    return qkv + output + gate_up + down


def measure_qk_repack(batch_size: int, config_path: Path) -> dict[str, Any]:
    """Measure PAP's exact two contiguous Q/K slice copies."""
    import torch

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden_size // num_heads)
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv = torch.empty(
        (batch_size, q_size + 2 * kv_size),
        dtype=torch.float16,
        device="cuda",
    )
    q = torch.empty((batch_size, q_size), dtype=torch.float16, device="cuda")
    k = torch.empty((batch_size, kv_size), dtype=torch.float16, device="cuda")

    def repack() -> None:
        qkv[:, :q_size].copy_(q)
        qkv[:, q_size : q_size + kv_size].copy_(k)

    for _ in range(100):
        repack()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(9):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(500):
            repack()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / 500)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "bytes": batch_size * (q_size + kv_size) * torch.float16.itemsize,
    }


def summarize_batch(
    samples: list[dict[str, Any]],
    *,
    batch_size: int,
    configured_context: int,
    config_path: Path,
) -> dict[str, Any]:
    expected = [
        sample
        for sample in samples
        if int(sample.get("batch_size", 0)) == batch_size
        and int(sample.get("configured_batch_size", 0)) == batch_size
        and int(sample.get("configured_prompt_len", 0)) == configured_context
    ]
    unexpected = [sample for sample in samples if sample not in expected]
    contaminated = [
        sample
        for sample in unexpected
        if int(sample.get("configured_batch_size", 0)) != batch_size
        or int(sample.get("configured_prompt_len", 0)) != configured_context
    ]
    if contaminated:
        raise RuntimeError("Projection sample directory mixes configurations")
    excluded_shapes = sorted(
        {
            (
                int(sample.get("batch_size", 0)),
                int(sample.get("configured_batch_size", 0)),
                int(sample.get("configured_prompt_len", 0)),
            )
            for sample in unexpected
        }
    )

    by_stage: dict[str, list[float]] = defaultdict(list)
    by_layer_stage: dict[tuple[int, str], list[float]] = defaultdict(list)
    for sample in expected:
        stage = str(sample["stage"])
        if stage not in STAGES:
            continue
        elapsed_ms = float(sample["elapsed_ms"])
        layer_index = int(sample["layer_index"])
        by_stage[stage].append(elapsed_ms)
        by_layer_stage[(layer_index, stage)].append(elapsed_ms)
    missing = [stage for stage in STAGES if not by_stage[stage]]
    if missing:
        raise RuntimeError(f"Projection profile omitted stages: {missing}")
    counts = {stage: len(by_stage[stage]) for stage in STAGES}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Projection stage counts differ: {counts}")
    if next(iter(counts.values())) % 36 != 0:
        raise RuntimeError(f"Projection stage count is not layer-aligned: {counts}")
    decode_steps = next(iter(counts.values())) // 36
    if decode_steps < 3:
        raise RuntimeError(f"Projection has too few full-batch steps: {counts}")

    layer_totals = []
    for layer_index in range(36):
        stage_values = {stage: by_layer_stage[(layer_index, stage)] for stage in STAGES}
        if any(not values for values in stage_values.values()):
            raise RuntimeError(f"Projection layer {layer_index} is incomplete")
        layer_totals.append(
            sum(statistics.median(stage_values[stage]) for stage in STAGES)
        )
    stage_summary = {stage: sample_stats(by_stage[stage]) for stage in STAGES}
    repack = measure_qk_repack(batch_size, config_path)
    model_ms = statistics.fmean(layer_totals)
    total_ms = model_ms + float(repack["median_ms"])
    flops_per_row = model_dense_flops_per_row(config_path)
    dense_flops = flops_per_row * batch_size
    return {
        "batch_size": batch_size,
        "configured_context_tokens": configured_context,
        "shape_audit": {
            "valid": True,
            "decode_steps": decode_steps,
            "layers": 36,
            "stage_counts": counts,
            "excluded_sample_count": len(unexpected),
            "excluded_shapes": excluded_shapes,
        },
        "stages": stage_summary,
        "per_layer_model_ms": model_ms,
        "qk_repack": repack,
        "per_layer_total_ms": total_ms,
        "rows_per_second": batch_size / (total_ms / 1_000.0),
        "dense_flops_per_row_per_layer": flops_per_row,
        "useful_dense_tflops": dense_flops / (total_ms / 1_000.0) / 1e12,
        "layer_total_distribution_ms": sample_stats(layer_totals),
    }


def run_matrix(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    runner = args.dense_runner.resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pap_projection_scaling",
        "status": "running",
        "model": str(args.model.resolve()),
        "model_config": str(args.model_config.resolve()),
        "dtype": args.dtype,
        "context_tokens": args.context_tokens,
        "batch_sizes": list(args.batch_sizes),
        "warmup_output_tokens": args.warmup_output_tokens,
        "measure_output_tokens": args.measure_output_tokens,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "results": [],
    }
    result_path = output_root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    base_env = dict(os.environ)
    base_env["VLLM_QWEN3_LAYER_PROFILE_ASYNC"] = "1"
    # Drain after one 36-layer decode step (seven instrumented stages/layer).
    # Depending on engine shutdown ordering, relying on atexit alone can lose
    # all pending CUDA-event samples.
    base_env["VLLM_QWEN3_LAYER_PROFILE_ASYNC_FLUSH_THRESHOLD"] = str(36 * 7)
    base_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    for batch_size in args.batch_sizes:
        run_id = f"qwen3_8b_projection_b{batch_size}"
        run_root = output_root / run_id
        sample_dir = run_root / "samples"
        sample_dir.mkdir(parents=True)
        command = [
            sys.executable,
            str(runner),
            "--single",
            "--model",
            str(args.model.resolve()),
            "--model-name",
            "Qwen3-8B",
            "--tp",
            "1",
            "--context-len",
            str(args.context_tokens),
            "--batch-size",
            str(batch_size),
            "--measure-output-len",
            str(args.measure_output_tokens),
            "--warmup-output-len",
            str(args.warmup_output_tokens),
            "--sample-dir",
            str(sample_dir),
            "--run-id",
            run_id,
            "--dtype",
            args.dtype,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-num-batched-tokens",
            str(args.max_num_batched_tokens),
            "--enforce-eager",
        ]
        log_path = run_root / "runner.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path.cwd(),
                env=base_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Projection batch {batch_size} failed; inspect {log_path}"
            )
        samples = iter_samples(sample_dir)
        summary = summarize_batch(
            samples,
            batch_size=batch_size,
            configured_context=args.context_tokens,
            config_path=args.model_config,
        )
        result["results"].append(summary)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"B={batch_size}: {summary['per_layer_total_ms']:.4f} ms/layer, "
            f"{summary['useful_dense_tflops']:.2f} useful TFLOP/s",
            flush=True,
        )
    result["status"] = "completed"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(result_path)


def dry_run(args: argparse.Namespace) -> None:
    print("batch\tcontext\tprompt_tokens\tmeasure_decode_steps")
    for batch_size in args.batch_sizes:
        print(
            f"{batch_size}\t{args.context_tokens}\t"
            f"{batch_size * args.context_tokens}\t"
            f"{max(0, args.measure_output_tokens - 1)}"
        )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--dense-runner",
        type=Path,
        default=root
        / "benchmarks"
        / "disagg_benchmarks"
        / "qwen3_dense_decode_layer_profile.py",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--warmup-output-tokens", type=int, default=2)
    parser.add_argument("--measure-output-tokens", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32_768)
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.context_tokens <= 0 or args.measure_output_tokens < 2:
        raise ValueError("context must be positive and measured output at least two")
    if args.dry_run:
        dry_run(args)
        return
    if args.output_root is None:
        raise ValueError("--output-root is required unless --dry-run is used")
    run_matrix(args)


if __name__ == "__main__":
    main()
