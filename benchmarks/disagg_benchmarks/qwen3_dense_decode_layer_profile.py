# SPDX-License-Identifier: Apache-2.0
"""Profile Qwen3 dense decode per-layer compute stages.

This benchmark intentionally runs the normal vLLM dense decode path, not PAP.
It enables the Qwen3 model-level CUDA-event profiler via environment variables
and aggregates the emitted JSONL samples into CSV/Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)

PROJECTION_CORE_STAGES = ("qkv_proj", "qk_norm_rope", "o_proj", "mlp")
PROJECTION_WITH_LN_STAGES = (
    "input_layernorm",
    "qkv_proj",
    "qk_norm_rope",
    "o_proj",
    "post_attention_layernorm",
    "mlp",
)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


def clean_env_for_dense_profile(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for key in list(env):
        if key.startswith("PAP_") or key in PROXY_ENV_KEYS:
            env.pop(key, None)
    return env


def make_prompt_tokens(prompt_len: int, request_index: int) -> list[int]:
    # Stay far from special Qwen token ids while avoiding identical prompts that
    # could accidentally benefit from prefix-cache behavior.
    base = 1000 + request_index * 17
    return [base + ((offset * 13 + request_index) % 30000) for offset in range(prompt_len)]


def run_single(args: argparse.Namespace) -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    for key in list(os.environ):
        if key.startswith("PAP_"):
            os.environ.pop(key, None)

    args.sample_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_QWEN3_LAYER_PROFILE"] = "1"
    os.environ["VLLM_QWEN3_LAYER_PROFILE_DIR"] = str(args.sample_dir)
    os.environ["VLLM_QWEN3_LAYER_PROFILE_RUN_ID"] = args.run_id
    os.environ["VLLM_QWEN3_LAYER_PROFILE_MODEL"] = args.model_name
    os.environ["VLLM_QWEN3_LAYER_PROFILE_PROMPT_LEN"] = str(args.context_len)
    os.environ["VLLM_QWEN3_LAYER_PROFILE_CONFIGURED_BATCH_SIZE"] = str(args.batch_size)

    from vllm import LLM, SamplingParams

    max_model_len = args.context_len + args.measure_output_len + args.warmup_output_len + 8
    max_num_batched_tokens = args.max_num_batched_tokens
    if max_num_batched_tokens <= 0:
        max_num_batched_tokens = max(args.batch_size, min(args.batch_size * args.context_len, 8192))

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=False,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )

    prompts = [
        {"prompt_token_ids": make_prompt_tokens(args.context_len, request_index)}
        for request_index in range(args.batch_size)
    ]

    if args.warmup_output_len > 0:
        warmup_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.warmup_output_len,
            ignore_eos=True,
        )
        llm.generate(prompts, warmup_params, use_tqdm=False)
        for sample_path in args.sample_dir.glob("samples_pid*_rank*.jsonl"):
            sample_path.unlink()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.measure_output_len,
        ignore_eos=True,
    )
    llm.generate(prompts, sampling_params, use_tqdm=False)


def iter_samples(output_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("**/samples_pid*_rank*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    return samples


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def write_raw_csv(samples: list[dict[str, Any]], output_dir: Path) -> None:
    if not samples:
        return
    keys = sorted({key for sample in samples for key in sample})
    with (output_dir / "raw_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(samples)


def write_stage_summary(samples: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for sample in samples:
        configured_batch_size = int(
            sample.get("configured_batch_size") or sample.get("batch_size", 0)
        )
        configured_prompt_len = int(
            sample.get("configured_prompt_len") or sample.get("context_len", 0)
        )
        key = (
            sample.get("run_id", ""),
            sample.get("model", ""),
            sample.get("tp_size", 1),
            configured_batch_size,
            configured_prompt_len,
            sample.get("layer_index", -1),
            sample.get("stage", ""),
        )
        grouped[key].append(float(sample["elapsed_ms"]))
        metadata[key] = {
            "run_id": key[0],
            "model": key[1],
            "tp_size": key[2],
            "batch_size": key[3],
            "context_len": key[4],
            "layer_index": key[5],
            "stage": key[6],
        }

    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        row = dict(metadata[key])
        row.update(
            {
                "count": len(values),
                "mean_ms": statistics.fmean(values),
                "p50_ms": percentile(values, 0.50),
                "p90_ms": percentile(values, 0.90),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        )
        rows.append(row)

    with (output_dir / "summary_by_stage.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_id",
            "model",
            "tp_size",
            "batch_size",
            "context_len",
            "layer_index",
            "stage",
            "count",
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "min_ms",
            "max_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_projection_attention_summary(stage_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    by_layer: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    base_meta: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in stage_rows:
        key = (
            row["run_id"],
            row["model"],
            row["tp_size"],
            row["batch_size"],
            row["context_len"],
            row["layer_index"],
        )
        by_layer[key][str(row["stage"])] = float(row["mean_ms"])
        base_meta[key] = {
            "run_id": key[0],
            "model": key[1],
            "tp_size": key[2],
            "batch_size": key[3],
            "context_len": key[4],
            "layer_index": key[5],
        }

    rows: list[dict[str, Any]] = []
    for key, stages in sorted(by_layer.items()):
        attention_ms = stages.get("attention", 0.0)
        projection_core_ms = sum(stages.get(stage, 0.0) for stage in PROJECTION_CORE_STAGES)
        projection_with_ln_ms = sum(stages.get(stage, 0.0) for stage in PROJECTION_WITH_LN_STAGES)
        row = dict(base_meta[key])
        row.update(
            {
                "attention_ms": attention_ms,
                "projection_core_ms": projection_core_ms,
                "projection_with_ln_ms": projection_with_ln_ms,
                "measured_total_with_ln_ms": attention_ms + projection_with_ln_ms,
                "qkv_proj_ms": stages.get("qkv_proj", 0.0),
                "qk_norm_rope_ms": stages.get("qk_norm_rope", 0.0),
                "o_proj_ms": stages.get("o_proj", 0.0),
                "mlp_ms": stages.get("mlp", 0.0),
            }
        )
        rows.append(row)

    with (output_dir / "summary_projection_attention.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "run_id",
            "model",
            "tp_size",
            "batch_size",
            "context_len",
            "layer_index",
            "attention_ms",
            "projection_core_ms",
            "projection_with_ln_ms",
            "measured_total_with_ln_ms",
            "qkv_proj_ms",
            "qk_norm_rope_ms",
            "o_proj_ms",
            "mlp_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_markdown_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["tp_size"], row["batch_size"], row["context_len"])
        grouped[key].append(row)

    lines = [
        "# Qwen3 Dense Decode Layer Profile",
        "",
        "Times are CUDA-event GPU times from the original dense vLLM decode path.",
        "Projection core = qkv_proj + qk_norm_rope + o_proj + mlp.",
        "Projection with LN additionally includes input_layernorm and post_attention_layernorm.",
        "",
        "| model | TP | batch | context | attention mean ms/layer | projection core mean ms/layer | projection+LN mean ms/layer | qkv ms | o_proj ms | mlp ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in sorted(grouped.items()):
        model, tp_size, batch_size, context_len = key
        lines.append(
            "| {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                model,
                tp_size,
                batch_size,
                context_len,
                statistics.fmean(float(v["attention_ms"]) for v in values),
                statistics.fmean(float(v["projection_core_ms"]) for v in values),
                statistics.fmean(float(v["projection_with_ln_ms"]) for v in values),
                statistics.fmean(float(v["qkv_proj_ms"]) for v in values),
                statistics.fmean(float(v["o_proj_ms"]) for v in values),
                statistics.fmean(float(v["mlp_ms"]) for v in values),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(output_dir: Path) -> None:
    samples = iter_samples(output_dir)
    write_raw_csv(samples, output_dir)
    stage_rows = write_stage_summary(samples, output_dir)
    projection_rows = write_projection_attention_summary(stage_rows, output_dir)
    write_markdown_summary(projection_rows, output_dir)


def run_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts = parse_int_list(args.contexts)
    batches = parse_int_list(args.batches)
    base_env = clean_env_for_dense_profile(os.environ)

    failures: list[tuple[int, int, int]] = []
    for context_len in contexts:
        for batch_size in batches:
            run_id = f"{args.model_name}_tp{args.tp}_ctx{context_len}_bs{batch_size}"
            combo_dir = output_dir / run_id
            sample_dir = combo_dir / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--single",
                "--model",
                args.model,
                "--model-name",
                args.model_name,
                "--tp",
                str(args.tp),
                "--context-len",
                str(context_len),
                "--batch-size",
                str(batch_size),
                "--measure-output-len",
                str(args.measure_output_len),
                "--warmup-output-len",
                str(args.warmup_output_len),
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
            ]
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            if args.disable_custom_all_reduce:
                cmd.append("--disable-custom-all-reduce")
            if args.trust_remote_code:
                cmd.append("--trust-remote-code")
            print(f"[profile] running {run_id}", flush=True)
            start = time.time()
            result = subprocess.run(cmd, env=base_env, cwd=Path.cwd())
            elapsed = time.time() - start
            print(
                f"[profile] finished {run_id} rc={result.returncode} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if result.returncode != 0:
                failures.append((context_len, batch_size, result.returncode))
            aggregate(output_dir)

    if failures:
        print("[profile] failed combinations:", flush=True)
        for context_len, batch_size, returncode in failures:
            print(
                f"  context={context_len} batch={batch_size} rc={returncode}",
                flush=True,
            )
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default="qwen3")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--contexts", default="128,512,1024,2048")
    parser.add_argument("--batches", default="1,8,16,32,64")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--measure-output-len", type=int, default=4)
    parser.add_argument("--warmup-output-len", type=int, default=2)
    parser.add_argument("--output-dir", default="benchmarks/disagg_benchmarks/results/qwen3_layer_profile")
    parser.add_argument("--sample-dir", type=Path, default=Path("/tmp/qwen3-layer-profile"))
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--single", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single:
        run_single(args)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
