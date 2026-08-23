#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure PAP Prefill compute saturation with controlled batch shapes."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import regex as re

BEGIN_PREFIX = "PAP_PREFILL_SAMPLE_BEGIN "
END_PREFIX = "PAP_PREFILL_SAMPLE_END "
ITERATION_RE = re.compile(
    r"Iteration\((?P<index>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)


@dataclass(frozen=True)
class Shape:
    """One controlled Prefill batch shape."""

    batch_size: int
    prompt_tokens: int
    groups: tuple[str, ...]

    @property
    def shape_id(self) -> str:
        return f"b{self.batch_size}_l{self.prompt_tokens}"

    @property
    def total_prompt_tokens(self) -> int:
        return self.batch_size * self.prompt_tokens


def build_shapes(selected_groups: set[str]) -> list[Shape]:
    """Build the three pre-registered experiment groups without duplicates."""
    definitions = {
        "serial": [
            (1, prompt_tokens)
            for prompt_tokens in (
                64,
                128,
                256,
                512,
                768,
                1_000,
                2_000,
                4_000,
                8_000,
                10_000,
                20_000,
                30_000,
            )
        ],
        "serial_refine": [
            (1, prompt_tokens)
            for prompt_tokens in (
                144,
                160,
                176,
                192,
                208,
                224,
                240,
                320,
                384,
            )
        ],
        "utilization": [
            (1, prompt_tokens) for prompt_tokens in (256, 1_000, 10_000, 30_000)
        ],
        "calibration": [(1, 256)],
        "step1": [(1, 10_000), (2, 10_000), (3, 10_000)],
        "step2": [
            *((batch, 1_000) for batch in (1, 2, 4, 8, 16, 24, 32)),
            *((batch, 2_000) for batch in (1, 2, 4, 8, 12, 16)),
            *((batch, 5_000) for batch in (1, 2, 3, 4, 6)),
            *((batch, 10_000) for batch in (1, 2, 3)),
        ],
        "step3": [(30, 1_000), (15, 2_000), (6, 5_000), (3, 10_000)],
    }
    ordered_keys: list[tuple[int, int]] = []
    groups_by_key: dict[tuple[int, int], list[str]] = defaultdict(list)
    for group in (
        "serial",
        "serial_refine",
        "utilization",
        "calibration",
        "step1",
        "step2",
        "step3",
    ):
        if group not in selected_groups:
            continue
        for key in definitions[group]:
            if key not in groups_by_key:
                ordered_keys.append(key)
            groups_by_key[key].append(group)
    return [
        Shape(
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            groups=tuple(groups_by_key[(batch_size, prompt_tokens)]),
        )
        for batch_size, prompt_tokens in ordered_keys
    ]


def parse_groups(value: str) -> set[str]:
    groups = {item.strip() for item in value.split(",") if item.strip()}
    unknown = groups - {
        "serial",
        "serial_refine",
        "utilization",
        "calibration",
        "step1",
        "step2",
        "step3",
    }
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown experiment groups: {', '.join(sorted(unknown))}"
        )
    if not groups:
        raise argparse.ArgumentTypeError("at least one experiment group is required")
    return groups


def load_vocab_size(model: Path) -> int:
    config = json.loads((model / "config.json").read_text())
    vocab_size = int(config["vocab_size"])
    if vocab_size <= 2_000:
        raise ValueError(f"unexpectedly small vocabulary: {vocab_size}")
    return vocab_size


def make_prompts(
    *,
    shape: Shape,
    vocab_size: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    rng = random.Random(seed)
    return [
        {
            "prompt_token_ids": [
                rng.randrange(1_000, vocab_size - 256)
                for _ in range(shape.prompt_tokens)
            ]
        }
        for _ in range(shape.batch_size)
    ]


def print_marker(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def run_benchmark(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    selected_groups = parse_groups(args.groups)
    shapes = build_shapes(selected_groups)
    if not shapes:
        raise ValueError("no shapes selected")
    if max(shape.total_prompt_tokens for shape in shapes) > args.max_batched_tokens:
        raise ValueError("a shape exceeds --max-batched-tokens")

    model = Path(args.model).resolve()
    vocab_size = load_vocab_size(model)
    result_path = Path(args.output).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "benchmark": "pap_prefill_saturation",
        "method": {
            "timing_primary": "vllm_engine_context_iteration_elapsed_ms",
            "timing_secondary": "llm_generate_wall_ms",
            "generated_tokens_per_request": 1,
            "prefix_caching": False,
            "chunked_prefill": True,
            "execution_mode": "eager",
            "async_scheduling": True,
            "batch_gate": "sleep_level_0_enqueue_wake_scheduling",
        },
        "config": {
            "model": str(model),
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_batched_tokens,
            "warmup_repeats": args.warmup_repeats,
            "measure_repeats": args.measure_repeats,
            "seed": args.seed,
            "groups": sorted(selected_groups),
        },
        "shapes": [asdict(shape) | {"shape_id": shape.shape_id} for shape in shapes],
        "samples": [],
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    nvtx_gate_raw = os.environ.get("VLLM_QWEN3_COMPONENT_NVTX_GATE_FILE", "")
    nvtx_gate = Path(nvtx_gate_raw) if nvtx_gate_raw else None
    if nvtx_gate is not None:
        nvtx_gate.unlink(missing_ok=True)

    llm = LLM(
        model=str(model),
        dtype=args.dtype,
        enforce_eager=True,
        generation_config="vllm",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enable_logging_iteration_details=True,
        disable_log_stats=True,
        async_scheduling=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
    )

    sample_ordinal = 0
    try:
        for shape in shapes:
            phases = (
                ("warmup", args.warmup_repeats),
                ("measure", args.measure_repeats),
            )
            for phase, repeats in phases:
                for repetition in range(repeats):
                    sample_id = f"{shape.shape_id}_{phase}_r{repetition:02d}"
                    prompt_seed = args.seed + sample_ordinal * 1_000_003
                    prompts = make_prompts(
                        shape=shape,
                        vocab_size=vocab_size,
                        seed=prompt_seed,
                    )
                    marker = {
                        "sample_id": sample_id,
                        "shape_id": shape.shape_id,
                        "phase": phase,
                        "repetition": repetition,
                        "batch_size": shape.batch_size,
                        "prompt_tokens": shape.prompt_tokens,
                        "total_prompt_tokens": shape.total_prompt_tokens,
                    }
                    print_marker(BEGIN_PREFIX, marker)
                    sleep_start = time.perf_counter()
                    llm.sleep(level=0, mode="keep")
                    sleep_ms = (time.perf_counter() - sleep_start) * 1_000.0
                    enqueue_start = time.perf_counter()
                    try:
                        request_ids = llm.enqueue(
                            prompts,
                            sampling_params=sampling_params,
                            use_tqdm=False,
                        )
                    except BaseException:
                        llm.wake_up(tags=["scheduling"])
                        raise
                    enqueue_ms = (time.perf_counter() - enqueue_start) * 1_000.0
                    if nvtx_gate is not None and phase == "measure":
                        nvtx_gate.parent.mkdir(parents=True, exist_ok=True)
                        nvtx_gate.touch()
                    try:
                        start = time.perf_counter()
                        llm.wake_up(tags=["scheduling"])
                        outputs = llm.wait_for_completion(use_tqdm=False)
                        wall_ms = (time.perf_counter() - start) * 1_000.0
                    finally:
                        if nvtx_gate is not None:
                            nvtx_gate.unlink(missing_ok=True)
                    if len(request_ids) != shape.batch_size:
                        raise RuntimeError(
                            f"{sample_id}: enqueued {len(request_ids)} requests, "
                            f"expected {shape.batch_size}"
                        )
                    if len(outputs) != shape.batch_size:
                        raise RuntimeError(
                            f"{sample_id}: returned {len(outputs)} outputs, "
                            f"expected {shape.batch_size}"
                        )
                    observed_prompt_tokens = sum(
                        len(output.prompt_token_ids) for output in outputs
                    )
                    if observed_prompt_tokens != shape.total_prompt_tokens:
                        raise RuntimeError(
                            f"{sample_id}: processed {observed_prompt_tokens} "
                            f"prompt tokens, expected {shape.total_prompt_tokens}"
                        )
                    sample = marker | {
                        "prompt_seed": prompt_seed,
                        "scheduler_pause_ms": sleep_ms,
                        "enqueue_ms": enqueue_ms,
                        "wall_ms": wall_ms,
                        "enqueued_requests": len(request_ids),
                        "returned_outputs": len(outputs),
                        "observed_prompt_tokens": observed_prompt_tokens,
                    }
                    result["samples"].append(sample)
                    result_path.write_text(json.dumps(result, indent=2) + "\n")
                    print_marker(
                        END_PREFIX,
                        {
                            "sample_id": sample_id,
                            "wall_ms": round(wall_ms, 3),
                        },
                    )
                    sample_ordinal += 1
    finally:
        result["status"] = "completed"
        result_path.write_text(json.dumps(result, indent=2) + "\n")


def parse_iteration_blocks(log_path: Path) -> dict[str, list[dict[str, Any]]]:
    blocks: dict[str, list[dict[str, Any]]] = {}
    active_sample: str | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if BEGIN_PREFIX in line:
            payload = json.loads(line.split(BEGIN_PREFIX, 1)[1])
            active_sample = str(payload["sample_id"])
            if active_sample in blocks:
                raise ValueError(f"duplicate sample marker: {active_sample}")
            blocks[active_sample] = []
            continue
        if END_PREFIX in line:
            payload = json.loads(line.split(END_PREFIX, 1)[1])
            if active_sample != str(payload["sample_id"]):
                raise ValueError("sample marker nesting is inconsistent")
            active_sample = None
            continue
        if active_sample is None:
            continue
        match = ITERATION_RE.search(line)
        if match:
            record: dict[str, Any] = {
                key: int(value)
                for key, value in match.groupdict().items()
                if key != "elapsed_ms"
            }
            record["elapsed_ms"] = float(match.group("elapsed_ms"))
            blocks[active_sample].append(record)
    if active_sample is not None:
        raise ValueError(f"unterminated sample marker: {active_sample}")
    return blocks


def annotate_samples(
    raw_result: dict[str, Any],
    blocks: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for raw_sample in raw_result["samples"]:
        sample = dict(raw_sample)
        sample_id = sample["sample_id"]
        iterations = blocks.get(sample_id)
        if iterations is None:
            raise ValueError(f"engine log omitted sample: {sample_id}")
        context_iterations = [
            record for record in iterations if record["context_tokens"] > 0
        ]
        observed_context_tokens = sum(
            record["context_tokens"] for record in context_iterations
        )
        prefill_engine_ms = sum(record["elapsed_ms"] for record in context_iterations)
        exact_single_iteration = (
            len(context_iterations) == 1
            and context_iterations[0]["context_requests"] == sample["batch_size"]
            and observed_context_tokens == sample["total_prompt_tokens"]
        )
        sample.update(
            {
                "engine_iterations": iterations,
                "context_iteration_count": len(context_iterations),
                "observed_context_tokens": observed_context_tokens,
                "prefill_engine_ms": prefill_engine_ms,
                "exact_single_context_iteration": exact_single_iteration,
            }
        )
        samples.append(sample)
    return samples


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    return float(statistics.median(values))


def summarize_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample["phase"] == "measure":
            by_shape[sample["shape_id"]].append(sample)

    summaries: list[dict[str, Any]] = []
    for shape_id, shape_samples in by_shape.items():
        first = shape_samples[0]
        engine_ms = median(
            [float(sample["prefill_engine_ms"]) for sample in shape_samples]
        )
        total_tokens = int(first["total_prompt_tokens"])
        summaries.append(
            {
                "shape_id": shape_id,
                "batch_size": int(first["batch_size"]),
                "prompt_tokens": int(first["prompt_tokens"]),
                "total_prompt_tokens": total_tokens,
                "sample_count": len(shape_samples),
                "prefill_engine_ms_median": engine_ms,
                "prefill_engine_ms_min": min(
                    float(sample["prefill_engine_ms"]) for sample in shape_samples
                ),
                "prefill_engine_ms_max": max(
                    float(sample["prefill_engine_ms"]) for sample in shape_samples
                ),
                "wall_ms_median": median(
                    [float(sample["wall_ms"]) for sample in shape_samples]
                ),
                "prompt_tokens_per_second": total_tokens / (engine_ms / 1_000.0),
                "all_exact_single_context_iteration": all(
                    bool(sample["exact_single_context_iteration"])
                    for sample in shape_samples
                ),
            }
        )
    return summaries


def saturation_knees(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_length[int(summary["prompt_tokens"])].append(summary)
    knees = []
    for prompt_tokens, rows in sorted(by_length.items()):
        rows.sort(key=lambda row: int(row["batch_size"]))
        max_throughput = max(float(row["prompt_tokens_per_second"]) for row in rows)
        threshold = max_throughput * 0.95
        knee = next(
            row for row in rows if float(row["prompt_tokens_per_second"]) >= threshold
        )
        knees.append(
            {
                "prompt_tokens": prompt_tokens,
                "max_prompt_tokens_per_second": max_throughput,
                "knee_fraction": 0.95,
                "knee_batch_size": int(knee["batch_size"]),
                "knee_total_prompt_tokens": int(knee["total_prompt_tokens"]),
            }
        )
    return knees


def fixed_10k_scaling(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = {
        int(row["batch_size"]): row
        for row in summaries
        if int(row["prompt_tokens"]) == 10_000 and int(row["batch_size"]) in {1, 2, 3}
    }
    if 1 not in rows:
        return []
    base_ms = float(rows[1]["prefill_engine_ms_median"])
    return [
        {
            "batch_size": batch_size,
            "time_ratio_vs_b1": (
                float(rows[batch_size]["prefill_engine_ms_median"]) / base_ms
            ),
            "throughput_scaling_vs_b1": (
                batch_size
                * base_ms
                / float(rows[batch_size]["prefill_engine_ms_median"])
            ),
        }
        for batch_size in sorted(rows)
    ]


def iso_total_30k(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in summaries if int(row["total_prompt_tokens"]) == 30_000],
        key=lambda row: int(row["prompt_tokens"]),
    )


def serial_length_scan(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        [dict(row) for row in summaries if int(row["batch_size"]) == 1],
        key=lambda row: int(row["prompt_tokens"]),
    )
    if not rows:
        return {"rows": [], "peak": None, "knee": None}
    peak = max(rows, key=lambda row: float(row["prompt_tokens_per_second"]))
    threshold = float(peak["prompt_tokens_per_second"]) * 0.95
    knee = next(
        row for row in rows if float(row["prompt_tokens_per_second"]) >= threshold
    )
    return {
        "rows": rows,
        "peak": {
            "prompt_tokens": int(peak["prompt_tokens"]),
            "prompt_tokens_per_second": float(peak["prompt_tokens_per_second"]),
        },
        "knee": {
            "definition": "first_length_at_or_above_95_percent_of_peak",
            "fraction": 0.95,
            "prompt_tokens": int(knee["prompt_tokens"]),
            "prompt_tokens_per_second": float(knee["prompt_tokens_per_second"]),
        },
    }


def markdown_report(result: dict[str, Any]) -> str:
    summaries = sorted(
        result["summaries"],
        key=lambda row: (row["prompt_tokens"], row["batch_size"]),
    )
    lines = [
        "# PAP Prefill saturation microbenchmark",
        "",
        "This is a standalone Prefill-compute benchmark on the PAP Prefill "
        "MPS partition. It does not exercise Decode, Attention offload, or "
        "NIXL data transfer.",
        "",
        "Primary latency is vLLM EngineCore context-iteration elapsed time. "
        "Every measured point must form one audited context iteration.",
        "",
        "| Prompt/request | Requests | Total prompt | Prefill ms | Prompt tok/s "
        "| Wall ms | Shape audit |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        audit = "PASS" if row["all_exact_single_context_iteration"] else "FAIL"
        lines.append(
            f"| {row['prompt_tokens']:,} | {row['batch_size']} | "
            f"{row['total_prompt_tokens']:,} | "
            f"{row['prefill_engine_ms_median']:.2f} | "
            f"{row['prompt_tokens_per_second']:,.0f} | "
            f"{row['wall_ms_median']:.2f} | {audit} |"
        )
    lines.extend(
        [
            "",
            "## Fixed 10K/request scaling",
            "",
            "| Requests | Time / B1 | Throughput / B1 |",
            "|---:|---:|---:|",
        ]
    )
    for row in result["fixed_10k_scaling"]:
        lines.append(
            f"| {row['batch_size']} | {row['time_ratio_vs_b1']:.3f}x | "
            f"{row['throughput_scaling_vs_b1']:.3f}x |"
        )
    serial_scan = result["serial_length_scan"]
    if serial_scan["rows"]:
        lines.extend(
            [
                "",
                "## Strictly serial B=1 length scan",
                "",
                "| Prompt tokens | Prefill ms | Prompt tok/s |",
                "|---:|---:|---:|",
            ]
        )
        for row in serial_scan["rows"]:
            lines.append(
                f"| {row['prompt_tokens']:,} | "
                f"{row['prefill_engine_ms_median']:.2f} | "
                f"{row['prompt_tokens_per_second']:,.0f} |"
            )
        peak = serial_scan["peak"]
        knee = serial_scan["knee"]
        lines.extend(
            [
                "",
                f"Peak throughput is {peak['prompt_tokens_per_second']:,.0f} "
                f"prompt tok/s at {peak['prompt_tokens']:,} tokens. "
                f"The first length reaching 95% of that peak is "
                f"{knee['prompt_tokens']:,} tokens.",
            ]
        )
    lines.extend(
        [
            "",
            "## Smallest batch within 95% of per-length peak",
            "",
            "| Prompt/request | Knee requests | Knee total prompt | "
            "Peak prompt tok/s |",
            "|---:|---:|---:|---:|",
        ]
    )
    for knee in result["saturation_knees"]:
        lines.append(
            f"| {knee['prompt_tokens']:,} | {knee['knee_batch_size']} | "
            f"{knee['knee_total_prompt_tokens']:,} | "
            f"{knee['max_prompt_tokens_per_second']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Approximately 30K total prompt tokens",
            "",
            "| Prompt/request | Requests | Prefill ms | Prompt tok/s |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["iso_total_30k"]:
        lines.append(
            f"| {row['prompt_tokens']:,} | {row['batch_size']} | "
            f"{row['prefill_engine_ms_median']:.2f} | "
            f"{row['prompt_tokens_per_second']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "For fixed 10K requests, `T(B) ≈ B×T(1)` means one request already "
            "saturates Prefill throughput. `T(B) < B×T(1)` means batching still "
            "improves compute efficiency. The ~30K iso-token points separate "
            "total-token effects from per-sequence shape effects.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    log_path = Path(args.engine_log).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    raw_result = json.loads(input_path.read_text())
    samples = annotate_samples(raw_result, parse_iteration_blocks(log_path))
    result = dict(raw_result)
    result["status"] = "analyzed"
    result["samples"] = samples
    result["summaries"] = summarize_samples(samples)
    result["fixed_10k_scaling"] = fixed_10k_scaling(result["summaries"])
    result["saturation_knees"] = saturation_knees(result["summaries"])
    result["iso_total_30k"] = iso_total_30k(result["summaries"])
    result["serial_length_scan"] = serial_length_scan(result["summaries"])
    measured = [sample for sample in samples if sample["phase"] == "measure"]
    invalid = [
        sample["sample_id"]
        for sample in measured
        if not sample["exact_single_context_iteration"]
    ]
    result["shape_audit"] = {
        "required": bool(args.require_single_context_iteration),
        "valid": not invalid,
        "invalid_samples": invalid,
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(markdown_report(result))
    if invalid and args.require_single_context_iteration:
        raise SystemExit(
            "measured shapes did not form one context iteration: " + ", ".join(invalid)
        )


def merge_results(args: argparse.Namespace) -> None:
    inputs = [Path(value).resolve() for value in args.inputs]
    results = [json.loads(path.read_text()) for path in inputs]
    if len(results) < 2:
        raise ValueError("merge requires at least two analyzed results")
    config_keys = (
        "model",
        "dtype",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "warmup_repeats",
        "measure_repeats",
    )
    reference = results[0]["config"]
    for result in results:
        if result.get("status") != "analyzed":
            raise ValueError("merge inputs must have status=analyzed")
        if not result.get("shape_audit", {}).get("valid"):
            raise ValueError("merge inputs must pass the shape audit")
        for key in config_keys:
            if result["config"].get(key) != reference.get(key):
                raise ValueError(f"merge input config mismatch: {key}")

    samples: list[dict[str, Any]] = []
    shapes_by_id: dict[str, dict[str, Any]] = {}
    sample_ids: set[str] = set()
    groups: set[str] = set()
    for result in results:
        groups.update(result["config"].get("groups", ()))
        for shape in result["shapes"]:
            shapes_by_id.setdefault(shape["shape_id"], shape)
        for sample in result["samples"]:
            sample_id = str(sample["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"duplicate merged sample: {sample_id}")
            sample_ids.add(sample_id)
            samples.append(sample)

    merged = dict(results[0])
    merged["status"] = "analyzed"
    merged["config"] = dict(reference) | {"groups": sorted(groups)}
    merged["sources"] = [str(path) for path in inputs]
    merged["shapes"] = sorted(
        shapes_by_id.values(),
        key=lambda shape: (shape["prompt_tokens"], shape["batch_size"]),
    )
    merged["samples"] = samples
    merged["summaries"] = summarize_samples(samples)
    merged["fixed_10k_scaling"] = fixed_10k_scaling(merged["summaries"])
    merged["saturation_knees"] = saturation_knees(merged["summaries"])
    merged["iso_total_30k"] = iso_total_30k(merged["summaries"])
    merged["serial_length_scan"] = serial_length_scan(merged["summaries"])
    merged["shape_audit"] = {
        "required": True,
        "valid": True,
        "invalid_samples": [],
    }
    Path(args.output).resolve().write_text(json.dumps(merged, indent=2) + "\n")
    Path(args.report).resolve().write_text(markdown_report(merged))


def dry_run(args: argparse.Namespace) -> None:
    shapes = build_shapes(parse_groups(args.groups))
    print("shape_id\tgroups\tbatch\tprompt_tokens\ttotal_prompt_tokens")
    for shape in shapes:
        print(
            f"{shape.shape_id}\t{','.join(shape.groups)}\t{shape.batch_size}\t"
            f"{shape.prompt_tokens}\t{shape.total_prompt_tokens}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--groups", default="step1,step2,step3")

    dry_parser = subparsers.add_parser("dry-run", parents=[common])
    dry_parser.set_defaults(func=dry_run)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--dtype", default="float16")
    run_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    run_parser.add_argument("--max-model-len", type=int, default=32_768)
    run_parser.add_argument("--max-num-seqs", type=int, default=256)
    run_parser.add_argument("--max-batched-tokens", type=int, default=32_768)
    run_parser.add_argument("--warmup-repeats", type=int, default=1)
    run_parser.add_argument("--measure-repeats", type=int, default=3)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.set_defaults(func=run_benchmark)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input", required=True)
    analyze_parser.add_argument("--engine-log", required=True)
    analyze_parser.add_argument("--output", required=True)
    analyze_parser.add_argument("--report", required=True)
    analyze_parser.add_argument(
        "--require-single-context-iteration",
        action="store_true",
    )
    analyze_parser.set_defaults(func=analyze)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.add_argument("--report", required=True)
    merge_parser.set_defaults(func=merge_results)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
