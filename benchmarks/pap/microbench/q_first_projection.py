#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark fused QKV projection against sequential Q-first projection."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    batch_sizes = tuple(int(item) for item in value.split(","))
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return batch_sizes


def measure(
    operation: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    operations_per_replay: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        samples.append(
            float(start.elapsed_time(end)) / iterations / operations_per_replay
        )
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def benchmark_batch(
    batch_size: int,
    *,
    hidden_size: int,
    q_width: int,
    kv_width: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
    weight_sets: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    hidden = torch.randn(
        (batch_size, hidden_size), dtype=dtype, device=device
    )
    weights = torch.randn(
        (weight_sets, q_width + kv_width, hidden_size),
        dtype=dtype,
        device=device,
    )
    fused_output = torch.empty(
        (batch_size, q_width + kv_width), dtype=dtype, device=device
    )
    q_output = torch.empty((batch_size, q_width), dtype=dtype, device=device)
    kv_output = torch.empty((batch_size, kv_width), dtype=dtype, device=device)

    def fused(layer_index: int) -> None:
        torch.mm(hidden, weights[layer_index].t(), out=fused_output)

    def q_only(layer_index: int) -> None:
        torch.mm(hidden, weights[layer_index, :q_width].t(), out=q_output)

    def kv_only(layer_index: int) -> None:
        torch.mm(hidden, weights[layer_index, q_width:].t(), out=kv_output)

    def q_then_kv(layer_index: int) -> None:
        q_only(layer_index)
        kv_only(layer_index)

    def capture(
        operation: Callable[[int], None],
    ) -> torch.cuda.CUDAGraph:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for layer_index in range(weight_sets):
                operation(layer_index)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            for layer_index in range(weight_sets):
                operation(layer_index)
        torch.cuda.current_stream().wait_stream(capture_stream)
        return graph

    layer_operations = {
        "fused_qkv": fused,
        "q_only": q_only,
        "kv_only": kv_only,
        "q_then_kv": q_then_kv,
    }
    operations = {
        name: capture(operation).replay
        for name, operation in layer_operations.items()
    }
    timings = {
        name: measure(
            operation,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            operations_per_replay=weight_sets,
        )
        for name, operation in operations.items()
    }

    fused(weight_sets - 1)
    q_then_kv(weight_sets - 1)
    torch.cuda.synchronize()
    split_output = torch.cat((q_output, kv_output), dim=-1)
    max_abs_error = float((fused_output - split_output).abs().max().item())
    fused_ms = float(timings["fused_qkv"]["median_ms"])
    split_ms = float(timings["q_then_kv"]["median_ms"])
    q_ms = float(timings["q_only"]["median_ms"])
    return {
        "batch_size": batch_size,
        "weight_sets": weight_sets,
        "timings": timings,
        "split_over_fused": split_ms / fused_ms,
        "split_extra_ms": split_ms - fused_ms,
        "q_ready_fraction_of_split": q_ms / split_ms,
        "max_abs_error": max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=parse_batch_sizes("1,2,4,6,8,12,16,24,32,60"),
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--q-width", type=int, default=4096)
    parser.add_argument("--kv-width", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--weight-sets", type=int, default=36)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    results = [
        benchmark_batch(
            batch_size,
            hidden_size=args.hidden_size,
            q_width=args.q_width,
            kv_width=args.kv_width,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            weight_sets=args.weight_sets,
        )
        for batch_size in args.batch_sizes
    ]
    report = {
        "schema_version": 1,
        "kind": "pap_q_first_projection",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "shape": {
            "hidden_size": args.hidden_size,
            "q_width": args.q_width,
            "kv_width": args.kv_width,
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
