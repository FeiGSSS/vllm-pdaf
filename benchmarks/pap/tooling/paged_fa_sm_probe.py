# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone paged-FlashAttention SM and memory-bandwidth probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ProbeShape:
    """P17 C4 Attention shape captured from the live decode path."""

    seq_lens: tuple[int, ...] = (17344, 17334, 17324)
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 16
    table_blocks: int = 1092
    num_layers: int = 36

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    @property
    def physical_blocks(self) -> int:
        return self.batch_size * self.table_blocks

    @property
    def logical_kv_bytes(self) -> int:
        return (
            sum(self.seq_lens)
            * self.num_kv_heads
            * self.head_dim
            * 2
            * torch.float16.itemsize
        )


@dataclass
class ProbeInputs:
    """Tensors passed to one paged FlashAttention invocation."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    block_table: torch.Tensor
    cross_layer_kv_cache: torch.Tensor


@cache
def get_flash_attn_varlen_func() -> Any:
    """Import FlashAttention only after the CLI has selected a CUDA device."""

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    return flash_attn_varlen_func


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("timing", "profile", "trace"),
        default="timing",
    )
    parser.add_argument("--num-splits", type=int, choices=(0, 1), default=0)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--warmup-calls", type=_positive_int, default=108)
    parser.add_argument("--samples", type=_positive_int, default=9)
    parser.add_argument("--calls-per-sample", type=_positive_int, default=180)
    parser.add_argument("--profile-seconds", type=float, default=2.0)
    parser.add_argument("--expected-sms", type=_positive_int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    return parser.parse_args()


def build_inputs(shape: ProbeShape, *, layer_index: int) -> ProbeInputs:
    """Allocate the same cross-layer paged KV layout used by P17."""

    if not 0 <= layer_index < shape.num_layers:
        raise ValueError(f"invalid layer index: {layer_index}")
    device = torch.device("cuda", 0)
    torch.manual_seed(0)
    query = torch.randn(
        (shape.batch_size, shape.num_q_heads, shape.head_dim),
        device=device,
        dtype=torch.float16,
    )
    cross_layer_kv_cache = torch.zeros(
        (
            shape.physical_blocks,
            shape.num_layers,
            2,
            shape.block_size,
            shape.num_kv_heads,
            shape.head_dim,
        ),
        device=device,
        dtype=torch.float16,
    )
    layer_kv_cache = cross_layer_kv_cache[:, layer_index]
    key_cache, value_cache = layer_kv_cache.unbind(1)
    output = torch.empty_like(query)
    cu_seqlens_q = torch.arange(
        shape.batch_size + 1,
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(shape.seq_lens, device=device, dtype=torch.int32)
    block_table = torch.arange(
        shape.physical_blocks,
        device=device,
        dtype=torch.int32,
    ).view(shape.batch_size, shape.table_blocks)
    return ProbeInputs(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        cross_layer_kv_cache=cross_layer_kv_cache,
    )


def run_attention(
    inputs: ProbeInputs,
    shape: ProbeShape,
    *,
    num_splits: int,
) -> torch.Tensor:
    """Run the exact FA2 paged-attention signature used by PAP."""

    flash_attn_varlen_func = get_flash_attn_varlen_func()
    return flash_attn_varlen_func(
        q=inputs.query,
        k=inputs.key_cache,
        v=inputs.value_cache,
        out=inputs.output,
        cu_seqlens_q=inputs.cu_seqlens_q,
        seqused_k=inputs.seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max(shape.seq_lens),
        softmax_scale=1.0 / math.sqrt(shape.head_dim),
        causal=True,
        block_table=inputs.block_table,
        softcap=0.0,
        return_softmax_lse=False,
        num_splits=num_splits,
        fa_version=2,
    )


def warm_up(
    inputs: ProbeInputs,
    shape: ProbeShape,
    *,
    num_splits: int,
    calls: int,
) -> None:
    for _ in range(calls):
        run_attention(inputs, shape, num_splits=num_splits)
    torch.cuda.synchronize()


def timing_samples(
    inputs: ProbeInputs,
    shape: ProbeShape,
    *,
    num_splits: int,
    samples: int,
    calls_per_sample: int,
) -> list[float]:
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            run_attention(inputs, shape, num_splits=num_splits)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / calls_per_sample)
    return values


def profile_for_duration(
    inputs: ProbeInputs,
    shape: ProbeShape,
    *,
    num_splits: int,
    seconds: float,
) -> int:
    """Keep the target kernel active long enough for GPU metric sampling."""

    if seconds <= 0:
        raise ValueError("profile duration must be positive")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    calls = 0
    torch.cuda.nvtx.range_push("pap_paged_fa_probe")
    try:
        start.record()
        while True:
            for _ in range(100):
                run_attention(inputs, shape, num_splits=num_splits)
                calls += 1
            end.record()
            end.synchronize()
            if float(start.elapsed_time(end)) >= seconds * 1000.0:
                break
    finally:
        torch.cuda.nvtx.range_pop()
    return calls


def trace_one_call(
    inputs: ProbeInputs,
    shape: ProbeShape,
    *,
    num_splits: int,
    output: Path,
) -> None:
    """Capture one target call without modifying the PAP runtime."""

    from torch.profiler import ProfilerActivity, profile

    output.parent.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as profiler:
        torch.cuda.nvtx.range_push("pap_paged_fa_probe")
        try:
            run_attention(inputs, shape, num_splits=num_splits)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
    profiler.export_chrome_trace(str(output))


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def base_result(
    shape: ProbeShape,
    inputs: ProbeInputs,
    *,
    mode: str,
    num_splits: int,
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "schema_version": 1,
        "kind": "pap_paged_fa_sm_probe",
        "mode": mode,
        "device_name": props.name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "visible_sms": props.multi_processor_count,
        "cuda_mps_sm_partition": os.environ.get("CUDA_MPS_SM_PARTITION"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fa_version": 2,
        "num_splits": num_splits,
        "shape": asdict(shape),
        "logical_min_kv_bytes": shape.logical_kv_bytes,
        "cross_layer_kv_cache": tensor_metadata(inputs.cross_layer_kv_cache),
        "layer_key_cache": tensor_metadata(inputs.key_cache),
        "layer_value_cache": tensor_metadata(inputs.value_cache),
    }


def write_result(result: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(output)


def main() -> None:
    args = parse_args()
    shape = ProbeShape()
    inputs = build_inputs(shape, layer_index=args.layer_index)
    visible_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if args.expected_sms is not None and visible_sms != args.expected_sms:
        raise RuntimeError(
            f"visible SM mismatch: expected {args.expected_sms}, got {visible_sms}"
        )
    warm_up(
        inputs,
        shape,
        num_splits=args.num_splits,
        calls=args.warmup_calls,
    )
    result = base_result(
        shape,
        inputs,
        mode=args.mode,
        num_splits=args.num_splits,
    )
    result["warmup_calls"] = args.warmup_calls
    if args.mode == "timing":
        samples = timing_samples(
            inputs,
            shape,
            num_splits=args.num_splits,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        mean_ms = statistics.mean(samples)
        result.update(
            {
                "calls_per_sample": args.calls_per_sample,
                "samples_ms_per_call": samples,
                "mean_ms_per_call": mean_ms,
                "median_ms_per_call": statistics.median(samples),
                "min_ms_per_call": min(samples),
                "max_ms_per_call": max(samples),
                "logical_min_kv_gbps": (
                    shape.logical_kv_bytes / (mean_ms / 1000.0) / 1e9
                ),
            }
        )
    elif args.mode == "profile":
        result.update(
            {
                "profile_seconds": args.profile_seconds,
                "profile_calls": profile_for_duration(
                    inputs,
                    shape,
                    num_splits=args.num_splits,
                    seconds=args.profile_seconds,
                ),
            }
        )
    else:
        if args.trace_output is None:
            raise ValueError("--trace-output is required in trace mode")
        trace_one_call(
            inputs,
            shape,
            num_splits=args.num_splits,
            output=args.trace_output,
        )
        result.update(
            {
                "trace_calls": 1,
                "trace_output": str(args.trace_output),
            }
        )
    write_result(result, args.output)


if __name__ == "__main__":
    main()
