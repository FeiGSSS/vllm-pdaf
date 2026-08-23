#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure production PAP paged-decode Attention scaling."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.cuda import nvtx

from vllm.platforms import current_platform


@dataclass(frozen=True)
class ModelShape:
    """Qwen Attention dimensions used by the production kernel."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16


@dataclass(frozen=True)
class ProbeShape:
    """One equal-length paged-decode batch shape."""

    batch_size: int
    context_tokens_per_request: int
    groups: tuple[str, ...]

    @property
    def total_context_tokens(self) -> int:
        return self.batch_size * self.context_tokens_per_request

    @property
    def shape_id(self) -> str:
        return f"b{self.batch_size}_l{self.context_tokens_per_request}"


@dataclass
class ProbeInputs:
    """Device tensors for one production paged-decode invocation."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    slot_mapping: torch.Tensor
    metadata: Any
    workspace: Any
    k_scale: torch.Tensor
    v_scale: torch.Tensor


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _dtype(value: str) -> torch.dtype:
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return mapping[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("dtype must be float16 or bfloat16") from exc


def load_model_shape(path: Path) -> ModelShape:
    """Load the Attention dimensions from a Hugging Face config."""
    config = json.loads(path.read_text(encoding="utf-8"))
    num_heads = int(config["num_attention_heads"])
    hidden_size = int(config["hidden_size"])
    return ModelShape(
        num_heads=num_heads,
        num_kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config.get("head_dim") or hidden_size // num_heads),
    )


def build_shapes(groups: set[str]) -> list[ProbeShape]:
    """Build the preregistered matrix and merge duplicate shapes."""
    definitions: dict[str, list[tuple[int, int]]] = {
        "context": [
            (batch_size, context_tokens)
            for batch_size in (1, 8, 32)
            for context_tokens in (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)
        ],
        "iso_total": [
            *(
                (batch_size, 32_768 // batch_size)
                for batch_size in (1, 2, 4, 8, 16, 32)
            ),
            *(
                (batch_size, 65_536 // batch_size)
                for batch_size in (2, 4, 8, 16, 32, 64)
            ),
            *(
                (batch_size, 131_072 // batch_size)
                for batch_size in (4, 8, 16, 32, 64, 128)
            ),
        ],
        "calibration": [(1, 1_024), (4, 32_768), (16, 8_192), (64, 2_048)],
    }
    unknown = groups - definitions.keys()
    if unknown:
        raise ValueError(f"unknown shape groups: {','.join(sorted(unknown))}")
    group_by_shape: dict[tuple[int, int], list[str]] = {}
    order: list[tuple[int, int]] = []
    for group in ("context", "iso_total", "calibration"):
        if group not in groups:
            continue
        for shape in definitions[group]:
            if shape not in group_by_shape:
                group_by_shape[shape] = []
                order.append(shape)
            group_by_shape[shape].append(group)
    return [
        ProbeShape(
            batch_size=batch_size,
            context_tokens_per_request=context_tokens,
            groups=tuple(group_by_shape[(batch_size, context_tokens)]),
        )
        for batch_size, context_tokens in order
    ]


def parse_groups(value: str) -> set[str]:
    groups = {item.strip() for item in value.split(",") if item.strip()}
    if not groups:
        raise argparse.ArgumentTypeError("at least one group is required")
    return groups


def parse_explicit_shape(value: str) -> ProbeShape:
    """Parse `B,L` for a single profile point."""
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("shape must be B,L")
    batch_size, context_tokens = map(int, fields)
    if batch_size <= 0 or context_tokens <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return ProbeShape(batch_size, context_tokens, ("explicit",))


def build_inputs(
    model: ModelShape,
    shape: ProbeShape,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> ProbeInputs:
    """Allocate the exact tensors accepted by the production PAP kernel."""
    from vllm.pap.attention.kernels import build_paged_decode_workspace
    from vllm.pap.kv.metadata import PAPPagedFlashMetadata

    blocks_per_request = math.ceil(shape.context_tokens_per_request / model.block_size)
    num_blocks = shape.batch_size * blocks_per_request
    kv_cache = torch.empty(
        (
            num_blocks,
            2,
            model.block_size,
            model.num_kv_heads,
            model.head_dim,
        ),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    key_cache, value_cache = kv_cache.unbind(1)
    query = torch.empty(
        (shape.batch_size, model.num_heads, model.head_dim),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    key = torch.empty(
        (shape.batch_size, model.num_kv_heads, model.head_dim),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    value = torch.empty_like(key).normal_(mean=0.0, std=0.02)
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=device,
    ).view(shape.batch_size, blocks_per_request)
    seq_lens = torch.full(
        (shape.batch_size,),
        shape.context_tokens_per_request,
        dtype=torch.int32,
        device=device,
    )
    slot_mapping = (
        torch.arange(shape.batch_size, dtype=torch.int64, device=device)
        * blocks_per_request
        * model.block_size
        + shape.context_tokens_per_request
        - 1
    )
    metadata = PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=torch.arange(
            shape.batch_size + 1,
            dtype=torch.int32,
            device=device,
        ),
        max_seq_len=shape.context_tokens_per_request,
    )
    return ProbeInputs(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        metadata=metadata,
        workspace=build_paged_decode_workspace(query),
        k_scale=torch.ones((), dtype=torch.float32, device=device),
        v_scale=torch.ones((), dtype=torch.float32, device=device),
    )


def append_kv(inputs: ProbeInputs) -> None:
    """Run the production KV append operation."""
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        inputs.key,
        inputs.value,
        inputs.key_cache,
        inputs.value_cache,
        inputs.slot_mapping,
        "auto",
        inputs.k_scale,
        inputs.v_scale,
    )


def run_attention(inputs: ProbeInputs, model: ModelShape) -> torch.Tensor:
    """Run the production low-SM-aware Triton paged-decode path."""
    from vllm.pap.attention.kernels import run_paged_decode_attention

    return run_paged_decode_attention(
        query=inputs.query,
        key_cache=inputs.key_cache,
        value_cache=inputs.value_cache,
        metadata=inputs.metadata,
        workspace=inputs.workspace,
        scale=1.0 / math.sqrt(model.head_dim),
        block_size=model.block_size,
    )


def combined(inputs: ProbeInputs, model: ModelShape) -> torch.Tensor:
    append_kv(inputs)
    return run_attention(inputs, model)


def time_cuda(
    function: Callable[[], Any],
    *,
    warmup_calls: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    """Measure one stream-ordered operation with CUDA events."""
    for _ in range(warmup_calls):
        function()
    torch.accelerator.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / calls_per_sample)
    return {
        "samples_ms": values,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def logical_kv_bytes(
    model: ModelShape,
    shape: ProbeShape,
    dtype: torch.dtype,
) -> int:
    element_bytes = torch.finfo(dtype).bits // 8
    return (
        shape.total_context_tokens
        * model.num_kv_heads
        * model.head_dim
        * 2
        * element_bytes
    )


def device_metadata(device: torch.device) -> dict[str, Any]:
    device_index = int(device.index or 0)
    capability = current_platform.get_device_capability(device_index)
    return {
        "index": device.index,
        "name": current_platform.get_device_name(device_index),
        "multiprocessor_count": current_platform.num_compute_units(device_index),
        "total_memory_bytes": current_platform.get_device_total_memory(device_index),
        "capability": list(capability) if capability is not None else None,
    }


def base_result(
    model: ModelShape,
    shape: ProbeShape,
    inputs: ProbeInputs,
    *,
    dtype: torch.dtype,
    mode: str,
) -> dict[str, Any]:
    bytes_read = logical_kv_bytes(model, shape, dtype)
    workspace = inputs.workspace
    return {
        "schema_version": 1,
        "kind": "pap_attention_scaling",
        "mode": mode,
        "shape": asdict(shape)
        | {
            "shape_id": shape.shape_id,
            "total_context_tokens": shape.total_context_tokens,
        },
        "model_shape": asdict(model),
        "dtype": str(dtype).removeprefix("torch."),
        "logical_kv_bytes": bytes_read,
        "workspace": {
            "partial_shape": list(workspace.partial.shape),
            "num_splits": workspace.kernel_config.num_splits,
            "block_h": workspace.kernel_config.block_h,
            "num_warps": workspace.kernel_config.num_warps,
            "num_stages": workspace.kernel_config.num_stages,
        },
        "block_table_shape": list(inputs.metadata.block_table.shape),
        "device": device_metadata(inputs.query.device),
        "cuda_mps_sm_partition": os.environ.get("CUDA_MPS_SM_PARTITION"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def run_timing(args: argparse.Namespace) -> None:
    model = load_model_shape(args.model_config)
    device = torch.device("cuda", 0)
    visible_sms = current_platform.num_compute_units(int(device.index or 0))
    if args.expected_sms is not None and visible_sms != args.expected_sms:
        raise RuntimeError(
            f"visible SM mismatch: expected {args.expected_sms}, got {visible_sms}"
        )
    shapes = build_shapes(args.groups)
    results = []
    for shape in shapes:
        inputs = build_inputs(model, shape, dtype=args.dtype, device=device)
        attention = time_cuda(
            lambda inputs=inputs: run_attention(inputs, model),
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        append = time_cuda(
            lambda inputs=inputs: append_kv(inputs),
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        total = time_cuda(
            lambda inputs=inputs: combined(inputs, model),
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        entry = base_result(
            model,
            shape,
            inputs,
            dtype=args.dtype,
            mode="timing",
        )
        entry.update(
            {
                "warmup_calls": args.warmup_calls,
                "samples": args.samples,
                "calls_per_sample": args.calls_per_sample,
                "paged_attention": attention,
                "kv_append": append,
                "append_then_attention": total,
                "logical_kv_gbps": (
                    entry["logical_kv_bytes"] / (attention["median_ms"] / 1_000.0) / 1e9
                ),
            }
        )
        results.append(entry)
        del inputs
        torch.accelerator.empty_cache()
    output = {
        "schema_version": 1,
        "kind": "pap_attention_scaling_matrix",
        "status": "completed",
        "groups": sorted(args.groups),
        "results": results,
    }
    write_json(output, args.output)


def run_profile(args: argparse.Namespace) -> None:
    model = load_model_shape(args.model_config)
    shape = args.shape
    device = torch.device("cuda", 0)
    inputs = build_inputs(model, shape, dtype=args.dtype, device=device)
    visible_sms = current_platform.num_compute_units(int(device.index or 0))
    if args.expected_sms is not None and visible_sms != args.expected_sms:
        raise RuntimeError(
            f"visible SM mismatch: expected {args.expected_sms}, got {visible_sms}"
        )
    for _ in range(args.warmup_calls):
        run_attention(inputs, model)
    torch.accelerator.synchronize()
    calls = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    nvtx.range_push("pap_attention_scaling_probe")
    try:
        start.record()
        while True:
            for _ in range(args.profile_chunk_calls):
                run_attention(inputs, model)
                calls += 1
            end.record()
            end.synchronize()
            if float(start.elapsed_time(end)) >= args.profile_seconds * 1_000.0:
                break
    finally:
        nvtx.range_pop()
    result = base_result(
        model,
        shape,
        inputs,
        dtype=args.dtype,
        mode="profile",
    )
    result.update(
        {
            "status": "completed",
            "profile_seconds_requested": args.profile_seconds,
            "profile_calls": calls,
            "profile_elapsed_ms": float(start.elapsed_time(end)),
        }
    )
    write_json(result, args.output)


def write_json(value: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(output)


def dry_run(args: argparse.Namespace) -> None:
    print("shape_id\tbatch\tcontext/request\ttotal_context\tgroups")
    for shape in build_shapes(args.groups):
        print(
            f"{shape.shape_id}\t{shape.batch_size}\t"
            f"{shape.context_tokens_per_request}\t"
            f"{shape.total_context_tokens}\t{','.join(shape.groups)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument(
        "--groups", type=parse_groups, default={"context", "iso_total"}
    )
    dry_parser.set_defaults(func=dry_run)

    timing_parser = subparsers.add_parser("timing")
    timing_parser.add_argument("--model-config", type=Path, required=True)
    timing_parser.add_argument(
        "--groups", type=parse_groups, default={"context", "iso_total"}
    )
    timing_parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    timing_parser.add_argument("--expected-sms", type=_positive_int)
    timing_parser.add_argument("--warmup-calls", type=_positive_int, default=20)
    timing_parser.add_argument("--samples", type=_positive_int, default=7)
    timing_parser.add_argument("--calls-per-sample", type=_positive_int, default=50)
    timing_parser.add_argument("--output", type=Path)
    timing_parser.set_defaults(func=run_timing)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--model-config", type=Path, required=True)
    profile_parser.add_argument("--shape", type=parse_explicit_shape, required=True)
    profile_parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    profile_parser.add_argument("--expected-sms", type=_positive_int)
    profile_parser.add_argument("--warmup-calls", type=_positive_int, default=20)
    profile_parser.add_argument("--profile-seconds", type=float, default=1.0)
    profile_parser.add_argument(
        "--profile-chunk-calls", type=_positive_int, default=100
    )
    profile_parser.add_argument("--output", type=Path)
    profile_parser.set_defaults(func=run_profile)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "profile_seconds", 1.0) <= 0:
        raise ValueError("profile duration must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
