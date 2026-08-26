#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure production PAP paged-decode Attention scaling."""

from __future__ import annotations

import argparse
import hashlib
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

MAX_TOTAL_CONTEXT_TOKENS = 262_144


@dataclass(frozen=True)
class ModelShape:
    """Qwen Attention dimensions used by the production kernel."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16


@dataclass(frozen=True)
class ProbeShape:
    """One paged-decode batch with an explicit context distribution."""

    context_lengths: tuple[int, ...]
    distribution: str
    groups: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return len(self.context_lengths)

    @property
    def total_context_tokens(self) -> int:
        return sum(self.context_lengths)

    @property
    def context_tokens_per_request(self) -> int | None:
        first = self.context_lengths[0]
        return first if all(value == first for value in self.context_lengths) else None

    @property
    def shape_id(self) -> str:
        digest = hashlib.sha1(
            ",".join(map(str, self.context_lengths)).encode()
        ).hexdigest()[:8]
        return (
            f"b{self.batch_size}_t{self.total_context_tokens}_"
            f"{self.distribution}_{digest}"
        )


@dataclass(frozen=True)
class KernelCandidate:
    """One Attention implementation and launch configuration."""

    implementation: str
    num_splits: int
    block_h: int
    num_warps: int
    num_stages: int
    block_n: int

    @property
    def config_id(self) -> str:
        if self.implementation == "auto":
            return "production_auto"
        if self.implementation == "vllm":
            return f"vllm_s{self.num_splits}"
        return (
            f"pap_s{self.num_splits}_h{self.block_h}_w{self.num_warps}_"
            f"g{self.num_stages}_n{self.block_n}"
        )


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


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
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


def _weighted_lengths(
    total_tokens: int,
    weights: tuple[float, ...],
    *,
    max_context_tokens: int = 32_768,
    min_context_tokens: int = 128,
) -> tuple[int, ...]:
    """Deterministically apportion an exact total under per-request bounds."""
    batch_size = len(weights)
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("context distribution weights must be positive")
    if not (
        batch_size * min_context_tokens
        <= total_tokens
        <= batch_size * max_context_tokens
    ):
        raise ValueError("context total is outside the per-request bounds")
    lengths = [min_context_tokens] * batch_size
    remaining = total_tokens - sum(lengths)
    while remaining:
        active = [
            index for index, length in enumerate(lengths) if length < max_context_tokens
        ]
        if not active:
            raise ValueError("context distribution exceeded its capacity")
        weight_sum = sum(weights[index] for index in active)
        raw = {index: remaining * weights[index] / weight_sum for index in active}
        grants = {
            index: min(
                max_context_tokens - lengths[index],
                math.floor(raw[index]),
            )
            for index in active
        }
        granted = sum(grants.values())
        if granted == 0:
            order = sorted(
                active,
                key=lambda index: (raw[index] % 1, weights[index]),
                reverse=True,
            )
            for index in order[:remaining]:
                grants[index] = 1
            granted = sum(grants.values())
        for index, grant in grants.items():
            lengths[index] += grant
        remaining -= granted
    return tuple(lengths)


def distribution_lengths(
    batch_size: int,
    total_tokens: int,
    distribution: str,
) -> tuple[int, ...]:
    """Create equal, bimodal, single-heavy, or Zipf context vectors."""
    if distribution == "equal":
        weights = (1.0,) * batch_size
    elif distribution == "bimodal":
        weights = tuple(
            0.5 if index < batch_size // 2 else 1.5 for index in range(batch_size)
        )
    elif distribution == "one_heavy":
        weights = (float(batch_size),) + (1.0,) * (batch_size - 1)
    elif distribution in ("zipf", "zipf_0_8", "zipf_1_2"):
        exponent = {
            "zipf": 1.2,
            "zipf_0_8": 0.8,
            "zipf_1_2": 1.2,
        }[distribution]
        weights = tuple(1.0 / ((index + 1) ** exponent) for index in range(batch_size))
    else:
        raise ValueError(f"unknown context distribution: {distribution}")
    return _weighted_lengths(total_tokens, weights)


def build_shapes(groups: set[str]) -> list[ProbeShape]:
    """Build the preregistered workload surface and merge duplicate vectors."""
    definitions: dict[str, list[tuple[tuple[int, ...], str]]] = {
        "context": [
            ((context_tokens,) * batch_size, "equal")
            for batch_size in (1, 2, 4, 8, 16, 32, 64)
            for context_tokens in (512, 2_048, 8_192, 16_384, 32_768)
            if batch_size * context_tokens <= MAX_TOTAL_CONTEXT_TOKENS
        ],
        "iso_total": [
            (
                distribution_lengths(batch_size, total_tokens, "equal"),
                "equal",
            )
            for total_tokens in (32_768, 65_536, 131_072, 262_144)
            for batch_size in (1, 2, 4, 8, 16, 32, 64)
            if 128 * batch_size <= total_tokens <= 32_768 * batch_size
        ],
        "distribution": [
            (
                distribution_lengths(
                    batch_size,
                    batch_size * mean_context,
                    distribution,
                ),
                distribution,
            )
            for batch_size in (4, 8, 16, 32)
            for mean_context in (2_048, 8_192, 16_384)
            for distribution in ("equal", "bimodal", "one_heavy", "zipf")
            if batch_size * mean_context <= MAX_TOTAL_CONTEXT_TOKENS
        ],
        "expanded": [
            (
                distribution_lengths(
                    batch_size,
                    batch_size * mean_context,
                    distribution,
                ),
                distribution,
            )
            for batch_size in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
            for mean_context in (
                512,
                1_024,
                2_048,
                4_096,
                8_192,
                12_288,
                16_384,
                24_576,
                32_768,
            )
            for distribution in (
                "equal",
                "bimodal",
                "one_heavy",
                "zipf_0_8",
                "zipf_1_2",
            )
            if batch_size * mean_context <= MAX_TOTAL_CONTEXT_TOKENS
        ],
        "calibration": [
            ((1_024,), "equal"),
            ((32_768,) * 4, "equal"),
            ((8_192,) * 16, "equal"),
            ((2_048,) * 64, "equal"),
        ],
    }
    unknown = groups - definitions.keys()
    if unknown:
        raise ValueError(f"unknown shape groups: {','.join(sorted(unknown))}")
    group_by_shape: dict[tuple[int, ...], list[str]] = {}
    distribution_by_shape: dict[tuple[int, ...], str] = {}
    order: list[tuple[int, ...]] = []
    for group in (
        "context",
        "iso_total",
        "distribution",
        "expanded",
        "calibration",
    ):
        if group not in groups:
            continue
        for context_lengths, distribution in definitions[group]:
            if context_lengths not in group_by_shape:
                group_by_shape[context_lengths] = []
                distribution_by_shape[context_lengths] = distribution
                order.append(context_lengths)
            if group not in group_by_shape[context_lengths]:
                group_by_shape[context_lengths].append(group)
    return [
        ProbeShape(
            context_lengths=context_lengths,
            distribution=distribution_by_shape[context_lengths],
            groups=tuple(group_by_shape[context_lengths]),
        )
        for context_lengths in order
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
    return ProbeShape(
        (context_tokens,) * batch_size,
        "equal",
        ("explicit",),
    )


def parse_context_lengths(value: str) -> ProbeShape:
    """Parse an explicit comma-separated per-request context vector."""
    lengths = tuple(int(field) for field in value.split(",") if field)
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths must be positive")
    return ProbeShape(lengths, "explicit", ("explicit",))


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

    blocks_per_request = tuple(
        math.ceil(length / model.block_size) for length in shape.context_lengths
    )
    num_blocks = sum(blocks_per_request)
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
    max_blocks_per_request = max(blocks_per_request)
    block_table_host = torch.zeros(
        (shape.batch_size, max_blocks_per_request),
        dtype=torch.int32,
    )
    slot_mapping_host = torch.empty(shape.batch_size, dtype=torch.int64)
    next_block = 0
    for index, (length, block_count) in enumerate(
        zip(shape.context_lengths, blocks_per_request)
    ):
        block_ids = torch.arange(
            next_block,
            next_block + block_count,
            dtype=torch.int32,
        )
        block_table_host[index, :block_count] = block_ids
        last_block = int(block_ids[-1])
        slot_mapping_host[index] = (
            last_block * model.block_size + (length - 1) % model.block_size
        )
        next_block += block_count
    block_table = block_table_host.to(device)
    seq_lens = torch.tensor(shape.context_lengths, dtype=torch.int32, device=device)
    slot_mapping = slot_mapping_host.to(device)
    metadata = PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=torch.arange(
            shape.batch_size + 1,
            dtype=torch.int32,
            device=device,
        ),
        max_seq_len=max(shape.context_lengths),
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


def run_attention(
    inputs: ProbeInputs,
    model: ModelShape,
    implementation: str = "auto",
) -> torch.Tensor:
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
        implementation=implementation,
    )


def combined(inputs: ProbeInputs, model: ModelShape) -> torch.Tensor:
    append_kv(inputs)
    return run_attention(inputs, model)


def build_kernel_candidates(kernel_set: str) -> list[KernelCandidate]:
    """Build production, practical sweep, or exhaustive launch candidates."""
    production = KernelCandidate("auto", 8, 4, 4, 1, 32)
    if kernel_set == "production":
        return [production]
    if kernel_set not in ("practical", "sweep", "full"):
        raise ValueError(f"unknown kernel set: {kernel_set}")
    candidates: dict[str, KernelCandidate] = {}

    def add(candidate: KernelCandidate) -> None:
        candidates[candidate.config_id] = candidate

    add(production)

    if kernel_set == "full":
        for num_splits in (1, 2, 4, 8, 16, 32):
            for block_h in (1, 2, 4):
                for num_warps in (2, 4, 8):
                    for num_stages in (1, 2):
                        for block_n in (16, 32, 64):
                            add(
                                KernelCandidate(
                                    "pap_grouped",
                                    num_splits,
                                    block_h,
                                    num_warps,
                                    num_stages,
                                    block_n,
                                )
                            )
    elif kernel_set == "sweep":
        for num_splits in (1, 2, 4, 8, 16, 32):
            for block_h in (1, 2, 4):
                add(
                    KernelCandidate(
                        "pap_grouped",
                        num_splits,
                        block_h,
                        4,
                        1,
                        32,
                    )
                )
        for num_warps in (2, 8):
            add(KernelCandidate("pap_grouped", 8, 4, num_warps, 1, 32))
        for num_stages in (2, 3):
            add(KernelCandidate("pap_grouped", 8, 4, 4, num_stages, 32))
        for block_n in (16, 64):
            add(KernelCandidate("pap_grouped", 8, 4, 4, 1, block_n))
    else:
        for num_splits in (1, 2, 4, 8, 16, 32):
            add(KernelCandidate("pap_grouped", num_splits, 4, 4, 1, 32))
        for num_warps in (2, 8):
            add(KernelCandidate("pap_grouped", 8, 4, num_warps, 1, 32))
        for num_stages in (2, 3):
            add(KernelCandidate("pap_grouped", 8, 4, 4, num_stages, 32))
        for block_n in (16, 64):
            add(KernelCandidate("pap_grouped", 8, 4, 4, 1, block_n))
    for num_splits in (1, 2, 4, 8, 16, 32):
        add(KernelCandidate("vllm", num_splits, 16, 2, 2, 64))
    return list(candidates.values())


def kernel_config(candidate: KernelCandidate) -> Any:
    from vllm.pap.attention.kernels import PAPPagedDecodeKernelConfig

    return PAPPagedDecodeKernelConfig(
        num_splits=candidate.num_splits,
        block_h=candidate.block_h,
        num_warps=candidate.num_warps,
        num_stages=candidate.num_stages,
        block_n=candidate.block_n,
    )


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


def _percentile(values: tuple[int, ...], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def context_features(shape: ProbeShape, block_size: int) -> dict[str, Any]:
    """Return prediction features for request count, volume, and skew."""
    lengths = shape.context_lengths
    mean = statistics.fmean(lengths)
    std = statistics.pstdev(lengths)
    total_blocks = sum(math.ceil(length / block_size) for length in lengths)
    allocated_tokens = total_blocks * block_size
    absolute_differences = sum(
        abs(left - right) for left in lengths for right in lengths
    )
    gini = absolute_differences / (2 * len(lengths) ** 2 * mean)
    return {
        "batch_size": shape.batch_size,
        "context_lengths": list(lengths),
        "distribution": shape.distribution,
        "total_context_tokens": shape.total_context_tokens,
        "allocated_context_tokens": allocated_tokens,
        "padding_context_tokens": allocated_tokens - shape.total_context_tokens,
        "total_blocks": total_blocks,
        "max_blocks_per_request": max(
            math.ceil(length / block_size) for length in lengths
        ),
        "min_context_tokens": min(lengths),
        "max_context_tokens": max(lengths),
        "mean_context_tokens": mean,
        "std_context_tokens": std,
        "cv_context_tokens": std / mean,
        "p50_context_tokens": _percentile(lengths, 0.50),
        "p90_context_tokens": _percentile(lengths, 0.90),
        "p99_context_tokens": _percentile(lengths, 0.99),
        "gini_context_tokens": gini,
    }


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
    candidate: KernelCandidate | None = None,
) -> dict[str, Any]:
    bytes_read = logical_kv_bytes(model, shape, dtype)
    workspace = inputs.workspace
    return {
        "schema_version": 2,
        "kind": "pap_attention_scaling",
        "mode": mode,
        "shape": {
            "groups": list(shape.groups),
            "shape_id": shape.shape_id,
        }
        | context_features(shape, model.block_size),
        "model_shape": asdict(model),
        "dtype": str(dtype).removeprefix("torch."),
        "logical_kv_bytes": bytes_read,
        "workspace": {
            "partial_shape": list(workspace.partial.shape),
            "num_splits": workspace.kernel_config.num_splits,
            "block_h": workspace.kernel_config.block_h,
            "num_warps": workspace.kernel_config.num_warps,
            "num_stages": workspace.kernel_config.num_stages,
            "block_n": workspace.kernel_config.block_n,
        },
        "kernel": asdict(candidate) | {"config_id": candidate.config_id}
        if candidate is not None
        else {"implementation": "auto", "config_id": "production_auto"},
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
    all_shapes = build_shapes(args.groups)
    if args.shard_index >= args.shard_count:
        raise ValueError("shard index must be smaller than shard count")
    shapes = [
        shape
        for index, shape in enumerate(all_shapes)
        if index % args.shard_count == args.shard_index
    ]
    if not shapes:
        raise ValueError("workload shard is empty")
    candidates = build_kernel_candidates(args.kernel_set)
    results = []
    best_by_shape = []
    for shape_index, shape in enumerate(shapes, start=1):
        inputs = build_inputs(model, shape, dtype=args.dtype, device=device)
        append = time_cuda(
            lambda inputs=inputs: append_kv(inputs),
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        from vllm.pap.attention.kernels import build_paged_decode_workspace

        inputs.workspace = build_paged_decode_workspace(inputs.query)
        reference = run_attention(inputs, model, "auto").detach().clone()
        torch.accelerator.synchronize()
        shape_entries = []
        for candidate in candidates:
            try:
                inputs.workspace = build_paged_decode_workspace(
                    inputs.query,
                    kernel_config(candidate),
                )
                actual = run_attention(
                    inputs,
                    model,
                    candidate.implementation,
                )
                torch.accelerator.synchronize()
                difference = (actual.float() - reference.float()).abs()
                correctness = {
                    "allclose": bool(
                        torch.allclose(
                            actual,
                            reference,
                            rtol=args.correctness_rtol,
                            atol=args.correctness_atol,
                        )
                    ),
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "rtol": args.correctness_rtol,
                    "atol": args.correctness_atol,
                }
                if not correctness["allclose"]:
                    raise RuntimeError("Attention candidate failed numerical parity")
                attention = time_cuda(
                    lambda inputs=inputs, implementation=candidate.implementation: (
                        run_attention(inputs, model, implementation)
                    ),
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
                    candidate=candidate,
                )
                entry.update(
                    {
                        "status": "completed",
                        "warmup_calls": args.warmup_calls,
                        "samples": args.samples,
                        "calls_per_sample": args.calls_per_sample,
                        "correctness": correctness,
                        "paged_attention": attention,
                        "kv_append": append,
                        "logical_kv_gbps": (
                            entry["logical_kv_bytes"]
                            / (attention["median_ms"] / 1_000.0)
                            / 1e9
                        ),
                    }
                )
            except Exception as exc:
                entry = {
                    "schema_version": 2,
                    "kind": "pap_attention_scaling",
                    "mode": "timing",
                    "status": "error",
                    "shape": {
                        "groups": list(shape.groups),
                        "shape_id": shape.shape_id,
                    }
                    | context_features(shape, model.block_size),
                    "kernel": asdict(candidate) | {"config_id": candidate.config_id},
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if args.fail_on_kernel_error:
                    raise
            results.append(entry)
            shape_entries.append(entry)
        completed = [entry for entry in shape_entries if entry["status"] == "completed"]
        if not completed:
            raise RuntimeError(f"no valid Attention kernel for {shape.shape_id}")
        best = min(
            completed,
            key=lambda entry: entry["paged_attention"]["median_ms"],
        )
        best_by_shape.append(
            {
                "shape_id": shape.shape_id,
                "config_id": best["kernel"]["config_id"],
                "median_ms": best["paged_attention"]["median_ms"],
            }
        )
        print(
            f"[{shape_index}/{len(shapes)}] {shape.shape_id} "
            f"best={best['kernel']['config_id']} "
            f"median_ms={best['paged_attention']['median_ms']:.6f}",
            flush=True,
        )
        if args.output is not None:
            checkpoint = {
                "schema_version": 2,
                "kind": "pap_attention_scaling_matrix",
                "status": "running",
                "groups": sorted(args.groups),
                "kernel_set": args.kernel_set,
                "candidate_count": len(candidates),
                "shape_count": len(shapes),
                "total_shape_count": len(all_shapes),
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "completed_shape_count": shape_index,
                "results": results,
                "best_by_shape": best_by_shape,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        del inputs
        torch.accelerator.empty_cache()
    output = {
        "schema_version": 2,
        "kind": "pap_attention_scaling_matrix",
        "status": "completed",
        "groups": sorted(args.groups),
        "kernel_set": args.kernel_set,
        "candidate_count": len(candidates),
        "shape_count": len(shapes),
        "total_shape_count": len(all_shapes),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "results": results,
        "best_by_shape": best_by_shape,
    }
    write_json(output, args.output)


def run_profile(args: argparse.Namespace) -> None:
    model = load_model_shape(args.model_config)
    shape = args.shape or args.context_lengths
    if shape is None:
        raise ValueError("profile requires --shape or --context-lengths")
    device = torch.device("cuda", 0)
    inputs = build_inputs(model, shape, dtype=args.dtype, device=device)
    candidate = KernelCandidate(
        args.implementation,
        args.num_splits,
        args.block_h,
        args.num_warps,
        args.num_stages,
        args.block_n,
    )
    from vllm.pap.attention.kernels import build_paged_decode_workspace

    inputs.workspace = build_paged_decode_workspace(
        inputs.query,
        kernel_config(candidate),
    )
    visible_sms = current_platform.num_compute_units(int(device.index or 0))
    if args.expected_sms is not None and visible_sms != args.expected_sms:
        raise RuntimeError(
            f"visible SM mismatch: expected {args.expected_sms}, got {visible_sms}"
        )
    for _ in range(args.warmup_calls):
        run_attention(inputs, model, candidate.implementation)
    torch.accelerator.synchronize()
    calls = 0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    nvtx.range_push("pap_attention_scaling_probe")
    try:
        start.record()
        while True:
            for _ in range(args.profile_chunk_calls):
                run_attention(inputs, model, candidate.implementation)
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
        candidate=candidate,
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
    print("shape_id\tbatch\ttotal_context\tmin\tmax\tdistribution\tgroups")
    for shape in build_shapes(args.groups):
        print(
            f"{shape.shape_id}\t{shape.batch_size}\t{shape.total_context_tokens}\t"
            f"{min(shape.context_lengths)}\t{max(shape.context_lengths)}\t"
            f"{shape.distribution}\t{','.join(shape.groups)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_parser = subparsers.add_parser("dry-run")
    dry_parser.add_argument(
        "--groups",
        type=parse_groups,
        default={"context", "iso_total", "distribution"},
    )
    dry_parser.set_defaults(func=dry_run)

    timing_parser = subparsers.add_parser("timing")
    timing_parser.add_argument("--model-config", type=Path, required=True)
    timing_parser.add_argument(
        "--groups",
        type=parse_groups,
        default={"context", "iso_total", "distribution"},
    )
    timing_parser.add_argument(
        "--kernel-set",
        choices=("production", "practical", "sweep", "full"),
        default="sweep",
    )
    timing_parser.add_argument("--shard-count", type=_positive_int, default=1)
    timing_parser.add_argument("--shard-index", type=_nonnegative_int, default=0)
    timing_parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    timing_parser.add_argument("--expected-sms", type=_positive_int)
    timing_parser.add_argument("--warmup-calls", type=_positive_int, default=20)
    timing_parser.add_argument("--samples", type=_positive_int, default=7)
    timing_parser.add_argument("--calls-per-sample", type=_positive_int, default=50)
    timing_parser.add_argument("--correctness-rtol", type=float, default=2e-2)
    timing_parser.add_argument("--correctness-atol", type=float, default=2e-2)
    timing_parser.add_argument("--fail-on-kernel-error", action="store_true")
    timing_parser.add_argument("--output", type=Path)
    timing_parser.set_defaults(func=run_timing)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--model-config", type=Path, required=True)
    profile_shape = profile_parser.add_mutually_exclusive_group(required=True)
    profile_shape.add_argument("--shape", type=parse_explicit_shape)
    profile_shape.add_argument("--context-lengths", type=parse_context_lengths)
    profile_parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    profile_parser.add_argument(
        "--implementation",
        choices=("auto", "pap_grouped", "vllm"),
        default="auto",
    )
    profile_parser.add_argument("--num-splits", type=_positive_int, default=8)
    profile_parser.add_argument("--block-h", type=_positive_int, default=4)
    profile_parser.add_argument("--num-warps", type=_positive_int, default=4)
    profile_parser.add_argument("--num-stages", type=_positive_int, default=1)
    profile_parser.add_argument("--block-n", type=_positive_int, default=32)
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
