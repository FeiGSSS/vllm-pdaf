# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure PAP Projection, local P2P, and Attention batch scaling."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16

    @property
    def q_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_size(self) -> int:
        return self.q_size + 2 * self.kv_size


@dataclass
class ProjectionInputs:
    hidden: torch.Tensor
    residual: torch.Tensor
    attention_output: torch.Tensor
    input_norm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    post_norm_weight: torch.Tensor
    qkv_weight: torch.Tensor
    output_weight: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass
class AttentionInputs:
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


def _batch_sizes(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(","))
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return parsed


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float16 or bfloat16"
        ) from exc


def _components(value: str) -> tuple[str, ...]:
    allowed = {"projection", "local_fast", "attention"}
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("components must be unique and non-empty")
    invalid = set(parsed) - allowed
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported components: {','.join(sorted(invalid))}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=(1, 2, 4, 8, 16, 32),
    )
    parser.add_argument("--seq-len", type=_positive_int, default=8192)
    parser.add_argument("--dtype", type=_dtype, default=torch.float16)
    parser.add_argument("--attention-device", type=int, default=0)
    parser.add_argument("--projection-device", type=int, default=1)
    parser.add_argument(
        "--components",
        type=_components,
        default=("projection", "local_fast", "attention"),
    )
    parser.add_argument("--warmup-calls", type=_positive_int, default=10)
    parser.add_argument("--samples", type=_positive_int, default=5)
    parser.add_argument("--calls-per-sample", type=_positive_int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_model_shape(path: Path) -> ModelShape:
    config = json.loads(path.read_text(encoding="utf-8"))
    num_heads = int(config["num_attention_heads"])
    hidden_size = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden_size // num_heads)
    return ModelShape(
        hidden_size=hidden_size,
        intermediate_size=int(config["intermediate_size"]),
        num_heads=num_heads,
        num_kv_heads=int(config["num_key_value_heads"]),
        head_dim=head_dim,
    )


def _timing_stats(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples_ms,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def time_cuda(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup_calls: int,
    samples: int,
    calls_per_sample: int,
    stream: torch.cuda.Stream | None = None,
) -> dict[str, Any]:
    with torch.cuda.device(device), torch.inference_mode():
        target_stream = stream or torch.cuda.current_stream(device)
        with torch.cuda.stream(target_stream):
            for _ in range(warmup_calls):
                function()
        target_stream.synchronize()
        values: list[float] = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(target_stream):
                start.record(target_stream)
                for _ in range(calls_per_sample):
                    function()
                end.record(target_stream)
            end.synchronize()
            values.append(
                float(start.elapsed_time(end)) / float(calls_per_sample)
            )
    return _timing_stats(values)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def build_projection_inputs(
    shape: ModelShape,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ProjectionInputs:
    def weight(rows: int, columns: int) -> torch.Tensor:
        return torch.empty(
            (rows, columns),
            dtype=dtype,
            device=device,
        ).normal_(mean=0.0, std=0.02)

    hidden = torch.empty(
        (batch_size, shape.hidden_size),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    return ProjectionInputs(
        hidden=hidden,
        residual=torch.empty_like(hidden).normal_(mean=0.0, std=0.02),
        attention_output=torch.empty_like(hidden).normal_(
            mean=0.0,
            std=0.02,
        ),
        input_norm_weight=torch.ones(
            shape.hidden_size,
            dtype=dtype,
            device=device,
        ),
        q_norm_weight=torch.ones(shape.head_dim, dtype=dtype, device=device),
        k_norm_weight=torch.ones(shape.head_dim, dtype=dtype, device=device),
        post_norm_weight=torch.ones(
            shape.hidden_size,
            dtype=dtype,
            device=device,
        ),
        qkv_weight=weight(shape.qkv_size, shape.hidden_size),
        output_weight=weight(shape.hidden_size, shape.hidden_size),
        gate_up_weight=weight(
            2 * shape.intermediate_size,
            shape.hidden_size,
        ),
        down_weight=weight(shape.hidden_size, shape.intermediate_size),
        cos=torch.empty(
            (1, 1, shape.head_dim),
            dtype=dtype,
            device=device,
        ).uniform_(-1.0, 1.0),
        sin=torch.empty(
            (1, 1, shape.head_dim),
            dtype=dtype,
            device=device,
        ).uniform_(-1.0, 1.0),
    )


def projection_qkv(
    inputs: ProjectionInputs,
    shape: ModelShape,
) -> torch.Tensor:
    hidden = F.rms_norm(
        inputs.hidden,
        (shape.hidden_size,),
        inputs.input_norm_weight,
    )
    qkv = F.linear(hidden, inputs.qkv_weight)
    q, k, _value = qkv.split(
        (shape.q_size, shape.kv_size, shape.kv_size),
        dim=-1,
    )
    q = F.rms_norm(
        q.view(-1, shape.num_heads, shape.head_dim),
        (shape.head_dim,),
        inputs.q_norm_weight,
    )
    k = F.rms_norm(
        k.view(-1, shape.num_kv_heads, shape.head_dim),
        (shape.head_dim,),
        inputs.k_norm_weight,
    )
    q = q * inputs.cos + _rotate_half(q) * inputs.sin
    k = k * inputs.cos + _rotate_half(k) * inputs.sin
    qkv[:, : shape.q_size].copy_(q.reshape(-1, shape.q_size))
    qkv[:, shape.q_size : shape.q_size + shape.kv_size].copy_(
        k.reshape(-1, shape.kv_size)
    )
    return qkv


def projection_post_attention(
    inputs: ProjectionInputs,
    shape: ModelShape,
) -> torch.Tensor:
    hidden = F.linear(inputs.attention_output, inputs.output_weight)
    hidden = F.rms_norm(
        hidden + inputs.residual,
        (shape.hidden_size,),
        inputs.post_norm_weight,
    )
    gate, up = F.linear(hidden, inputs.gate_up_weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, inputs.down_weight)


def benchmark_projection(
    shape: ModelShape,
    *,
    batch_sizes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    warmup_calls: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    weights: ProjectionInputs | None = None
    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        current = build_projection_inputs(
            shape,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        if weights is None:
            weights = current
        else:
            current.qkv_weight = weights.qkv_weight
            current.output_weight = weights.output_weight
            current.gate_up_weight = weights.gate_up_weight
            current.down_weight = weights.down_weight

        qkv = time_cuda(
            lambda: projection_qkv(current, shape),
            device=device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
        )
        post = time_cuda(
            lambda: projection_post_attention(current, shape),
            device=device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
        )
        results[str(batch_size)] = {
            "qkv_norm_rope_repack": qkv,
            "output_projection_norm_mlp": post,
            "sum_of_medians_ms": (
                float(qkv["median_ms"]) + float(post["median_ms"])
            ),
            "rows_per_ms": (
                batch_size
                / (
                    float(qkv["median_ms"])
                    + float(post["median_ms"])
                )
            ),
        }
    return results


def benchmark_p2p(
    shape: ModelShape,
    *,
    batch_sizes: tuple[int, ...],
    dtype: torch.dtype,
    projection_device: torch.device,
    attention_device: torch.device,
    warmup_calls: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    if not torch.cuda.can_device_access_peer(
        projection_device.index,
        attention_device.index,
    ):
        raise RuntimeError("selected GPUs do not support CUDA peer access")
    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        qkv_source = torch.empty(
            (batch_size, shape.qkv_size),
            dtype=dtype,
            device=projection_device,
        )
        qkv_target = torch.empty_like(qkv_source, device=attention_device)
        output_source = torch.empty(
            (batch_size, shape.hidden_size),
            dtype=dtype,
            device=attention_device,
        )
        output_target = torch.empty_like(
            output_source,
            device=projection_device,
        )
        qkv_stream = torch.cuda.Stream(device=attention_device)
        output_stream = torch.cuda.Stream(device=projection_device)
        qkv = time_cuda(
            lambda: qkv_target.copy_(qkv_source, non_blocking=True),
            device=attention_device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
            stream=qkv_stream,
        )
        output = time_cuda(
            lambda: output_target.copy_(output_source, non_blocking=True),
            device=projection_device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
            stream=output_stream,
        )
        element_bytes = torch.finfo(dtype).bits // 8
        results[str(batch_size)] = {
            "qkv_p2p": qkv,
            "output_p2p": output,
            "qkv_bytes": batch_size * shape.qkv_size * element_bytes,
            "output_bytes": (
                batch_size * shape.hidden_size * element_bytes
            ),
            "sum_of_medians_ms": (
                float(qkv["median_ms"]) + float(output["median_ms"])
            ),
        }
        total_bytes = (
            int(results[str(batch_size)]["qkv_bytes"])
            + int(results[str(batch_size)]["output_bytes"])
        )
        total_ms = float(results[str(batch_size)]["sum_of_medians_ms"])
        results[str(batch_size)]["effective_bidirectional_gbps"] = (
            total_bytes / (total_ms / 1000.0) / 1e9
        )
    return results


def build_attention_inputs(
    shape: ModelShape,
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> AttentionInputs:
    from vllm.pap.attention.kernels import build_paged_decode_workspace
    from vllm.pap.kv.metadata import PAPPagedFlashMetadata

    result_seq_len = seq_len + 1
    blocks_per_sequence = math.ceil(result_seq_len / shape.block_size)
    num_blocks = batch_size * blocks_per_sequence
    kv_cache = torch.empty(
        (
            num_blocks,
            2,
            shape.block_size,
            shape.num_kv_heads,
            shape.head_dim,
        ),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    key_cache, value_cache = kv_cache.unbind(1)
    query = torch.empty(
        (batch_size, shape.num_heads, shape.head_dim),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    key = torch.empty(
        (batch_size, shape.num_kv_heads, shape.head_dim),
        dtype=dtype,
        device=device,
    ).normal_(mean=0.0, std=0.02)
    value = torch.empty_like(key).normal_(mean=0.0, std=0.02)
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_sequence)
    seq_lens = torch.full(
        (batch_size,),
        result_seq_len,
        dtype=torch.int32,
        device=device,
    )
    slot_mapping = (
        torch.arange(batch_size, dtype=torch.int64, device=device)
        * blocks_per_sequence
        * shape.block_size
        + seq_len
    )
    metadata = PAPPagedFlashMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        ),
        max_seq_len=result_seq_len,
    )
    return AttentionInputs(
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


def append_kv(inputs: AttentionInputs) -> None:
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
    inputs: AttentionInputs,
    shape: ModelShape,
) -> torch.Tensor:
    from vllm.pap.attention.kernels import run_paged_decode_attention

    return run_paged_decode_attention(
        query=inputs.query,
        key_cache=inputs.key_cache,
        value_cache=inputs.value_cache,
        metadata=inputs.metadata,
        workspace=inputs.workspace,
        scale=1.0 / math.sqrt(shape.head_dim),
        block_size=shape.block_size,
    )


def benchmark_attention(
    shape: ModelShape,
    *,
    batch_sizes: tuple[int, ...],
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup_calls: int,
    samples: int,
    calls_per_sample: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for batch_size in batch_sizes:
        inputs = build_attention_inputs(
            shape,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
        )
        append = time_cuda(
            lambda: append_kv(inputs),
            device=device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
        )
        attention = time_cuda(
            lambda: run_attention(inputs, shape),
            device=device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
        )

        def combined() -> torch.Tensor:
            append_kv(inputs)
            return run_attention(inputs, shape)

        append_attention = time_cuda(
            combined,
            device=device,
            warmup_calls=warmup_calls,
            samples=samples,
            calls_per_sample=calls_per_sample,
        )
        logical_kv_bytes = (
            batch_size
            * (seq_len + 1)
            * shape.num_kv_heads
            * shape.head_dim
            * 2
            * (torch.finfo(dtype).bits // 8)
        )
        results[str(batch_size)] = {
            "kv_append": append,
            "paged_attention": attention,
            "append_then_attention": append_attention,
            "logical_kv_bytes": logical_kv_bytes,
            "logical_kv_gbps": (
                logical_kv_bytes
                / (float(attention["median_ms"]) / 1000.0)
                / 1e9
            ),
        }
        del inputs
        torch.cuda.empty_cache()
    return results


def add_scaling(results: dict[str, Any], metric_path: tuple[str, ...]) -> None:
    first_key = next(iter(results))
    first: Any = results[first_key]
    for name in metric_path:
        first = first[name]
    baseline = float(first)
    for entry in results.values():
        value: Any = entry
        for name in metric_path:
            value = value[name]
        entry["latency_vs_first_batch"] = float(value) / baseline


def device_metadata(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "index": device.index,
        "name": properties.name,
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "capability": list(torch.cuda.get_device_capability(device)),
    }


def main() -> None:
    args = parse_args()
    if (
        "local_fast" in args.components
        and args.attention_device == args.projection_device
    ):
        raise ValueError("attention and projection devices must differ")
    shape = load_model_shape(args.model_config)
    attention_device = torch.device("cuda", args.attention_device)
    projection_device = torch.device("cuda", args.projection_device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    projection = (
        benchmark_projection(
            shape,
            batch_sizes=args.batch_sizes,
            dtype=args.dtype,
            device=projection_device,
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        if "projection" in args.components
        else {}
    )
    local_fast = (
        benchmark_p2p(
            shape,
            batch_sizes=args.batch_sizes,
            dtype=args.dtype,
            projection_device=projection_device,
            attention_device=attention_device,
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        if "local_fast" in args.components
        else {}
    )
    attention = (
        benchmark_attention(
            shape,
            batch_sizes=args.batch_sizes,
            seq_len=args.seq_len,
            dtype=args.dtype,
            device=attention_device,
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        if "attention" in args.components
        else {}
    )
    if projection:
        add_scaling(projection, ("sum_of_medians_ms",))
    if local_fast:
        add_scaling(local_fast, ("sum_of_medians_ms",))
    if attention:
        add_scaling(
            attention,
            ("append_then_attention", "median_ms"),
        )
    result = {
        "schema_version": 1,
        "kind": "pap_batch_scaling_probe",
        "model_config": str(args.model_config),
        "model_shape": asdict(shape),
        "dtype": str(args.dtype).removeprefix("torch."),
        "batch_sizes": list(args.batch_sizes),
        "components": list(args.components),
        "seq_len": args.seq_len,
        "attention_device": args.attention_device,
        "projection_device": args.projection_device,
        "devices": {
            "attention": (
                device_metadata(attention_device)
                if "attention" in args.components or "local_fast" in args.components
                else None
            ),
            "projection": (
                device_metadata(projection_device)
                if "projection" in args.components or "local_fast" in args.components
                else None
            ),
        },
        "cuda_mps_sm_partition": os.environ.get("CUDA_MPS_SM_PARTITION"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "peer_access": torch.cuda.can_device_access_peer(
            args.projection_device,
            args.attention_device,
        ),
        "warmup_calls": args.warmup_calls,
        "samples": args.samples,
        "calls_per_sample": args.calls_per_sample,
        "projection": projection,
        "local_fast": local_fast,
        "attention": attention,
        "scope": {
            "projection": (
                "synthetic eager Qwen3 layer math; excludes remote Attention"
            ),
            "local_fast": (
                "batch-dependent raw CUDA P2P copies; excludes constant "
                "doorbell and stream-signal control cost"
            ),
            "attention": (
                "production reshape_and_cache_flash plus PAP paged-decode kernel"
            ),
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
