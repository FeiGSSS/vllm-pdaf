# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare paged decode-attention kernels on an archived matched shape."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from paged_fa_sm_probe import (
    ProbeInputs,
    ProbeShape,
    build_inputs,
    run_attention as run_fa2,
    tensor_metadata,
)


@dataclass
class KernelCase:
    """One reusable kernel invocation."""

    name: str
    run: Callable[[], torch.Tensor]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _split_list(value: str) -> tuple[int, ...]:
    splits = tuple(int(item) for item in value.split(","))
    if not splits or any(item <= 0 for item in splits):
        raise argparse.ArgumentTypeError("splits must be positive integers")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triton-splits", type=_split_list, default=(4, 8, 16))
    parser.add_argument("--warmup-calls", type=_positive_int, default=20)
    parser.add_argument("--samples", type=_positive_int, default=5)
    parser.add_argument("--calls-per-sample", type=_positive_int, default=60)
    parser.add_argument("--expected-sms", type=_positive_int)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _build_cases(
    inputs: ProbeInputs,
    shape: ProbeShape,
    triton_splits: tuple[int, ...],
) -> list[KernelCase]:
    from vllm.pap.attention.kernels import (
        PAP_TRITON_DECODE_NUM_SPLITS,
        build_paged_decode_workspace,
        run_paged_decode_attention,
    )
    from vllm.pap.kv.metadata import PAPPagedFlashMetadata
    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd,
    )

    scale = 1.0 / math.sqrt(shape.head_dim)

    def run_reference() -> torch.Tensor:
        run_fa2(inputs, shape, num_splits=0)
        return inputs.output

    cases = [KernelCase("fa2_auto", run_reference)]
    metadata = PAPPagedFlashMetadata(
        block_table=inputs.block_table,
        seq_lens=inputs.seqused_k,
        cu_seqlens_q=inputs.cu_seqlens_q,
        max_seq_len=max(shape.seq_lens),
    )
    workspace = build_paged_decode_workspace(inputs.query)
    k_scale = torch.ones((), dtype=torch.float32, device=inputs.query.device)
    v_scale = torch.ones((), dtype=torch.float32, device=inputs.query.device)
    for num_splits in triton_splits:
        if num_splits == PAP_TRITON_DECODE_NUM_SPLITS:

            def run_pap_triton() -> torch.Tensor:
                return run_paged_decode_attention(
                    query=inputs.query,
                    key_cache=inputs.key_cache,
                    value_cache=inputs.value_cache,
                    metadata=metadata,
                    workspace=workspace,
                    scale=scale,
                    block_size=shape.block_size,
                )

            cases.append(
                KernelCase(
                    f"pap_triton_decode_splits{num_splits}",
                    run_pap_triton,
                )
            )
            continue

        output = torch.empty_like(inputs.query)
        lse = torch.empty(
            (shape.batch_size, shape.num_q_heads),
            dtype=torch.float32,
            device=inputs.query.device,
        )
        partial = torch.empty(
            (
                shape.batch_size,
                shape.num_q_heads,
                num_splits,
                shape.head_dim + 1,
            ),
            dtype=torch.float32,
            device=inputs.query.device,
        )

        def run_triton(
            *,
            output: torch.Tensor = output,
            lse: torch.Tensor = lse,
            partial: torch.Tensor = partial,
            num_splits: int = num_splits,
        ) -> torch.Tensor:
            decode_attention_fwd(
                inputs.query,
                inputs.key_cache,
                inputs.value_cache,
                output,
                lse,
                inputs.block_table,
                inputs.seqused_k,
                partial,
                num_splits,
                scale,
                page_size=shape.block_size,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            return output

        cases.append(
            KernelCase(f"triton_decode_splits{num_splits}", run_triton)
        )
    return cases


def _correctness(cases: list[KernelCase]) -> dict[str, dict[str, float | bool]]:
    reference_output = cases[0].run()
    torch.cuda.synchronize()
    reference = reference_output.float().clone()
    result: dict[str, dict[str, float | bool]] = {}
    for case in cases[1:]:
        actual_output = case.run()
        torch.cuda.synchronize()
        actual = actual_output.float()
        difference = (actual - reference).abs()
        result[case.name] = {
            "allclose_atol_5e-3_rtol_5e-2": bool(
                torch.allclose(actual, reference, atol=5e-3, rtol=5e-2)
            ),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
        }
    return result


def _time_case(
    case: KernelCase,
    *,
    warmup_calls: int,
    samples: int,
    calls_per_sample: int,
) -> list[float]:
    for _ in range(warmup_calls):
        case.run()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            case.run()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / calls_per_sample)
    return values


def main() -> None:
    args = parse_args()
    shape = ProbeShape()
    inputs = build_inputs(shape, layer_index=args.layer_index)
    visible_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if args.expected_sms is not None and visible_sms != args.expected_sms:
        raise RuntimeError(
            f"visible SM mismatch: expected {args.expected_sms}, got {visible_sms}"
        )

    inputs.key_cache.normal_(mean=0.0, std=0.1)
    inputs.value_cache.normal_(mean=0.0, std=0.1)
    torch.cuda.synchronize()
    cases = _build_cases(inputs, shape, args.triton_splits)
    correctness = _correctness(cases)

    measurements: dict[str, dict[str, object]] = {}
    for case in cases:
        samples_ms = _time_case(
            case,
            warmup_calls=args.warmup_calls,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )
        mean_ms = statistics.mean(samples_ms)
        measurements[case.name] = {
            "samples_ms_per_call": samples_ms,
            "mean_ms_per_call": mean_ms,
            "median_ms_per_call": statistics.median(samples_ms),
            "min_ms_per_call": min(samples_ms),
            "max_ms_per_call": max(samples_ms),
            "logical_min_kv_gbps": (
                shape.logical_kv_bytes / (mean_ms / 1000.0) / 1e9
            ),
        }

    props = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "kind": "pap_paged_attention_backend_probe",
        "device_name": props.name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "visible_sms": visible_sms,
        "shape": asdict(shape),
        "logical_min_kv_bytes": shape.logical_kv_bytes,
        "layer_key_cache": tensor_metadata(inputs.key_cache),
        "layer_value_cache": tensor_metadata(inputs.value_cache),
        "warmup_calls": args.warmup_calls,
        "samples": args.samples,
        "calls_per_sample": args.calls_per_sample,
        "correctness_vs_fa2": correctness,
        "measurements": measurements,
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
