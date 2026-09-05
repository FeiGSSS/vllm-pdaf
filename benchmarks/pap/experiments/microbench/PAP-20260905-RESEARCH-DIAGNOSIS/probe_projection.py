# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure Projection's dense backbone, not serving or attention latency.

Uses 36 distinct weight sets to avoid a single-layer cache-residency artifact.
The measured region contains O, gate/up, down and next QKV linear operations,
SiLU and two residual RMS norms. It excludes Q/K norm, RoPE, communication,
attention, scheduling and the final vocabulary head. Splitting measures GPU
service demand (sequential lanes), NOT pipelined end-to-end latency.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,4,8,16,24,32,48,64,96,128")
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=5)
    args = parser.parse_args()
    assert args.layers == 36, "This probe is fixed to Qwen3-8B's 36-layer shape"
    assert args.samples > 0 and args.replays > 0
    torch.manual_seed(42)
    device_module = torch.get_device_module("cuda")
    torch.accelerator.set_device_index(0)
    props = device_module.get_device_properties(0)
    assert props.name == "NVIDIA L20", props.name
    assert props.multi_processor_count == 92, props.multi_processor_count
    dtype = torch.float16
    h, f, qkv = 4096, 12288, 6144
    weights = []
    for _ in range(args.layers):
        layer = []
        for out_dim, in_dim in ((h, h), (2 * f, h), (h, f), (qkv, h)):
            layer.append(
                torch.randn(out_dim, in_dim, device="cuda", dtype=dtype)
                * (in_dim**-0.5)
            )
        weights.append(layer)
    norm_weight = torch.ones(h, device="cuda", dtype=dtype)
    weight_bytes = sum(t.numel() * t.element_size() for w in weights for t in w)
    stream = device_module.Stream()
    stream.wait_stream(torch.accelerator.current_stream())
    results = []

    def backbone(inputs: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        lanes = list(inputs)
        # Lane order stays identical for each layer, as in a two-lane server.
        for wo, wgu, wd, wqkv in weights:
            for index, attention_output in enumerate(lanes):
                residual = attention_output.clone()
                x = F.linear(attention_output, wo)
                ops.fused_add_rms_norm(x, residual, norm_weight, 1e-6)
                gate_up = F.linear(x, wgu)
                activated = torch.empty((x.shape[0], f), device=x.device, dtype=x.dtype)
                torch.ops._C.silu_and_mul(activated, gate_up)
                x = F.linear(activated, wd)
                ops.fused_add_rms_norm(x, residual, norm_weight, 1e-6)
                next_qkv = F.linear(x, wqkv)
                # Dummy dependency, not an attention approximation in serving.
                lanes[index] = next_qkv[:, :h]
        return lanes

    with torch.inference_mode(), device_module.stream(stream):
        for batch in map(int, args.batches.split(",")):
            assert batch > 0
            for lane_count in (1, 2):
                if batch < lane_count or batch % lane_count:
                    continue
                inputs = tuple(
                    torch.randn(batch // lane_count, h, device="cuda", dtype=dtype)
                    for _ in range(lane_count)
                )
                for _ in range(3):
                    eager = backbone(inputs)
                stream.synchronize()
                graph = device_module.CUDAGraph()
                with device_module.graph(graph, stream=stream):
                    outputs = backbone(inputs)
                graph.replay()
                stream.synchronize()
                for actual, expected in zip(outputs, eager, strict=True):
                    assert torch.isfinite(actual).all().item()
                    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
                samples = []
                for _ in range(args.samples):
                    start = device_module.Event(enable_timing=True)
                    end = device_module.Event(enable_timing=True)
                    start.record(stream)
                    for _ in range(args.replays):
                        graph.replay()
                    end.record(stream)
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / args.replays)
                med = statistics.median(samples)
                result = {
                    "total_batch": batch,
                    "lanes": lane_count,
                    "lane_batch": batch // lane_count,
                    "service_ms_36layers_samples": samples,
                    "service_ms_36layers_median": med,
                    "service_ms_per_layer_median": med / args.layers,
                    "logical_weight_GBps": weight_bytes * lane_count / med / 1e6,
                }
                results.append(result)
                print(json.dumps(result), flush=True)
                del graph, eager, outputs, inputs
                gc.collect()
    payload = {
        "unit": "36-layer dense Projection backbone GPU service demand",
        "not_measured": ["attention", "NVSHMEM", "QK norm", "RoPE", "lm_head"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu": props.name,
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(dtype),
        "model_shape": {"layers": 36, "hidden": h, "ffn": f, "qkv": qkv},
        "weight_bytes": weight_bytes,
        "weight_sets_distinct": True,
        "samples": args.samples,
        "replays_per_sample": args.replays,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
