# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure unchanged production Attention on fixed, checked KV layouts."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import prefix_attn
import torch
from torch.cuda import nvtx

from vllm.pap.attention.pat_backend import PAPPATPlanner
from vllm.platforms import current_platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SPEC = importlib.util.spec_from_file_location(
    "pap_existing_probe",
    HERE.parent / "PAP-20260824-ATTENTION-LATENCY-SURFACE/probe.py",
)
existing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = existing
SPEC.loader.exec_module(existing)

CASES = {
    "triton_b1_16k": (1, 16384, 0, "triton"),
    "triton_b1_32k": (1, 32768, 0, "triton"),
    "triton_b2_16k_shared8k": (2, 16384, 8192, "triton"),
    "pat_b2_16k_shared8k": (2, 16384, 8192, "pat"),
    "pat_b2_32k_shared16k": (2, 32768, 16384, "pat"),
}


def measure(name, profile, output_dir, cache_mode):
    print(f"CASE {name}", flush=True)
    batch, length, shared, backend = CASES[name]
    torch.manual_seed(42)
    model = existing.load_model_shape(
        Path("/data/ssd1/llm-models/Qwen3-8B/config.json")
    )
    shape = existing.ProbeShape((length,) * batch, "fixed", ("bandwidth",))
    inputs = existing.build_inputs(
        model, shape, dtype=torch.float16, device=torch.device("cuda", 0)
    )
    inputs.query.normal_()
    table = inputs.metadata.block_table.cpu()
    perm = torch.randperm(int(table.max()) + 1)
    table = perm[table.long()].to(torch.int32)
    shared_blocks = shared // model.block_size
    for i in range(1, batch):
        table[i, :shared_blocks] = table[0, :shared_blocks]
    assert all(len(set(row.tolist())) == len(row) for row in table)
    inputs.metadata.block_table.copy_(table)
    cache_pairs = [(inputs.key_cache, inputs.value_cache)]
    if cache_mode == "rotating":
        for _ in range(3):
            storage = torch.stack((inputs.key_cache, inputs.value_cache), dim=1)
            cache_pairs.append(tuple(storage.unbind(1)))
    cache_index = 0

    def next_cache():
        nonlocal cache_index
        pair = cache_pairs[cache_index]
        cache_index = (cache_index + 1) % len(cache_pairs)
        return pair

    if backend == "pat":
        planner = PAPPATPlanner()
        states = [
            SimpleNamespace(block_size=model.block_size, block_ids=row.tolist())
            for row in table
        ]
        plan = planner.plan(
            states=states,
            seq_lens=[length] * batch,
            num_heads=model.num_heads,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            scale=1 / math.sqrt(model.head_dim),
            reused_kv_tokens=(batch - 1) * shared,
            dtype=torch.float16,
            device=torch.device("cuda", 0),
        )
        assert plan is not None, "PAT was requested but no plan was created"

        def call():
            key_cache, value_cache = next_cache()
            return plan.run_attention(inputs.query, key_cache, value_cache)
    else:

        def call():
            inputs.key_cache, inputs.value_cache = next_cache()
            return existing.run_attention(inputs, model)

    references = []
    for i, blocks in enumerate(table):
        key = inputs.key_cache[blocks.long()].reshape(-1, 8, 128)[:length].float()
        value = inputs.value_cache[blocks.long()].reshape(-1, 8, 128)[:length].float()
        query = inputs.query[i].float().reshape(8, 4, 128)
        probability = torch.softmax(
            torch.einsum("hgd,thd->hgt", query, key) / math.sqrt(128), dim=-1
        )
        references.append(
            torch.einsum("hgt,thd->hgd", probability, value).reshape(32, 128)
        )
    reference = torch.stack(references)
    actual = call()
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual.float(), reference, rtol=1e-2, atol=1e-4)
    error = float((actual.float() - reference).abs().max())
    del references, reference, key, value, probability, query
    warm_start, warm_end = (
        torch.Event(enable_timing=True),
        torch.Event(enable_timing=True),
    )
    warm_start.record()
    while True:
        for _ in range(100):
            call()
        warm_end.record()
        warm_end.synchronize()
        if warm_start.elapsed_time(warm_end) >= 1000:
            break
    torch.accelerator.synchronize()
    samples = []
    if profile:
        nvtx.range_push("pap_bandwidth")
        for _ in range(32):
            call()
        torch.accelerator.synchronize()
        nvtx.range_pop()
    else:
        for _ in range(7):
            start, end = (
                torch.Event(enable_timing=True),
                torch.Event(enable_timing=True),
            )
            start.record()
            for _ in range(100):
                call()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) / 100)
    unique = batch * length - (batch - 1) * shared
    result = {
        "case": name,
        "batch": batch,
        "length": length,
        "shared_prefix": shared,
        "backend": backend,
        "logical_kv_bytes": batch * length * 4096,
        "unique_kv_bytes": unique * 4096,
        "seed": 42,
        "max_abs_error": error,
        "samples_ms": samples,
        "scope": "isolated current kernel chain; CUDA "
        "events include eager launch gaps, no Prefill/transport",
        "cache_mode": cache_mode,
        "kv_cache_copies": len(cache_pairs),
        "unique_working_set_bytes": unique * 4096 * len(cache_pairs),
        "profile_calls": 32 if profile else None,
    }
    if samples:
        duration = sum(samples) / len(samples)
        result.update(
            {
                "mean_ms": duration,
                "logical_GBs": result["logical_kv_bytes"] / duration / 1e6,
                "unique_GBs": result["unique_kv_bytes"] / duration / 1e6,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--expected-sms", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cache-mode", choices=("warm", "rotating"), default="warm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sms = current_platform.num_compute_units(0)
    assert sms == args.expected_sms, (sms, args.expected_sms)
    binary = next(Path(prefix_attn.__file__).parent.glob("_prefix_attn*.so"))
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "visible_sms": sms,
        "partition": os.environ["CUDA_MPS_SM_PARTITION"],
        "pat_binary": str(binary),
        "pat_binary_sha256": digest,
        "warmup": "at least one GPU-second per case",
        "python": sys.executable,
        "source": str(ROOT),
        "device": str(current_platform.get_device_name(0)),
        "results": [
            measure(name, args.profile, args.output.parent, args.cache_mode)
            for name in ([args.case] if args.case else CASES)
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
