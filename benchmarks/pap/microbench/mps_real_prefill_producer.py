#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a continuous queue of real Qwen Prefill forwards under MPS."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests <= 0 or args.prompt_tokens <= 0:
        raise ValueError("requests and prompt tokens must be positive")
    if args.max_batched_tokens <= 0:
        raise ValueError("max batched tokens must be positive")

    from vllm import LLM, SamplingParams

    config = json.loads((args.model / "config.json").read_text())
    vocab_size = int(config["vocab_size"])
    rng = random.Random(args.seed)
    prompts = [
        {
            "prompt_token_ids": [
                rng.randrange(1_000, vocab_size - 256)
                for _ in range(args.prompt_tokens)
            ]
        }
        for _ in range(args.requests)
    ]

    load_start = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        dtype="float16",
        enforce_eager=True,
        generation_config="vllm",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(4096, args.prompt_tokens + 2),
        max_num_seqs=256,
        max_num_batched_tokens=args.max_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        async_scheduling=True,
        seed=args.seed,
    )
    load_ms = (time.perf_counter() - load_start) * 1_000.0
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        detokenize=False,
    )

    llm.sleep(level=0, mode="keep")
    request_ids = llm.enqueue(prompts, sampling_params=params, use_tqdm=False)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text("ready\n")
    run_start = time.perf_counter()
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(use_tqdm=False)
    run_ms = (time.perf_counter() - run_start) * 1_000.0

    if len(request_ids) != args.requests or len(outputs) != args.requests:
        raise RuntimeError(
            f"expected {args.requests} completed requests; "
            f"got {len(request_ids)} IDs and {len(outputs)} outputs"
        )
    result = {
        "role": "real_prefill_producer",
        "status": "passed",
        "requests": args.requests,
        "prompt_tokens": args.prompt_tokens,
        "max_batched_tokens": args.max_batched_tokens,
        "load_ms": load_ms,
        "run_ms": run_ms,
    }
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
