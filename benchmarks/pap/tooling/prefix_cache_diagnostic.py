# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict two-turn audit for PAP native prefix-cache reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx


def build_second_prompt(
    first_prompt_token_ids: Sequence[int],
    first_output_token_ids: Sequence[int],
    suffix_token_ids: Sequence[int],
) -> list[int]:
    return [
        *(int(token_id) for token_id in first_prompt_token_ids),
        *(int(token_id) for token_id in first_output_token_ids),
        *(int(token_id) for token_id in suffix_token_ids),
    ]


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right))


def expected_prefix_hit_tokens(
    first_prompt_token_ids: Sequence[int],
    first_output_token_ids: Sequence[int],
    second_prompt_token_ids: Sequence[int],
    *,
    block_size: int,
) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    materialized_history = [
        *(int(token_id) for token_id in first_prompt_token_ids),
        *(int(token_id) for token_id in first_output_token_ids[:-1]),
    ]
    common_prefix = _common_prefix_length(
        materialized_history,
        second_prompt_token_ids,
    )
    max_cache_hit = max(0, len(second_prompt_token_ids) - 1)
    aligned_hit = min(common_prefix, max_cache_hit) // block_size * block_size
    return int(aligned_hit)


def _token_digest(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode()).hexdigest()


def _required_header_int(headers: Mapping[str, str], name: str) -> int:
    value = headers.get(name)
    if value is None:
        raise RuntimeError(f"missing required response header: {name}")
    parsed = int(value)
    if parsed < 0:
        raise RuntimeError(f"negative response header {name}: {parsed}")
    return parsed


def _completion_payload(
    *,
    model: str,
    prompt_token_ids: Sequence[int],
    max_tokens: int,
    cache_salt: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": [int(token_id) for token_id in prompt_token_ids],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }


def _run_completion(
    client: httpx.Client,
    *,
    endpoint: str,
    request_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        endpoint,
        json=payload,
        headers={"X-Request-Id": request_id},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError(f"expected one completion choice, got {len(choices)}")
    choice = choices[0]
    prompt_token_ids = choice.get("prompt_token_ids")
    output_token_ids = choice.get("token_ids")
    if not isinstance(prompt_token_ids, list) or not isinstance(output_token_ids, list):
        raise RuntimeError("completion response did not return token IDs")
    return {
        "request_id": request_id,
        "elapsed_ms": elapsed_ms,
        "prefill_ms": _required_header_int(
            response.headers,
            "X-PAP-Prefill-Ms",
        ),
        "prefill_prompt_tokens": _required_header_int(
            response.headers,
            "X-PAP-Prefill-Prompt-Tokens",
        ),
        "prefill_cached_tokens": _required_header_int(
            response.headers,
            "X-PAP-Prefill-Cached-Tokens",
        ),
        "prefill_computed_tokens": _required_header_int(
            response.headers,
            "X-PAP-Prefill-Computed-Tokens",
        ),
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        "output_token_ids": [int(token_id) for token_id in output_token_ids],
        "finish_reason": choice.get("finish_reason"),
    }


def _build_prompt_tokens(
    *,
    model: str,
    prompt_text: str,
    suffix_text: str,
    prompt_tokens: int,
) -> tuple[list[int], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=False,
    )
    repeated_text = prompt_text
    token_ids = tokenizer.encode(repeated_text, add_special_tokens=False)
    while len(token_ids) < prompt_tokens:
        repeated_text = f"{repeated_text} {prompt_text}"
        token_ids = tokenizer.encode(repeated_text, add_special_tokens=False)
    suffix_token_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    if not suffix_token_ids:
        raise RuntimeError("suffix_text produced no tokens")
    return [int(token_id) for token_id in token_ids[:prompt_tokens]], [
        int(token_id) for token_id in suffix_token_ids
    ]


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    first_prompt, suffix = _build_prompt_tokens(
        model=args.model,
        prompt_text=args.prompt_text,
        suffix_text=args.suffix_text,
        prompt_tokens=args.prompt_tokens,
    )
    run_nonce = uuid.uuid4().hex
    warm_salt = f"pap-multiturn-warm-{run_nonce}"
    cold_salt = f"pap-multiturn-cold-{run_nonce}"
    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"

    with httpx.Client(timeout=args.timeout) as client:
        first = _run_completion(
            client,
            endpoint=endpoint,
            request_id=f"pap-multiturn-{run_nonce}-turn1",
            payload=_completion_payload(
                model=args.model,
                prompt_token_ids=first_prompt,
                max_tokens=args.first_output_tokens,
                cache_salt=warm_salt,
            ),
        )
        if first["prompt_token_ids"] != first_prompt:
            raise RuntimeError("first response prompt token IDs changed")
        if len(first["output_token_ids"]) != args.first_output_tokens:
            raise RuntimeError(
                "first completion length differs from requested length: "
                f"{len(first['output_token_ids'])} != {args.first_output_tokens}"
            )

        second_prompt = build_second_prompt(
            first_prompt,
            first["output_token_ids"],
            suffix,
        )
        warm = _run_completion(
            client,
            endpoint=endpoint,
            request_id=f"pap-multiturn-{run_nonce}-warm",
            payload=_completion_payload(
                model=args.model,
                prompt_token_ids=second_prompt,
                max_tokens=args.second_output_tokens,
                cache_salt=warm_salt,
            ),
        )
        cold = _run_completion(
            client,
            endpoint=endpoint,
            request_id=f"pap-multiturn-{run_nonce}-cold",
            payload=_completion_payload(
                model=args.model,
                prompt_token_ids=second_prompt,
                max_tokens=args.second_output_tokens,
                cache_salt=cold_salt,
            ),
        )

    expected_hit = expected_prefix_hit_tokens(
        first_prompt,
        first["output_token_ids"],
        second_prompt,
        block_size=args.block_size,
    )
    warm_hit = int(warm["prefill_cached_tokens"])
    cold_hit = int(cold["prefill_cached_tokens"])
    prompt_block_boundary = len(first_prompt) // args.block_size * args.block_size
    decode_hit = max(0, warm_hit - prompt_block_boundary)

    if warm["prompt_token_ids"] != second_prompt:
        raise RuntimeError("warm response prompt token IDs changed")
    if cold["prompt_token_ids"] != second_prompt:
        raise RuntimeError("cold response prompt token IDs changed")
    if warm["output_token_ids"] != cold["output_token_ids"]:
        raise RuntimeError("warm and cold output token IDs differ")
    if warm_hit != expected_hit:
        raise RuntimeError(
            f"warm prefix hit {warm_hit} != expected block-aligned hit {expected_hit}"
        )
    if cold_hit != 0:
        raise RuntimeError(f"cold salted request unexpectedly hit {cold_hit} tokens")
    minimum_decode_hit = args.min_decode_hit_blocks * args.block_size
    if decode_hit < minimum_decode_hit:
        raise RuntimeError(
            f"decode-derived hit {decode_hit} < required {minimum_decode_hit}"
        )

    return {
        "status": "passed",
        "model": args.model,
        "base_url": args.base_url,
        "block_size": args.block_size,
        "first_prompt_tokens": len(first_prompt),
        "first_output_tokens": len(first["output_token_ids"]),
        "suffix_tokens": len(suffix),
        "second_prompt_tokens": len(second_prompt),
        "materialized_history_tokens": (
            len(first_prompt) + max(0, len(first["output_token_ids"]) - 1)
        ),
        "expected_prefix_hit_tokens": expected_hit,
        "actual_prefix_hit_tokens": warm_hit,
        "decode_derived_hit_tokens": decode_hit,
        "cold_prefix_hit_tokens": cold_hit,
        "first_prompt_digest": _token_digest(first_prompt),
        "first_output_digest": _token_digest(first["output_token_ids"]),
        "second_prompt_digest": _token_digest(second_prompt),
        "second_output_digest": _token_digest(warm["output_token_ids"]),
        "first": {
            key: value
            for key, value in first.items()
            if key not in {"prompt_token_ids", "output_token_ids"}
        },
        "warm": {
            key: value
            for key, value in warm.items()
            if key not in {"prompt_token_ids", "output_token_ids"}
        },
        "cold": {
            key: value
            for key, value in cold.items()
            if key not in {"prompt_token_ids", "output_token_ids"}
        },
        "warm_output_matches_cold": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit two-turn PAP native prefix-cache reuse"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9460")
    parser.add_argument(
        "--model",
        default="/data/ssd1/llm-models/Qwen3-8B",
    )
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--first-output-tokens", type=int, default=48)
    parser.add_argument("--second-output-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--min-decode-hit-blocks", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--prompt-text",
        default=(
            "Explain why a content-addressed KV cache can reuse prior "
            "autoregressive decoding work."
        ),
    )
    parser.add_argument(
        "--suffix-text",
        default=" Now give one concrete implementation invariant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_audit(args)
    except Exception as exc:
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        args.result_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    args.result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
