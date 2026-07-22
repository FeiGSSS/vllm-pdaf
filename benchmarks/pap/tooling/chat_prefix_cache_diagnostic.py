# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict two-turn Chat Completions audit for PAP prefix-cache reuse."""

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

from benchmarks.pap.tooling.prefix_cache_diagnostic import (
    expected_prefix_hit_tokens,
)


def build_second_turn_messages(
    first_messages: Sequence[Mapping[str, str]],
    assistant_content: str,
    second_user_content: str,
) -> list[dict[str, str]]:
    return [
        *(
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in first_messages
        ),
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": second_user_content},
    ]


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right))


def chat_prefix_metrics(
    first_prompt_token_ids: Sequence[int],
    first_output_token_ids: Sequence[int],
    second_prompt_token_ids: Sequence[int],
    *,
    block_size: int,
) -> dict[str, int]:
    committed_history = [
        *(int(token_id) for token_id in first_prompt_token_ids),
        *(int(token_id) for token_id in first_output_token_ids[:-1]),
    ]
    committed_lcp = _common_prefix_length(
        committed_history,
        second_prompt_token_ids,
    )
    expected_hit = expected_prefix_hit_tokens(
        first_prompt_token_ids,
        first_output_token_ids,
        second_prompt_token_ids,
        block_size=block_size,
    )
    prompt_boundary = len(first_prompt_token_ids) // block_size * block_size
    return {
        "committed_lcp_tokens": committed_lcp,
        "expected_prefix_hit_tokens": expected_hit,
        "decode_derived_hit_tokens": max(0, expected_hit - prompt_boundary),
    }


def build_chat_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    cache_salt: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
        "return_token_ids": True,
        "cache_salt": cache_salt,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _required_header_int(headers: Mapping[str, str], name: str) -> int:
    value = headers.get(name)
    if value is None:
        raise RuntimeError(f"missing required response header: {name}")
    parsed = int(value)
    if parsed < 0:
        raise RuntimeError(f"negative response header {name}: {parsed}")
    return parsed


def _token_digest(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode()).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _render_chat_prompt_token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if not isinstance(rendered, list):
        raise RuntimeError("local chat template did not return token IDs")
    return [int(token_id) for token_id in rendered]


def _build_first_messages(
    tokenizer: Any,
    *,
    prompt_text: str,
    min_prompt_tokens: int,
) -> tuple[list[dict[str, str]], list[int]]:
    if min_prompt_tokens <= 0:
        raise ValueError("min_prompt_tokens must be positive")
    if not prompt_text.strip():
        raise ValueError("prompt_text must not be empty")

    content = prompt_text
    for _ in range(128):
        messages = [{"role": "user", "content": content}]
        prompt_token_ids = _render_chat_prompt_token_ids(tokenizer, messages)
        if len(prompt_token_ids) >= min_prompt_tokens:
            return messages, prompt_token_ids
        content = f"{content} {prompt_text}"
    raise RuntimeError("failed to build the minimum Chat Completions prompt")


def _run_chat_completion(
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

    prompt_token_ids = body.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list):
        raise RuntimeError("chat response did not return prompt token IDs")
    choices = body.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError(f"expected one chat choice, got {len(choices)}")
    choice = choices[0]
    output_token_ids = choice.get("token_ids")
    if not isinstance(output_token_ids, list):
        raise RuntimeError("chat response did not return output token IDs")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("chat response did not return a message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise RuntimeError("chat response assistant content is empty")

    result = {
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
        "assistant_content": content,
        "finish_reason": choice.get("finish_reason"),
    }
    prompt_tokens = len(result["prompt_token_ids"])
    if result["prefill_prompt_tokens"] != prompt_tokens:
        raise RuntimeError("Prefill prompt count differs from returned token IDs")
    if (
        result["prefill_cached_tokens"]
        + result["prefill_computed_tokens"]
        != prompt_tokens
    ):
        raise RuntimeError("Prefill cached/computed counts do not cover the prompt")
    return result


def _validate_fixed_length(
    response: Mapping[str, Any],
    *,
    expected_output_tokens: int,
) -> None:
    actual = len(response["output_token_ids"])
    if actual != expected_output_tokens:
        raise RuntimeError(
            "chat completion length differs from requested length: "
            f"{actual} != {expected_output_tokens}"
        )
    if response["finish_reason"] != "length":
        raise RuntimeError(
            "chat completion did not finish by length: "
            f"{response['finish_reason']}"
        )


def _safe_response_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "assistant_content",
        "output_token_ids",
        "prompt_token_ids",
    }
    return {key: value for key, value in response.items() if key not in excluded}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    first_messages, local_first_prompt = _build_first_messages(
        tokenizer,
        prompt_text=args.first_user_text,
        min_prompt_tokens=args.min_first_prompt_tokens,
    )
    run_nonce = uuid.uuid4().hex
    warm_salt = f"pap-chat-multiturn-warm-{run_nonce}"
    cold_salt = f"pap-chat-multiturn-cold-{run_nonce}"
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"

    with httpx.Client(timeout=args.timeout) as client:
        first = _run_chat_completion(
            client,
            endpoint=endpoint,
            request_id=f"pap-chat-multiturn-{run_nonce}-turn1",
            payload=build_chat_payload(
                model=args.model,
                messages=first_messages,
                max_tokens=args.first_output_tokens,
                cache_salt=warm_salt,
            ),
        )
        _validate_fixed_length(
            first,
            expected_output_tokens=args.first_output_tokens,
        )
        if first["prompt_token_ids"] != local_first_prompt:
            raise RuntimeError("server and local first chat prompt tokens differ")
        if first["prefill_cached_tokens"] != 0:
            raise RuntimeError("first salted chat request unexpectedly hit cache")

        second_messages = build_second_turn_messages(
            first_messages,
            first["assistant_content"],
            args.second_user_text,
        )
        local_second_prompt = _render_chat_prompt_token_ids(
            tokenizer,
            second_messages,
        )
        warm = _run_chat_completion(
            client,
            endpoint=endpoint,
            request_id=f"pap-chat-multiturn-{run_nonce}-warm",
            payload=build_chat_payload(
                model=args.model,
                messages=second_messages,
                max_tokens=args.second_output_tokens,
                cache_salt=warm_salt,
            ),
        )
        cold = _run_chat_completion(
            client,
            endpoint=endpoint,
            request_id=f"pap-chat-multiturn-{run_nonce}-cold",
            payload=build_chat_payload(
                model=args.model,
                messages=second_messages,
                max_tokens=args.second_output_tokens,
                cache_salt=cold_salt,
            ),
        )

    _validate_fixed_length(
        warm,
        expected_output_tokens=args.second_output_tokens,
    )
    _validate_fixed_length(
        cold,
        expected_output_tokens=args.second_output_tokens,
    )
    if warm["prompt_token_ids"] != local_second_prompt:
        raise RuntimeError("server and local second chat prompt tokens differ")
    if cold["prompt_token_ids"] != local_second_prompt:
        raise RuntimeError("warm and cold chat prompt token IDs differ")
    if warm["output_token_ids"] != cold["output_token_ids"]:
        raise RuntimeError("warm and cold chat output token IDs differ")
    if warm["assistant_content"] != cold["assistant_content"]:
        raise RuntimeError("warm and cold chat assistant content differs")

    metrics = chat_prefix_metrics(
        first["prompt_token_ids"],
        first["output_token_ids"],
        warm["prompt_token_ids"],
        block_size=args.block_size,
    )
    expected_hit = metrics["expected_prefix_hit_tokens"]
    warm_hit = int(warm["prefill_cached_tokens"])
    cold_hit = int(cold["prefill_cached_tokens"])
    if warm_hit != expected_hit:
        raise RuntimeError(
            f"warm chat prefix hit {warm_hit} != expected {expected_hit}"
        )
    if cold_hit != 0:
        raise RuntimeError(
            f"cold salted chat request unexpectedly hit {cold_hit} tokens"
        )
    minimum_decode_hit = args.min_decode_hit_blocks * args.block_size
    if metrics["decode_derived_hit_tokens"] < minimum_decode_hit:
        raise RuntimeError(
            "chat decode-derived hit "
            f"{metrics['decode_derived_hit_tokens']} < required "
            f"{minimum_decode_hit}"
        )

    first_prompt_tokens = len(first["prompt_token_ids"])
    first_output_tokens = len(first["output_token_ids"])
    second_prompt_tokens = len(warm["prompt_token_ids"])
    return {
        "status": "passed",
        "model": args.model,
        "base_url": args.base_url,
        "api_path": "/v1/chat/completions",
        "enable_thinking": True,
        "block_size": args.block_size,
        "minimum_first_prompt_tokens": args.min_first_prompt_tokens,
        "first_prompt_tokens": first_prompt_tokens,
        "first_output_tokens": first_output_tokens,
        "second_prompt_tokens": second_prompt_tokens,
        "materialized_history_tokens": (
            first_prompt_tokens + max(0, first_output_tokens - 1)
        ),
        "committed_lcp_tokens": metrics["committed_lcp_tokens"],
        "first_prompt_block_boundary": (
            first_prompt_tokens // args.block_size * args.block_size
        ),
        "expected_prefix_hit_tokens": expected_hit,
        "actual_prefix_hit_tokens": warm_hit,
        "decode_derived_hit_tokens": metrics["decode_derived_hit_tokens"],
        "cold_prefix_hit_tokens": cold_hit,
        "first_prompt_digest": _token_digest(first["prompt_token_ids"]),
        "first_output_digest": _token_digest(first["output_token_ids"]),
        "second_prompt_digest": _token_digest(warm["prompt_token_ids"]),
        "second_output_digest": _token_digest(warm["output_token_ids"]),
        "first_assistant_text_digest": _text_digest(
            first["assistant_content"]
        ),
        "second_assistant_text_digest": _text_digest(
            warm["assistant_content"]
        ),
        "first": _safe_response_summary(first),
        "warm": _safe_response_summary(warm),
        "cold": _safe_response_summary(cold),
        "warm_output_matches_cold": True,
        "local_template_matches_server": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit two-turn PAP Chat Completions prefix-cache reuse"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9460")
    parser.add_argument(
        "--model",
        default="/data/ssd1/llm-models/Qwen3-8B",
    )
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--min-first-prompt-tokens", type=int, default=128)
    parser.add_argument("--first-output-tokens", type=int, default=48)
    parser.add_argument("--second-output-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--min-decode-hit-blocks", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--first-user-text",
        default=(
            "Explain how content-addressed KV cache reuse works in an "
            "autoregressive language model serving system."
        ),
    )
    parser.add_argument(
        "--second-user-text",
        default="Now give one concrete implementation invariant.",
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
