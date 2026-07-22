"""Shared token and streaming helpers for PAP multi-turn benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def calculate_tpot_ms(
    latency_ms: float,
    ttft_ms: float,
    completion_tokens: int,
) -> float:
    """Calculate time per output token after the first generated token."""
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError(f"invalid latency_ms: {latency_ms}")
    if not math.isfinite(ttft_ms) or ttft_ms <= 0:
        raise ValueError(f"invalid ttft_ms: {ttft_ms}")
    if latency_ms < ttft_ms:
        raise ValueError(
            f"latency_ms must be >= ttft_ms: {latency_ms} < {ttft_ms}"
        )
    if completion_tokens <= 0:
        raise ValueError(
            f"completion_tokens must be positive: {completion_tokens}"
        )
    return (latency_ms - ttft_ms) / max(completion_tokens - 1, 1)


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if int(left_token) != int(right_token):
            return index
    return min(len(left), len(right))


def block_aligned_prefix_metrics(
    first_prompt_ids: Sequence[int],
    first_output_ids: Sequence[int],
    second_prompt_ids: Sequence[int],
    block_size: int = 16,
) -> dict[str, int]:
    """Return the materialized, retokenized prefix-cache reuse boundary."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive: {block_size}")
    if not first_prompt_ids:
        raise ValueError("first_prompt_ids must not be empty")
    if not first_output_ids:
        raise ValueError("first_output_ids must not be empty")
    if not second_prompt_ids:
        raise ValueError("second_prompt_ids must not be empty")

    materialized = [
        *(int(token_id) for token_id in first_prompt_ids),
        *(int(token_id) for token_id in first_output_ids[:-1]),
    ]
    lcp_tokens = _common_prefix_length(materialized, second_prompt_ids)
    expected_cached = lcp_tokens // block_size * block_size
    first_prompt_boundary = len(first_prompt_ids) // block_size * block_size
    return {
        "materialized_history_tokens": len(materialized),
        "committed_lcp_tokens": lcp_tokens,
        "expected_cached_tokens": expected_cached,
        "first_prompt_block_boundary": first_prompt_boundary,
        "decode_derived_hit_tokens": max(
            0,
            expected_cached - first_prompt_boundary,
        ),
    }


_PREFILL_HEADER_NAMES = {
    "prompt_tokens": "x-pap-prefill-prompt-tokens",
    "cached_tokens": "x-pap-prefill-cached-tokens",
    "computed_tokens": "x-pap-prefill-computed-tokens",
    "prefill_ms": "x-pap-prefill-ms",
}


def parse_prefill_headers(
    headers: Mapping[str, str],
) -> dict[str, int | None]:
    """Parse optional PAP Prefill accounting headers."""
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    parsed: dict[str, int | None] = {}
    for field, header_name in _PREFILL_HEADER_NAMES.items():
        raw_value = normalized.get(header_name)
        if raw_value is None:
            parsed[field] = None
            continue
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"invalid {header_name}: {raw_value!r}") from exc
        if value < 0:
            raise ValueError(f"negative {header_name}: {value}")
        parsed[field] = value

    prompt = parsed["prompt_tokens"]
    cached = parsed["cached_tokens"]
    computed = parsed["computed_tokens"]
    if (
        prompt is not None
        and cached is not None
        and computed is not None
        and cached + computed != prompt
    ):
        raise ValueError(
            "cached and computed Prefill tokens do not cover prompt tokens: "
            f"{cached} + {computed} != {prompt}"
        )
    return parsed


def profile_fingerprint(profile: Mapping[str, object]) -> str:
    """Return a stable digest for an architecture-independent profile."""
    payload = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def consume_sse_lines(
    lines: Iterable[str],
    *,
    started_at: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Consume a vLLM SSE response through EOF and return token metrics."""
    prompt_token_ids: list[int] | None = None
    output_token_ids: list[int] = []
    assistant_parts: list[str] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    saw_done = False

    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line.removeprefix("data: ")
        if raw == "[DONE]":
            saw_done = True
            continue
        try:
            chunk: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid SSE JSON: {raw!r}") from exc

        raw_prompt_ids = chunk.get("prompt_token_ids")
        if raw_prompt_ids is not None:
            if not isinstance(raw_prompt_ids, list):
                raise ValueError("prompt_token_ids must be a list")
            current_prompt_ids = [int(token_id) for token_id in raw_prompt_ids]
            if prompt_token_ids is not None and current_prompt_ids != prompt_token_ids:
                raise ValueError("conflicting prompt_token_ids chunks")
            prompt_token_ids = current_prompt_ids

        usage = chunk.get("usage")
        if isinstance(usage, Mapping):
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

        choices = chunk.get("choices") or []
        if not isinstance(choices, list):
            raise ValueError("SSE choices must be a list")
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise ValueError("SSE choice must be an object")
            raw_choice_prompt_ids = choice.get("prompt_token_ids")
            if raw_choice_prompt_ids is not None:
                if not isinstance(raw_choice_prompt_ids, list):
                    raise ValueError("choice prompt_token_ids must be a list")
                current_prompt_ids = [
                    int(token_id) for token_id in raw_choice_prompt_ids
                ]
                if (
                    prompt_token_ids is not None
                    and current_prompt_ids != prompt_token_ids
                ):
                    raise ValueError("conflicting prompt_token_ids chunks")
                prompt_token_ids = current_prompt_ids
            raw_token_ids = choice.get("token_ids") or []
            if not isinstance(raw_token_ids, list):
                raise ValueError("choice token_ids must be a list")
            if raw_token_ids:
                token_at = clock()
                if first_token_at is None:
                    first_token_at = token_at
                last_token_at = token_at
            output_token_ids.extend(int(token_id) for token_id in raw_token_ids)

            delta = choice.get("delta") or {}
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    assistant_parts.append(content)
            completion_text = choice.get("text")
            if isinstance(completion_text, str) and completion_text:
                assistant_parts.append(completion_text)
            current_finish = choice.get("finish_reason")
            if current_finish is not None:
                finish_reason = str(current_finish)

    eof_at = clock()
    if not saw_done:
        raise ValueError("stream ended without [DONE]")
    if prompt_token_ids is None:
        raise ValueError("stream did not return prompt_token_ids")
    if first_token_at is None or last_token_at is None or not output_token_ids:
        raise ValueError("stream did not return output token IDs")
    if prompt_tokens is None or completion_tokens is None:
        raise ValueError("stream did not return final usage")
    if prompt_tokens != len(prompt_token_ids):
        raise ValueError(
            "prompt token usage differs from token IDs: "
            f"{prompt_tokens} != {len(prompt_token_ids)}"
        )
    if completion_tokens != len(output_token_ids):
        raise ValueError(
            "completion token usage differs from token IDs: "
            f"{completion_tokens} != {len(output_token_ids)}"
        )

    ttft_ms = (first_token_at - started_at) * 1000.0
    latency_ms = (last_token_at - started_at) * 1000.0
    eof_latency_ms = (eof_at - started_at) * 1000.0
    post_token_stream_ms = (eof_at - last_token_at) * 1000.0
    if post_token_stream_ms < 0:
        raise ValueError("HTTP stream EOF preceded the final output token timestamp")
    return {
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": output_token_ids,
        "assistant_text": "".join(assistant_parts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "saw_done": saw_done,
        "ttft_ms": ttft_ms,
        "latency_ms": latency_ms,
        "eof_latency_ms": eof_latency_ms,
        "post_token_stream_ms": post_token_stream_ms,
        "tpot_ms": calculate_tpot_ms(latency_ms, ttft_ms, completion_tokens),
    }
