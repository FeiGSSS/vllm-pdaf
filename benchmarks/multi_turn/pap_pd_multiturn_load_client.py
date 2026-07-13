"""Versioned concurrent long-context multi-turn client for PAP and PD."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.multi_turn.pap_pd_multiturn_client import (
    block_aligned_prefix_metrics,
    consume_sse_lines,
    parse_prefill_headers,
    profile_fingerprint,
)

DEFAULT_DOCUMENT_TOKENS = 16000
DEFAULT_APPEND_TOKENS = 120
DEFAULT_OUTPUT_TOKENS = 256
DEFAULT_ROUNDS = 5
DEFAULT_ACTIVE_CONVERSATIONS = 4
DEFAULT_REQUEST_RATE = 2.0
DEFAULT_BLOCK_SIZE = 16
PROFILE_VERSION = 2

_INITIAL_INSTRUCTION = (
    "Read the following document and summarize its main argument.\n\n"
)
_INITIAL_GENERATION_MARKER = "\n\nAssistant:"
_LATER_USER_MARKER = (
    "\n\nUser: Using this additional passage, refine the summary and "
    "identify one changed conclusion.\n\n"
)
_LATER_GENERATION_MARKER = "\n\nAssistant:"


@dataclass(frozen=True)
class LoadConfig:
    """Frozen workload and implementation metadata for one load run."""

    base_url: str
    model: str
    corpus_path: str
    result_path: str
    architecture: str
    topology: str
    conversation_id_prefix: str
    cache_salt_prefix: str
    hardware_signature: str
    git_commit: str
    git_tracked_worktree_dirty: bool
    offload_exec_transport: str
    direct_mailbox_output: bool
    unified_md_fast_key: bool
    document_tokens: int = DEFAULT_DOCUMENT_TOKENS
    append_tokens: int = DEFAULT_APPEND_TOKENS
    output_tokens: int = DEFAULT_OUTPUT_TOKENS
    rounds: int = DEFAULT_ROUNDS
    active_conversations: int = DEFAULT_ACTIVE_CONVERSATIONS
    request_rate: float = DEFAULT_REQUEST_RATE
    block_size: int = DEFAULT_BLOCK_SIZE
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 20000
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 4
    request_timeout_seconds: float = 600.0


@dataclass(frozen=True)
class ConversationIdentity:
    """Unique routing and cache identity for one conversation."""

    index: int
    conversation_id: str
    cache_salt: str


TurnExecutor = Callable[
    [Any, LoadConfig, ConversationIdentity, Sequence[int], int],
    Awaitable[tuple[dict[str, object], dict[str, int | None]]],
]


def build_profile_id(config: LoadConfig) -> str:
    """Return the versioned profile id, including the concurrency lane."""

    if config.document_tokens % 1000 == 0:
        document_shape = f"{config.document_tokens // 1000}k"
    else:
        document_shape = str(config.document_tokens)
    return (
        f"qwen3_8b_token_{document_shape}_{config.rounds}turn_"
        f"o{config.output_tokens}_c{config.active_conversations}_v{PROFILE_VERSION}"
    )


def validate_config(config: LoadConfig) -> None:
    """Reject workload shapes that cannot satisfy the result contract."""

    if config.architecture not in ("pap", "pd"):
        raise ValueError(f"unsupported architecture: {config.architecture}")
    if config.rounds < 2:
        raise ValueError("multi-turn load requires at least two rounds")
    if config.active_conversations <= 0:
        raise ValueError("active_conversations must be positive")
    if not math.isfinite(config.request_rate) or config.request_rate <= 0:
        raise ValueError("request_rate must be finite and positive")
    if config.document_tokens <= 0 or config.append_tokens <= 0:
        raise ValueError("document_tokens and append_tokens must be positive")
    if config.output_tokens <= 1:
        raise ValueError("output_tokens must be greater than one")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if config.max_num_seqs < config.active_conversations:
        raise ValueError(
            "max_num_seqs must cover all active conversations: "
            f"{config.max_num_seqs} < {config.active_conversations}"
        )
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")


def build_load_workload(
    tokenizer: Any,
    corpus: str,
    *,
    document_tokens: int = DEFAULT_DOCUMENT_TOKENS,
    append_tokens: int = DEFAULT_APPEND_TOKENS,
    rounds: int = DEFAULT_ROUNDS,
) -> tuple[list[int], tuple[tuple[int, ...], ...]]:
    """Build an exact token-continuous prompt and later-turn suffixes."""

    if document_tokens <= 0 or append_tokens <= 0 or rounds < 2:
        raise ValueError("invalid multi-turn workload shape")
    corpus_token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    required_tokens = document_tokens + (rounds - 1) * append_tokens
    if len(corpus_token_ids) < required_tokens:
        raise ValueError(
            "corpus is too short for multi-turn load: "
            f"{len(corpus_token_ids)} < {required_tokens}"
        )
    instruction_ids = tokenizer.encode(
        _INITIAL_INSTRUCTION,
        add_special_tokens=False,
    )
    initial_generation_ids = tokenizer.encode(
        _INITIAL_GENERATION_MARKER,
        add_special_tokens=False,
    )
    later_user_ids = tokenizer.encode(
        _LATER_USER_MARKER,
        add_special_tokens=False,
    )
    later_generation_ids = tokenizer.encode(
        _LATER_GENERATION_MARKER,
        add_special_tokens=False,
    )
    initial_prompt_ids = [
        *(int(token_id) for token_id in instruction_ids),
        *(int(token_id) for token_id in corpus_token_ids[:document_tokens]),
        *(int(token_id) for token_id in initial_generation_ids),
    ]
    append_suffixes = tuple(
        (
            *(int(token_id) for token_id in later_user_ids),
            *(
                int(token_id)
                for token_id in corpus_token_ids[
                    document_tokens + index * append_tokens :
                    document_tokens + (index + 1) * append_tokens
                ]
            ),
            *(int(token_id) for token_id in later_generation_ids),
        )
        for index in range(rounds - 1)
    )
    return initial_prompt_ids, append_suffixes


def build_completion_payload(
    *,
    config: LoadConfig,
    identity: ConversationIdentity,
    prompt_token_ids: Sequence[int],
) -> dict[str, object]:
    """Build one exact-token OpenAI Completions request."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    payload: dict[str, object] = {
        "model": config.model,
        "prompt": [int(token_id) for token_id in prompt_token_ids],
        "add_special_tokens": False,
        "cache_salt": identity.cache_salt,
        "max_tokens": config.output_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    payload["conversation_id"] = identity.conversation_id
    return payload


def build_conversation_identities(
    config: LoadConfig,
) -> tuple[ConversationIdentity, ...]:
    """Create a unique conversation id and cache salt for every active lane."""

    return tuple(
        ConversationIdentity(
            index=index,
            conversation_id=f"{config.conversation_id_prefix}-conv-{index}",
            cache_salt=f"{config.cache_salt_prefix}-conv-{index}",
        )
        for index in range(config.active_conversations)
    )


def _required_prefill_value(
    prefill: Mapping[str, int | None],
    field: str,
) -> int:
    value = prefill.get(field)
    if not isinstance(value, int):
        raise ValueError(f"missing PAP Prefill {field}")
    return value


def validate_pap_first_round(
    prefill: Mapping[str, int | None],
) -> dict[str, int | str]:
    """Require a cold first request in each cache-salt namespace."""

    prompt = _required_prefill_value(prefill, "prompt_tokens")
    cached = _required_prefill_value(prefill, "cached_tokens")
    computed = _required_prefill_value(prefill, "computed_tokens")
    if cached != 0 or computed != prompt:
        raise ValueError(
            "PAP first round must be cold: "
            f"prompt={prompt} cached={cached} computed={computed}"
        )
    return {
        "status": "passed",
        "expected_cached_tokens": 0,
        "actual_cached_tokens": cached,
        "actual_computed_tokens": computed,
    }


def validate_pap_transition(
    *,
    previous_prompt_ids: Sequence[int],
    previous_output_ids: Sequence[int],
    current_prompt_ids: Sequence[int],
    current_prefill: Mapping[str, int | None],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, int | str]:
    """Require exact block-aligned reuse across one conversation transition."""

    metrics = block_aligned_prefix_metrics(
        previous_prompt_ids,
        previous_output_ids,
        current_prompt_ids,
        block_size=block_size,
    )
    actual_cached = _required_prefill_value(current_prefill, "cached_tokens")
    actual_computed = _required_prefill_value(current_prefill, "computed_tokens")
    prompt_tokens = _required_prefill_value(current_prefill, "prompt_tokens")
    expected_cached = int(metrics["expected_cached_tokens"])
    if actual_cached != expected_cached:
        raise ValueError(
            "PAP cached tokens differ from retokenized LCP: "
            f"{actual_cached} != {expected_cached}"
        )
    if actual_cached + actual_computed != prompt_tokens:
        raise ValueError("PAP cached and computed tokens do not cover the prompt")
    decode_derived = int(metrics["decode_derived_hit_tokens"])
    if decode_derived < block_size:
        raise ValueError(
            "PAP transition has no full Decode-derived cache block: "
            f"{decode_derived} < {block_size}"
        )
    return {
        "status": "passed",
        **{str(key): int(value) for key, value in metrics.items()},
        "actual_cached_tokens": actual_cached,
        "actual_computed_tokens": actual_computed,
    }


def _pd_external_cache_gate(
    *,
    previous_prompt_ids: Sequence[int] | None,
    previous_output_ids: Sequence[int] | None,
    current_prompt_ids: Sequence[int],
    block_size: int,
) -> dict[str, int | str | None]:
    if previous_prompt_ids is None or previous_output_ids is None:
        expected: dict[str, int] = {
            "expected_cached_tokens": 0,
            "decode_derived_hit_tokens": 0,
        }
    else:
        expected = block_aligned_prefix_metrics(
            previous_prompt_ids,
            previous_output_ids,
            current_prompt_ids,
            block_size=block_size,
        )
    return {
        "status": "requires_external_validation",
        **{str(key): int(value) for key, value in expected.items()},
        "actual_cached_tokens": None,
    }


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _digest_tokens(token_ids: Sequence[int]) -> str:
    return _digest_text(",".join(str(int(token_id)) for token_id in token_ids))


def _validate_turn(
    observation: Mapping[str, object],
    *,
    expected_output_tokens: int,
    max_model_len: int,
) -> None:
    if observation.get("finish_reason") != "length":
        raise ValueError(
            "load turn did not finish by length: "
            f"{observation.get('finish_reason')}"
        )
    if observation.get("completion_tokens") != expected_output_tokens:
        raise ValueError(
            "load completion length mismatch: "
            f"{observation.get('completion_tokens')} != {expected_output_tokens}"
        )
    prompt_tokens = int(observation["prompt_tokens"])
    if prompt_tokens + expected_output_tokens > max_model_len:
        raise ValueError(
            "load turn exceeds max_model_len: "
            f"{prompt_tokens} + {expected_output_tokens} > {max_model_len}"
        )


def _run_stream_turn_sync(
    client: Any,
    config: LoadConfig,
    identity: ConversationIdentity,
    prompt_token_ids: Sequence[int],
    round_index: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, object], dict[str, int | None]]:
    request_id = f"{identity.conversation_id}-turn-{round_index}"
    payload = build_completion_payload(
        config=config,
        identity=identity,
        prompt_token_ids=prompt_token_ids,
    )
    request_started_at = clock()
    with client.stream(
        "POST",
        f"{config.base_url.rstrip('/')}/v1/completions",
        json=payload,
        headers={"X-Request-Id": request_id},
    ) as response:
        response.raise_for_status()
        prefill = parse_prefill_headers(response.headers)
        observation = consume_sse_lines(
            response.iter_lines(),
            started_at=request_started_at,
            clock=clock,
        )
    returned_prompt_ids = observation.get("prompt_token_ids")
    if returned_prompt_ids != [int(token_id) for token_id in prompt_token_ids]:
        raise ValueError("server prompt token IDs differ from submitted token IDs")
    _validate_turn(
        observation,
        expected_output_tokens=config.output_tokens,
        max_model_len=config.max_model_len,
    )
    header_prompt_tokens = prefill.get("prompt_tokens")
    if (
        header_prompt_tokens is not None
        and header_prompt_tokens != observation["prompt_tokens"]
    ):
        raise ValueError(
            "PAP Prefill header prompt tokens differ from stream usage: "
            f"{header_prompt_tokens} != {observation['prompt_tokens']}"
        )
    observation.update(
        {
            "request_id": request_id,
            "request_started_at": request_started_at,
            "first_token_at": request_started_at
            + float(observation["ttft_ms"]) / 1000.0,
            "last_token_at": request_started_at
            + float(observation["latency_ms"]) / 1000.0,
            "eof_at": request_started_at
            + float(observation["eof_latency_ms"]) / 1000.0,
        }
    )
    return observation, prefill


async def _default_turn_executor(
    client: Any,
    config: LoadConfig,
    identity: ConversationIdentity,
    prompt_token_ids: Sequence[int],
    round_index: int,
) -> tuple[dict[str, object], dict[str, int | None]]:
    return await asyncio.to_thread(
        _run_stream_turn_sync,
        client,
        config,
        identity,
        prompt_token_ids,
        round_index,
    )


def _metric_stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("metric stats require at least one value")
    sorted_values = sorted(float(value) for value in values)
    p90_index = max(0, math.ceil(0.9 * len(sorted_values)) - 1)
    return {
        "count": len(sorted_values),
        "median": statistics.median(sorted_values),
        "p90": sorted_values[p90_index],
        "max": sorted_values[-1],
    }


def _request_metric_summary(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float | int]]:
    fields = (
        "ttft_ms",
        "tpot_ms",
        "latency_ms",
        "eof_latency_ms",
        "post_token_stream_ms",
    )
    return {
        field: _metric_stats(
            [float(observation[field]) for observation in observations]
        )
        for field in fields
    }


def interval_concurrency(
    intervals: Sequence[tuple[float, float]],
) -> dict[str, float | int]:
    """Return peak and time-weighted concurrency for half-open intervals."""

    if not intervals:
        return {
            "interval_count": 0,
            "peak": 0,
            "time_weighted_average": 0.0,
            "span_ms": 0.0,
        }
    deltas: dict[float, int] = {}
    for start, end in intervals:
        if end < start:
            raise ValueError(f"invalid interval: {start} > {end}")
        deltas[start] = deltas.get(start, 0) + 1
        deltas[end] = deltas.get(end, 0) - 1
    current = 0
    peak = 0
    area = 0.0
    ordered_times = sorted(deltas)
    previous = ordered_times[0]
    for timestamp in ordered_times:
        area += current * (timestamp - previous)
        current += deltas[timestamp]
        peak = max(peak, current)
        previous = timestamp
    span = ordered_times[-1] - ordered_times[0]
    return {
        "interval_count": len(intervals),
        "peak": peak,
        "time_weighted_average": area / span if span > 0 else 0.0,
        "span_ms": span * 1000.0,
    }


def _effective_concurrency(
    observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    http_intervals = [
        (
            float(observation["request_started_at"]),
            float(observation["eof_at"]),
        )
        for observation in observations
    ]
    decode_intervals = [
        (
            float(observation["first_token_at"]),
            float(observation["last_token_at"]),
        )
        for observation in observations
    ]
    return {
        "http": interval_concurrency(http_intervals),
        "decode": interval_concurrency(decode_intervals),
    }


def _topology_metadata(architecture: str, topology: str) -> dict[str, object]:
    if architecture == "pap":
        match = re.fullmatch(r"([1-9][0-9]*)pa([1-9][0-9]*)p", topology)
        if match is None:
            raise ValueError(f"invalid PAP topology: {topology}")
        return {
            "name": topology,
            "pa_count": int(match.group(1)),
            "projection_count": int(match.group(2)),
            "pd_prefill_count": 0,
            "pd_decode_count": 0,
        }
    match = re.fullmatch(r"([1-9][0-9]*)p([1-9][0-9]*)d", topology)
    if match is None:
        raise ValueError(f"invalid PD topology: {topology}")
    return {
        "name": topology,
        "pa_count": 0,
        "projection_count": 0,
        "pd_prefill_count": int(match.group(1)),
        "pd_decode_count": int(match.group(2)),
    }


def _profile(config: LoadConfig, corpus: str) -> dict[str, object]:
    return {
        "profile_id": build_profile_id(config),
        "profile_version": PROFILE_VERSION,
        "model": config.model,
        "corpus_path": config.corpus_path,
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "api": "/v1/completions",
        "workload_semantics": "exact_token_continuous_multiturn",
        "prompt_input": "token_ids",
        "history_rule": "previous_prompt_plus_full_output_plus_suffix",
        "document_tokens": config.document_tokens,
        "append_tokens_per_later_round": config.append_tokens,
        "output_tokens_per_round": config.output_tokens,
        "rounds": config.rounds,
        "active_conversations": config.active_conversations,
        "request_rate_per_round": config.request_rate,
        "arrival_mode": "fixed_rate_round_barrier_closed_loop",
        "arrival": {
            "mode": "fixed_rate_round_barrier_closed_loop",
            "request_rate_per_round": config.request_rate,
        },
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "block_size": config.block_size,
        "dtype": config.dtype,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_model_len": config.max_model_len,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_num_seqs": config.max_num_seqs,
    }


def _public_request_summary(
    observation: Mapping[str, object],
    *,
    identity: ConversationIdentity,
    round_index: int,
    prefill: Mapping[str, int | None],
    cache_validation: Mapping[str, object],
    origin: float,
) -> dict[str, object]:
    prompt_ids = observation["prompt_token_ids"]
    output_ids = observation["output_token_ids"]
    assistant_text = observation["assistant_text"]
    assert isinstance(prompt_ids, list)
    assert isinstance(output_ids, list)
    assert isinstance(assistant_text, str)
    return {
        "round": round_index,
        "conversation_index": identity.index,
        "request_id": observation["request_id"],
        "conversation_id_digest": _digest_text(identity.conversation_id),
        "cache_salt_digest": _digest_text(identity.cache_salt),
        "prompt_tokens": observation["prompt_tokens"],
        "completion_tokens": observation["completion_tokens"],
        "ttft_ms": observation["ttft_ms"],
        "tpot_ms": observation["tpot_ms"],
        "latency_ms": observation["latency_ms"],
        "eof_latency_ms": observation["eof_latency_ms"],
        "post_token_stream_ms": observation["post_token_stream_ms"],
        "finish_reason": observation["finish_reason"],
        "saw_done": observation["saw_done"],
        "timeline_ms": {
            "request_start": (
                float(observation["request_started_at"]) - origin
            )
            * 1000.0,
            "first_token": (float(observation["first_token_at"]) - origin)
            * 1000.0,
            "last_token": (float(observation["last_token_at"]) - origin)
            * 1000.0,
            "eof": (float(observation["eof_at"]) - origin) * 1000.0,
        },
        "shape": {
            "prompt_tokens": observation["prompt_tokens"],
            "completion_tokens": observation["completion_tokens"],
            "expected_cached_tokens": cache_validation.get(
                "expected_cached_tokens"
            ),
        },
        "prompt_token_digest": _digest_tokens(prompt_ids),
        "output_token_digest": _digest_tokens(output_ids),
        "assistant_text_digest": _digest_text(assistant_text),
        "prefill": dict(prefill),
        "cache_validation": dict(cache_validation),
    }


async def execute_load(
    config: LoadConfig,
    *,
    tokenizer: Any,
    corpus: str,
    clients: Sequence[Any],
    turn_executor: TurnExecutor | None = None,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, object]:
    """Execute fixed-rate concurrent waves with a barrier between rounds."""

    validate_config(config)
    if len(clients) != config.active_conversations:
        raise ValueError(
            "one HTTP client is required per active conversation: "
            f"{len(clients)} != {config.active_conversations}"
        )
    initial_prompt_ids, append_suffixes = build_load_workload(
        tokenizer,
        corpus,
        document_tokens=config.document_tokens,
        append_tokens=config.append_tokens,
        rounds=config.rounds,
    )
    identities = build_conversation_identities(config)
    prompt_ids_by_conversation = [
        list(initial_prompt_ids)
        for _ in range(config.active_conversations)
    ]
    previous_prompt_ids: list[list[int] | None] = [
        None for _ in range(config.active_conversations)
    ]
    previous_output_ids: list[list[int] | None] = [
        None for _ in range(config.active_conversations)
    ]
    executor = turn_executor or _default_turn_executor
    all_records: list[
        tuple[
            int,
            ConversationIdentity,
            dict[str, object],
            dict[str, int | None],
            dict[str, object],
        ]
    ] = []
    round_observations: list[list[dict[str, object]]] = []

    for round_index in range(1, config.rounds + 1):

        async def run_conversation(
            identity: ConversationIdentity,
        ) -> tuple[
            ConversationIdentity,
            dict[str, object],
            dict[str, int | None],
        ]:
            delay = identity.index / config.request_rate
            if delay > 0:
                await sleeper(delay)
            observation, prefill = await executor(
                clients[identity.index],
                config,
                identity,
                tuple(prompt_ids_by_conversation[identity.index]),
                round_index,
            )
            _validate_turn(
                observation,
                expected_output_tokens=config.output_tokens,
                max_model_len=config.max_model_len,
            )
            return identity, observation, prefill

        wave = await asyncio.gather(
            *(run_conversation(identity) for identity in identities)
        )
        current_round: list[dict[str, object]] = []
        for identity, observation, prefill in wave:
            current_prompt_ids = observation["prompt_token_ids"]
            current_output_ids = observation["output_token_ids"]
            assistant_text = observation["assistant_text"]
            assert isinstance(current_prompt_ids, list)
            assert isinstance(current_output_ids, list)
            assert isinstance(assistant_text, str)
            prior_prompt = previous_prompt_ids[identity.index]
            prior_output = previous_output_ids[identity.index]
            if config.architecture == "pap":
                if prior_prompt is None or prior_output is None:
                    cache_validation: dict[str, object] = (
                        validate_pap_first_round(prefill)
                    )
                else:
                    cache_validation = validate_pap_transition(
                        previous_prompt_ids=prior_prompt,
                        previous_output_ids=prior_output,
                        current_prompt_ids=current_prompt_ids,
                        current_prefill=prefill,
                        block_size=config.block_size,
                    )
            else:
                cache_validation = _pd_external_cache_gate(
                    previous_prompt_ids=prior_prompt,
                    previous_output_ids=prior_output,
                    current_prompt_ids=current_prompt_ids,
                    block_size=config.block_size,
                )
            all_records.append(
                (round_index, identity, observation, prefill, cache_validation)
            )
            current_round.append(observation)
            previous_prompt_ids[identity.index] = list(current_prompt_ids)
            previous_output_ids[identity.index] = list(current_output_ids)
            if round_index < config.rounds:
                prompt_ids_by_conversation[identity.index] = [
                    *current_prompt_ids,
                    *current_output_ids,
                    *append_suffixes[round_index - 1],
                ]
        round_observations.append(current_round)

    observations = [record[2] for record in all_records]
    origin = min(
        float(observation["request_started_at"]) for observation in observations
    )
    round_results: list[dict[str, object]] = []
    public_requests: list[dict[str, object]] = []
    for round_index, current_round in enumerate(round_observations, start=1):
        records = [record for record in all_records if record[0] == round_index]
        public_round_requests = [
            _public_request_summary(
                observation,
                identity=identity,
                round_index=round_index,
                prefill=prefill,
                cache_validation=cache_validation,
                origin=origin,
            )
            for _, identity, observation, prefill, cache_validation in records
        ]
        public_requests.extend(public_round_requests)
        starts = [
            float(observation["request_started_at"]) for observation in current_round
        ]
        round_results.append(
            {
                "round": round_index,
                "request_count": len(current_round),
                "request_start_skew_ms": (max(starts) - min(starts)) * 1000.0,
                "metrics": _request_metric_summary(current_round),
                "effective_concurrency": _effective_concurrency(current_round),
                "requests": public_round_requests,
            }
        )

    profile = _profile(config, corpus)
    implementation = {
        "offload_exec_transport": config.offload_exec_transport,
        "direct_mailbox_output": config.direct_mailbox_output,
        "unified_md_fast_key": config.unified_md_fast_key,
    }
    starts = [float(observation["request_started_at"]) for observation in observations]
    eof_times = [float(observation["eof_at"]) for observation in observations]
    duration_seconds = max(eof_times) - min(starts)
    total_output_tokens = sum(
        int(observation["completion_tokens"]) for observation in observations
    )
    shape_payload = [request["shape"] for request in public_requests]
    requests_by_key = {
        (int(request["conversation_index"]), int(request["round"])): request
        for request in public_requests
    }
    transitions = []
    for request in public_requests:
        round_index = int(request["round"])
        if round_index == 1:
            continue
        conversation_index = int(request["conversation_index"])
        previous = requests_by_key[(conversation_index, round_index - 1)]
        cache = request["cache_validation"]
        assert isinstance(cache, Mapping)
        transitions.append(
            {
                "conversation_index": conversation_index,
                "from_round": round_index - 1,
                "to_round": round_index,
                "previous_prompt_tokens": previous["prompt_tokens"],
                "materialized_history_tokens": cache.get(
                    "materialized_history_tokens"
                ),
                "expected_cached_tokens": cache["expected_cached_tokens"],
                "decode_derived_hit_tokens": cache[
                    "decode_derived_hit_tokens"
                ],
                "actual_cached_tokens": cache.get("actual_cached_tokens"),
            }
        )
    cache_gate = (
        "passed" if config.architecture == "pap" else "requires_external_validation"
    )
    return {
        "schema_version": 1,
        "metric_definition": "last_output_token_v2",
        "profile": profile,
        "profile_fingerprint": profile_fingerprint(profile),
        "architecture": config.architecture,
        "topology": _topology_metadata(config.architecture, config.topology),
        "hardware_signature": config.hardware_signature,
        "git_commit": config.git_commit,
        "git_tracked_worktree_dirty": config.git_tracked_worktree_dirty,
        "implementation": implementation,
        "implementation_fingerprint": profile_fingerprint(implementation),
        "conversation_identity_digests": [
            {
                "conversation_index": identity.index,
                "conversation_id_digest": _digest_text(identity.conversation_id),
                "cache_salt_digest": _digest_text(identity.cache_salt),
            }
            for identity in identities
        ],
        "workload_shape_digest": profile_fingerprint({"requests": shape_payload}),
        "requests": public_requests,
        "rounds": round_results,
        "overall": {
            "completed_requests": len(observations),
            "failed_requests": 0,
            "duration_seconds": duration_seconds,
            "request_throughput": (
                len(observations) / duration_seconds if duration_seconds > 0 else 0.0
            ),
            "output_throughput": (
                total_output_tokens / duration_seconds
                if duration_seconds > 0
                else 0.0
            ),
            "metrics": _request_metric_summary(observations),
            "effective_concurrency": _effective_concurrency(observations),
        },
        "cache_validation": {
            "status": cache_gate,
            "request_count": len(observations),
            "transition_count": config.active_conversations * (config.rounds - 1),
            "transitions": transitions,
        },
        "validity": {"status": "passed", "cache_gate": cache_gate},
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    temporary.replace(path)


def run_load(config: LoadConfig) -> dict[str, object]:
    """Load local dependencies and execute the concurrent workload."""

    import httpx
    from transformers import AutoTokenizer

    validate_config(config)
    corpus = Path(config.corpus_path).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    with ExitStack() as stack:
        clients = [
            stack.enter_context(
                httpx.Client(
                    timeout=config.request_timeout_seconds,
                    trust_env=False,
                )
            )
            for _ in range(config.active_conversations)
        ]
        return asyncio.run(
            execute_load(
                config,
                tokenizer=tokenizer,
                corpus=corpus,
                clients=clients,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--architecture", choices=("pap", "pd"), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--conversation-id-prefix", required=True)
    parser.add_argument("--cache-salt-prefix", required=True)
    parser.add_argument("--hardware-signature", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument(
        "--git-tracked-worktree-dirty",
        choices=("0", "1"),
        required=True,
    )
    parser.add_argument("--offload-exec-transport", required=True)
    parser.add_argument(
        "--direct-mailbox-output",
        choices=("0", "1"),
        required=True,
    )
    parser.add_argument(
        "--unified-md-fast-key",
        choices=("0", "1"),
        required=True,
    )
    parser.add_argument("--document-tokens", type=int, default=16000)
    parser.add_argument("--append-tokens", type=int, default=120)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--active-conversations", type=int, default=4)
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> LoadConfig:
    return LoadConfig(
        base_url=args.base_url,
        model=args.model,
        corpus_path=args.corpus,
        result_path=args.result,
        architecture=args.architecture,
        topology=args.topology,
        conversation_id_prefix=args.conversation_id_prefix,
        cache_salt_prefix=args.cache_salt_prefix,
        hardware_signature=args.hardware_signature,
        git_commit=args.git_commit,
        git_tracked_worktree_dirty=args.git_tracked_worktree_dirty == "1",
        offload_exec_transport=args.offload_exec_transport,
        direct_mailbox_output=args.direct_mailbox_output == "1",
        unified_md_fast_key=args.unified_md_fast_key == "1",
        document_tokens=args.document_tokens,
        append_tokens=args.append_tokens,
        output_tokens=args.output_tokens,
        rounds=args.rounds,
        active_conversations=args.active_conversations,
        request_rate=args.request_rate,
        block_size=args.block_size,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        request_timeout_seconds=args.request_timeout_seconds,
    )


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    result_path = Path(config.result_path)
    try:
        result = run_load(config)
    except Exception as exc:
        _atomic_write_json(
            result_path,
            {
                "schema_version": 1,
                "metric_definition": "last_output_token_v2",
                "profile_id": build_profile_id(config),
                "architecture": config.architecture,
                "git_commit": config.git_commit,
                "git_tracked_worktree_dirty": config.git_tracked_worktree_dirty,
                "validity": {"status": "failed"},
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _atomic_write_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
