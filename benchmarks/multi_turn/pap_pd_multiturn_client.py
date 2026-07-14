"""Fixed two-turn streaming client for the PAP/PD north-star workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_ID = "qwen3_8b_chat_16k_2turn_o256_c1_v1"
DEFAULT_DOCUMENT_TOKENS = 16000
DEFAULT_APPEND_TOKENS = 120
DEFAULT_OUTPUT_TOKENS = 256
DEFAULT_BLOCK_SIZE = 16


@dataclass(frozen=True)
class NorthStarConfig:
    """Frozen workload and engine metadata for one repetition."""

    base_url: str
    model: str
    corpus_path: str
    result_path: str
    architecture: str
    topology: str
    conversation_id: str
    cache_salt: str
    hardware_signature: str
    git_commit: str
    git_tracked_worktree_dirty: bool
    offload_exec_transport: str
    direct_mailbox_output: bool
    unified_md_fast_key: bool
    prefill_ipc_profile: bool = False
    kv_handoff_mode: str = "layer_descriptor"
    document_tokens: int = DEFAULT_DOCUMENT_TOKENS
    append_tokens: int = DEFAULT_APPEND_TOKENS
    output_tokens: int = DEFAULT_OUTPUT_TOKENS
    block_size: int = DEFAULT_BLOCK_SIZE
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 20000
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 2


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


def build_workload(
    tokenizer: Any,
    corpus: str,
    *,
    document_tokens: int = DEFAULT_DOCUMENT_TOKENS,
    append_tokens: int = DEFAULT_APPEND_TOKENS,
) -> tuple[list[dict[str, str]], str]:
    """Build the fixed first turn and non-overlapping second-turn append."""
    if document_tokens <= 0 or append_tokens <= 0:
        raise ValueError("document_tokens and append_tokens must be positive")
    corpus_token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    required_tokens = document_tokens + append_tokens
    if len(corpus_token_ids) < required_tokens:
        raise ValueError(
            "corpus is too short for north-star workload: "
            f"{len(corpus_token_ids)} < {required_tokens}"
        )
    document = tokenizer.decode(corpus_token_ids[:document_tokens])
    append_text = tokenizer.decode(corpus_token_ids[document_tokens:required_tokens])
    first_messages = [
        {
            "role": "user",
            "content": (
                "Read the following document and summarize its main argument.\n\n"
                f"{document}"
            ),
        }
    ]
    return first_messages, append_text


def build_chat_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    conversation_id: str,
    cache_salt: str,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> dict[str, object]:
    """Build one frozen north-star Chat Completions request."""
    if not conversation_id:
        raise ValueError("conversation_id must not be empty")
    if not cache_salt:
        raise ValueError("cache_salt must not be empty")
    if output_tokens <= 0:
        raise ValueError(f"output_tokens must be positive: {output_tokens}")
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "conversation_id": conversation_id,
        "cache_salt": cache_salt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def _required_prefill_value(
    prefill: Mapping[str, int | None],
    field: str,
) -> int:
    value = prefill.get(field)
    if not isinstance(value, int):
        raise ValueError(f"missing PAP Prefill {field}")
    return value


def validate_pap_cache_reuse(
    *,
    first_prefill: Mapping[str, int | None],
    second_prefill: Mapping[str, int | None],
    prefix_metrics: Mapping[str, int],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, int | str]:
    """Validate actual PAP cached tokens against the retokenized LCP."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive: {block_size}")
    first_cached = _required_prefill_value(first_prefill, "cached_tokens")
    if first_cached != 0:
        raise ValueError(
            f"first north-star request unexpectedly hit {first_cached} tokens"
        )
    actual_cached = _required_prefill_value(second_prefill, "cached_tokens")
    expected_cached = int(prefix_metrics["expected_cached_tokens"])
    if actual_cached != expected_cached:
        raise ValueError(
            "PAP second-turn cached tokens differ from retokenized LCP: "
            f"{actual_cached} != {expected_cached}"
        )
    decode_derived = int(prefix_metrics["decode_derived_hit_tokens"])
    if decode_derived < block_size:
        raise ValueError(
            "PAP second turn has no full Decode-derived cache block: "
            f"{decode_derived} < {block_size}"
        )
    return {
        "status": "passed",
        **{str(key): int(value) for key, value in prefix_metrics.items()},
        "actual_cached_tokens": actual_cached,
    }


def consume_sse_lines(
    lines: Iterable[str],
    *,
    started_at: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Consume a vLLM SSE response through EOF and return token-level metrics."""
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
        raise ValueError(
            "HTTP stream EOF preceded the final output token timestamp"
        )
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
        "tpot_ms": calculate_tpot_ms(
            latency_ms,
            ttft_ms,
            completion_tokens,
        ),
    }


def _token_digest(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode()).hexdigest()


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
    if architecture == "pd":
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
    raise ValueError(f"unsupported architecture: {architecture}")


def _profile(config: NorthStarConfig, corpus: str) -> dict[str, object]:
    return {
        "profile_id": PROFILE_ID,
        "model": config.model,
        "corpus_path": config.corpus_path,
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "api": "/v1/chat/completions",
        "document_tokens": config.document_tokens,
        "append_tokens": config.append_tokens,
        "output_tokens_per_round": config.output_tokens,
        "rounds": 2,
        "active_conversations": 1,
        "enable_thinking": True,
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


def _validate_turn(
    observation: Mapping[str, object],
    *,
    expected_output_tokens: int,
) -> None:
    if observation.get("finish_reason") != "length":
        raise ValueError(
            "north-star turn did not finish by length: "
            f"{observation.get('finish_reason')}"
        )
    if observation.get("completion_tokens") != expected_output_tokens:
        raise ValueError(
            "north-star completion length mismatch: "
            f"{observation.get('completion_tokens')} != {expected_output_tokens}"
        )
    if not observation.get("assistant_text"):
        raise ValueError("north-star assistant text is empty")


def _run_stream_turn(
    client: Any,
    *,
    config: NorthStarConfig,
    messages: Sequence[Mapping[str, str]],
    round_index: int,
    clock: Callable[[], float],
) -> tuple[dict[str, object], dict[str, int | None]]:
    request_id = f"{config.conversation_id}-turn-{round_index}"
    payload = build_chat_payload(
        model=config.model,
        messages=messages,
        conversation_id=config.conversation_id,
        cache_salt=config.cache_salt,
        output_tokens=config.output_tokens,
    )
    started_at = clock()
    with client.stream(
        "POST",
        f"{config.base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        headers={"X-Request-Id": request_id},
    ) as response:
        response.raise_for_status()
        prefill = parse_prefill_headers(response.headers)
        observation = consume_sse_lines(
            response.iter_lines(),
            started_at=started_at,
            clock=clock,
        )
    _validate_turn(
        observation,
        expected_output_tokens=config.output_tokens,
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
    observation["request_id"] = request_id
    return observation, prefill


def _round_summary(
    round_index: int,
    observation: Mapping[str, object],
    prefill: Mapping[str, int | None],
) -> dict[str, object]:
    prompt_ids = observation["prompt_token_ids"]
    output_ids = observation["output_token_ids"]
    assert isinstance(prompt_ids, list)
    assert isinstance(output_ids, list)
    assistant_text = observation["assistant_text"]
    assert isinstance(assistant_text, str)
    return {
        "round": round_index,
        "request_id": observation["request_id"],
        "prompt_tokens": observation["prompt_tokens"],
        "completion_tokens": observation["completion_tokens"],
        "ttft_ms": observation["ttft_ms"],
        "tpot_ms": observation["tpot_ms"],
        "latency_ms": observation["latency_ms"],
        "eof_latency_ms": observation["eof_latency_ms"],
        "post_token_stream_ms": observation["post_token_stream_ms"],
        "finish_reason": observation["finish_reason"],
        "saw_done": observation["saw_done"],
        "prompt_token_digest": _token_digest(prompt_ids),
        "output_token_digest": _token_digest(output_ids),
        "assistant_text_digest": _text_digest(assistant_text),
        "prefill": dict(prefill),
    }


def execute_two_turn(
    config: NorthStarConfig,
    *,
    tokenizer: Any,
    corpus: str,
    client: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    """Execute the fixed two-turn workload with injected dependencies."""
    first_messages, append_text = build_workload(
        tokenizer,
        corpus,
        document_tokens=config.document_tokens,
        append_tokens=config.append_tokens,
    )
    first, first_prefill = _run_stream_turn(
        client,
        config=config,
        messages=first_messages,
        round_index=1,
        clock=clock,
    )
    first_assistant = first["assistant_text"]
    assert isinstance(first_assistant, str)
    second_messages = [
        *first_messages,
        {"role": "assistant", "content": first_assistant},
        {
            "role": "user",
            "content": (
                "Using this additional passage, refine the summary and identify "
                f"one changed conclusion.\n\n{append_text}"
            ),
        },
    ]
    second, second_prefill = _run_stream_turn(
        client,
        config=config,
        messages=second_messages,
        round_index=2,
        clock=clock,
    )

    first_prompt_ids = first["prompt_token_ids"]
    first_output_ids = first["output_token_ids"]
    second_prompt_ids = second["prompt_token_ids"]
    assert isinstance(first_prompt_ids, list)
    assert isinstance(first_output_ids, list)
    assert isinstance(second_prompt_ids, list)
    prefix_metrics = block_aligned_prefix_metrics(
        first_prompt_ids,
        first_output_ids,
        second_prompt_ids,
        block_size=config.block_size,
    )
    if config.architecture == "pap":
        cache_validation: dict[str, object] = validate_pap_cache_reuse(
            first_prefill=first_prefill,
            second_prefill=second_prefill,
            prefix_metrics=prefix_metrics,
            block_size=config.block_size,
        )
        cache_gate = "passed"
    else:
        cache_validation = {
            "status": "requires_official_log",
            **prefix_metrics,
            "actual_cached_tokens": None,
        }
        cache_gate = "requires_official_log"

    profile = _profile(config, corpus)
    rounds = [
        _round_summary(1, first, first_prefill),
        _round_summary(2, second, second_prefill),
    ]
    implementation = {
        "offload_exec_transport": config.offload_exec_transport,
        "direct_mailbox_output": config.direct_mailbox_output,
        "unified_md_fast_key": config.unified_md_fast_key,
        "prefill_kv_async": config.architecture == "pap",
        "prefill_ipc_profile": config.prefill_ipc_profile,
        "kv_handoff_mode": config.kv_handoff_mode,
    }
    return {
        "schema_version": 2,
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
        "conversation_id_digest": _text_digest(config.conversation_id),
        "rounds": rounds,
        "conversation_latency_ms": (
            float(rounds[0]["eof_latency_ms"])
            + float(rounds[1]["latency_ms"])
        ),
        "conversation_eof_latency_ms": sum(
            float(round_result["eof_latency_ms"]) for round_result in rounds
        ),
        "cache_validation": cache_validation,
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


def run_two_turn(config: NorthStarConfig) -> dict[str, object]:
    """Load local dependencies and execute one north-star repetition."""
    import httpx
    from transformers import AutoTokenizer

    corpus = Path(config.corpus_path).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    with httpx.Client(timeout=600.0, trust_env=False) as client:
        return execute_two_turn(
            config,
            tokenizer=tokenizer,
            corpus=corpus,
            client=client,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed PAP/PD multi-turn north-star workload"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--architecture", choices=("pap", "pd"), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--cache-salt", required=True)
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
    parser.add_argument("--prefill-ipc-profile", choices=("0", "1"), default="0")
    parser.add_argument(
        "--kv-handoff-mode",
        choices=("layer_descriptor", "sealed_manifest"),
        default="layer_descriptor",
    )
    parser.add_argument("--document-tokens", type=int, default=16000)
    parser.add_argument("--append-tokens", type=int, default=120)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> NorthStarConfig:
    return NorthStarConfig(
        base_url=args.base_url,
        model=args.model,
        corpus_path=args.corpus,
        result_path=args.result,
        architecture=args.architecture,
        topology=args.topology,
        conversation_id=args.conversation_id,
        cache_salt=args.cache_salt,
        hardware_signature=args.hardware_signature,
        git_commit=args.git_commit,
        git_tracked_worktree_dirty=args.git_tracked_worktree_dirty == "1",
        offload_exec_transport=args.offload_exec_transport,
        direct_mailbox_output=args.direct_mailbox_output == "1",
        unified_md_fast_key=args.unified_md_fast_key == "1",
        prefill_ipc_profile=args.prefill_ipc_profile == "1",
        kv_handoff_mode=args.kv_handoff_mode,
        document_tokens=args.document_tokens,
        append_tokens=args.append_tokens,
        output_tokens=args.output_tokens,
        block_size=args.block_size,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    result_path = Path(config.result_path)
    try:
        result = run_two_turn(config)
    except Exception as exc:
        _atomic_write_json(
            result_path,
            {
                "schema_version": 2,
                "metric_definition": "last_output_token_v2",
                "architecture": config.architecture,
                "git_commit": config.git_commit,
                "git_tracked_worktree_dirty": (
                    config.git_tracked_worktree_dirty
                ),
                "implementation": {
                    "offload_exec_transport": config.offload_exec_transport,
                    "direct_mailbox_output": config.direct_mailbox_output,
                    "unified_md_fast_key": config.unified_md_fast_key,
                    "prefill_kv_async": config.architecture == "pap",
                    "prefill_ipc_profile": config.prefill_ipc_profile,
                },
                "validity": {"status": "failed"},
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _atomic_write_json(result_path, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
