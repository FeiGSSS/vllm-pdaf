"""Generate an AIPerf multi-turn workload shared by PAP and PD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INITIAL_INSTRUCTION = "Read the following document and summarize its main argument.\n\n"
FOLLOWUP_INSTRUCTION = (
    "Using this additional passage, refine the summary and identify one "
    "changed conclusion.\n\n"
)


@dataclass(frozen=True)
class TokenLengthDistribution:
    """AIPerf-compatible log-normal token-length distribution."""

    mean: int
    median: int | None = None
    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        median = self.effective_median
        if min(self.mean, median, self.minimum) <= 0:
            raise ValueError("length distribution values must be positive")
        if median > self.mean:
            raise ValueError("length distribution median must not exceed mean")
        if self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("length distribution maximum is below minimum")

    @property
    def effective_median(self) -> int:
        return self.mean if self.median is None else self.median

    @property
    def randomized(self) -> bool:
        return self.effective_median < self.mean

    def sample(self, rng: random.Random) -> int:
        """Sample using AIPerf's mean/median log-normal parameterization."""

        median = self.effective_median
        if median == self.mean:
            value = float(self.mean)
        else:
            sigma = math.sqrt(2.0 * math.log(self.mean / median))
            value = math.exp(rng.gauss(math.log(median), sigma))
        value = max(float(self.minimum), value)
        if self.maximum is not None:
            value = min(float(self.maximum), value)
        return max(1, math.ceil(value))

    def manifest_config(self) -> dict[str, object]:
        return {
            "type": "lognormal" if self.randomized else "fixed",
            "mean": self.mean,
            "median": self.effective_median,
            "min": self.minimum,
            "max": self.maximum,
        }


def _derived_rng(seed: int, stream: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("cannot summarize an empty length sample")
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "stddev": round(statistics.pstdev(values), 3),
        "min": min(values),
        "p90": round(_percentile(values, 0.90), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "max": max(values),
    }


def _sample_lengths(
    distribution: TokenLengthDistribution,
    count: int,
    *,
    seed: int,
    stream: str,
    mean_tolerance: float,
) -> tuple[list[int], dict[str, object]]:
    rng = _derived_rng(seed, stream)
    values = [distribution.sample(rng) for _ in range(count)]
    sampled = _sample_summary(values)
    if distribution.randomized and count > 1 and min(values) == max(values):
        raise ValueError(f"{stream} distribution produced no variability")
    relative_error = abs(float(sampled["mean"]) - distribution.mean) / (
        distribution.mean
    )
    if relative_error > mean_tolerance:
        raise ValueError(
            f"{stream} sampled mean differs from its target by "
            f"{relative_error:.1%}, above {mean_tolerance:.1%}"
        )
    return values, {
        "configured": distribution.manifest_config(),
        "sampled": sampled,
        "sampled_mean_relative_error": round(relative_error, 6),
    }


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _estimate_input_sequence_lengths(
    tokenizer: Any,
    turn_texts: list[str],
    output_lengths: list[int],
) -> list[int]:
    """Estimate each full-history prompt with empty assistant placeholders."""

    messages: list[dict[str, str]] = []
    prior_output_tokens = 0
    estimates = []
    for text, output_length in zip(turn_texts, output_lengths, strict=True):
        messages.append({"role": "user", "content": text})
        if not hasattr(tokenizer, "apply_chat_template"):
            user_tokens = sum(
                len(tokenizer.encode(item["content"], add_special_tokens=False))
                for item in messages
                if item["role"] == "user"
            )
            estimates.append(
                user_tokens + prior_output_tokens + 64 + 16 * len(messages)
            )
            messages.append({"role": "assistant", "content": ""})
            prior_output_tokens += output_length
            continue
        tokenized_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(tokenized_prompt, Mapping):
            tokenized_prompt = tokenized_prompt["input_ids"]
        if (
            isinstance(tokenized_prompt, list)
            and tokenized_prompt
            and isinstance(tokenized_prompt[0], list)
        ):
            tokenized_prompt = tokenized_prompt[0]
        shape = getattr(tokenized_prompt, "shape", ())
        token_count = int(shape[-1]) if shape else len(tokenized_prompt)
        estimates.append(token_count + prior_output_tokens)
        messages.append({"role": "assistant", "content": ""})
        prior_output_tokens += output_length
    return estimates


def build_delay_schedule(
    turns: int,
    *,
    think_time_ms: int,
    tool_time_ms: int,
    tool_every: int,
) -> list[int]:
    """Return per-turn delays; the first turn never waits."""

    if turns <= 0:
        raise ValueError("turns must be positive")
    if min(think_time_ms, tool_time_ms) < 0:
        raise ValueError("think and tool delays must be non-negative")
    if tool_every <= 0:
        raise ValueError("tool_every must be positive")

    return [
        0
        if turn_index == 0
        else (tool_time_ms if turn_index % tool_every == 0 else think_time_ms)
        for turn_index in range(turns)
    ]


def build_records(
    tokenizer: Any,
    corpus: str,
    *,
    sessions: int,
    turns: int,
    document_tokens: int,
    append_tokens: int,
    output_tokens: int,
    session_prefix: str,
    document_tokens_median: int | None = None,
    document_tokens_min: int = 1,
    document_tokens_max: int | None = None,
    append_tokens_median: int | None = None,
    append_tokens_min: int = 1,
    append_tokens_max: int | None = None,
    output_tokens_median: int | None = None,
    output_tokens_min: int = 1,
    output_tokens_max: int | None = None,
    random_seed: int = 42,
    sampled_mean_tolerance: float = 0.10,
    max_model_len: int | None = None,
    think_time_ms: int = 0,
    tool_time_ms: int = 0,
    tool_every: int = 3,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build reproducible conversations with sampled per-turn lengths."""

    if min(sessions, turns, document_tokens, append_tokens, output_tokens) <= 0:
        raise ValueError("all workload dimensions must be positive")

    if not 0 < sampled_mean_tolerance < 1:
        raise ValueError("sampled mean tolerance must be between zero and one")

    distributions = {
        "document_content_tokens": TokenLengthDistribution(
            document_tokens,
            document_tokens_median,
            document_tokens_min,
            document_tokens_max,
        ),
        "append_content_tokens": TokenLengthDistribution(
            append_tokens,
            append_tokens_median,
            append_tokens_min,
            append_tokens_max,
        ),
        "output_tokens": TokenLengthDistribution(
            output_tokens,
            output_tokens_median,
            output_tokens_min,
            output_tokens_max,
        ),
    }
    document_lengths, document_summary = _sample_lengths(
        distributions["document_content_tokens"],
        sessions,
        seed=random_seed,
        stream="document_content_tokens",
        mean_tolerance=sampled_mean_tolerance,
    )
    append_lengths, append_summary = _sample_lengths(
        distributions["append_content_tokens"],
        sessions * (turns - 1),
        seed=random_seed,
        stream="append_content_tokens",
        mean_tolerance=sampled_mean_tolerance,
    )
    output_lengths, output_summary = _sample_lengths(
        distributions["output_tokens"],
        sessions * turns,
        seed=random_seed,
        stream="output_tokens",
        mean_tolerance=sampled_mean_tolerance,
    )

    required_by_session = [
        document_lengths[index]
        + sum(append_lengths[index * (turns - 1) : (index + 1) * (turns - 1)])
        for index in range(sessions)
    ]
    corpus_token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    required_tokens = max(required_by_session)
    if len(corpus_token_ids) < required_tokens:
        raise ValueError(
            "corpus is too short for requested workload: "
            f"{len(corpus_token_ids)} < {required_tokens}"
        )
    delay_schedule = build_delay_schedule(
        turns,
        think_time_ms=think_time_ms,
        tool_time_ms=tool_time_ms,
        tool_every=tool_every,
    )

    records: list[dict[str, object]] = []
    initial_text_counts: list[int] = []
    followup_text_counts: list[int] = []
    input_sequence_estimates: list[int] = []
    input_sequence_estimates_by_turn = [[] for _ in range(turns)]
    request_token_budget_estimates: list[int] = []
    for session_index in range(sessions):
        session_id = f"{session_prefix}-{session_index:03d}"
        append_start = session_index * (turns - 1)
        session_append_lengths = append_lengths[append_start : append_start + turns - 1]
        output_start = session_index * turns
        session_output_lengths = output_lengths[output_start : output_start + turns]
        document_length = document_lengths[session_index]
        initial_text = INITIAL_INSTRUCTION + _decode_tokens(
            tokenizer,
            corpus_token_ids[:document_length],
        )
        token_offset = document_length
        followup_texts = []
        for append_length in session_append_lengths:
            followup_texts.append(
                FOLLOWUP_INSTRUCTION
                + _decode_tokens(
                    tokenizer,
                    corpus_token_ids[token_offset : token_offset + append_length],
                )
            )
            token_offset += append_length
        turn_texts = [initial_text, *followup_texts]
        text_counts = [
            len(tokenizer.encode(text, add_special_tokens=False)) for text in turn_texts
        ]
        initial_text_counts.append(text_counts[0])
        followup_text_counts.extend(text_counts[1:])

        session_input_estimates = _estimate_input_sequence_lengths(
            tokenizer,
            turn_texts,
            session_output_lengths,
        )
        input_sequence_estimates.extend(session_input_estimates)
        for turn_index, estimate in enumerate(session_input_estimates):
            input_sequence_estimates_by_turn[turn_index].append(estimate)

        turn_records = []
        for turn_index, (text, sampled_output_tokens) in enumerate(
            zip(
                turn_texts,
                session_output_lengths,
                strict=True,
            )
        ):
            request_token_budget_estimates.append(
                session_input_estimates[turn_index] + sampled_output_tokens
            )
            turn_records.append(
                {
                    "text": text,
                    "role": "user",
                    "output_length": sampled_output_tokens,
                    "delay": delay_schedule[turn_index],
                    "extra": {
                        "cache_salt": session_id,
                        "ignore_eos": True,
                        "min_tokens": sampled_output_tokens,
                        "seed": 0,
                        "temperature": 0,
                    },
                }
            )
        records.append(
            {
                "session_id": session_id,
                "turns": turn_records,
            }
        )

    maximum_budget = max(request_token_budget_estimates)
    if max_model_len is not None and maximum_budget > max_model_len:
        raise ValueError(
            "sampled conversation exceeds the model context budget: "
            f"{maximum_budget} > {max_model_len}"
        )
    generation_summary = {
        "distribution_semantics": "aiperf_lognormal_mean_median",
        "random_seed": random_seed,
        "sampled_mean_tolerance": sampled_mean_tolerance,
        "length_distributions": {
            "document_content_tokens": document_summary,
            "append_content_tokens": append_summary,
            "output_tokens": output_summary,
        },
        "actual_text_token_counts": {
            "initial_user_text_tokens": _sample_summary(initial_text_counts),
            "followup_user_text_tokens": _sample_summary(followup_text_counts),
        },
        "estimated_input_sequence_tokens": {
            "all_requests": _sample_summary(input_sequence_estimates),
            "by_turn": [
                {"turn_index": turn_index, **_sample_summary(values)}
                for turn_index, values in enumerate(input_sequence_estimates_by_turn)
            ],
            "method": "tokenizer_chat_template_with_empty_assistant_placeholders",
        },
        "context_budget": {
            "estimate_includes_current_output": True,
            "max_estimated_request_tokens": maximum_budget,
            "max_model_len": max_model_len,
            "headroom_tokens": (
                max_model_len - maximum_budget if max_model_len is not None else None
            ),
        },
        "validation": {
            "status": "passed",
            "all_requested_lengths_within_bounds": True,
            "all_randomized_dimensions_non_degenerate": True,
            "sampled_means_within_tolerance": True,
            "context_budget_within_limit": (
                max_model_len is None or maximum_budget <= max_model_len
            ),
        },
    }
    return records, generation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--document-tokens", type=int, default=8192)
    parser.add_argument("--document-tokens-median", type=int)
    parser.add_argument("--document-tokens-min", type=int, default=1)
    parser.add_argument("--document-tokens-max", type=int)
    parser.add_argument("--append-tokens", type=int, default=512)
    parser.add_argument("--append-tokens-median", type=int)
    parser.add_argument("--append-tokens-min", type=int, default=1)
    parser.add_argument("--append-tokens-max", type=int)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--output-tokens-median", type=int)
    parser.add_argument("--output-tokens-min", type=int, default=1)
    parser.add_argument("--output-tokens-max", type=int)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sampled-mean-tolerance", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--session-prefix", default="pap-aiperf-session")
    parser.add_argument("--think-time-ms", type=int, default=0)
    parser.add_argument("--tool-time-ms", type=int, default=0)
    parser.add_argument("--tool-every", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    corpus = args.corpus.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    records, generation_summary = build_records(
        tokenizer,
        corpus,
        sessions=args.sessions,
        turns=args.turns,
        document_tokens=args.document_tokens,
        append_tokens=args.append_tokens,
        output_tokens=args.output_tokens,
        session_prefix=args.session_prefix,
        document_tokens_median=args.document_tokens_median,
        document_tokens_min=args.document_tokens_min,
        document_tokens_max=args.document_tokens_max,
        append_tokens_median=args.append_tokens_median,
        append_tokens_min=args.append_tokens_min,
        append_tokens_max=args.append_tokens_max,
        output_tokens_median=args.output_tokens_median,
        output_tokens_min=args.output_tokens_min,
        output_tokens_max=args.output_tokens_max,
        random_seed=args.random_seed,
        sampled_mean_tolerance=args.sampled_mean_tolerance,
        max_model_len=args.max_model_len,
        think_time_ms=args.think_time_ms,
        tool_time_ms=args.tool_time_ms,
        tool_every=args.tool_every,
    )
    delay_schedule = build_delay_schedule(
        args.turns,
        think_time_ms=args.think_time_ms,
        tool_time_ms=args.tool_time_ms,
        tool_every=args.tool_every,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode()
    args.output.write_bytes(encoded)

    manifest = {
        "schema_version": 2,
        "format": "aiperf_multi_turn_jsonl",
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "total_requests": args.sessions * args.turns,
        "requested_document_tokens": args.document_tokens,
        "requested_append_tokens": args.append_tokens,
        "output_tokens": args.output_tokens,
        "delay_profile": {
            "think_time_ms": args.think_time_ms,
            "tool_time_ms": args.tool_time_ms,
            "tool_every": args.tool_every,
            "schedule_ms": delay_schedule,
            "total_delay_per_session_ms": sum(delay_schedule),
        },
        **generation_summary,
        "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
