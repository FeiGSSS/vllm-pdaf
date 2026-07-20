"""Generate an AIPerf multi-turn workload shared by PAP and PD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INITIAL_INSTRUCTION = (
    "Read the following document and summarize its main argument.\n\n"
)
FOLLOWUP_INSTRUCTION = (
    "Using this additional passage, refine the summary and identify one "
    "changed conclusion.\n\n"
)


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


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
        else (
            tool_time_ms
            if turn_index % tool_every == 0
            else think_time_ms
        )
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
    think_time_ms: int = 0,
    tool_time_ms: int = 0,
    tool_every: int = 3,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build deterministic conversations with long per-turn user deltas."""

    if min(sessions, turns, document_tokens, append_tokens, output_tokens) <= 0:
        raise ValueError("all workload dimensions must be positive")

    corpus_token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    required_tokens = document_tokens + (turns - 1) * append_tokens
    if len(corpus_token_ids) < required_tokens:
        raise ValueError(
            "corpus is too short for requested workload: "
            f"{len(corpus_token_ids)} < {required_tokens}"
        )

    initial_text = INITIAL_INSTRUCTION + _decode_tokens(
        tokenizer,
        corpus_token_ids[:document_tokens],
    )
    followup_texts = [
        FOLLOWUP_INSTRUCTION
        + _decode_tokens(
            tokenizer,
            corpus_token_ids[
                document_tokens + index * append_tokens :
                document_tokens + (index + 1) * append_tokens
            ],
        )
        for index in range(turns - 1)
    ]
    delay_schedule = build_delay_schedule(
        turns,
        think_time_ms=think_time_ms,
        tool_time_ms=tool_time_ms,
        tool_every=tool_every,
    )

    records: list[dict[str, object]] = []
    for session_index in range(sessions):
        session_id = f"{session_prefix}-{session_index:03d}"
        extra = {
            "cache_salt": session_id,
            "ignore_eos": True,
            "min_tokens": output_tokens,
            "seed": 0,
            "temperature": 0,
        }
        turn_texts = [initial_text, *followup_texts]
        turn_records = [
            {
                "text": text,
                "role": "user",
                "output_length": output_tokens,
                "delay": delay_schedule[turn_index],
                "extra": extra,
            }
            for turn_index, text in enumerate(turn_texts)
        ]
        records.append(
            {
                "session_id": session_id,
                "turns": turn_records,
            }
        )

    followup_counts = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in followup_texts
    ]
    actual_counts = {
        "initial_user_text_tokens": len(
            tokenizer.encode(initial_text, add_special_tokens=False)
        ),
        "min_followup_user_text_tokens": min(followup_counts, default=0),
        "max_followup_user_text_tokens": max(followup_counts, default=0),
    }
    return records, actual_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--document-tokens", type=int, default=8192)
    parser.add_argument("--append-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
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
    records, actual_counts = build_records(
        tokenizer,
        corpus,
        sessions=args.sessions,
        turns=args.turns,
        document_tokens=args.document_tokens,
        append_tokens=args.append_tokens,
        output_tokens=args.output_tokens,
        session_prefix=args.session_prefix,
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
        "schema_version": 1,
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
        "actual_text_token_counts": actual_counts,
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
