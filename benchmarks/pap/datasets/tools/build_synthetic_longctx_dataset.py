# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build nested synthetic long-context multi-turn datasets for AIPerf."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class SessionSummary:
    """Compact description of one generated session."""

    rank: int
    session_id: str
    turns: int
    input_lengths: list[int]
    output_lengths: list[int]
    total_input_tokens: int
    total_output_tokens: int
    final_context_upper_bound: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
        "sum": sum(values),
    }


def _split_input(total_tokens: int, turns: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total_tokens), turns - 1))
    boundaries = [0, *cuts, total_tokens]
    return [end - start for start, end in pairwise(boundaries)]


def _sample_outputs(
    turns: int,
    minimum: int,
    maximum: int,
    max_total: int,
    rng: random.Random,
) -> list[int]:
    while True:
        outputs = [rng.randint(minimum, maximum) for _ in range(turns)]
        if sum(outputs) <= max_total:
            return outputs


def _generate_sessions(
    count: int,
    *,
    seed: int,
    min_turns: int,
    max_turns: int,
    total_input_tokens: int,
    min_output_tokens: int,
    max_output_tokens: int,
    max_total_output_tokens: int,
) -> tuple[list[list[dict[str, Any]]], list[SessionSummary]]:
    rng = random.Random(seed)
    sessions = []
    summaries = []

    for session_index in range(count):
        turns = rng.randint(min_turns, max_turns)
        inputs = _split_input(total_input_tokens, turns, rng)
        outputs = _sample_outputs(
            turns,
            min_output_tokens,
            max_output_tokens,
            max_total_output_tokens,
            rng,
        )
        session_id = f"pap-synthetic-longctx-{session_index + 1:04d}"
        records = []
        for input_length, output_length in zip(inputs, outputs, strict=True):
            records.append(
                {
                    "session_id": session_id,
                    "input_length": input_length,
                    "output_length": output_length,
                    "extra": {
                        "ignore_eos": True,
                        "min_tokens": output_length,
                        "seed": 0,
                        "temperature": 0,
                    },
                }
            )
        sessions.append(records)
        summaries.append(
            SessionSummary(
                rank=session_index + 1,
                session_id=session_id,
                turns=turns,
                input_lengths=inputs,
                output_lengths=outputs,
                total_input_tokens=sum(inputs),
                total_output_tokens=sum(outputs),
                final_context_upper_bound=sum(inputs) + sum(outputs),
            )
        )
    return sessions, summaries


def _write_subset(
    output_root: Path,
    sessions: list[list[dict[str, Any]]],
    summaries: list[SessionSummary],
    size: int,
) -> dict[str, Any]:
    dataset_path = output_root / f"synthetic_longctx_s{size}.jsonl"
    session_ids_path = output_root / f"synthetic_longctx_s{size}.sessions.txt"
    selected_sessions = sessions[:size]
    selected_summaries = summaries[:size]

    with dataset_path.open("w", encoding="utf-8") as output:
        for session in selected_sessions:
            for record in session:
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
    session_ids_path.write_text(
        "".join(f"{summary.session_id}\n" for summary in selected_summaries),
        encoding="utf-8",
    )

    turns = [summary.turns for summary in selected_summaries]
    input_chunks = [
        value for summary in selected_summaries for value in summary.input_lengths
    ]
    output_lengths = [
        value for summary in selected_summaries for value in summary.output_lengths
    ]
    final_contexts = [
        summary.final_context_upper_bound for summary in selected_summaries
    ]
    return {
        "sessions": size,
        "requests": sum(turns),
        "dataset_file": dataset_path.name,
        "dataset_sha256": _sha256(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "session_ids_file": session_ids_path.name,
        "session_ids_sha256": _sha256(session_ids_path),
        "turns_per_session": _summary(turns),
        "incremental_input_tokens": _summary(input_chunks),
        "output_tokens": _summary(output_lengths),
        "total_input_tokens_per_session": _summary(
            [summary.total_input_tokens for summary in selected_summaries]
        ),
        "total_output_tokens_per_session": _summary(
            [summary.total_output_tokens for summary in selected_summaries]
        ),
        "final_context_upper_bound": _summary(final_contexts),
        "delay_fields": 0,
    }


def build(args: argparse.Namespace) -> None:
    sizes = sorted(set(args.sizes))
    sessions, summaries = _generate_sessions(
        sizes[-1],
        seed=args.seed,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        total_input_tokens=args.total_input_tokens,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        max_total_output_tokens=args.max_total_output_tokens,
    )

    output_root = args.output.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    subsets = {
        str(size): _write_subset(
            output_root,
            sessions,
            summaries,
            size,
        )
        for size in sizes
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "policy": {
            "seed": args.seed,
            "nested_subsets": True,
            "sizes": sizes,
            "turn_distribution": (
                f"uniform integer [{args.min_turns}, {args.max_turns}]"
            ),
            "input_split": (
                f"{args.total_input_tokens} incremental tokens split by "
                "uniformly sampled integer cut points"
            ),
            "output_distribution": (
                f"uniform integer [{args.min_output_tokens}, "
                f"{args.max_output_tokens}] conditioned on per-session total "
                f"<= {args.max_total_output_tokens}"
            ),
            "delay": "omitted",
            "hash_ids": "omitted; AIPerf synthesizes independent prompts",
            "generation": (
                "temperature=0, seed=0, ignore_eos=true, and exact output "
                "length locked with min_tokens"
            ),
        },
        "subsets": subsets,
        "sessions": [asdict(summary) for summary in summaries],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote nested synthetic datasets {sizes} under {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--total-input-tokens", type=int, default=28_000)
    parser.add_argument("--min-output-tokens", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=200)
    parser.add_argument("--max-total-output-tokens", type=int, default=2_800)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) <= 0:
        parser.error("--sizes must contain positive integers")
    if not 0 < args.min_turns <= args.max_turns:
        parser.error("--min-turns must be positive and <= --max-turns")
    if args.max_turns > args.total_input_tokens:
        parser.error("--max-turns cannot exceed --total-input-tokens")
    if not 0 < args.min_output_tokens <= args.max_output_tokens:
        parser.error("--min-output-tokens must be positive and <= --max-output-tokens")
    if args.max_total_output_tokens < (args.max_turns * args.min_output_tokens):
        parser.error(
            "--max-total-output-tokens cannot accommodate the maximum turn count"
        )
    return args


if __name__ == "__main__":
    build(parse_args())
