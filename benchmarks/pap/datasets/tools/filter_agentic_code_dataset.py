# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build a deterministic bounded-turn subset of a Mooncake trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _summary(values: list[int]) -> dict[str, int | float]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "min": min(values),
        "max": max(values),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaled_row(
    row: dict[str, Any],
    *,
    length_divisor: int,
    hash_block_size: int,
) -> dict[str, Any]:
    scaled = dict(row)
    scaled.pop("delay", None)
    for field in ("input_length", "output_length"):
        scaled[field] = max(1, int(scaled[field]) // length_divisor)
    hash_ids = scaled.get("hash_ids")
    if isinstance(hash_ids, list):
        input_blocks = (scaled["input_length"] + hash_block_size - 1) // hash_block_size
        scaled["hash_ids"] = hash_ids[:input_blocks]
    return scaled


def build_subset(
    input_path: Path,
    output_path: Path,
    *,
    sessions: int,
    min_turns: int,
    max_turns: int,
    take_first_turns: int | None,
    length_divisor: int,
    hash_block_size: int,
    seed: int,
    final_context_min: int | None = None,
    final_context_max: int | None = None,
    final_context_strata: int | None = None,
) -> dict[str, Any]:
    """Filter complete sessions and write a deterministic sample."""

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with input_path.open() as file:
        for line in file:
            row = json.loads(line)
            grouped.setdefault(row["session_id"], []).append(row)

    if take_first_turns is None:
        eligible = [
            (session_id, rows)
            for session_id, rows in grouped.items()
            if min_turns <= len(rows) <= max_turns
        ]
    else:
        eligible = [
            (session_id, rows)
            for session_id, rows in grouped.items()
            if len(rows) >= take_first_turns
        ]
    if len(eligible) < sessions:
        raise ValueError(
            f"only {len(eligible)} sessions satisfy the turn bounds; need {sessions}"
        )

    rng = random.Random(seed)
    stratum_counts: list[dict[str, int]] | None = None
    if final_context_strata is None:
        chosen = rng.sample(eligible, sessions)
        sampling = "random_without_replacement"
    else:
        if final_context_min is None or final_context_max is None:
            raise ValueError("stratified sampling requires final context bounds")
        if final_context_max <= final_context_min:
            raise ValueError("final context bounds must be ordered")
        if final_context_strata <= 0:
            raise ValueError("final-context-strata must be positive")
        if sessions % final_context_strata:
            raise ValueError("sessions must be divisible by final-context-strata")

        per_stratum = sessions // final_context_strata
        width = (final_context_max - final_context_min) / final_context_strata
        chosen = []
        stratum_counts = []
        for index in range(final_context_strata):
            lower = round(final_context_min + index * width)
            upper = round(final_context_min + (index + 1) * width)
            candidates = [
                item
                for item in eligible
                if lower
                <= sum(
                    row["input_length"] + row["output_length"]
                    for row in (
                        item[1]
                        if take_first_turns is None
                        else item[1][:take_first_turns]
                    )
                )
                < upper + (index == final_context_strata - 1)
            ]
            if len(candidates) < per_stratum:
                raise ValueError(
                    f"final-context stratum [{lower}, {upper}) has "
                    f"{len(candidates)} candidates; need {per_stratum}"
                )
            chosen.extend(rng.sample(candidates, per_stratum))
            stratum_counts.append(
                {
                    "minimum": lower,
                    "maximum": upper,
                    "eligible": len(candidates),
                    "selected": per_stratum,
                }
            )
        sampling = "equal_count_final_context_strata"

    chosen_ids = {session_id for session_id, _ in chosen}
    selected = [
        (
            session_id,
            [
                _scaled_row(
                    row,
                    length_divisor=length_divisor,
                    hash_block_size=hash_block_size,
                )
                for row in (
                    rows if take_first_turns is None else rows[:take_first_turns]
                )
            ],
        )
        for session_id, rows in grouped.items()
        if session_id in chosen_ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for _, rows in selected:
            for row in rows:
                file.write(json.dumps(row, separators=(",", ":")) + "\n")

    turn_counts = [len(rows) for _, rows in selected]
    initial_contexts = [rows[0]["input_length"] for _, rows in selected]
    input_lengths = [row["input_length"] for _, rows in selected for row in rows]
    output_lengths = [row["output_length"] for _, rows in selected for row in rows]
    prompt_contexts = []
    per_turn = []
    for turn_index in range(max(turn_counts)):
        turn_inputs = [rows[turn_index]["input_length"] for _, rows in selected]
        turn_outputs = [rows[turn_index]["output_length"] for _, rows in selected]
        per_turn.append(
            {
                "turn": turn_index + 1,
                "input_tokens": _summary(turn_inputs),
                "output_tokens": _summary(turn_outputs),
            }
        )
    for _, rows in selected:
        context = 0
        for row in rows:
            context += row["input_length"]
            prompt_contexts.append(context)
            context += row["output_length"]
    final_contexts = [
        sum(row["input_length"] + row["output_length"] for row in rows)
        for _, rows in selected
    ]
    return {
        "source": str(input_path),
        "source_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "selection": {
            "sessions": sessions,
            "take_first_turns": take_first_turns,
            "length_divisor": length_divisor,
            "hash_block_size": hash_block_size,
            "seed": seed,
            "eligible_sessions": len(eligible),
            "sampling": sampling,
            "final_context_min": final_context_min,
            "final_context_max": final_context_max,
            "final_context_strata": final_context_strata,
            "stratum_counts": stratum_counts,
            "eligibility": (
                {"minimum_source_turns": take_first_turns}
                if take_first_turns is not None
                else {"minimum_turns": min_turns, "maximum_turns": max_turns}
            ),
        },
        "observed": {
            "sessions": len(selected),
            "turns": sum(turn_counts),
            "turns_per_session": _summary(turn_counts),
            "total_input_tokens": sum(input_lengths),
            "total_output_tokens": sum(output_lengths),
            "input_tokens_per_request": _summary(input_lengths),
            "output_tokens_per_request": _summary(output_lengths),
            "prompt_context_tokens_per_request": _summary(prompt_contexts),
            "initial_context_tokens": _summary(initial_contexts),
            "final_context_tokens": _summary(final_contexts),
            "per_turn": per_turn,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=128)
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--take-first-turns", type=int)
    parser.add_argument("--length-divisor", type=int, default=1)
    parser.add_argument("--hash-block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-context-min", type=int)
    parser.add_argument("--final-context-max", type=int)
    parser.add_argument("--final-context-strata", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.min_turns <= args.max_turns:
        raise ValueError("turn bounds must be positive and ordered")
    if args.sessions <= 0:
        raise ValueError("sessions must be positive")
    if args.take_first_turns is not None and args.take_first_turns <= 0:
        raise ValueError("take-first-turns must be positive")
    if args.length_divisor <= 0:
        raise ValueError("length-divisor must be positive")
    if args.hash_block_size <= 0:
        raise ValueError("hash-block-size must be positive")
    manifest = build_subset(
        args.input,
        args.output,
        sessions=args.sessions,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        take_first_turns=args.take_first_turns,
        length_divisor=args.length_divisor,
        hash_block_size=args.hash_block_size,
        seed=args.seed,
        final_context_min=args.final_context_min,
        final_context_max=args.final_context_max,
        final_context_strata=args.final_context_strata,
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
