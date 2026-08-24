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


def build_subset(
    input_path: Path,
    output_path: Path,
    *,
    sessions: int,
    min_turns: int,
    max_turns: int,
    seed: int,
) -> dict[str, Any]:
    """Filter complete sessions and write a deterministic sample."""

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with input_path.open() as file:
        for line in file:
            row = json.loads(line)
            grouped.setdefault(row["session_id"], []).append(row)

    eligible = [
        (session_id, rows)
        for session_id, rows in grouped.items()
        if min_turns <= len(rows) <= max_turns
    ]
    if len(eligible) < sessions:
        raise ValueError(
            f"only {len(eligible)} sessions satisfy the turn bounds; "
            f"need {sessions}"
        )

    chosen_ids = {
        session_id
        for session_id, _ in random.Random(seed).sample(eligible, sessions)
    }
    selected = [
        (session_id, rows)
        for session_id, rows in grouped.items()
        if session_id in chosen_ids
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        for _, rows in selected:
            for row in rows:
                row.pop("delay", None)
                file.write(json.dumps(row, separators=(",", ":")) + "\n")

    turn_counts = [len(rows) for _, rows in selected]
    initial_contexts = [rows[0]["input_length"] for _, rows in selected]
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
            "min_turns": min_turns,
            "max_turns": max_turns,
            "seed": seed,
            "eligible_sessions": len(eligible),
            "sampling": "random_without_replacement",
        },
        "observed": {
            "sessions": len(selected),
            "turns": sum(turn_counts),
            "turns_per_session": _summary(turn_counts),
            "initial_context_tokens": _summary(initial_contexts),
            "final_context_tokens": _summary(final_contexts),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=128)
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.min_turns <= args.max_turns:
        raise ValueError("turn bounds must be positive and ordered")
    if args.sessions <= 0:
        raise ValueError("sessions must be positive")
    manifest = build_subset(
        args.input,
        args.output,
        sessions=args.sessions,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        seed=args.seed,
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
