# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate PAP's registered replay formats without tokenizing prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inspect_replay(
    path: Path, dataset_type: str, sessions: int, routing_policy: str | None = None
) -> dict:
    """Count the first sessions in AIPerf sequential loader order.

    This validates the project's explicit-session, text-only workload contract,
    not every schema supported by AIPerf. Counts describe the available replay;
    a timed run can stop before these requests have all been sent.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8").strip()
    records = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if not isinstance(records, list) or not records:
        raise ValueError("dataset must contain records")
    if dataset_type not in {"multi-turn", "mooncake-trace"}:
        raise ValueError(f"unsupported dataset type: {dataset_type}")
    counts: dict[str, int] = {}
    max_output: dict[str, int] = {}
    salted_sessions: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be an object")
        session = record.get("session_id")
        if not isinstance(session, str) or not session:
            raise ValueError(f"record {index} requires an explicit session_id")
        if dataset_type == "multi-turn":
            turns = record.get("turns")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"record {index} requires nonempty turns")
        else:
            if "turns" in record:
                raise ValueError(f"record {index} is not mooncake-trace")
            length = record.get("input_length")
            if type(length) is not int or length < 1:
                raise ValueError(f"record {index} requires positive input_length")
            turns = [record]
        for turn in turns:
            output = turn.get("output_length") if isinstance(turn, dict) else None
            if type(output) is not int or output < 1:
                raise ValueError(f"record {index} requires positive output_length")
            extra = turn.get("extra") or {}
            if not isinstance(extra, dict):
                raise ValueError(f"record {index} requires an object for extra")
            if (
                turn.get("cache_salt") is not None
                or extra.get("cache_salt") is not None
            ):
                salted_sessions.add(session)
            counts[session] = counts.get(session, 0) + 1
            max_output[session] = max(max_output.get(session, 0), output)
    if sessions < 1 or sessions > len(counts):
        raise ValueError(f"requested {sessions} sessions; dataset has {len(counts)}")
    selected = list(counts)[:sessions]
    selected_salted = len(salted_sessions.intersection(selected))
    if routing_policy == "dynamo" and selected_salted:
        raise ValueError(
            "PAP Dynamo routing cannot index cache_salt-isolated prefixes; "
            f"{selected_salted} selected sessions require this unsupported feature"
        )
    return {
        "dataset": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_type": dataset_type,
        "sampling": "sequential",
        "available_sessions": len(counts),
        "available_requests": sum(counts.values()),
        "selected_sessions": sessions,
        "selected_salted_sessions": selected_salted,
        "selected_requests": sum(counts[session] for session in selected),
        "selected_max_output_tokens": max(max_output[session] for session in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--dataset-type", required=True)
    parser.add_argument("--sessions", required=True, type=int)
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--routing-policy")
    args = parser.parse_args()
    try:
        result = inspect_replay(
            args.dataset, args.dataset_type, args.sessions, args.routing_policy
        )
        if (
            args.expected_requests is not None
            and result["selected_requests"] != args.expected_requests
        ):
            raise ValueError(
                f"expected {args.expected_requests} requests, "
                f"dataset selects {result['selected_requests']}"
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
