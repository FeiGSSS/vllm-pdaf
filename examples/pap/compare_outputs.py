# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare fused, native-PD, and PAP completion responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_response(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def normalize_completion_response(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("completion response does not contain choices")
    choice = choices[0]
    usage = response.get("usage") or {}
    return {
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def compare_architecture_outputs(
    responses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = {
        name: normalize_completion_response(response)
        for name, response in responses.items()
    }
    reference_name = "fused" if "fused" in normalized else next(iter(normalized))
    reference = normalized[reference_name]
    mismatches: list[dict[str, Any]] = []
    for name, value in normalized.items():
        if name == reference_name:
            continue
        for field, expected in reference.items():
            actual = value.get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "architecture": name,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {
        "match": not mismatches,
        "reference": reference_name,
        "architectures": normalized,
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fused, native-PD, and PAP completion responses"
    )
    parser.add_argument("--fused", required=True)
    parser.add_argument("--native-pd", required=True)
    parser.add_argument("--pap", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_architecture_outputs(
        {
            "fused": load_response(args.fused),
            "native_pd": load_response(args.native_pd),
            "pap": load_response(args.pap),
        }
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)
    if not report["match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
