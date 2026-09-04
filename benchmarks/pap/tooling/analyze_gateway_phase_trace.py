# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize PAP Gateway phases against AIPerf request metrics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import regex as re

_PREFILL = re.compile(
    r"prefill IPC profile request_id=(?P<id>\S+) "
    r"register_ms=(?P<register>[\d.]+) "
    r"tokenization_ms=(?P<tokenization>[\d.]+) "
    r"routing_ms=(?P<routing>[\d.]+) "
    r"prefill_payload_ms=(?P<prefill_payload>[\d.]+) "
    r"prefill_ms=(?P<prefill>[\d.]+) "
    r"readiness_ms=(?P<readiness>[\d.]+) "
    r"projection_payload_ms=(?P<projection_payload>[\d.]+) "
    r"pre_projection_ms=(?P<pre_projection>[\d.]+)"
)
_ADMISSION = re.compile(
    r"projection admission profile request_id=(?P<id>\S+) "
    r"admission_ms=(?P<admission>[\d.]+) "
    r"request_to_admission_ms=(?P<request_to_admission>[\d.]+)"
)
_FIRST_CHUNK = re.compile(
    r"projection stream profile request_id=(?P<id>\S+) "
    r"first_chunk_ms=(?P<projection_first_chunk>[\d.]+) "
    r"request_to_first_chunk_ms=(?P<gateway_ttft>[\d.]+)"
)
_PLACEMENT = re.compile(
    r"Dynamo placement request_id=(?P<id>\S+) selected_pa=(?P<pa>\d+) "
    r"prompt_tokens=(?P<prompt_tokens>\d+) "
    r"effective_prefill_tokens=(?P<effective_prefill_tokens>\d+)"
)


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
        "max": max(values),
    }


def _read_aiperf(path: Path) -> dict[str, dict[str, float | int]]:
    records = {}
    for line in path.read_text().splitlines():
        payload = json.loads(line)
        metadata = payload["metadata"]
        metrics = payload["metrics"]
        records[metadata["x_request_id"]] = {
            "turn": int(metadata["turn_index"]),
            "client_ttft": float(metrics["time_to_first_token"]["value"]),
            "input_tokens": int(metrics["input_sequence_length"]["value"]),
            "cache_read_tokens": int(
                metrics["usage_prompt_cache_read_tokens"]["value"]
            ),
        }
    return records


def analyze(log_path: Path, profile_path: Path) -> dict[str, object]:
    rows: dict[str, dict[str, float | int]] = defaultdict(dict)
    patterns = (_PREFILL, _ADMISSION, _FIRST_CHUNK, _PLACEMENT)
    for line in log_path.read_text().splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match is None:
                continue
            request_id = match.group("id")
            for key, value in match.groupdict().items():
                if key == "id":
                    continue
                rows[request_id][key] = float(value)
            break

    aiperf = _read_aiperf(profile_path)
    required = {
        "register",
        "tokenization",
        "routing",
        "prefill_payload",
        "prefill",
        "readiness",
        "projection_payload",
        "pre_projection",
        "admission",
        "request_to_admission",
        "projection_first_chunk",
        "gateway_ttft",
    }
    joined = []
    for request_id, client in aiperf.items():
        row = rows.get(request_id, {})
        if not required.issubset(row):
            continue
        result = dict(row)
        result.update(client)
        result["pre_projection_other"] = result["pre_projection"] - sum(
            result[key]
            for key in (
                "register",
                "tokenization",
                "routing",
                "prefill_payload",
                "prefill",
                "readiness",
                "projection_payload",
            )
        )
        result["post_admission_to_first_chunk"] = (
            result["gateway_ttft"] - result["request_to_admission"]
        )
        result["client_gateway_delta"] = result["client_ttft"] - result["gateway_ttft"]
        result["actual_prefill_tokens"] = (
            result["input_tokens"] - result["cache_read_tokens"]
        )
        joined.append(result)

    metric_names = sorted(
        key
        for key, value in joined[0].items()
        if key not in {"turn", "pa"} and isinstance(value, (float, int))
    )

    def summarize(source: list[dict[str, float | int]]) -> dict[str, object]:
        return {
            name: _summary([float(row[name]) for row in source if name in row])
            for name in metric_names
            if any(name in row for row in source)
        }

    by_turn = {
        str(turn): summarize([row for row in joined if row["turn"] == turn])
        for turn in sorted({int(row["turn"]) for row in joined})
    }

    def correlation(left: str, right: str) -> float:
        left_values = [float(row[left]) for row in joined]
        right_values = [float(row[right]) for row in joined]
        left_mean = sum(left_values) / len(left_values)
        right_mean = sum(right_values) / len(right_values)
        covariance = sum(
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in zip(left_values, right_values)
        )
        left_variance = sum((value - left_mean) ** 2 for value in left_values)
        right_variance = sum((value - right_mean) ** 2 for value in right_values)
        return covariance / math.sqrt(left_variance * right_variance)

    return {
        "joined_requests": len(joined),
        "aiperf_requests": len(aiperf),
        "all": summarize(joined),
        "by_turn": by_turn,
        "correlation": {
            "prefill_vs_effective_prefill_tokens": correlation(
                "prefill", "effective_prefill_tokens"
            ),
            "prefill_vs_actual_prefill_tokens": correlation(
                "prefill", "actual_prefill_tokens"
            ),
            "client_ttft_vs_prefill": correlation("client_ttft", "prefill"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gateway_log", type=Path)
    parser.add_argument("aiperf_profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = analyze(args.gateway_log, args.aiperf_profile)
    output = args.output or args.gateway_log.with_name("gateway_phase_trace.json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
