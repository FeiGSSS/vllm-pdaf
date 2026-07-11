"""Validate token-source evidence for the official streaming PD baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


STATUS = "official_streaming_one_way_metrics_passed"
MODE = "official_streaming_one_way"
SOURCES = (
    "local_compute",
    "local_cache_hit",
    "external_kv_transfer",
)
_METRIC_RE = re.compile(
    r"^vllm:prompt_tokens_by_source_total\{(?P<labels>[^\n]*)\}\s+"
    r"(?P<value>[\d.eE+\-]+)$",
    re.MULTILINE,
)
_SOURCE_RE = re.compile(r'(?:^|,)source="(?P<source>[^"]+)"(?:,|$)')


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer: {value}")
    return value


def parse_prompt_token_sources(metrics_text: str) -> dict[str, int]:
    """Parse one model/engine series for every prompt-token source."""
    values: dict[str, list[float]] = {source: [] for source in SOURCES}
    for metric_match in _METRIC_RE.finditer(metrics_text):
        source_match = _SOURCE_RE.search(metric_match.group("labels"))
        if source_match is None:
            continue
        source = source_match.group("source")
        if source in values:
            values[source].append(float(metric_match.group("value")))

    parsed: dict[str, int] = {}
    for source, source_values in values.items():
        if len(source_values) != 1:
            raise ValueError(
                f"expected exactly one {source} metric series, "
                f"found {len(source_values)}"
            )
        value = source_values[0]
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"invalid {source} metric value: {value}")
        parsed[source] = int(value)
    return parsed


def _require_conservation(
    node: str,
    sources: Mapping[str, int],
    expected_total: int,
) -> None:
    actual_total = sum(sources[source] for source in SOURCES)
    if actual_total != expected_total:
        raise ValueError(
            f"{node} prompt-source conservation failed: "
            f"{actual_total} != {expected_total}; {dict(sources)}"
        )


def validate_official_streaming_one_way(
    result: Mapping[str, object],
    *,
    proxy_log: str,
    prefill_metrics: str,
    decode_metrics: str,
) -> dict[str, object]:
    """Validate exact two-turn reuse without modifying the official PD lane."""
    if result.get("architecture") != "pd":
        raise ValueError("PD reuse validation requires architecture=pd")

    proxy_misses = len(re.findall(r"cache MISS", proxy_log))
    proxy_hits = len(re.findall(r"cache HIT", proxy_log))
    if proxy_misses != 2 or proxy_hits != 0:
        raise ValueError(
            "official streaming one-way semantics changed: "
            f"misses={proxy_misses}, hits={proxy_hits}"
        )

    profile = _mapping(result.get("profile"), "profile")
    block_size = _positive_integer(profile.get("block_size"), "block size")
    rounds = result.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise ValueError("PD result must contain exactly two rounds")
    first_round = _mapping(rounds[0], "first round")
    second_round = _mapping(rounds[1], "second round")
    first_prompt = _positive_integer(
        first_round.get("prompt_tokens"),
        "first-round prompt tokens",
    )
    second_prompt = _positive_integer(
        second_round.get("prompt_tokens"),
        "second-round prompt tokens",
    )
    total_prompt_tokens = first_prompt + second_prompt

    cache = _mapping(result.get("cache_validation"), "cache validation")
    first_boundary = _positive_integer(
        cache.get("first_prompt_block_boundary"),
        "first prompt block boundary",
    )
    expected_first_boundary = first_prompt // block_size * block_size
    if first_boundary != expected_first_boundary:
        raise ValueError(
            "first prompt block boundary mismatch: "
            f"{first_boundary} != {expected_first_boundary}"
        )
    expected_cached = _positive_integer(
        cache.get("expected_cached_tokens"),
        "expected cached tokens",
    )
    decode_derived = _positive_integer(
        cache.get("decode_derived_hit_tokens"),
        "Decode-derived hit tokens",
    )
    if expected_cached <= first_boundary or decode_derived < block_size:
        raise ValueError(
            "second-turn LCP does not contain a Decode-derived cache block"
        )

    prefill_sources = parse_prompt_token_sources(prefill_metrics)
    decode_sources = parse_prompt_token_sources(decode_metrics)
    _require_conservation("Prefill", prefill_sources, total_prompt_tokens)
    _require_conservation("Decode", decode_sources, total_prompt_tokens)

    if prefill_sources["local_cache_hit"] != first_boundary:
        raise ValueError(
            "Prefill local cache hit mismatch: "
            f"{prefill_sources['local_cache_hit']} != {first_boundary}"
        )
    if prefill_sources["external_kv_transfer"] != 0:
        raise ValueError(
            "one-way Prefill unexpectedly received external KV: "
            f"{prefill_sources}"
        )
    if decode_sources["local_cache_hit"] != expected_cached:
        raise ValueError(
            "Decode local cache did not cover the exact materialized LCP: "
            f"{decode_sources['local_cache_hit']} != {expected_cached}"
        )
    if decode_sources["external_kv_transfer"] <= first_boundary:
        raise ValueError(
            "Decode metrics do not prove a second-turn Prefill-to-Decode "
            f"transfer: {decode_sources}"
        )

    return {
        "status": STATUS,
        "mode": MODE,
        "proxy_cache_misses": proxy_misses,
        "proxy_cache_hits": proxy_hits,
        "total_prompt_tokens": total_prompt_tokens,
        "prefill_prompt_tokens_by_source": prefill_sources,
        "decode_prompt_tokens_by_source": decode_sources,
    }


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate official streaming PD multi-turn reuse metrics"
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--prefill-metrics", type=Path, required=True)
    parser.add_argument("--decode-metrics", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result JSON root must be an object")
    evidence = validate_official_streaming_one_way(
        result,
        proxy_log=args.proxy_log.read_text(encoding="utf-8"),
        prefill_metrics=args.prefill_metrics.read_text(encoding="utf-8"),
        decode_metrics=args.decode_metrics.read_text(encoding="utf-8"),
    )
    cache = dict(_mapping(result.get("cache_validation"), "cache validation"))
    cache["status"] = STATUS
    result["cache_validation"] = cache
    result["pd_reuse_validation"] = evidence
    result["validity"] = {"status": "passed", "cache_gate": STATUS}
    _atomic_write(args.result, result)


if __name__ == "__main__":
    main()
