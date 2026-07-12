"""Audit prompt reuse and NIXL transfers for the five-turn PD load test."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


STATUS = "pd_multiturn_load_reuse_metrics_passed"
MODE = "pd_multiturn_load"
REQUIRED_ROUNDS = 5
SOURCES = (
    "local_compute",
    "local_cache_hit",
    "external_kv_transfer",
)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SOURCE_RE = re.compile(r'(?:^|,)source="(?P<source>[^"]+)"(?:,|$)')
_FALSE_VALUES = frozenset({"0", "false", "n", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "y", "yes", "on"})
_PUSH_MODES = frozenset(
    {"push", "nixl_push", "nixl-push", "nixlpushconnector"}
)
_SERVICE_ERROR_RE = re.compile(
    r"NIXL (?:transfer|notification) (?:failure|failed)",
    re.IGNORECASE,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}: {value}")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive number: {value}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive number: {value}")
    return parsed


def _metric_samples(metrics_text: str, metric_name: str) -> list[tuple[str, float]]:
    pattern = re.compile(
        rf"^{re.escape(metric_name)}"
        rf"(?:\{{(?P<labels>[^\n}}]*)\}})?\s+"
        rf"(?P<value>{_NUMBER})(?:\s+\d+)?$",
        re.MULTILINE,
    )
    samples: list[tuple[str, float]] = []
    for match in pattern.finditer(metrics_text):
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {metric_name} sample: {value}")
        samples.append((match.group("labels") or "", value))
    if not samples:
        raise ValueError(f"missing Prometheus metric: {metric_name}")
    return samples


def _integral_metric_total(metrics_text: str, metric_name: str) -> int:
    value = sum(value for _, value in _metric_samples(metrics_text, metric_name))
    if not value.is_integer():
        raise ValueError(f"non-integral Prometheus metric {metric_name}: {value}")
    return int(value)


def _float_metric_total(metrics_text: str, metric_name: str) -> float:
    return sum(value for _, value in _metric_samples(metrics_text, metric_name))


def parse_prompt_token_sources(metrics_text: str) -> dict[str, int]:
    """Aggregate prompt-token source counters across all metric series."""
    values: dict[str, list[float]] = {source: [] for source in SOURCES}
    for labels, value in _metric_samples(
        metrics_text,
        "vllm:prompt_tokens_by_source_total",
    ):
        source_match = _SOURCE_RE.search(labels)
        if source_match is None:
            continue
        source = source_match.group("source")
        if source in values:
            values[source].append(value)

    parsed: dict[str, int] = {}
    for source, samples in values.items():
        if not samples:
            raise ValueError(f"missing prompt-token source metric: {source}")
        total = sum(samples)
        if not total.is_integer():
            raise ValueError(f"non-integral prompt-token source {source}: {total}")
        parsed[source] = int(total)
    return parsed


def parse_nixl_metrics(metrics_text: str) -> dict[str, int | float]:
    """Parse and validate aggregate NIXL histogram/counter evidence."""
    transfer_count = _integral_metric_total(
        metrics_text,
        "vllm:nixl_xfer_time_seconds_count",
    )
    transfer_time_seconds = _float_metric_total(
        metrics_text,
        "vllm:nixl_xfer_time_seconds_sum",
    )
    bytes_count = _integral_metric_total(
        metrics_text,
        "vllm:nixl_bytes_transferred_count",
    )
    bytes_transferred = _float_metric_total(
        metrics_text,
        "vllm:nixl_bytes_transferred_sum",
    )
    descriptor_count = _integral_metric_total(
        metrics_text,
        "vllm:nixl_num_descriptors_count",
    )
    descriptors = _integral_metric_total(
        metrics_text,
        "vllm:nixl_num_descriptors_sum",
    )
    failed_transfers = _integral_metric_total(
        metrics_text,
        "vllm:nixl_num_failed_transfers_total",
    )
    failed_notifications = _integral_metric_total(
        metrics_text,
        "vllm:nixl_num_failed_notifications_total",
    )
    expired_requests = _integral_metric_total(
        metrics_text,
        "vllm:nixl_num_kv_expired_reqs_total",
    )

    if not (transfer_count == bytes_count == descriptor_count):
        raise ValueError(
            "NIXL histogram counts disagree: "
            f"xfer={transfer_count}, bytes={bytes_count}, "
            f"descriptors={descriptor_count}"
        )
    if transfer_count == 0:
        if transfer_time_seconds != 0 or bytes_transferred != 0 or descriptors != 0:
            raise ValueError("zero-count NIXL histograms have nonzero sums")
    else:
        if transfer_time_seconds <= 0 or bytes_transferred <= 0:
            raise ValueError("positive NIXL transfer count has an empty payload/time")
        if descriptors < transfer_count:
            raise ValueError(
                "NIXL descriptor sum is smaller than transfer count: "
                f"{descriptors} < {transfer_count}"
            )

    transferred_mib = bytes_transferred / (2**20)
    throughput_mib_s = (
        transferred_mib / transfer_time_seconds
        if transfer_time_seconds > 0
        else 0.0
    )
    return {
        "transfer_count": transfer_count,
        "transfer_time_seconds": transfer_time_seconds,
        "bytes_histogram_count": bytes_count,
        "bytes_transferred": bytes_transferred,
        "descriptor_histogram_count": descriptor_count,
        "descriptors": descriptors,
        "failed_transfers": failed_transfers,
        "failed_notifications": failed_notifications,
        "expired_requests": expired_requests,
        "transferred_mib": transferred_mib,
        "aggregate_throughput_mib_s": throughput_mib_s,
    }


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


def _parse_effective_config(text: str | None) -> dict[str, str]:
    if text is None:
        raise ValueError(
            "effective config is required to prove UCX emulation is disabled"
        )
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value.strip().strip("'\"")
    return values


def _config_bool(config: Mapping[str, str], key: str) -> bool:
    if key not in config:
        raise ValueError(f"effective config is missing {key}")
    value = config[key].lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ValueError(f"effective config has invalid {key}: {config[key]!r}")


def _validate_requests(
    result: Mapping[str, object],
    *,
    rounds: int,
    active_conversations: int,
) -> tuple[dict[tuple[int, int], Mapping[str, Any]], int]:
    raw_requests = result.get("requests")
    if not isinstance(raw_requests, list):
        raise ValueError("load result requests must be a list")
    expected_count = rounds * active_conversations
    if len(raw_requests) != expected_count:
        raise ValueError(
            f"load result must contain {expected_count} requests, "
            f"found {len(raw_requests)}"
        )

    requests: dict[tuple[int, int], Mapping[str, Any]] = {}
    total_prompt_tokens = 0
    for index, raw_request in enumerate(raw_requests):
        request = _mapping(raw_request, f"request {index}")
        conversation = _integer(
            request.get("conversation_index"),
            f"request {index} conversation_index",
        )
        round_index = _integer(
            request.get("round"),
            f"request {index} round",
            minimum=1,
        )
        if conversation >= active_conversations or round_index > rounds:
            raise ValueError(
                f"request {index} is outside the declared conversation/round range"
            )
        key = (conversation, round_index)
        if key in requests:
            raise ValueError(f"duplicate request for conversation/round {key}")
        requests[key] = request

        prompt_tokens = _integer(
            request.get("prompt_tokens"),
            f"request {index} prompt_tokens",
            minimum=1,
        )
        _integer(
            request.get("completion_tokens"),
            f"request {index} completion_tokens",
            minimum=1,
        )
        for field in ("ttft_ms", "tpot_ms", "latency_ms", "eof_latency_ms"):
            _positive_float(request.get(field), f"request {index} {field}")
        _mapping(request.get("prefill"), f"request {index} prefill")
        total_prompt_tokens += prompt_tokens

    expected_keys = {
        (conversation, round_index)
        for conversation in range(active_conversations)
        for round_index in range(1, rounds + 1)
    }
    if set(requests) != expected_keys:
        raise ValueError("requests do not cover every declared conversation/round")
    return requests, total_prompt_tokens


def _validate_transitions(
    cache_validation: Mapping[str, Any],
    *,
    requests: Mapping[tuple[int, int], Mapping[str, Any]],
    rounds: int,
    active_conversations: int,
    block_size: int,
) -> tuple[int, int, int]:
    raw_transitions = cache_validation.get("transitions")
    if not isinstance(raw_transitions, list):
        raise ValueError("cache_validation.transitions must be a list")
    expected_count = active_conversations * (rounds - 1)
    if len(raw_transitions) != expected_count:
        raise ValueError(
            f"expected {expected_count} transition records, "
            f"found {len(raw_transitions)}"
        )

    transition_keys: set[tuple[int, int, int]] = set()
    prefill_local_hits = 0
    decode_local_hits = 0
    decode_derived_hits = 0
    for index, raw_transition in enumerate(raw_transitions):
        transition = _mapping(raw_transition, f"transition {index}")
        conversation = _integer(
            transition.get("conversation_index"),
            f"transition {index} conversation_index",
        )
        from_round = _integer(
            transition.get("from_round"),
            f"transition {index} from_round",
            minimum=1,
        )
        to_round = _integer(
            transition.get("to_round"),
            f"transition {index} to_round",
            minimum=2,
        )
        if (
            conversation >= active_conversations
            or from_round >= rounds
            or to_round != from_round + 1
        ):
            raise ValueError(f"transition {index} is not an adjacent valid round")
        key = (conversation, from_round, to_round)
        if key in transition_keys:
            raise ValueError(f"duplicate transition {key}")
        transition_keys.add(key)

        previous_prompt = _integer(
            transition.get("previous_prompt_tokens"),
            f"transition {index} previous_prompt_tokens",
            minimum=1,
        )
        request_prompt = _integer(
            requests[(conversation, from_round)].get("prompt_tokens"),
            f"transition {index} source request prompt_tokens",
            minimum=1,
        )
        if previous_prompt != request_prompt:
            raise ValueError(
                f"transition {index} previous prompt mismatch: "
                f"{previous_prompt} != {request_prompt}"
            )
        previous_boundary = previous_prompt // block_size * block_size
        expected_cached = _integer(
            transition.get("expected_cached_tokens"),
            f"transition {index} expected_cached_tokens",
            minimum=1,
        )
        if expected_cached % block_size:
            raise ValueError(
                f"transition {index} expected_cached_tokens is not block aligned"
            )
        target_prompt = _integer(
            requests[(conversation, to_round)].get("prompt_tokens"),
            f"transition {index} target request prompt_tokens",
            minimum=1,
        )
        if not previous_boundary <= expected_cached < target_prompt:
            raise ValueError(
                f"transition {index} cached-token boundary is invalid: "
                f"{previous_boundary} <= {expected_cached} < {target_prompt}"
            )
        decode_derived = _integer(
            transition.get("decode_derived_hit_tokens"),
            f"transition {index} decode_derived_hit_tokens",
        )
        expected_decode_derived = expected_cached - previous_boundary
        if decode_derived != expected_decode_derived:
            raise ValueError(
                f"transition {index} Decode-derived hit mismatch: "
                f"{decode_derived} != {expected_decode_derived}"
            )
        if decode_derived < block_size:
            raise ValueError(
                f"transition {index} has no full Decode-derived cache block: "
                f"{decode_derived} < {block_size}"
            )
        if transition.get("actual_cached_tokens") is not None:
            raise ValueError(
                "PD transition actual_cached_tokens must be null; "
                "runtime evidence comes from Prometheus"
            )

        prefill_local_hits += previous_boundary
        decode_local_hits += expected_cached
        decode_derived_hits += decode_derived

    expected_keys = {
        (conversation, round_index, round_index + 1)
        for conversation in range(active_conversations)
        for round_index in range(1, rounds)
    }
    if transition_keys != expected_keys:
        raise ValueError("transitions do not cover every adjacent round")
    return prefill_local_hits, decode_local_hits, decode_derived_hits


def _nixl_total(
    prefill: Mapping[str, int | float],
    decode: Mapping[str, int | float],
) -> dict[str, int | float]:
    integer_fields = (
        "transfer_count",
        "bytes_histogram_count",
        "descriptor_histogram_count",
        "descriptors",
        "failed_transfers",
        "failed_notifications",
        "expired_requests",
    )
    float_fields = (
        "transfer_time_seconds",
        "bytes_transferred",
        "transferred_mib",
    )
    total: dict[str, int | float] = {
        field: int(prefill[field]) + int(decode[field])
        for field in integer_fields
    }
    total.update(
        {
            field: float(prefill[field]) + float(decode[field])
            for field in float_fields
        }
    )
    transfer_time = float(total["transfer_time_seconds"])
    total["aggregate_throughput_mib_s"] = (
        float(total["transferred_mib"]) / transfer_time
        if transfer_time > 0
        else 0.0
    )
    return total


def validate_pd_multiturn_load_reuse(
    result: Mapping[str, object],
    *,
    prefill_metrics: str,
    decode_metrics: str,
    effective_config: str | None,
    service_logs: Sequence[str] = (),
) -> dict[str, object]:
    """Validate five-turn prompt reuse and bounded push-transfer evidence."""
    if result.get("architecture") != "pd":
        raise ValueError("PD load reuse validation requires architecture=pd")

    profile = _mapping(result.get("profile"), "profile")
    rounds = _integer(profile.get("rounds"), "profile rounds", minimum=1)
    if rounds != REQUIRED_ROUNDS:
        raise ValueError(f"PD load audit requires exactly {REQUIRED_ROUNDS} rounds")
    if profile.get("api") != "/v1/completions" or profile.get(
        "workload_semantics"
    ) != "exact_token_continuous_multiturn":
        raise ValueError("PD load audit requires the exact-token workload")
    active_conversations = _integer(
        profile.get("active_conversations"),
        "active conversations",
        minimum=1,
    )
    block_size = _integer(profile.get("block_size"), "block size", minimum=1)
    implementation = _mapping(result.get("implementation"), "implementation")
    transfer_mode_raw = implementation.get("offload_exec_transport")
    if not isinstance(transfer_mode_raw, str) or not transfer_mode_raw.strip():
        raise ValueError("profile transfer_mode must be a non-empty string")
    transfer_mode = transfer_mode_raw.strip().lower()
    if "arrival" not in profile:
        raise ValueError("profile is missing arrival evidence")

    requests, total_prompt_tokens = _validate_requests(
        result,
        rounds=rounds,
        active_conversations=active_conversations,
    )
    cache_validation = _mapping(
        result.get("cache_validation"),
        "cache validation",
    )
    (
        expected_prefill_hits,
        expected_decode_hits,
        decode_derived_hits,
    ) = _validate_transitions(
        cache_validation,
        requests=requests,
        rounds=rounds,
        active_conversations=active_conversations,
        block_size=block_size,
    )

    prefill_sources = parse_prompt_token_sources(prefill_metrics)
    decode_sources = parse_prompt_token_sources(decode_metrics)
    _require_conservation("Prefill", prefill_sources, total_prompt_tokens)
    _require_conservation("Decode", decode_sources, total_prompt_tokens)
    if prefill_sources["local_cache_hit"] != expected_prefill_hits:
        raise ValueError(
            "Prefill local cache hit mismatch: "
            f"{prefill_sources['local_cache_hit']} != {expected_prefill_hits}"
        )
    if prefill_sources["external_kv_transfer"] != 0:
        raise ValueError("PD Prefill unexpectedly received external KV")
    expected_prefill_compute = total_prompt_tokens - expected_prefill_hits
    if prefill_sources["local_compute"] != expected_prefill_compute:
        raise ValueError(
            "Prefill local compute mismatch: "
            f"{prefill_sources['local_compute']} != {expected_prefill_compute}"
        )
    if decode_sources["local_cache_hit"] != expected_decode_hits:
        raise ValueError(
            "Decode local cache hit mismatch: "
            f"{decode_sources['local_cache_hit']} != {expected_decode_hits}"
        )
    if decode_sources["local_compute"] != 0:
        raise ValueError(
            "Decode performed unexpected local compute: "
            f"{decode_sources['local_compute']} != 0"
        )
    expected_decode_external = total_prompt_tokens - expected_decode_hits
    if decode_sources["external_kv_transfer"] != expected_decode_external:
        raise ValueError(
            "Decode external transfer mismatch: "
            f"{decode_sources['external_kv_transfer']} != "
            f"{expected_decode_external}"
        )

    config = _parse_effective_config(effective_config)
    if _config_bool(config, "UCX_PROTO_EMULATION_ENABLE"):
        raise ValueError("UCX software protocol emulation must be disabled")
    cross_layers_enabled = _config_bool(
        config,
        "ENABLE_CROSS_LAYERS_BLOCKS",
    )

    prefill_nixl = parse_nixl_metrics(prefill_metrics)
    decode_nixl = parse_nixl_metrics(decode_metrics)
    total_nixl = _nixl_total(prefill_nixl, decode_nixl)
    for node, metrics in (
        ("Prefill", prefill_nixl),
        ("Decode", decode_nixl),
    ):
        if metrics["failed_transfers"] != 0:
            raise ValueError(f"{node} recorded failed NIXL transfers")
        if metrics["failed_notifications"] != 0:
            raise ValueError(f"{node} recorded failed NIXL notifications")
        if metrics["expired_requests"] != 0:
            raise ValueError(f"{node} recorded expired NIXL requests")

    request_count = rounds * active_conversations
    if transfer_mode in _PUSH_MODES:
        if total_nixl["transfer_count"] != request_count:
            raise ValueError(
                "push-mode NIXL transfer count mismatch: "
                f"{total_nixl['transfer_count']} != {request_count}"
            )
        if cross_layers_enabled:
            descriptor_limit = request_count * active_conversations
            descriptors = int(total_nixl["descriptors"])
            if not request_count <= descriptors <= descriptor_limit:
                raise ValueError(
                    "cross-layer NIXL descriptor sum is outside the bounded "
                    f"range: {descriptors} not in "
                    f"[{request_count}, {descriptor_limit}]"
                )

    error_matches = sum(
        len(_SERVICE_ERROR_RE.findall(log_text)) for log_text in service_logs
    )
    if error_matches:
        raise ValueError(
            f"service logs contain {error_matches} NIXL transfer errors"
        )

    return {
        "status": STATUS,
        "mode": MODE,
        "rounds": rounds,
        "active_conversations": active_conversations,
        "request_count": request_count,
        "transition_count": active_conversations * (rounds - 1),
        "total_prompt_tokens": total_prompt_tokens,
        "expected_prefill_local_cache_hit_tokens": expected_prefill_hits,
        "expected_decode_local_cache_hit_tokens": expected_decode_hits,
        "decode_derived_hit_tokens": decode_derived_hits,
        "prefill_prompt_tokens_by_source": prefill_sources,
        "decode_prompt_tokens_by_source": decode_sources,
        "nixl": {
            "transfer_mode": transfer_mode_raw,
            "cross_layers_enabled": cross_layers_enabled,
            "prefill": prefill_nixl,
            "decode": decode_nixl,
            "total": total_nixl,
            "descriptor_upper_bound": request_count * active_conversations,
        },
        "ucx": {
            "software_emulation_disabled": True,
            "proto_emulation_value": config["UCX_PROTO_EMULATION_ENABLE"],
        },
        "service_logs_checked": len(service_logs),
        "service_log_error_matches": error_matches,
    }


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit five-turn PD load reuse and NIXL metrics"
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--prefill-metrics", type=Path, required=True)
    parser.add_argument("--decode-metrics", type=Path, required=True)
    parser.add_argument("--effective-config", type=Path)
    parser.add_argument("--service-log", type=Path, action="append", default=[])
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result JSON root must be an object")
    effective_config = (
        args.effective_config.read_text(encoding="utf-8")
        if args.effective_config is not None
        else None
    )
    evidence = validate_pd_multiturn_load_reuse(
        result,
        prefill_metrics=args.prefill_metrics.read_text(encoding="utf-8"),
        decode_metrics=args.decode_metrics.read_text(encoding="utf-8"),
        effective_config=effective_config,
        service_logs=tuple(
            path.read_text(encoding="utf-8") for path in args.service_log
        ),
    )
    cache = dict(_mapping(result.get("cache_validation"), "cache validation"))
    cache["status"] = STATUS
    result["cache_validation"] = cache
    result["pd_reuse_validation"] = evidence
    validity = dict(_mapping(result.get("validity", {}), "validity"))
    validity.update({"status": "passed", "cache_gate": STATUS})
    result["validity"] = validity
    _atomic_write(args.result, result)


if __name__ == "__main__":
    main()
