"""Aggregate and compare request-level PAP/PD multi-turn load results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any


EXPECTED_ROUNDS = (1, 2, 3, 4, 5)
EXPECTED_COMPLETION_TOKENS = 256
PRIMARY_METRICS = ("ttft_ms", "tpot_ms", "latency_ms")
DIAGNOSTIC_METRICS = ("eof_latency_ms",)
DIGEST_FIELDS = (
    "prompt_token_digest",
    "output_token_digest",
    "assistant_text_digest",
)
ROUND_SCOPES = tuple(f"round_{round_index}" for round_index in EXPECTED_ROUNDS)
STEADY_SCOPE = "steady_rounds_2_5"
ALL_SCOPES = (*ROUND_SCOPES, STEADY_SCOPE)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _status_passed(value: object, name: str) -> Mapping[str, Any]:
    gate = _mapping(value, name)
    if gate.get("status") != "passed":
        raise ValueError(f"{name} did not pass: {gate}")
    return gate


def _external_validation_passed(
    value: object,
    name: str,
) -> Mapping[str, Any]:
    validation = _status_passed(value, name)
    gates = _mapping(validation.get("gates"), f"{name} gates")
    if not gates:
        raise ValueError(f"{name} gates must not be empty")
    for gate_name, raw_gate in gates.items():
        gate_status = (
            raw_gate.get("status")
            if isinstance(raw_gate, Mapping)
            else raw_gate
        )
        if gate_status != "passed":
            raise ValueError(
                f"{name} gate {gate_name} did not pass: {raw_gate}"
            )
    return validation


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}: {value}")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive: {value}")
    return parsed


def _canonical_fingerprint(value: Mapping[str, object]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a nearest-rank percentile, matching the load client."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0 <= quantile <= 1:
        raise ValueError(f"quantile must be in [0, 1]: {quantile}")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[index])


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric sequence")
    parsed = [_positive_finite(value, "request metric") for value in values]
    return {
        "count": len(parsed),
        "median": float(median(parsed)),
        "p90": _percentile(parsed, 0.90),
        "max": float(max(parsed)),
    }


def _same_value(
    results: Sequence[Mapping[str, object]],
    key: str,
    label: str,
) -> object:
    values = [result.get(key) for result in results]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"mixed {label}: {values}")
    return deepcopy(first)


def _validate_profile(result: Mapping[str, object]) -> Mapping[str, Any]:
    profile = _mapping(result.get("profile"), "profile")
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("profile_id must be a non-empty string")
    if profile.get("rounds") != len(EXPECTED_ROUNDS):
        raise ValueError(
            "profile must contain exactly five rounds: "
            f"{profile.get('rounds')}"
        )
    _integer(
        profile.get("active_conversations"),
        "profile active_conversations",
        minimum=1,
    )
    _positive_finite(
        profile.get("request_rate_per_round"),
        "profile request_rate_per_round",
    )
    if profile.get("output_tokens_per_round") != EXPECTED_COMPLETION_TOKENS:
        raise ValueError("profile output_tokens_per_round must be 256")
    fingerprint = _digest(
        result.get("profile_fingerprint"),
        "profile_fingerprint",
    )
    expected = _canonical_fingerprint(dict(profile))
    if fingerprint != expected:
        raise ValueError(
            "profile fingerprint mismatch: " f"{fingerprint} != {expected}"
        )
    return profile


def _validate_validity(result: Mapping[str, object]) -> None:
    _status_passed(result.get("validity"), "client validity")
    if "client_validity" in result:
        _status_passed(result.get("client_validity"), "client_validity")
    _external_validation_passed(
        result.get("external_validation"),
        "external validation",
    )
    cache_validation = result.get("cache_validation")
    if cache_validation is not None:
        cache = _mapping(cache_validation, "cache validation")
        cache_status = cache.get("status")
        cache_passed = cache_status == "passed" or (
            isinstance(cache_status, str) and cache_status.endswith("_passed")
        )
        if "status" in cache and not cache_passed:
            raise ValueError(f"cache validation did not pass: {cache}")


def _normalize_request(
    raw_request: object,
    request_position: int,
) -> dict[str, object]:
    request = _mapping(raw_request, f"request {request_position}")
    conversation_index = _integer(
        request.get("conversation_index"),
        f"request {request_position} conversation_index",
    )
    round_index = _integer(
        request.get("round"),
        f"request {request_position} round",
        minimum=1,
    )
    if round_index not in EXPECTED_ROUNDS:
        raise ValueError(
            f"request {request_position} has unsupported round {round_index}"
        )
    prompt_tokens = _integer(
        request.get("prompt_tokens"),
        f"request {request_position} prompt_tokens",
        minimum=1,
    )
    completion_tokens = _integer(
        request.get("completion_tokens"),
        f"request {request_position} completion_tokens",
        minimum=1,
    )
    if completion_tokens != EXPECTED_COMPLETION_TOKENS:
        raise ValueError(
            "completion token count must be 256 for "
            f"conversation {conversation_index} round {round_index}: "
            f"{completion_tokens}"
        )

    normalized: dict[str, object] = {
        "conversation_index": conversation_index,
        "round": round_index,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS):
        normalized[metric] = _positive_finite(
            request.get(metric),
            f"conversation {conversation_index} round {round_index} {metric}",
        )

    ttft_ms = float(normalized["ttft_ms"])
    tpot_ms = float(normalized["tpot_ms"])
    latency_ms = float(normalized["latency_ms"])
    eof_latency_ms = float(normalized["eof_latency_ms"])
    expected_tpot = (latency_ms - ttft_ms) / (completion_tokens - 1)
    if not math.isclose(
        tpot_ms,
        expected_tpot,
        rel_tol=1e-6,
        abs_tol=1e-3,
    ):
        raise ValueError(
            "TPOT accounting mismatch for conversation "
            f"{conversation_index} round {round_index}: "
            f"{tpot_ms} != {expected_tpot}"
        )
    if eof_latency_ms + 1e-6 < latency_ms:
        raise ValueError(
            "EOF latency precedes last-token latency for conversation "
            f"{conversation_index} round {round_index}"
        )

    for digest_name in DIGEST_FIELDS:
        normalized[digest_name] = _digest(
            request.get(digest_name),
            "conversation "
            f"{conversation_index} round {round_index} {digest_name}",
        )
    return normalized


def _normalize_requests(
    result: Mapping[str, object],
) -> dict[tuple[int, int], dict[str, object]]:
    raw_requests = result.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("requests must be a non-empty list")
    requests: dict[tuple[int, int], dict[str, object]] = {}
    for position, raw_request in enumerate(raw_requests, start=1):
        request = _normalize_request(raw_request, position)
        key = (int(request["conversation_index"]), int(request["round"]))
        if key in requests:
            raise ValueError(
                "duplicate request for conversation "
                f"{key[0]} round {key[1]}"
            )
        requests[key] = request

    conversation_indices = sorted({key[0] for key in requests})
    if not conversation_indices:
        raise ValueError("result contains no conversations")
    for conversation_index in conversation_indices:
        observed = {
            round_index
            for conv, round_index in requests
            if conv == conversation_index
        }
        if observed != set(EXPECTED_ROUNDS):
            raise ValueError(
                f"conversation {conversation_index} round set mismatch: "
                f"{sorted(observed)} != {list(EXPECTED_ROUNDS)}"
            )

    active_conversations = int(
        _mapping(result["profile"], "profile")["active_conversations"]
    )
    if active_conversations != len(conversation_indices):
        raise ValueError(
            "profile active_conversations differs from observed conversations: "
            f"{active_conversations} != {len(conversation_indices)}"
        )
    _validate_round_summaries(result, requests)
    return requests


def _validate_round_summaries(
    result: Mapping[str, object],
    requests: Mapping[tuple[int, int], Mapping[str, object]],
) -> None:
    summaries = result.get("rounds")
    if not isinstance(summaries, list) or len(summaries) != len(EXPECTED_ROUNDS):
        raise ValueError("round summaries must contain exactly five entries")
    observed: set[int] = set()
    conversation_count = len({key[0] for key in requests})
    for position, raw_summary in enumerate(summaries, start=1):
        summary = _mapping(raw_summary, f"round summary {position}")
        round_index = _integer(
            summary.get("round"),
            f"round summary {position} round",
            minimum=1,
        )
        if round_index not in EXPECTED_ROUNDS or round_index in observed:
            raise ValueError(f"invalid or duplicate round summary: {round_index}")
        observed.add(round_index)
        for count_field in ("request_count", "completed"):
            if count_field in summary:
                count = _integer(
                    summary.get(count_field),
                    f"round {round_index} {count_field}",
                )
                if count != conversation_count:
                    raise ValueError(
                        f"round {round_index} {count_field} mismatch: "
                        f"{count} != {conversation_count}"
                    )
        raw_round_requests = summary.get("requests")
        if raw_round_requests is not None:
            if not isinstance(raw_round_requests, list):
                raise ValueError(
                    f"round {round_index} requests summary must be a list"
                )
            normalized_round: dict[tuple[int, int], dict[str, object]] = {}
            for request_position, raw_request in enumerate(
                raw_round_requests,
                start=1,
            ):
                request = _normalize_request(raw_request, request_position)
                key = (
                    int(request["conversation_index"]),
                    int(request["round"]),
                )
                if key[1] != round_index:
                    raise ValueError(
                        f"round {round_index} summary contains round {key[1]}"
                    )
                if key in normalized_round:
                    raise ValueError(
                        f"round {round_index} summary duplicates conversation "
                        f"{key[0]}"
                    )
                normalized_round[key] = request
            expected_round = {
                key: value for key, value in requests.items() if key[1] == round_index
            }
            if normalized_round != expected_round:
                raise ValueError(
                    f"round {round_index} nested request summary mismatch"
                )
    if observed != set(EXPECTED_ROUNDS):
        raise ValueError(f"round summary set mismatch: {sorted(observed)}")


def validate_repetition(result: Mapping[str, object]) -> None:
    """Fail closed unless one load-test repetition is complete and valid."""
    if result.get("schema_version") != 1:
        raise ValueError(
            f"unsupported repetition schema: {result.get('schema_version')}"
        )
    if result.get("metric_definition") != "last_output_token_v2":
        raise ValueError(
            f"unexpected metric definition: {result.get('metric_definition')}"
        )
    architecture = result.get("architecture")
    if architecture not in {"pd", "pap"}:
        raise ValueError(f"unsupported architecture: {architecture}")
    _validate_profile(result)
    hardware = result.get("hardware_signature")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("hardware_signature must be a non-empty string")
    if not isinstance(result.get("git_tracked_worktree_dirty"), bool):
        raise ValueError("git_tracked_worktree_dirty must be boolean")
    _validate_validity(result)
    _normalize_requests(result)


def _request_shape(
    requests: Mapping[tuple[int, int], Mapping[str, object]],
) -> list[dict[str, int]]:
    return [
        {
            "conversation_index": conversation_index,
            "round": round_index,
            "prompt_tokens": int(request["prompt_tokens"]),
            "completion_tokens": int(request["completion_tokens"]),
        }
        for (conversation_index, round_index), request in sorted(requests.items())
    ]


def _shape_mismatch(
    expected: Sequence[Mapping[str, object]],
    observed: Sequence[Mapping[str, object]],
) -> str:
    expected_by_key = {
        (int(item["conversation_index"]), int(item["round"])): item
        for item in expected
    }
    observed_by_key = {
        (int(item["conversation_index"]), int(item["round"])): item
        for item in observed
    }
    if set(expected_by_key) != set(observed_by_key):
        return (
            "request key set differs: "
            f"{sorted(expected_by_key)} != {sorted(observed_by_key)}"
        )
    for key in sorted(expected_by_key):
        expected_item = expected_by_key[key]
        observed_item = observed_by_key[key]
        for field in ("prompt_tokens", "completion_tokens"):
            if expected_item.get(field) != observed_item.get(field):
                return (
                    f"conversation {key[0]} round {key[1]} {field} differs: "
                    f"{expected_item.get(field)} != {observed_item.get(field)}"
                )
    return "unknown shape mismatch"


def _metric_samples(
    normalized_repetitions: Sequence[
        Mapping[tuple[int, int], Mapping[str, object]]
    ],
) -> dict[str, dict[str, list[float]]]:
    samples: dict[str, dict[str, list[float]]] = {}
    for round_index in EXPECTED_ROUNDS:
        scope = f"round_{round_index}"
        samples[scope] = {
            metric: [
                float(request[metric])
                for repetition in normalized_repetitions
                for key, request in sorted(repetition.items())
                if key[1] == round_index
            ]
            for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS)
        }
    samples[STEADY_SCOPE] = {
        metric: [
            value
            for round_index in EXPECTED_ROUNDS[1:]
            for value in samples[f"round_{round_index}"][metric]
        ]
        for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS)
    }
    return samples


def _metric_summaries(
    samples: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        scope: {
            metric: _summary(scope_samples[metric])
            for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS)
        }
        for scope, scope_samples in samples.items()
    }


def _per_repetition_metrics(
    normalized_repetitions: Sequence[
        Mapping[tuple[int, int], Mapping[str, object]]
    ],
) -> list[dict[str, object]]:
    return [
        {
            "repetition_index": repetition_index,
            "metrics": _metric_summaries(_metric_samples([requests])),
        }
        for repetition_index, requests in enumerate(
            normalized_repetitions,
            start=1,
        )
    ]


def _digest_observations(
    normalized_repetitions: Sequence[
        Mapping[tuple[int, int], Mapping[str, object]]
    ],
) -> tuple[dict[str, list[dict[str, object]]], list[str]]:
    observations: dict[str, list[dict[str, object]]] = {
        digest_name: [] for digest_name in DIGEST_FIELDS
    }
    warnings: list[str] = []
    keys = sorted(normalized_repetitions[0])
    for digest_name in DIGEST_FIELDS:
        for conversation_index, round_index in keys:
            values = sorted(
                {
                    str(repetition[(conversation_index, round_index)][digest_name])
                    for repetition in normalized_repetitions
                }
            )
            observations[digest_name].append(
                {
                    "conversation_index": conversation_index,
                    "round": round_index,
                    "digests": values,
                }
            )
            if len(values) > 1:
                warnings.append(
                    f"conversation {conversation_index} round {round_index} "
                    f"{digest_name} differs across repetitions"
                )
    return observations, warnings


def _repetition_identity(result: Mapping[str, object]) -> str:
    for key in ("repetition_id", "run_id", "conversation_id_digest"):
        value = result.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return f"{key}:{value}"
    return "content_sha256:" + _canonical_fingerprint(dict(result))


def aggregate_repetitions(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate one quick or three formal load-test repetitions."""
    if len(results) not in {1, 3}:
        raise ValueError(
            "load aggregate requires one quick or three formal repetitions, "
            f"got {len(results)}"
        )
    for result in results:
        validate_repetition(result)

    profile = _same_value(results, "profile", "profile")
    profile_fingerprint = _same_value(
        results,
        "profile_fingerprint",
        "profile fingerprint",
    )
    architecture = _same_value(results, "architecture", "architecture")
    implementation = _same_value(
        results,
        "implementation",
        "implementation",
    )
    hardware = _same_value(results, "hardware_signature", "hardware signature")
    formal = len(results) == 3
    dirty_states = [bool(result["git_tracked_worktree_dirty"]) for result in results]
    if formal and any(dirty_states):
        raise ValueError("formal aggregate cannot use a dirty tracked worktree")

    repetition_ids = [_repetition_identity(result) for result in results]
    if len(set(repetition_ids)) != len(repetition_ids):
        raise ValueError("aggregate requires distinct repetitions")

    external_evidence = [
        deepcopy(dict(_mapping(result["external_validation"], "external validation")))
        for result in results
    ]
    external_gates = [
        deepcopy(
            dict(
                _mapping(
                    evidence.get("gates"),
                    "external validation gates",
                )
            )
        )
        for evidence in external_evidence
    ]
    if any(gates != external_gates[0] for gates in external_gates[1:]):
        raise ValueError(f"mixed external validation gates: {external_gates}")

    normalized_repetitions = [_normalize_requests(result) for result in results]
    shape = _request_shape(normalized_repetitions[0])
    for repetition_index, requests in enumerate(
        normalized_repetitions[1:],
        start=2,
    ):
        observed_shape = _request_shape(requests)
        if observed_shape != shape:
            raise ValueError(
                f"prompt token shape mismatch in repetition {repetition_index}: "
                + _shape_mismatch(shape, observed_shape)
            )

    samples = _metric_samples(normalized_repetitions)
    digest_observations, warnings = _digest_observations(
        normalized_repetitions
    )
    if not formal and dirty_states[0]:
        warnings.append("quick repetition used a dirty tracked worktree")
    conversation_count = len({item["conversation_index"] for item in shape})
    return {
        "schema_version": 1,
        "kind": "multiturn_load_aggregate",
        "metric_definition": "last_output_token_v2",
        "profile": profile,
        "profile_fingerprint": profile_fingerprint,
        "architecture": architecture,
        "implementation": implementation,
        "implementation_fingerprint": _canonical_fingerprint(implementation),
        "hardware_signature": hardware,
        "mode": "formal" if formal else "quick",
        "repetition_count": len(results),
        "repetition_ids": repetition_ids,
        "git_commits": [result.get("git_commit") for result in results],
        "git_tracked_worktree_dirty": any(dirty_states),
        "validity": {"status": "passed"},
        "client_validation": {
            "status": "passed",
            "repetitions": [deepcopy(result["validity"]) for result in results],
        },
        "external_validation": {
            "status": "passed",
            "gates": external_gates[0],
            "repetitions": external_evidence,
        },
        "cache_validation": {
            "status": "passed",
            "repetitions": [
                deepcopy(result.get("cache_validation")) for result in results
            ],
        },
        "request_shape": {
            "conversation_count": conversation_count,
            "requests_per_repetition": len(shape),
            "entries": shape,
            "fingerprint": _canonical_fingerprint({"entries": shape}),
        },
        "aggregation_method": (
            "pooled request-level samples across repetitions; "
            "nearest-rank p90"
        ),
        "request_samples": samples,
        "metrics": _metric_summaries(samples),
        "per_repetition_metrics": _per_repetition_metrics(
            normalized_repetitions
        ),
        "digest_observations": digest_observations,
        "warnings": warnings,
    }


def _parse_shape(aggregate: Mapping[str, object]) -> list[dict[str, int]]:
    shape = _mapping(aggregate.get("request_shape"), "request shape")
    raw_entries = shape.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("request shape entries must be a non-empty list")
    entries: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for position, raw_entry in enumerate(raw_entries, start=1):
        entry = _mapping(raw_entry, f"request shape entry {position}")
        conversation_index = _integer(
            entry.get("conversation_index"),
            f"request shape entry {position} conversation_index",
        )
        round_index = _integer(
            entry.get("round"),
            f"request shape entry {position} round",
            minimum=1,
        )
        prompt_tokens = _integer(
            entry.get("prompt_tokens"),
            f"request shape entry {position} prompt_tokens",
            minimum=1,
        )
        completion_tokens = _integer(
            entry.get("completion_tokens"),
            f"request shape entry {position} completion_tokens",
            minimum=1,
        )
        if round_index not in EXPECTED_ROUNDS:
            raise ValueError(f"invalid request shape round: {round_index}")
        if completion_tokens != EXPECTED_COMPLETION_TOKENS:
            raise ValueError("request shape completion token count must be 256")
        key = (conversation_index, round_index)
        if key in seen:
            raise ValueError(f"duplicate request shape key: {key}")
        seen.add(key)
        entries.append(
            {
                "conversation_index": conversation_index,
                "round": round_index,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )
    entries.sort(key=lambda item: (item["conversation_index"], item["round"]))
    conversations = {entry["conversation_index"] for entry in entries}
    expected_keys = {
        (conversation_index, round_index)
        for conversation_index in conversations
        for round_index in EXPECTED_ROUNDS
    }
    if seen != expected_keys:
        raise ValueError("request shape does not contain five rounds per conversation")
    expected_fingerprint = _canonical_fingerprint({"entries": entries})
    if shape.get("fingerprint") != expected_fingerprint:
        raise ValueError("request shape fingerprint mismatch")
    if shape.get("conversation_count") != len(conversations):
        raise ValueError("request shape conversation_count mismatch")
    if shape.get("requests_per_repetition") != len(entries):
        raise ValueError("request shape requests_per_repetition mismatch")
    return entries


def _validate_summary(
    actual: object,
    values: Sequence[float],
    name: str,
) -> None:
    summary = _mapping(actual, name)
    expected = _summary(values)
    if set(summary) != set(expected):
        raise ValueError(f"{name} fields mismatch")
    for field, expected_value in expected.items():
        actual_value = summary.get(field)
        if field == "count":
            if actual_value != expected_value:
                raise ValueError(f"{name} count mismatch")
            continue
        parsed = _positive_finite(actual_value, f"{name} {field}")
        if not math.isclose(
            parsed,
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name} {field} mismatch")


def _validate_digest_observations(aggregate: Mapping[str, object]) -> None:
    observations = _mapping(
        aggregate.get("digest_observations"),
        "digest observations",
    )
    shape = _parse_shape(aggregate)
    expected_keys = {
        (entry["conversation_index"], entry["round"]) for entry in shape
    }
    for digest_name in DIGEST_FIELDS:
        raw_entries = observations.get(digest_name)
        if not isinstance(raw_entries, list):
            raise ValueError(f"missing {digest_name} observations")
        observed_keys: set[tuple[int, int]] = set()
        for position, raw_entry in enumerate(raw_entries, start=1):
            entry = _mapping(
                raw_entry,
                f"{digest_name} observation {position}",
            )
            key = (
                _integer(entry.get("conversation_index"), "conversation_index"),
                _integer(entry.get("round"), "round", minimum=1),
            )
            if key in observed_keys:
                raise ValueError(f"duplicate {digest_name} observation: {key}")
            observed_keys.add(key)
            digests = entry.get("digests")
            if not isinstance(digests, list) or not digests:
                raise ValueError(f"empty {digest_name} observation: {key}")
            parsed = [_digest(value, digest_name) for value in digests]
            if parsed != sorted(set(parsed)):
                raise ValueError(f"non-canonical {digest_name} observation: {key}")
        if observed_keys != expected_keys:
            raise ValueError(f"{digest_name} observation shape mismatch")


def _validate_aggregate(aggregate: Mapping[str, object]) -> None:
    if aggregate.get("schema_version") != 1:
        raise ValueError(
            f"unsupported aggregate schema: {aggregate.get('schema_version')}"
        )
    if aggregate.get("kind") != "multiturn_load_aggregate":
        raise ValueError("payload is not a multi-turn load aggregate")
    if aggregate.get("metric_definition") != "last_output_token_v2":
        raise ValueError("aggregate metric definition is invalid")
    mode_count = (aggregate.get("mode"), aggregate.get("repetition_count"))
    if mode_count not in {("quick", 1), ("formal", 3)}:
        raise ValueError(f"invalid aggregate mode/count pair: {mode_count}")
    if aggregate.get("validity") != {"status": "passed"}:
        raise ValueError("aggregate validity did not pass")
    repetition_count = int(mode_count[1])
    for validation_name in ("client_validation", "external_validation"):
        validator = (
            _external_validation_passed
            if validation_name == "external_validation"
            else _status_passed
        )
        validation = validator(
            aggregate.get(validation_name),
            f"aggregate {validation_name}",
        )
        evidence = validation.get("repetitions")
        if not isinstance(evidence, list) or len(evidence) != repetition_count:
            raise ValueError(
                f"aggregate {validation_name} repetition evidence mismatch"
            )
        for index, item in enumerate(evidence, start=1):
            evidence_validator = (
                _external_validation_passed
                if validation_name == "external_validation"
                else _status_passed
            )
            evidence_validator(
                item,
                f"aggregate {validation_name} repetition {index}",
            )
    _status_passed(aggregate.get("cache_validation"), "aggregate cache validation")
    profile = _validate_profile(aggregate)
    hardware = aggregate.get("hardware_signature")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("aggregate hardware_signature is invalid")
    if aggregate.get("architecture") not in {"pd", "pap"}:
        raise ValueError("aggregate architecture is invalid")
    dirty = aggregate.get("git_tracked_worktree_dirty")
    if not isinstance(dirty, bool):
        raise ValueError("aggregate tracked worktree state is invalid")
    if mode_count == ("formal", 3) and dirty:
        raise ValueError("formal aggregate cannot use a dirty tracked worktree")
    repetition_ids = aggregate.get("repetition_ids")
    if not isinstance(repetition_ids, list) or len(repetition_ids) != mode_count[1]:
        raise ValueError("aggregate repetition identity count mismatch")
    if len(set(repetition_ids)) != len(repetition_ids):
        raise ValueError("aggregate repetitions are not distinct")

    shape = _parse_shape(aggregate)
    conversation_count = len({entry["conversation_index"] for entry in shape})
    expected_round_samples = conversation_count * repetition_count
    raw_samples = _mapping(aggregate.get("request_samples"), "request samples")
    metrics = _mapping(aggregate.get("metrics"), "aggregate metrics")
    parsed_samples: dict[str, dict[str, list[float]]] = {}
    for scope in ALL_SCOPES:
        scope_samples = _mapping(raw_samples.get(scope), f"{scope} samples")
        scope_metrics = _mapping(metrics.get(scope), f"{scope} metrics")
        parsed_samples[scope] = {}
        expected_count = (
            expected_round_samples
            if scope != STEADY_SCOPE
            else expected_round_samples * 4
        )
        for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS):
            raw_values = scope_samples.get(metric)
            if not isinstance(raw_values, list) or len(raw_values) != expected_count:
                raise ValueError(f"{scope} {metric} sample count mismatch")
            values = [
                _positive_finite(value, f"{scope} {metric}")
                for value in raw_values
            ]
            parsed_samples[scope][metric] = values
            _validate_summary(
                scope_metrics.get(metric),
                values,
                f"{scope} {metric}",
            )
    for metric in (*PRIMARY_METRICS, *DIAGNOSTIC_METRICS):
        expected_steady = [
            value
            for round_index in EXPECTED_ROUNDS[1:]
            for value in parsed_samples[f"round_{round_index}"][metric]
        ]
        if parsed_samples[STEADY_SCOPE][metric] != expected_steady:
            raise ValueError(f"steady {metric} samples do not match rounds 2-5")
    _validate_digest_observations(aggregate)
    warnings = aggregate.get("warnings")
    if not isinstance(warnings, list) or not all(
        isinstance(warning, str) for warning in warnings
    ):
        raise ValueError("aggregate warnings must be a list of strings")


def _digest_map(
    aggregate: Mapping[str, object],
    digest_name: str,
) -> dict[tuple[int, int], tuple[str, ...]]:
    observations = _mapping(
        aggregate.get("digest_observations"),
        "digest observations",
    )
    entries = observations[digest_name]
    return {
        (int(entry["conversation_index"]), int(entry["round"])): tuple(
            str(value) for value in entry["digests"]
        )
        for entry in entries
    }


def compare_aggregates(
    pd_aggregate: Mapping[str, object],
    pap_aggregate: Mapping[str, object],
) -> dict[str, object]:
    """Compare valid PD and PAP load aggregates at identical request shape."""
    try:
        _validate_aggregate(pd_aggregate)
    except ValueError as exc:
        raise ValueError(f"invalid PD aggregate: {exc}") from exc
    try:
        _validate_aggregate(pap_aggregate)
    except ValueError as exc:
        raise ValueError(f"invalid PAP aggregate: {exc}") from exc
    if pd_aggregate.get("architecture") != "pd":
        raise ValueError("PD aggregate architecture must be pd")
    if pap_aggregate.get("architecture") != "pap":
        raise ValueError("PAP aggregate architecture must be pap")
    if pd_aggregate.get("profile_fingerprint") != pap_aggregate.get(
        "profile_fingerprint"
    ):
        raise ValueError("PD/PAP profile fingerprint mismatch")
    if pd_aggregate.get("hardware_signature") != pap_aggregate.get(
        "hardware_signature"
    ):
        raise ValueError("PD/PAP hardware signature mismatch")
    pd_mode = (pd_aggregate.get("mode"), pd_aggregate.get("repetition_count"))
    pap_mode = (
        pap_aggregate.get("mode"),
        pap_aggregate.get("repetition_count"),
    )
    if pd_mode != pap_mode:
        raise ValueError(f"PD/PAP repetition mode mismatch: {pd_mode} != {pap_mode}")

    pd_shape = _parse_shape(pd_aggregate)
    pap_shape = _parse_shape(pap_aggregate)
    if pd_shape != pap_shape:
        raise ValueError(
            "PD/PAP prompt token shape mismatch: "
            + _shape_mismatch(pd_shape, pap_shape)
        )

    warnings = [
        f"PD: {warning}" for warning in pd_aggregate.get("warnings", [])
    ]
    warnings.extend(
        f"PAP: {warning}" for warning in pap_aggregate.get("warnings", [])
    )
    digest_checks: dict[str, dict[str, object]] = {}
    for digest_name in DIGEST_FIELDS:
        pd_digests = _digest_map(pd_aggregate, digest_name)
        pap_digests = _digest_map(pap_aggregate, digest_name)
        mismatches: list[dict[str, object]] = []
        for key in sorted(pd_digests):
            if pd_digests[key] == pap_digests[key]:
                continue
            mismatch = {
                "conversation_index": key[0],
                "round": key[1],
                "pd_digests": list(pd_digests[key]),
                "pap_digests": list(pap_digests[key]),
            }
            mismatches.append(mismatch)
            warnings.append(
                f"conversation {key[0]} round {key[1]} {digest_name} "
                "differs between PD and PAP"
            )
        digest_checks[digest_name] = {
            "status": "warning" if mismatches else "matched",
            "mismatches": mismatches,
        }

    pd_metrics = _mapping(pd_aggregate.get("metrics"), "PD metrics")
    pap_metrics = _mapping(pap_aggregate.get("metrics"), "PAP metrics")
    comparison_metrics: dict[str, object] = {}
    for scope in ALL_SCOPES:
        pd_scope = _mapping(pd_metrics.get(scope), f"PD {scope}")
        pap_scope = _mapping(pap_metrics.get(scope), f"PAP {scope}")
        scope_comparison: dict[str, object] = {}
        for metric in PRIMARY_METRICS:
            pd_summary = _mapping(pd_scope.get(metric), f"PD {scope} {metric}")
            pap_summary = _mapping(
                pap_scope.get(metric),
                f"PAP {scope} {metric}",
            )
            ratios = {
                statistic: float(pap_summary[statistic])
                / float(pd_summary[statistic])
                for statistic in ("median", "p90", "max")
            }
            scope_comparison[metric] = {
                "pd": deepcopy(dict(pd_summary)),
                "pap": deepcopy(dict(pap_summary)),
                "pap_over_pd": ratios,
            }
        comparison_metrics[scope] = scope_comparison

    return {
        "schema_version": 1,
        "kind": "multiturn_load_comparison",
        "status": "valid",
        "profile_id": _mapping(
            pd_aggregate.get("profile"),
            "profile",
        ).get("profile_id"),
        "profile_fingerprint": pd_aggregate.get("profile_fingerprint"),
        "hardware_signature": pd_aggregate.get("hardware_signature"),
        "mode": pd_mode[0],
        "repetition_count": pd_mode[1],
        "shape_parity": {
            "status": "passed",
            "fingerprint": _mapping(
                pd_aggregate.get("request_shape"),
                "request shape",
            ).get("fingerprint"),
            "conversation_count": len(
                {entry["conversation_index"] for entry in pd_shape}
            ),
            "requests_per_repetition": len(pd_shape),
            "completion_tokens": EXPECTED_COMPLETION_TOKENS,
        },
        "metrics": comparison_metrics,
        "digest_checks": digest_checks,
        "warnings": list(dict.fromkeys(warnings)),
    }


def _aggregate_transport(
    aggregate: Mapping[str, object],
    lane_name: str,
) -> str:
    implementation = _mapping(
        aggregate.get("implementation"),
        f"{lane_name} implementation",
    )
    transport = implementation.get("offload_exec_transport")
    if not isinstance(transport, str) or not transport:
        raise ValueError(f"{lane_name} aggregate transport is invalid")
    return transport


def compare_three_aggregates(
    pd_oneway: Mapping[str, object],
    pd_twoway: Mapping[str, object],
    pap: Mapping[str, object],
) -> dict[str, object]:
    """Compare PD-oneway, PD-twoway, and PAP at an identical load shape."""
    lanes = {
        "pd_oneway": pd_oneway,
        "pd_twoway": pd_twoway,
        "pap": pap,
    }
    for lane_name, aggregate in lanes.items():
        try:
            _validate_aggregate(aggregate)
        except ValueError as exc:
            raise ValueError(f"invalid {lane_name} aggregate: {exc}") from exc

    expected_identity = {
        "pd_oneway": ("pd", "nixl-oneway"),
        "pd_twoway": ("pd", "nixl-twoway"),
        "pap": ("pap", "local_fast"),
    }
    for lane_name, aggregate in lanes.items():
        expected_architecture, expected_transport = expected_identity[lane_name]
        if aggregate.get("architecture") != expected_architecture:
            raise ValueError(
                f"{lane_name} architecture must be {expected_architecture}"
            )
        transport = _aggregate_transport(aggregate, lane_name)
        if transport != expected_transport:
            display_name = lane_name.replace("pd_", "PD-").replace("pap", "PAP")
            raise ValueError(
                f"{display_name} aggregate transport must be "
                f"{expected_transport}, got {transport}"
            )

    reference = pd_oneway
    reference_profile = reference.get("profile_fingerprint")
    reference_hardware = reference.get("hardware_signature")
    reference_mode = (
        reference.get("mode"),
        reference.get("repetition_count"),
    )
    reference_shape = _parse_shape(reference)
    for lane_name, aggregate in lanes.items():
        if aggregate.get("profile_fingerprint") != reference_profile:
            raise ValueError(f"{lane_name} profile fingerprint mismatch")
        if aggregate.get("hardware_signature") != reference_hardware:
            raise ValueError(f"{lane_name} hardware signature mismatch")
        mode = (aggregate.get("mode"), aggregate.get("repetition_count"))
        if mode != reference_mode:
            raise ValueError(
                f"{lane_name} repetition mode mismatch: {mode} != "
                f"{reference_mode}"
            )
        shape = _parse_shape(aggregate)
        if shape != reference_shape:
            raise ValueError(
                f"{lane_name} prompt token shape mismatch: "
                + _shape_mismatch(reference_shape, shape)
            )

    prompt_maps = {
        lane_name: _digest_map(aggregate, "prompt_token_digest")
        for lane_name, aggregate in lanes.items()
    }
    for key in sorted(prompt_maps["pd_oneway"]):
        values = {mapping[key] for mapping in prompt_maps.values()}
        if len(values) != 1:
            raise ValueError(
                "prompt token digest mismatch at conversation "
                f"{key[0]} round {key[1]}"
            )

    warnings = [
        f"{lane_name}: {warning}"
        for lane_name, aggregate in lanes.items()
        for warning in aggregate.get("warnings", [])
    ]
    digest_checks: dict[str, object] = {}
    for digest_name in ("output_token_digest", "assistant_text_digest"):
        maps = {
            lane_name: _digest_map(aggregate, digest_name)
            for lane_name, aggregate in lanes.items()
        }
        mismatches = []
        for key in sorted(maps["pd_oneway"]):
            values = {
                lane_name: list(mapping[key])
                for lane_name, mapping in maps.items()
            }
            if len({tuple(value) for value in values.values()}) == 1:
                continue
            mismatches.append(
                {
                    "conversation_index": key[0],
                    "round": key[1],
                    "digests": values,
                }
            )
            warnings.append(
                f"conversation {key[0]} round {key[1]} {digest_name} "
                "differs across the three lanes"
            )
        digest_checks[digest_name] = {
            "status": "warning" if mismatches else "matched",
            "mismatches": mismatches,
        }

    lane_metrics = {
        lane_name: _mapping(aggregate.get("metrics"), f"{lane_name} metrics")
        for lane_name, aggregate in lanes.items()
    }
    ratio_lanes = {
        "pd_twoway_over_pd_oneway": ("pd_twoway", "pd_oneway"),
        "pap_over_pd_oneway": ("pap", "pd_oneway"),
        "pap_over_pd_twoway": ("pap", "pd_twoway"),
    }
    ratios: dict[str, dict[str, object]] = {
        ratio_name: {} for ratio_name in ratio_lanes
    }
    metrics: dict[str, object] = {}
    for scope in ALL_SCOPES:
        metrics[scope] = {}
        for metric in PRIMARY_METRICS:
            summaries = {
                lane_name: deepcopy(
                    dict(
                        _mapping(
                            _mapping(values.get(scope), f"{lane_name} {scope}").get(
                                metric
                            ),
                            f"{lane_name} {scope} {metric}",
                        )
                    )
                )
                for lane_name, values in lane_metrics.items()
            }
            metric_ratios = {}
            for ratio_name, (numerator, denominator) in ratio_lanes.items():
                values = {
                    statistic: float(summaries[numerator][statistic])
                    / float(summaries[denominator][statistic])
                    for statistic in ("median", "p90", "max")
                }
                metric_ratios[ratio_name] = values
                ratios[ratio_name].setdefault(scope, {})[metric] = values
            metrics[scope][metric] = {
                **summaries,
                "ratios": metric_ratios,
            }

    return {
        "schema_version": 1,
        "kind": "multiturn_load_three_lane_comparison",
        "status": "valid",
        "profile_id": _mapping(reference.get("profile"), "profile").get(
            "profile_id"
        ),
        "profile_fingerprint": reference_profile,
        "hardware_signature": reference_hardware,
        "mode": reference_mode[0],
        "repetition_count": reference_mode[1],
        "shape_parity": {
            "status": "passed",
            "fingerprint": _mapping(
                reference.get("request_shape"),
                "request shape",
            ).get("fingerprint"),
            "conversation_count": len(
                {entry["conversation_index"] for entry in reference_shape}
            ),
            "requests_per_repetition": len(reference_shape),
            "completion_tokens": EXPECTED_COMPLETION_TOKENS,
        },
        "lane_transports": {
            lane_name: _aggregate_transport(aggregate, lane_name)
            for lane_name, aggregate in lanes.items()
        },
        "metrics": metrics,
        "ratios": ratios,
        "digest_checks": digest_checks,
        "warnings": list(dict.fromkeys(warnings)),
    }


def render_three_lane_markdown(comparison: Mapping[str, object]) -> str:
    """Render the three-lane multi-turn load matrix as Markdown."""
    if comparison.get("kind") != "multiturn_load_three_lane_comparison":
        raise ValueError("payload is not a three-lane load comparison")
    metrics = _mapping(comparison.get("metrics"), "comparison metrics")
    shape = _mapping(comparison.get("shape_parity"), "shape parity")
    lines = [
        "# PD-oneway / PD-twoway / PAP Multi-turn Load Comparison",
        "",
        f"- Profile: `{comparison.get('profile_id')}`",
        f"- Hardware: `{comparison.get('hardware_signature')}`",
        f"- Mode/repetitions: `{comparison.get('mode')}` / "
        f"`{comparison.get('repetition_count')}`",
        "- Shape parity: `passed` "
        f"({shape.get('conversation_count')} conversations, "
        f"{shape.get('requests_per_repetition')} requests/repetition)",
        "",
        "| Scope | Metric | Statistic | PD-oneway (ms) | "
        "PD-twoway (ms) | PAP (ms) | PD-twoway/PD-oneway | "
        "PAP/PD-oneway | PAP/PD-twoway |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    scope_labels = {
        **{f"round_{index}": f"R{index}" for index in EXPECTED_ROUNDS},
        STEADY_SCOPE: "R2-R5 steady",
    }
    metric_labels = {
        "ttft_ms": "TTFT",
        "tpot_ms": "TPOT",
        "latency_ms": "Latency",
    }
    for scope in ALL_SCOPES:
        scope_metrics = _mapping(metrics.get(scope), scope)
        for metric in PRIMARY_METRICS:
            values = _mapping(scope_metrics.get(metric), f"{scope} {metric}")
            one = _mapping(values.get("pd_oneway"), "PD-oneway summary")
            two = _mapping(values.get("pd_twoway"), "PD-twoway summary")
            pap = _mapping(values.get("pap"), "PAP summary")
            ratio_values = _mapping(values.get("ratios"), "ratios")
            for statistic in ("median", "p90", "max"):
                lines.append(
                    "| {scope} | {metric} | {statistic} | {one:.3f} | "
                    "{two:.3f} | {pap:.3f} | {two_one:.3f}x | "
                    "{pap_one:.3f}x | {pap_two:.3f}x |".format(
                        scope=scope_labels[scope],
                        metric=metric_labels[metric],
                        statistic=statistic,
                        one=float(one[statistic]),
                        two=float(two[statistic]),
                        pap=float(pap[statistic]),
                        two_one=float(
                            _mapping(
                                ratio_values.get(
                                    "pd_twoway_over_pd_oneway"
                                ),
                                "PD-twoway/PD-oneway",
                            )[statistic]
                        ),
                        pap_one=float(
                            _mapping(
                                ratio_values.get("pap_over_pd_oneway"),
                                "PAP/PD-oneway",
                            )[statistic]
                        ),
                        pap_two=float(
                            _mapping(
                                ratio_values.get("pap_over_pd_twoway"),
                                "PAP/PD-twoway",
                            )[statistic]
                        ),
                    )
                )
    warnings = comparison.get("warnings")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.extend(["", "No digest or provenance warnings."])
    return "\n".join(lines) + "\n"


def render_markdown(comparison: Mapping[str, object]) -> str:
    """Render a request-level PD/PAP load comparison as Markdown."""
    if comparison.get("kind") != "multiturn_load_comparison":
        raise ValueError("payload is not a multi-turn load comparison")
    metrics = _mapping(comparison.get("metrics"), "comparison metrics")
    shape = _mapping(comparison.get("shape_parity"), "shape parity")
    lines = [
        "# PAP/PD Multi-turn Load Comparison",
        "",
        f"- Profile: `{comparison.get('profile_id')}`",
        f"- Hardware: `{comparison.get('hardware_signature')}`",
        f"- Mode/repetitions: `{comparison.get('mode')}` / "
        f"`{comparison.get('repetition_count')}`",
        "- Shape parity: `passed` "
        f"({shape.get('conversation_count')} conversations, "
        f"{shape.get('requests_per_repetition')} requests/repetition, "
        f"{shape.get('completion_tokens')} completion tokens/request)",
        "",
        "| Scope | Metric | Statistic | PD (ms) | PAP (ms) | PAP/PD |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    scope_labels = {
        **{f"round_{index}": f"R{index}" for index in EXPECTED_ROUNDS},
        STEADY_SCOPE: "R2-R5 steady",
    }
    metric_labels = {
        "ttft_ms": "TTFT",
        "tpot_ms": "TPOT",
        "latency_ms": "Latency",
    }
    for scope in ALL_SCOPES:
        scope_metrics = _mapping(metrics.get(scope), scope)
        for metric in PRIMARY_METRICS:
            values = _mapping(scope_metrics.get(metric), f"{scope} {metric}")
            pd_values = _mapping(values.get("pd"), "PD summary")
            pap_values = _mapping(values.get("pap"), "PAP summary")
            ratios = _mapping(values.get("pap_over_pd"), "PAP/PD ratios")
            for statistic in ("median", "p90", "max"):
                lines.append(
                    "| {scope} | {metric} | {statistic} | {pd:.3f} | "
                    "{pap:.3f} | {ratio:.3f}x |".format(
                        scope=scope_labels[scope],
                        metric=metric_labels[metric],
                        statistic=statistic,
                        pd=float(pd_values[statistic]),
                        pap=float(pap_values[statistic]),
                        ratio=float(ratios[statistic]),
                    )
                )
    warnings = comparison.get("warnings")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.extend(["", "No digest or provenance warnings."])
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate and compare PAP/PD multi-turn load results"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("results", nargs="*", type=Path)
    aggregate_parser.add_argument(
        "--result",
        dest="result_options",
        action="append",
        type=Path,
        default=[],
    )
    aggregate_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--pd", type=Path, required=True)
    compare_parser.add_argument("--pap", type=Path, required=True)
    compare_parser.add_argument("--output-json", type=Path, required=True)
    compare_parser.add_argument("--output-markdown", type=Path, required=True)

    compare_three_parser = subparsers.add_parser("compare-three")
    compare_three_parser.add_argument("--pd-oneway", type=Path, required=True)
    compare_three_parser.add_argument("--pd-twoway", type=Path, required=True)
    compare_three_parser.add_argument("--pap", type=Path, required=True)
    compare_three_parser.add_argument("--output-json", type=Path, required=True)
    compare_three_parser.add_argument(
        "--output-markdown",
        type=Path,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "aggregate":
        result_paths = [*args.results, *args.result_options]
        if len(result_paths) not in {1, 3}:
            raise ValueError(
                "aggregate requires one or three result paths, "
                f"got {len(result_paths)}"
            )
        resolved = [path.resolve() for path in result_paths]
        if len(set(resolved)) != len(resolved):
            raise ValueError("aggregate requires distinct result paths")
        aggregate = aggregate_repetitions(
            [_load_json(path) for path in result_paths]
        )
        aggregate["source_results"] = [str(path) for path in result_paths]
        _write_json(args.output, aggregate)
        print(json.dumps(aggregate, indent=2))
        return

    if args.command == "compare-three":
        comparison = compare_three_aggregates(
            _load_json(args.pd_oneway),
            _load_json(args.pd_twoway),
            _load_json(args.pap),
        )
        markdown = render_three_lane_markdown(comparison)
    else:
        comparison = compare_aggregates(
            _load_json(args.pd),
            _load_json(args.pap),
        )
        markdown = render_markdown(comparison)
    _write_json(args.output_json, comparison)
    _write_text(args.output_markdown, markdown)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
