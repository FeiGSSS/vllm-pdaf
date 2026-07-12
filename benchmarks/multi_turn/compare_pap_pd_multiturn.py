"""Aggregate and compare PAP/PD multi-turn north-star results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


ROUND_METRICS = ("ttft_ms", "tpot_ms", "latency_ms")
ROUND_DIAGNOSTICS = ("eof_latency_ms", "post_token_stream_ms")
DIGEST_FIELDS = (
    ("prompt_token_digest", "prompt token digest"),
    ("output_token_digest", "output token digest"),
    ("assistant_text_digest", "assistant text digest"),
)
PROFILE_ID = "qwen3_8b_chat_16k_2turn_o256_c1_v1"
METRIC_DEFINITION = "last_output_token_v2"
REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_EXTERNAL_GATES = {
    "pap": frozenset(
        {
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats_capture",
        }
    ),
    "pd": frozenset({"pd_reuse_metrics", "correctness_logs"}),
}
REQUIRED_EXTERNAL_ARTIFACTS = {
    "pap": frozenset(
        {
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats",
            "run_metadata",
            "tracked_worktree_patch",
            "tracked_index_patch",
        }
    ),
    "pd": frozenset(
        {
            "proxy_log",
            "prefill_metrics",
            "decode_metrics",
            "effective_config",
            "correctness_logs",
            "tracked_worktree_patch",
            "tracked_index_patch",
        }
    ),
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive: {value}")
    return parsed


def _nonnegative_finite(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be finite and nonnegative: {value}")
    return parsed


def _fingerprint(value: Mapping[str, object]) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_implementation(result: Mapping[str, object]) -> None:
    commit = result.get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"invalid git commit: {commit}")
    if not isinstance(result.get("git_tracked_worktree_dirty"), bool):
        raise ValueError("git_tracked_worktree_dirty must be boolean")
    implementation = _mapping(result.get("implementation"), "implementation")
    transport = implementation.get("offload_exec_transport")
    if not isinstance(transport, str) or not transport:
        raise ValueError("implementation transport is missing")
    if not isinstance(implementation.get("direct_mailbox_output"), bool):
        raise ValueError("implementation direct_mailbox_output must be boolean")
    expected = _fingerprint(dict(implementation))
    if result.get("implementation_fingerprint") != expected:
        raise ValueError("implementation fingerprint mismatch")


def _validate_external_gates(result: Mapping[str, object]) -> None:
    architecture = result.get("architecture")
    if architecture not in REQUIRED_EXTERNAL_GATES:
        raise ValueError(f"unsupported architecture: {architecture}")
    validation = _mapping(
        result.get("external_validation"),
        "external validation",
    )
    if validation.get("status") != "passed":
        raise ValueError(f"external validation did not pass: {validation}")
    gates = _mapping(validation.get("gates"), "external validation gates")
    required = REQUIRED_EXTERNAL_GATES[str(architecture)]
    if set(gates) != required:
        raise ValueError(
            f"external gate set mismatch: {sorted(gates)} != {sorted(required)}"
        )
    for gate in sorted(required):
        if gates.get(gate) != "passed":
            raise ValueError(f"external gate {gate} did not pass")

    def validate_artifacts(
        raw_artifacts: object,
        evidence_name: str,
    ) -> None:
        artifacts = _mapping(raw_artifacts, f"{evidence_name} artifacts")
        required_artifacts = REQUIRED_EXTERNAL_ARTIFACTS[str(architecture)]
        if not required_artifacts.issubset(artifacts):
            missing = sorted(required_artifacts - set(artifacts))
            raise ValueError(f"{evidence_name} missing artifacts: {missing}")
        for artifact_name in sorted(required_artifacts):
            evidence = _mapping(
                artifacts.get(artifact_name),
                f"{evidence_name} artifact {artifact_name}",
            )
            path = evidence.get("path")
            digest = evidence.get("sha256")
            if (
                not isinstance(path, str)
                or not path
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                raise ValueError(
                    f"{evidence_name} artifact path must be relative: {path}"
                )
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                raise ValueError(
                    f"{evidence_name} artifact digest is invalid: {digest}"
                )

    repetition_count = result.get("repetition_count")
    if repetition_count is None:
        validate_artifacts(validation.get("artifacts"), "repetition")
        return
    evidence = validation.get("repetition_evidence")
    if not isinstance(evidence, list) or len(evidence) != repetition_count:
        raise ValueError("external repetition evidence count mismatch")
    for index, raw_evidence in enumerate(evidence, start=1):
        item = _mapping(raw_evidence, f"external evidence {index}")
        if item.get("status") != "passed" or item.get("gates") != gates:
            raise ValueError(f"external evidence {index} did not pass")
        validate_artifacts(item.get("artifacts"), f"external evidence {index}")


def validate_repetition(result: Mapping[str, object]) -> None:
    """Fail closed unless one repetition satisfies all client-side gates."""
    if result.get("schema_version") != 2:
        raise ValueError(
            f"unsupported repetition schema: {result.get('schema_version')}"
        )
    if result.get("metric_definition") != METRIC_DEFINITION:
        raise ValueError(
            f"unexpected metric definition: {result.get('metric_definition')}"
        )
    validity = _mapping(result.get("validity"), "validity")
    if validity.get("status") != "passed":
        raise ValueError(f"repetition validity did not pass: {validity}")
    profile = _mapping(result.get("profile"), "profile")
    if profile.get("profile_id") != PROFILE_ID:
        raise ValueError(f"unexpected profile ID: {profile.get('profile_id')}")
    expected_output = profile.get("output_tokens_per_round")
    if not isinstance(expected_output, int) or expected_output <= 0:
        raise ValueError("profile output_tokens_per_round must be positive")
    fingerprint = result.get("profile_fingerprint")
    if fingerprint != _fingerprint(dict(profile)):
        raise ValueError("profile fingerprint mismatch")
    hardware = result.get("hardware_signature")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("missing hardware signature")
    _validate_implementation(result)
    _validate_external_gates(result)
    conversation_id_digest = result.get("conversation_id_digest")
    if not isinstance(conversation_id_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", conversation_id_digest
    ) is None:
        raise ValueError("invalid conversation identity digest")

    rounds = result.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise ValueError("repetition must contain exactly two rounds")
    for expected_round, raw_round in enumerate(rounds, start=1):
        round_result = _mapping(raw_round, f"round {expected_round}")
        if round_result.get("round") != expected_round:
            raise ValueError(f"round index mismatch: {round_result.get('round')}")
        if round_result.get("completion_tokens") != expected_output:
            raise ValueError(
                "completion token count mismatch: "
                f"{round_result.get('completion_tokens')} != {expected_output}"
            )
        if round_result.get("finish_reason") != "length":
            raise ValueError(
                f"round {expected_round} finish reason is not length"
            )
        if round_result.get("saw_done") is not True:
            raise ValueError(f"round {expected_round} did not consume [DONE]")
        for metric in ROUND_METRICS:
            _positive_finite(
                round_result.get(metric),
                f"round {expected_round} {metric}",
            )
        eof_latency = _positive_finite(
            round_result.get("eof_latency_ms"),
            f"round {expected_round} eof_latency_ms",
        )
        post_token = _nonnegative_finite(
            round_result.get("post_token_stream_ms"),
            f"round {expected_round} post_token_stream_ms",
        )
        latency = _positive_finite(
            round_result.get("latency_ms"),
            f"round {expected_round} latency_ms",
        )
        tpot = _positive_finite(
            round_result.get("tpot_ms"),
            f"round {expected_round} tpot_ms",
        )
        expected_tpot = (latency - float(round_result["ttft_ms"])) / (
            expected_output - 1
        )
        if not math.isclose(
            tpot,
            expected_tpot,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"round {expected_round} TPOT accounting mismatch"
            )
        if not math.isclose(
            eof_latency - latency,
            post_token,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"round {expected_round} EOF timing accounting mismatch"
            )
        for digest_name, _digest_label in DIGEST_FIELDS:
            digest = round_result.get(digest_name)
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                raise ValueError(
                    f"round {expected_round} has invalid {digest_name}"
                )
    conversation_latency = _positive_finite(
        result.get("conversation_latency_ms"),
        "conversation_latency_ms",
    )
    conversation_eof_latency = _positive_finite(
        result.get("conversation_eof_latency_ms"),
        "conversation_eof_latency_ms",
    )
    first_round = _mapping(rounds[0], "round 1")
    second_round = _mapping(rounds[1], "round 2")
    expected_conversation_latency = float(first_round["eof_latency_ms"]) + float(
        second_round["latency_ms"]
    )
    expected_conversation_eof = float(first_round["eof_latency_ms"]) + float(
        second_round["eof_latency_ms"]
    )
    if not math.isclose(
        conversation_latency,
        expected_conversation_latency,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("conversation last-token latency accounting mismatch")
    if not math.isclose(
        conversation_eof_latency,
        expected_conversation_eof,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("conversation EOF latency accounting mismatch")

    cache = _mapping(result.get("cache_validation"), "cache validation")
    architecture = result.get("architecture")
    allowed_cache_statuses = (
        {"passed"}
        if architecture == "pap"
        else {"official_streaming_one_way_metrics_passed"}
    )
    if cache.get("status") not in allowed_cache_statuses:
        raise ValueError(f"cache validation did not pass: {cache}")
    if int(cache.get("decode_derived_hit_tokens", 0)) < 16:
        raise ValueError("cache validation has no Decode-derived block")


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


def aggregate_repetitions(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate one quick or three formal repetitions."""
    if len(results) not in {1, 3}:
        raise ValueError(
            "north-star aggregate requires one quick or three formal "
            f"repetitions, got {len(results)}"
        )
    for result in results:
        validate_repetition(result)

    fingerprint = _same_value(
        results,
        "profile_fingerprint",
        "profile fingerprint",
    )
    architecture = _same_value(results, "architecture", "architecture")
    hardware = _same_value(results, "hardware_signature", "hardware signature")
    profile = _same_value(results, "profile", "profile")
    topology = _same_value(results, "topology", "topology")
    metric_definition = _same_value(
        results,
        "metric_definition",
        "metric definition",
    )
    git_commit = _same_value(results, "git_commit", "git commit")
    git_tracked_worktree_dirty = _same_value(
        results,
        "git_tracked_worktree_dirty",
        "tracked worktree state",
    )
    if len(results) == 3 and git_tracked_worktree_dirty is not False:
        raise ValueError("formal aggregate cannot use a dirty tracked worktree")
    repetition_ids = [str(result["conversation_id_digest"]) for result in results]
    if len(set(repetition_ids)) != len(repetition_ids):
        raise ValueError("formal inputs must be distinct repetitions")
    implementation = _same_value(results, "implementation", "implementation")
    implementation_fingerprint = _same_value(
        results,
        "implementation_fingerprint",
        "implementation fingerprint",
    )
    external_gates = [
        dict(
            _mapping(
                _mapping(result["external_validation"], "external validation").get(
                    "gates"
                ),
                "external validation gates",
            )
        )
        for result in results
    ]
    if any(gates != external_gates[0] for gates in external_gates[1:]):
        raise ValueError(f"mixed external validation gates: {external_gates}")

    raw_metrics: dict[str, object] = {}
    metrics: dict[str, object] = {}
    for round_index in (1, 2):
        raw_round: dict[str, list[float]] = {}
        aggregate_round: dict[str, float] = {}
        for metric in (*ROUND_METRICS, *ROUND_DIAGNOSTICS):
            values = [
                float(result["rounds"][round_index - 1][metric])  # type: ignore[index]
                for result in results
            ]
            raw_round[metric] = values
            aggregate_round[metric] = float(median(values))
        raw_metrics[f"round_{round_index}"] = raw_round
        metrics[f"round_{round_index}"] = aggregate_round
    conversation_values = [
        float(result["conversation_latency_ms"]) for result in results
    ]
    raw_metrics["conversation_latency_ms"] = conversation_values
    metrics["conversation_latency_ms"] = float(median(conversation_values))
    conversation_eof_values = [
        float(result["conversation_eof_latency_ms"]) for result in results
    ]
    raw_metrics["conversation_eof_latency_ms"] = conversation_eof_values
    metrics["conversation_eof_latency_ms"] = float(
        median(conversation_eof_values)
    )

    correctness_signatures: dict[str, dict[str, str]] = {}
    for round_index in (1, 2):
        round_signatures: dict[str, str] = {}
        for digest_name, digest_label in DIGEST_FIELDS:
            values = [
                str(
                    result["rounds"][round_index - 1][  # type: ignore[index]
                        digest_name
                    ]
                )
                for result in results
            ]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(
                    f"mixed round {round_index} {digest_label}: {values}"
                )
            round_signatures[digest_name] = values[0]
        correctness_signatures[f"round_{round_index}"] = round_signatures

    return {
        "schema_version": 2,
        "kind": "aggregate",
        "metric_definition": metric_definition,
        "profile": profile,
        "profile_fingerprint": fingerprint,
        "architecture": architecture,
        "topology": topology,
        "hardware_signature": hardware,
        "git_commit": git_commit,
        "git_tracked_worktree_dirty": git_tracked_worktree_dirty,
        "implementation": implementation,
        "implementation_fingerprint": implementation_fingerprint,
        "mode": "formal" if len(results) == 3 else "quick",
        "repetition_count": len(results),
        "repetition_ids": repetition_ids,
        "validity": {"status": "passed"},
        "external_validation": {
            "status": "passed",
            "gates": external_gates[0],
            "repetition_evidence": [
                deepcopy(result["external_validation"]) for result in results
            ],
        },
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "correctness_signatures": correctness_signatures,
    }


def classify_tpot(
    candidate_ms: float,
    reference_ms: float,
    threshold: float = 0.03,
) -> str:
    """Classify a formal candidate at the symmetric percentage boundary."""
    candidate = _positive_finite(candidate_ms, "candidate TPOT")
    reference = _positive_finite(reference_ms, "reference TPOT")
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between zero and one: {threshold}")
    ratio = candidate / reference
    if ratio <= 1.0 - threshold:
        return "improved"
    if ratio >= 1.0 + threshold:
        return "regressed"
    return "neutral"


def _validate_aggregate(aggregate: Mapping[str, object]) -> None:
    if aggregate.get("schema_version") != 2:
        raise ValueError(
            f"unsupported aggregate schema: {aggregate.get('schema_version')}"
        )
    if aggregate.get("validity") != {"status": "passed"}:
        raise ValueError("aggregate validity did not pass")
    if aggregate.get("metric_definition") != METRIC_DEFINITION:
        raise ValueError(
            f"unexpected metric definition: {aggregate.get('metric_definition')}"
        )
    mode_count = (aggregate.get("mode"), aggregate.get("repetition_count"))
    if mode_count not in {("quick", 1), ("formal", 3)}:
        raise ValueError(f"invalid aggregate mode/count pair: {mode_count}")
    if mode_count == ("formal", 3) and aggregate.get(
        "git_tracked_worktree_dirty"
    ) is not False:
        raise ValueError("formal aggregate cannot use a dirty tracked worktree")
    repetition_count = int(aggregate["repetition_count"])
    repetition_ids = aggregate.get("repetition_ids")
    if not isinstance(repetition_ids, list) or len(repetition_ids) != repetition_count:
        raise ValueError("repetition identity count mismatch")
    for repetition_id in repetition_ids:
        if not isinstance(repetition_id, str) or re.fullmatch(
            r"[0-9a-f]{64}", repetition_id
        ) is None:
            raise ValueError("aggregate has an invalid repetition identity")
    if len(set(repetition_ids)) != repetition_count:
        raise ValueError("aggregate does not contain distinct repetitions")
    profile = _mapping(aggregate.get("profile"), "profile")
    if profile.get("profile_id") != PROFILE_ID:
        raise ValueError(f"unexpected profile ID: {profile.get('profile_id')}")
    if aggregate.get("profile_fingerprint") != _fingerprint(dict(profile)):
        raise ValueError("aggregate profile fingerprint mismatch")
    hardware = aggregate.get("hardware_signature")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("missing or invalid hardware signature")
    _validate_implementation(aggregate)
    _validate_external_gates(aggregate)
    metrics = _mapping(aggregate.get("metrics"), "aggregate metrics")
    raw_metrics = _mapping(aggregate.get("raw_metrics"), "raw metrics")
    for round_name in ("round_1", "round_2"):
        round_metrics = _mapping(metrics.get(round_name), round_name)
        raw_round = _mapping(raw_metrics.get(round_name), f"raw {round_name}")
        for metric in (*ROUND_METRICS, *ROUND_DIAGNOSTICS):
            parser = (
                _nonnegative_finite
                if metric == "post_token_stream_ms"
                else _positive_finite
            )
            aggregate_value = parser(
                round_metrics.get(metric),
                f"{round_name} {metric}",
            )
            raw_values = raw_round.get(metric)
            if not isinstance(raw_values, list) or len(raw_values) != repetition_count:
                raise ValueError(f"raw {round_name} {metric} count mismatch")
            parsed_values = [
                parser(value, f"raw {round_name} {metric}")
                for value in raw_values
            ]
            if not math.isclose(
                aggregate_value,
                float(median(parsed_values)),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(f"{round_name} {metric} median mismatch")
    for metric in ("conversation_latency_ms", "conversation_eof_latency_ms"):
        aggregate_value = _positive_finite(metrics.get(metric), metric)
        raw_values = raw_metrics.get(metric)
        if not isinstance(raw_values, list) or len(raw_values) != repetition_count:
            raise ValueError(f"raw {metric} count mismatch")
        parsed_values = [
            _positive_finite(value, f"raw {metric}") for value in raw_values
        ]
        if not math.isclose(
            aggregate_value,
            float(median(parsed_values)),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{metric} median mismatch")
    output_tokens = profile.get("output_tokens_per_round")
    if not isinstance(output_tokens, int) or output_tokens <= 1:
        raise ValueError("aggregate output token count must exceed one")
    raw_round_1 = _mapping(raw_metrics.get("round_1"), "raw round_1")
    raw_round_2 = _mapping(raw_metrics.get("round_2"), "raw round_2")
    for index in range(repetition_count):
        for round_name, raw_round in (
            ("round_1", raw_round_1),
            ("round_2", raw_round_2),
        ):
            ttft = float(raw_round["ttft_ms"][index])
            latency = float(raw_round["latency_ms"][index])
            tpot = float(raw_round["tpot_ms"][index])
            eof_latency = float(raw_round["eof_latency_ms"][index])
            post_token = float(raw_round["post_token_stream_ms"][index])
            expected_tpot = (latency - ttft) / (output_tokens - 1)
            if not math.isclose(
                tpot,
                expected_tpot,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"raw {round_name} repetition {index + 1} "
                    "TPOT accounting mismatch"
                )
            if not math.isclose(
                eof_latency - latency,
                post_token,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"raw {round_name} repetition {index + 1} "
                    "EOF accounting mismatch"
                )
        conversation = float(raw_metrics["conversation_latency_ms"][index])
        conversation_eof = float(
            raw_metrics["conversation_eof_latency_ms"][index]
        )
        expected_conversation = float(raw_round_1["eof_latency_ms"][index]) + float(
            raw_round_2["latency_ms"][index]
        )
        expected_conversation_eof = float(
            raw_round_1["eof_latency_ms"][index]
        ) + float(raw_round_2["eof_latency_ms"][index])
        if not math.isclose(
            conversation,
            expected_conversation,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"raw repetition {index + 1} conversation accounting mismatch"
            )
        if not math.isclose(
            conversation_eof,
            expected_conversation_eof,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"raw repetition {index + 1} conversation EOF accounting mismatch"
            )
    signatures = _mapping(
        aggregate.get("correctness_signatures"),
        "correctness signatures",
    )
    for round_name in ("round_1", "round_2"):
        round_signatures = _mapping(signatures.get(round_name), round_name)
        for digest_name, _digest_label in DIGEST_FIELDS:
            digest = round_signatures.get(digest_name)
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                raise ValueError(f"invalid {round_name} {digest_name}")


def _validate_source_results(payload: Mapping[str, object]) -> None:
    repetition_count = payload.get("repetition_count")
    source_results = payload.get("source_results")
    if not isinstance(repetition_count, int) or not isinstance(source_results, list):
        raise ValueError("reference source results are missing")
    if len(source_results) != repetition_count:
        raise ValueError("reference source result count mismatch")
    for source in source_results:
        if (
            not isinstance(source, str)
            or not source
            or Path(source).is_absolute()
            or ".." in Path(source).parts
        ):
            raise ValueError(f"reference source path must be relative: {source}")
    if len(set(source_results)) != len(source_results):
        raise ValueError("reference requires distinct source results")


def _validate_reference(
    reference: Mapping[str, object],
    architecture: str,
) -> None:
    _validate_aggregate(reference)
    _validate_source_results(reference)
    if reference.get("kind") != "reference":
        raise ValueError("payload is not a reference")
    if reference.get("mode") != "formal" or reference.get("repetition_count") != 3:
        raise ValueError("reference must contain three formal repetitions")
    if reference.get("architecture") != architecture:
        raise ValueError(f"reference architecture is not {architecture}")
    if reference.get("reference_architecture") != architecture:
        raise ValueError(f"reference_architecture is not {architecture}")


def validate_reference_pair(
    pd_reference: Mapping[str, object],
    pap_reference: Mapping[str, object],
) -> None:
    """Validate that the tracked PD and PAP references are comparable."""
    for label, reference, architecture in (
        ("PD reference", pd_reference, "pd"),
        ("PAP reference", pap_reference, "pap"),
    ):
        try:
            _validate_reference(reference, architecture)
        except ValueError as exc:
            raise ValueError(f"invalid {label}: {exc}") from exc

    fingerprints = {
        pd_reference.get("profile_fingerprint"),
        pap_reference.get("profile_fingerprint"),
    }
    if len(fingerprints) != 1:
        raise ValueError(f"profile fingerprint mismatch: {fingerprints}")
    hardware = {
        pd_reference.get("hardware_signature"),
        pap_reference.get("hardware_signature"),
    }
    if len(hardware) != 1:
        raise ValueError(f"hardware signature mismatch: {hardware}")

    pd_signatures = _mapping(
        pd_reference.get("correctness_signatures"),
        "PD correctness signatures",
    )
    pap_signatures = _mapping(
        pap_reference.get("correctness_signatures"),
        "PAP correctness signatures",
    )
    for round_name in ("round_1", "round_2"):
        pd_prompt_digest = _mapping(
            pd_signatures.get(round_name),
            f"PD {round_name}",
        ).get("prompt_token_digest")
        pap_prompt_digest = _mapping(
            pap_signatures.get(round_name),
            f"PAP {round_name}",
        ).get("prompt_token_digest")
        if pd_prompt_digest != pap_prompt_digest:
            raise ValueError(
                f"{round_name} prompt token digest mismatch: "
                f"{pd_prompt_digest} != {pap_prompt_digest}"
            )


def make_reference(
    aggregate: Mapping[str, object],
    *,
    architecture: str,
) -> dict[str, object]:
    """Create a tracked reference payload from a valid formal aggregate."""
    _validate_aggregate(aggregate)
    if aggregate.get("kind") != "aggregate":
        raise ValueError("reference source must be an aggregate")
    if aggregate.get("mode") != "formal" or aggregate.get("repetition_count") != 3:
        raise ValueError("reference requires a three-repetition formal aggregate")
    if aggregate.get("architecture") != architecture:
        raise ValueError(
            "reference architecture mismatch: "
            f"{aggregate.get('architecture')} != {architecture}"
        )
    _validate_source_results(aggregate)
    reference = deepcopy(dict(aggregate))
    reference["kind"] = "reference"
    reference["reference_architecture"] = architecture
    reference["created_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    return reference


def _metric_value(
    result: Mapping[str, object],
    round_name: str,
    metric: str,
) -> float:
    metrics = _mapping(result.get("metrics"), "metrics")
    round_metrics = _mapping(metrics.get(round_name), round_name)
    return _positive_finite(round_metrics.get(metric), f"{round_name} {metric}")


def _diagnostic_value(
    result: Mapping[str, object],
    round_name: str,
    metric: str,
) -> float:
    metrics = _mapping(result.get("metrics"), "metrics")
    round_metrics = _mapping(metrics.get(round_name), round_name)
    parser = (
        _nonnegative_finite
        if metric == "post_token_stream_ms"
        else _positive_finite
    )
    return parser(round_metrics.get(metric), f"{round_name} {metric}")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def _round_signatures(
    aggregate: Mapping[str, object],
    round_name: str,
) -> Mapping[str, Any]:
    signatures = _mapping(
        aggregate.get("correctness_signatures"),
        "correctness signatures",
    )
    return _mapping(signatures.get(round_name), round_name)


def compare_candidate(
    candidate: Mapping[str, object],
    pd_reference: Mapping[str, object],
    pap_reference: Mapping[str, object],
) -> dict[str, object]:
    """Compare a PAP candidate with fixed PD and PAP references."""
    try:
        _validate_aggregate(candidate)
    except ValueError as exc:
        raise ValueError(f"invalid candidate: {exc}") from exc
    if candidate.get("kind") != "aggregate":
        raise ValueError("candidate payload is not an aggregate")
    validate_reference_pair(pd_reference, pap_reference)
    if candidate.get("architecture") != "pap":
        raise ValueError("candidate architecture must be pap")

    fingerprints = {
        candidate.get("profile_fingerprint"),
        pd_reference.get("profile_fingerprint"),
        pap_reference.get("profile_fingerprint"),
    }
    if len(fingerprints) != 1:
        raise ValueError(f"profile fingerprint mismatch: {fingerprints}")
    hardware = {
        candidate.get("hardware_signature"),
        pd_reference.get("hardware_signature"),
        pap_reference.get("hardware_signature"),
    }
    if len(hardware) != 1:
        raise ValueError(f"hardware signature mismatch: {hardware}")

    comparison_metrics: dict[str, object] = {}
    diagnostic_metrics: dict[str, object] = {}
    warnings: list[str] = []
    if candidate.get("git_tracked_worktree_dirty") is True:
        warnings.append("candidate used a dirty tracked worktree")
    correctness: dict[str, object] = {}
    for round_name in ("round_1", "round_2"):
        candidate_signatures = _round_signatures(candidate, round_name)
        pd_signatures = _round_signatures(pd_reference, round_name)
        pap_signatures = _round_signatures(pap_reference, round_name)
        prompt_digests = {
            candidate_signatures["prompt_token_digest"],
            pd_signatures["prompt_token_digest"],
            pap_signatures["prompt_token_digest"],
        }
        prompt_digest_match = len(prompt_digests) == 1
        if not prompt_digest_match:
            raise ValueError(
                f"{round_name} prompt token digest mismatch: {prompt_digests}"
            )
        candidate_output = candidate_signatures["output_token_digest"]
        pd_output = pd_signatures["output_token_digest"]
        pap_output = pap_signatures["output_token_digest"]
        candidate_text = candidate_signatures["assistant_text_digest"]
        pd_text = pd_signatures["assistant_text_digest"]
        pap_text = pap_signatures["assistant_text_digest"]
        candidate_matches_pd = candidate_output == pd_output
        candidate_matches_pap = candidate_output == pap_output
        pd_matches_pap = pd_output == pap_output
        candidate_text_matches_pd = candidate_text == pd_text
        candidate_text_matches_pap = candidate_text == pap_text
        pd_text_matches_pap = pd_text == pap_text
        correctness[round_name] = {
            "prompt_digest_match": prompt_digest_match,
            "candidate_output_matches_pd": candidate_matches_pd,
            "candidate_output_matches_pap_reference": candidate_matches_pap,
            "pd_output_matches_pap_reference": pd_matches_pap,
            "candidate_text_matches_pd": candidate_text_matches_pd,
            "candidate_text_matches_pap_reference": candidate_text_matches_pap,
            "pd_text_matches_pap_reference": pd_text_matches_pap,
        }
        if not pd_matches_pap:
            warnings.append(
                f"{round_name} PD output digest differs from PAP reference"
            )
        if not pd_text_matches_pap:
            warnings.append(
                f"{round_name} PD assistant text digest differs from PAP reference"
            )
        if not candidate_matches_pap or not candidate_text_matches_pap:
            raise ValueError(
                f"{round_name} candidate exact output differs from PAP reference"
            )
    for round_name in ("round_1", "round_2"):
        round_comparison: dict[str, object] = {}
        for metric in ROUND_METRICS:
            pd_value = _metric_value(pd_reference, round_name, metric)
            pap_value = _metric_value(pap_reference, round_name, metric)
            candidate_value = _metric_value(candidate, round_name, metric)
            round_comparison[metric] = {
                "pd_reference": pd_value,
                "pap_reference": pap_value,
                "candidate": candidate_value,
                "candidate_over_pd": _ratio(candidate_value, pd_value),
                "candidate_over_pap_reference": _ratio(
                    candidate_value,
                    pap_value,
                ),
            }
            if metric != "tpot_ms" or round_name != "round_2":
                if candidate_value > pap_value * 1.03:
                    warnings.append(
                        f"{round_name} {metric} regressed more than 3%"
                    )
        comparison_metrics[round_name] = round_comparison
        diagnostic_metrics[round_name] = {
            metric: {
                "pd_reference": _diagnostic_value(
                    pd_reference,
                    round_name,
                    metric,
                ),
                "pap_reference": _diagnostic_value(
                    pap_reference,
                    round_name,
                    metric,
                ),
                "candidate": _diagnostic_value(
                    candidate,
                    round_name,
                    metric,
                ),
            }
            for metric in ROUND_DIAGNOSTICS
        }

    for metric in ("conversation_latency_ms", "conversation_eof_latency_ms"):
        pd_value = _positive_finite(
            _mapping(pd_reference.get("metrics"), "PD metrics").get(metric),
            f"PD {metric}",
        )
        pap_value = _positive_finite(
            _mapping(pap_reference.get("metrics"), "PAP metrics").get(metric),
            f"PAP {metric}",
        )
        candidate_value = _positive_finite(
            _mapping(candidate.get("metrics"), "candidate metrics").get(metric),
            f"candidate {metric}",
        )
        comparison_metrics[metric] = {
            "pd_reference": pd_value,
            "pap_reference": pap_value,
            "candidate": candidate_value,
            "candidate_over_pd": _ratio(candidate_value, pd_value),
            "candidate_over_pap_reference": _ratio(candidate_value, pap_value),
        }
        if candidate_value > pap_value * 1.03:
            warnings.append(f"{metric} regressed more than 3%")

    candidate_tpot = _metric_value(candidate, "round_2", "tpot_ms")
    pap_tpot = _metric_value(pap_reference, "round_2", "tpot_ms")
    pd_tpot = _metric_value(pd_reference, "round_2", "tpot_ms")
    classification = (
        "diagnostic"
        if candidate.get("mode") == "quick"
        else classify_tpot(candidate_tpot, pap_tpot)
    )
    return {
        "schema_version": 2,
        "status": "valid",
        "profile_id": _mapping(candidate.get("profile"), "profile").get(
            "profile_id"
        ),
        "profile_fingerprint": candidate.get("profile_fingerprint"),
        "hardware_signature": candidate.get("hardware_signature"),
        "candidate_mode": candidate.get("mode"),
        "candidate_repetitions": candidate.get("repetition_count"),
        "classification": classification,
        "north_star_target_met": candidate_tpot < 2.0 * pd_tpot,
        "metrics": comparison_metrics,
        "diagnostics": diagnostic_metrics,
        "implementations": {
            label: {
                "git_commit": result.get("git_commit"),
                "git_tracked_worktree_dirty": result.get(
                    "git_tracked_worktree_dirty"
                ),
                "implementation": deepcopy(result.get("implementation")),
            }
            for label, result in (
                ("candidate", candidate),
                ("pd_reference", pd_reference),
                ("pap_reference", pap_reference),
            )
        },
        "correctness": correctness,
        "warnings": warnings,
    }


def render_markdown(comparison: Mapping[str, object]) -> str:
    """Render a concise human-readable north-star report."""
    metrics = _mapping(comparison.get("metrics"), "comparison metrics")
    lines = [
        "# PAP/PD Multi-turn North-star Report",
        "",
        f"- Classification: `{comparison.get('classification')}`",
        "- Target `PAP TPOT < 2 * PD TPOT`: "
        + ("passed" if comparison.get("north_star_target_met") else "not met"),
        f"- Candidate mode: `{comparison.get('candidate_mode')}`",
        "",
        "| Round | Metric | PD | PAP reference | Candidate | PAP/PD | Candidate/PAP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "ttft_ms": "TTFT (ms)",
        "tpot_ms": "TPOT (ms)",
        "latency_ms": "Latency (ms)",
    }
    for round_name in ("round_1", "round_2"):
        round_metrics = _mapping(metrics.get(round_name), round_name)
        for metric in ROUND_METRICS:
            values = _mapping(round_metrics.get(metric), metric)
            lines.append(
                "| {round_label} | {metric_label} | {pd:.3f} | {pap:.3f} | "
                "{candidate:.3f} | {over_pd:.3f}x | {over_pap:.3f}x |".format(
                    round_label=round_name.replace("_", " ").title(),
                    metric_label=labels[metric],
                    pd=float(values["pd_reference"]),
                    pap=float(values["pap_reference"]),
                    candidate=float(values["candidate"]),
                    over_pd=float(values["candidate_over_pd"]),
                    over_pap=float(values["candidate_over_pap_reference"]),
                )
            )
    lines.extend(
        [
            "",
            "## Conversation metrics",
            "",
            "| Metric | PD | PAP reference | Candidate | PAP/PD | Candidate/PAP |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric, label in (
        ("conversation_latency_ms", "Last-token latency (ms)"),
        ("conversation_eof_latency_ms", "HTTP EOF latency (ms)"),
    ):
        values = _mapping(metrics.get(metric), metric)
        lines.append(
            "| {label} | {pd:.3f} | {pap:.3f} | {candidate:.3f} | "
            "{over_pd:.3f}x | {over_pap:.3f}x |".format(
                label=label,
                pd=float(values["pd_reference"]),
                pap=float(values["pap_reference"]),
                candidate=float(values["candidate"]),
                over_pd=float(values["candidate_over_pd"]),
                over_pap=float(values["candidate_over_pap_reference"]),
            )
        )
    diagnostics = _mapping(comparison.get("diagnostics"), "diagnostics")
    lines.extend(
        [
            "",
            "## Stream-tail diagnostics",
            "",
            "| Round | Metric | PD | PAP reference | Candidate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    diagnostic_labels = {
        "eof_latency_ms": "HTTP EOF latency (ms)",
        "post_token_stream_ms": "Post-token stream tail (ms)",
    }
    for round_name in ("round_1", "round_2"):
        round_diagnostics = _mapping(diagnostics.get(round_name), round_name)
        for metric in ROUND_DIAGNOSTICS:
            values = _mapping(round_diagnostics.get(metric), metric)
            lines.append(
                "| {round_label} | {metric_label} | {pd:.3f} | "
                "{pap:.3f} | {candidate:.3f} |".format(
                    round_label=round_name.replace("_", " ").title(),
                    metric_label=diagnostic_labels[metric],
                    pd=float(values["pd_reference"]),
                    pap=float(values["pap_reference"]),
                    candidate=float(values["candidate"]),
                )
            )
    correctness = _mapping(comparison.get("correctness"), "correctness")
    lines.extend(
        [
            "",
            "## Exact-token checks",
            "",
            "| Round | Prompt match | Candidate=PD | Candidate=PAP ref | PD=PAP ref |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for round_name in ("round_1", "round_2"):
        checks = _mapping(correctness.get(round_name), round_name)
        lines.append(
            "| {round_label} | {prompt} | {candidate_pd} | "
            "{candidate_pap} | {pd_pap} |".format(
                round_label=round_name.replace("_", " ").title(),
                prompt=checks["prompt_digest_match"],
                candidate_pd=checks["candidate_output_matches_pd"],
                candidate_pap=checks[
                    "candidate_output_matches_pap_reference"
                ],
                pd_pap=checks["pd_output_matches_pap_reference"],
            )
        )
    implementations = _mapping(
        comparison.get("implementations"),
        "implementations",
    )
    lines.extend(["", "## Implementation provenance", ""])
    for label in ("pd_reference", "pap_reference", "candidate"):
        provenance = _mapping(implementations.get(label), label)
        implementation = _mapping(
            provenance.get("implementation"),
            f"{label} implementation",
        )
        lines.append(
            "- `{label}`: commit `{commit}`, transport `{transport}`, "
            "direct output `{direct}`, tracked dirty `{dirty}`".format(
                label=label,
                commit=provenance.get("git_commit"),
                transport=implementation.get("offload_exec_transport"),
                direct=implementation.get("direct_mailbox_output"),
                dirty=provenance.get("git_tracked_worktree_dirty"),
            )
        )
    warnings = comparison.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def write_reference_atomic(
    path: Path,
    reference: Mapping[str, object],
) -> None:
    """Atomically write one already-validated reference payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(reference, file_obj, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _repo_relative_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"source result is outside repository: {path}") from exc
    return str(relative)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_reference_atomic(path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate and compare multi-turn north-star results"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--result", action="append", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--pd-reference", type=Path, required=True)
    compare_parser.add_argument("--pap-reference", type=Path, required=True)
    compare_parser.add_argument("--output-json", type=Path, required=True)
    compare_parser.add_argument("--output-markdown", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate-references")
    validate_parser.add_argument("--pd-reference", type=Path, required=True)
    validate_parser.add_argument("--pap-reference", type=Path, required=True)

    reference_parser = subparsers.add_parser("write-reference")
    reference_parser.add_argument(
        "--architecture", choices=("pap", "pd"), required=True
    )
    reference_parser.add_argument("--aggregate", type=Path, required=True)
    reference_parser.add_argument("--output", type=Path, required=True)
    reference_parser.add_argument("--allow-reference-write", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "aggregate":
        resolved_results = [path.resolve() for path in args.result]
        if len(set(resolved_results)) != len(resolved_results):
            raise ValueError("aggregate requires distinct source result files")
        aggregate = aggregate_repetitions(
            [_load_json(path) for path in args.result]
        )
        aggregate["source_results"] = [
            _repo_relative_path(path) for path in args.result
        ]
        _write_json(args.output, aggregate)
        print(json.dumps(aggregate, indent=2))
        return
    if args.command == "compare":
        comparison = compare_candidate(
            _load_json(args.candidate),
            _load_json(args.pd_reference),
            _load_json(args.pap_reference),
        )
        _write_json(args.output_json, comparison)
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            render_markdown(comparison),
            encoding="utf-8",
        )
        print(json.dumps(comparison, indent=2))
        return
    if args.command == "validate-references":
        try:
            validate_reference_pair(
                _load_json(args.pd_reference),
                _load_json(args.pap_reference),
            )
        except ValueError as exc:
            raise SystemExit(f"invalid north-star references: {exc}") from exc
        print("north-star references are valid")
        return
    if not args.allow_reference_write:
        raise SystemExit(
            "write-reference requires the explicit --allow-reference-write flag"
        )
    aggregate = _load_json(args.aggregate)
    reference = make_reference(aggregate, architecture=args.architecture)
    write_reference_atomic(args.output, reference)
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
