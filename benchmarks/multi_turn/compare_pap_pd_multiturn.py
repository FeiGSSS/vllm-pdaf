"""Aggregate and compare PAP/PD multi-turn north-star results."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any


ROUND_METRICS = ("ttft_ms", "tpot_ms", "latency_ms")
PROFILE_ID = "qwen3_8b_chat_16k_2turn_o256_c1_v1"


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


def validate_repetition(result: Mapping[str, object]) -> None:
    """Fail closed unless one repetition satisfies all client-side gates."""
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
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("missing profile fingerprint")
    hardware = result.get("hardware_signature")
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("missing hardware signature")

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
        for digest_name in (
            "prompt_token_digest",
            "output_token_digest",
            "assistant_text_digest",
        ):
            digest = round_result.get(digest_name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"round {expected_round} has invalid {digest_name}"
                )
    _positive_finite(
        result.get("conversation_latency_ms"),
        "conversation_latency_ms",
    )

    cache = _mapping(result.get("cache_validation"), "cache validation")
    architecture = result.get("architecture")
    allowed_cache_statuses = (
        {"passed"}
        if architecture == "pap"
        else {"passed", "official_log_passed"}
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

    raw_metrics: dict[str, object] = {}
    metrics: dict[str, object] = {}
    for round_index in (1, 2):
        raw_round: dict[str, list[float]] = {}
        aggregate_round: dict[str, float] = {}
        for metric in ROUND_METRICS:
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

    return {
        "schema_version": 1,
        "kind": "aggregate",
        "profile": profile,
        "profile_fingerprint": fingerprint,
        "architecture": architecture,
        "topology": topology,
        "hardware_signature": hardware,
        "mode": "formal" if len(results) == 3 else "quick",
        "repetition_count": len(results),
        "validity": {"status": "passed"},
        "metrics": metrics,
        "raw_metrics": raw_metrics,
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
    if aggregate.get("validity") != {"status": "passed"}:
        raise ValueError("aggregate validity did not pass")
    if aggregate.get("mode") not in {"quick", "formal"}:
        raise ValueError(f"invalid aggregate mode: {aggregate.get('mode')}")
    if aggregate.get("repetition_count") not in {1, 3}:
        raise ValueError("invalid aggregate repetition count")
    _mapping(aggregate.get("metrics"), "aggregate metrics")


def make_reference(
    aggregate: Mapping[str, object],
    *,
    architecture: str,
) -> dict[str, object]:
    """Create a tracked reference payload from a valid formal aggregate."""
    _validate_aggregate(aggregate)
    if aggregate.get("mode") != "formal" or aggregate.get("repetition_count") != 3:
        raise ValueError("reference requires a three-repetition formal aggregate")
    if aggregate.get("architecture") != architecture:
        raise ValueError(
            "reference architecture mismatch: "
            f"{aggregate.get('architecture')} != {architecture}"
        )
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


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def compare_candidate(
    candidate: Mapping[str, object],
    pd_reference: Mapping[str, object],
    pap_reference: Mapping[str, object],
) -> dict[str, object]:
    """Compare a PAP candidate with fixed PD and PAP references."""
    for label, result in (
        ("candidate", candidate),
        ("PD reference", pd_reference),
        ("PAP reference", pap_reference),
    ):
        try:
            _validate_aggregate(result)
        except ValueError as exc:
            raise ValueError(f"invalid {label}: {exc}") from exc
    if candidate.get("architecture") != "pap":
        raise ValueError("candidate architecture must be pap")
    if pd_reference.get("architecture") != "pd":
        raise ValueError("PD reference architecture must be pd")
    if pap_reference.get("architecture") != "pap":
        raise ValueError("PAP reference architecture must be pap")

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
    warnings: list[str] = []
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

    candidate_tpot = _metric_value(candidate, "round_2", "tpot_ms")
    pap_tpot = _metric_value(pap_reference, "round_2", "tpot_ms")
    pd_tpot = _metric_value(pd_reference, "round_2", "tpot_ms")
    classification = (
        "diagnostic"
        if candidate.get("mode") == "quick"
        else classify_tpot(candidate_tpot, pap_tpot)
    )
    return {
        "schema_version": 1,
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
    labels = {"ttft_ms": "TTFT (ms)", "tpot_ms": "TPOT (ms)", "latency_ms": "Latency (ms)"}
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
        aggregate = aggregate_repetitions(
            [_load_json(path) for path in args.result]
        )
        aggregate["source_results"] = [str(path) for path in args.result]
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
