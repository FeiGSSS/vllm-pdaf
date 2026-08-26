# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate and summarize one PAP/PD AIPerf capacity run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLO_TIERS = {
    "strict": {"ttft_ms": 5_000.0, "itl_ms": 50.0},
    "standard": {"ttft_ms": 10_000.0, "itl_ms": 75.0},
    "relaxed": {"ttft_ms": 20_000.0, "itl_ms": 100.0},
}
MIN_GOOD_REQUEST_FRACTION = 0.95
EARLY_STOP_EXIT_CODES = {130, 143}


@dataclass(frozen=True)
class WorkloadExpectation:
    """Expected sessions, turns, and output-length semantics."""

    output_lengths: dict[tuple[str, int], int]
    turns_by_session: dict[str, int]
    output_length_mode: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _metric(record: dict[str, Any], name: str) -> float | None:
    raw_value = record.get("metrics", {}).get(name, {}).get("value")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return None
    value = float(raw_value)
    return value if math.isfinite(value) else None


def _aggregate_metric(profile: dict[str, Any], name: str) -> float | None:
    raw_value = profile.get(name, {}).get("avg")
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
        return None
    value = float(raw_value)
    return value if math.isfinite(value) else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_summary(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": _percentile([float(value) for value in values], 0.5),
        "min": min(values) if values else None,
        "p95": _percentile([float(value) for value in values], 0.95),
        "max": max(values) if values else None,
    }


def _load_expected_workload(path: Path) -> WorkloadExpectation:
    expected: dict[tuple[str, int], int] = {}
    turns_by_session: dict[str, int] = {}
    output_modes: set[str] = set()
    for conversation in _iter_jsonl(path):
        session_id = conversation.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("dataset conversation has no session_id")
        turns = conversation.get("turns")
        if turns is None and (
            "messages" in conversation or "input_length" in conversation
        ):
            turn_index = turns_by_session.get(session_id, 0)
            turns = [conversation]
            turns_by_session[session_id] = turn_index + 1
        elif isinstance(turns, list):
            if session_id in turns_by_session:
                raise ValueError(f"duplicate dataset session: {session_id}")
            turn_index = 0
            turns_by_session[session_id] = len(turns)
        else:
            raise ValueError(f"dataset session {session_id} has no turns")
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"dataset session {session_id} turn is not an object")
            output_length = turn.get("output_length")
            if not isinstance(output_length, int) or output_length <= 0:
                raise ValueError(
                    f"dataset session {session_id} turn {turn_index} "
                    "has an invalid output length"
                )
            extra = turn.get("extra")
            exact_output = (
                isinstance(extra, dict)
                and extra.get("ignore_eos") is True
                and extra.get("min_tokens") == output_length
            )
            output_modes.add("exact" if exact_output else "cap")
            if conversation.get("turns") is not None and not exact_output:
                raise ValueError(
                    f"dataset session {session_id} turn {turn_index} "
                    "does not force its requested output length"
                )
            elif len(turns) == 1:
                turn_index = turns_by_session[session_id] - 1
            key = (session_id, turn_index)
            if key in expected:
                raise ValueError(f"duplicate dataset request: {key}")
            expected[key] = output_length
    if len(output_modes) != 1:
        raise ValueError("dataset mixes exact and capped output semantics")
    return WorkloadExpectation(
        output_lengths=expected,
        turns_by_session=turns_by_session,
        output_length_mode=next(iter(output_modes)),
    )


def _read_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("STATUS="):
            return line.partition("=")[2].strip()
    return None


def _read_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _request_error_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        error = record.get("error")
        if isinstance(error, dict):
            error_type = error.get("type")
            if isinstance(error_type, str) and error_type:
                counts[error_type] += 1
        if record.get("metadata", {}).get("was_cancelled"):
            counts["Cancelled"] += 1
    return dict(sorted(counts.items()))


def _classify_run_status(
    *,
    records: list[dict[str, Any]],
    expected_requests: int,
    relaxed_good_requests: int,
    launcher_exit_code: int,
) -> dict[str, Any]:
    request_errors = _request_error_counts(records)
    minimum_good_requests = math.ceil(expected_requests * MIN_GOOD_REQUEST_FRACTION)
    maximum_bad_requests = expected_requests - minimum_good_requests
    observed_bad_requests = len(records) - relaxed_good_requests
    relaxed_pass_still_possible = observed_bad_requests <= maximum_bad_requests
    completed = len(records) == expected_requests

    if completed:
        if request_errors:
            state = "completed_with_errors"
        elif launcher_exit_code != 0:
            state = "completed_launcher_failed"
        else:
            state = "completed"
    elif not relaxed_pass_still_possible:
        state = (
            "early_stopped_slo_impossible"
            if launcher_exit_code in EARLY_STOP_EXIT_CODES
            else "incomplete_slo_impossible"
        )
    elif request_errors.get("TimeoutError"):
        state = "request_timeout"
    elif launcher_exit_code != 0:
        state = "service_failed"
    else:
        state = "incomplete"

    return {
        "state": state,
        "launcher_exit_code": launcher_exit_code,
        "request_error_counts": request_errors,
        "relaxed_slo": {
            "observed_bad_requests": observed_bad_requests,
            "maximum_bad_requests": maximum_bad_requests,
            "pass_still_possible": relaxed_pass_still_possible,
        },
    }


def _check_status(
    path: Path,
    label: str,
    errors: list[str],
) -> None:
    status = _read_status(path)
    if status is None:
        errors.append(f"missing {label}: {path.name}")
    elif status != "passed":
        errors.append(f"{label} status is {status!r}")


def _balanced(values: list[int]) -> bool:
    return bool(values) and max(values) - min(values) <= 1


def _check_pap_routing(
    run_root: Path,
    *,
    sessions: int,
    turns: int | None,
    expected_requests: int,
    errors: list[str],
) -> dict[str, Any]:
    audit_path = run_root / "routing_audit.json"
    stats_path = run_root / "topology_runtime_stats.json"
    if not audit_path.is_file() or not stats_path.is_file():
        errors.append("missing PAP structured routing audit")
        return {"passed": False}

    audit = _load_json(audit_path)
    stats = _load_json(stats_path)
    if audit.get("status") != "passed":
        errors.append("PAP routing audit did not pass")
    if audit.get("route_count") != expected_requests:
        errors.append("PAP route count does not match expected requests")
    if stats.get("total_requests") != expected_requests:
        errors.append("PAP runtime route count does not match")

    conversation = stats.get("conversation_routing", {})
    assignments_raw = conversation.get("pa_assignments", {})
    requests_raw = conversation.get("pa_requests", {})
    initial_loads_raw = conversation.get("pa_initial_context_characters", {})
    assignments = [int(value) for value in assignments_raw.values()]
    requests = [int(value) for value in requests_raw.values()]
    initial_loads = [int(value) for value in initial_loads_raw.values()]
    affinity_passed = True
    if conversation.get("conversations") != sessions:
        errors.append("PAP routed conversation count does not match")
        affinity_passed = False
    if sum(assignments) != sessions:
        errors.append("PAP PA assignment count does not match")
        affinity_passed = False
    if initial_loads_raw:
        if set(initial_loads_raw) != set(assignments_raw):
            errors.append("PAP initial-context load keys do not match PA assignments")
            affinity_passed = False
        elif not initial_loads or sum(initial_loads) <= 0 or min(initial_loads) < 0:
            errors.append("PAP initial-context loads are invalid")
            affinity_passed = False
    elif not _balanced(assignments):
        errors.append("legacy PAP PA assignments are not balanced")
        affinity_passed = False
    if sum(requests) != expected_requests:
        errors.append("PAP conversation-affinity request count does not match")
        affinity_passed = False
    elif turns is not None and sorted(requests) != sorted(
        value * turns for value in assignments
    ):
        errors.append("PAP conversation affinity was not retained")
        affinity_passed = False
    elif turns is None and any(
        requests[index] < assignments[index] for index in range(len(requests))
    ):
        errors.append("PAP has fewer requests than assigned sessions")
        affinity_passed = False

    passed = (
        audit.get("status") == "passed"
        and audit.get("route_count") == expected_requests
        and stats.get("total_requests") == expected_requests
        and affinity_passed
    )
    return {
        "passed": passed,
        "conversations": conversation.get("conversations"),
        "owner_assignments": assignments,
        "initial_context_characters": initial_loads,
    }


def _check_runtime_audits(
    run_root: Path,
    architecture: str,
    topology: str,
    errors: list[str],
) -> None:
    _check_status(
        run_root / "correctness_audit.env",
        "correctness audit",
        errors,
    )
    if architecture != "pap":
        return
    for filename, label in (
        ("routing_audit.env", "routing audit"),
        ("session_drain.env", "session drain audit"),
        ("decode_token_join_audit.env", "decode-token join audit"),
    ):
        _check_status(run_root / filename, label, errors)
    effective_config = _read_env(run_root / "effective_config.env")
    try:
        prefill_chunks = int(effective_config["PAP_STATIC_PREFILL_CHUNKS"])
        attention_chunks = int(effective_config["PAP_STATIC_ATTENTION_CHUNKS"])
        prefill_sms = int(effective_config["PAP_STATIC_PREFILL_EXPECTED_SMS"])
        attention_sms = int(effective_config["PAP_STATIC_ATTENTION_EXPECTED_SMS"])
    except (KeyError, ValueError):
        errors.append("PAP static-MPS effective configuration is invalid")
        return
    if (
        prefill_chunks + attention_chunks != 23
        or prefill_sms != prefill_chunks * 4
        or attention_sms != attention_chunks * 4
    ):
        errors.append("PAP static-MPS effective allocation is invalid")
        return
    mps_audits = sorted(run_root.glob("mps_static_audit_pa_*.env"))
    expected_pa_audits = int(topology.partition("pa")[0])
    if len(mps_audits) != expected_pa_audits:
        errors.append("PAP static-MPS audit count does not match PA count")
    for path in mps_audits:
        values = _read_env(path)
        if (
            values.get("MPS_MODE") != "static"
            or values.get("PREFILL_CHUNKS") != str(prefill_chunks)
            or values.get("ATTENTION_CHUNKS") != str(attention_chunks)
            or values.get("PREFILL_VISIBLE_SMS") != str(prefill_sms)
            or values.get("ATTENTION_VISIBLE_SMS") != str(attention_sms)
        ):
            errors.append(f"static-MPS audit is invalid: {path.name}")


def summarize_run(
    run_root: Path,
    *,
    aiperf_root: Path | None = None,
    architecture: str,
    topology: str,
    concurrency: int,
    sessions: int,
    turns: int,
    output_tokens: int | None,
    dataset_file: Path | None = None,
    repetition: int = 1,
    runtime_repetitions: int = 1,
    launcher_exit_code: int = 0,
) -> dict[str, Any]:
    """Build a compact correctness and three-tier SLO summary."""

    expected_requests = sessions * turns
    expected_turns_by_session: dict[str, int] = {}
    output_length_mode = "exact"
    fixed_turns: int | None = turns
    if runtime_repetitions <= 0:
        raise ValueError("runtime_repetitions must be positive")
    errors: list[str] = []
    if launcher_exit_code != 0:
        errors.append(f"launcher exited with code {launcher_exit_code}")

    expected_output_lengths: dict[tuple[str, int], int] = {}
    if dataset_file is not None:
        try:
            workload = _load_expected_workload(dataset_file)
            expected_output_lengths = workload.output_lengths
            expected_turns_by_session = workload.turns_by_session
            output_length_mode = workload.output_length_mode
            expected_requests = len(expected_output_lengths)
            if len(expected_turns_by_session) != sessions:
                errors.append(
                    "workload dataset session count does not match expected sessions"
                )
            turn_counts = set(expected_turns_by_session.values())
            fixed_turns = next(iter(turn_counts)) if len(turn_counts) == 1 else None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid or missing workload dataset: {exc}")
    elif output_tokens is not None:
        expected_output_lengths = {
            (f"__index__{index}", 0): output_tokens
            for index in range(expected_requests)
        }
    else:
        errors.append("no expected output lengths were provided")
    runtime_sessions = sessions * runtime_repetitions
    runtime_expected_requests = expected_requests * runtime_repetitions

    aiperf_root = aiperf_root or run_root / "aiperf"
    records_path = aiperf_root / "profile.jsonl"
    profile_path = aiperf_root / "profile.json"
    records: list[dict[str, Any]] = []
    profile: dict[str, Any] = {}
    try:
        records = _load_jsonl(records_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid or missing AIPerf records: {exc}")
    try:
        raw_profile = _load_json(profile_path)
        if isinstance(raw_profile, dict):
            profile = raw_profile
        else:
            errors.append("AIPerf aggregate profile is not an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid or missing AIPerf aggregate: {exc}")

    if len(records) != expected_requests:
        errors.append(
            f"completed {len(records)} of {expected_requests} expected requests"
        )
    if profile.get("error_summary"):
        errors.append("AIPerf aggregate contains request errors")
    if profile.get("was_cancelled"):
        errors.append("AIPerf run was cancelled")
    aggregate_count = _aggregate_metric(profile, "request_count")
    if aggregate_count is not None and aggregate_count != expected_requests:
        errors.append("AIPerf aggregate request count does not match")

    turns_by_conversation: dict[str, list[int]] = defaultdict(list)
    valid_request_indices: set[int] = set()
    ttft_values: list[float] = []
    itl_values: list[float] = []
    for index, record in enumerate(records):
        metadata = record.get("metadata", {})
        conversation_id = metadata.get("conversation_id")
        turn_index = metadata.get("turn_index")
        record_valid = True
        if not isinstance(conversation_id, str) or not conversation_id:
            errors.append(f"record {index} has no conversation id")
            record_valid = False
        elif not isinstance(turn_index, int):
            errors.append(f"record {index} has no integer turn index")
            record_valid = False
        else:
            turns_by_conversation[conversation_id].append(turn_index)
        if metadata.get("was_cancelled"):
            errors.append(f"record {index} was cancelled")
            record_valid = False

        if dataset_file is None:
            expected_output_tokens = output_tokens
        elif isinstance(conversation_id, str) and isinstance(turn_index, int):
            expected_output_tokens = expected_output_lengths.get(
                (conversation_id, turn_index)
            )
        else:
            expected_output_tokens = None
        actual_output_tokens = _metric(record, "output_token_count")
        if expected_output_tokens is None:
            errors.append(f"record {index} has no matching dataset request")
            record_valid = False
        elif output_length_mode == "exact" and (
            actual_output_tokens != expected_output_tokens
        ):
            errors.append(
                f"record {index} output tokens are {actual_output_tokens!r}; "
                f"expected {expected_output_tokens}"
            )
            record_valid = False
        elif output_length_mode == "cap" and (
            actual_output_tokens is None
            or actual_output_tokens <= 0
            or actual_output_tokens > expected_output_tokens
        ):
            errors.append(
                f"record {index} output tokens are {actual_output_tokens!r}; "
                f"expected 1..{expected_output_tokens}"
            )
            record_valid = False
        ttft = _metric(record, "time_to_first_token")
        itl = _metric(record, "inter_token_latency")
        if ttft is None or itl is None:
            errors.append(f"record {index} is missing TTFT or ITL")
            record_valid = False
        else:
            ttft_values.append(ttft)
            itl_values.append(itl)
        if record_valid:
            valid_request_indices.add(index)

    if len(turns_by_conversation) != sessions:
        errors.append("completed conversation count does not match expected sessions")
    for conversation_id, observed_turns in turns_by_conversation.items():
        expected_turn_count = (
            expected_turns_by_session.get(conversation_id)
            if expected_turns_by_session
            else turns
        )
        if expected_turn_count is None:
            errors.append(f"session {conversation_id} is absent from the workload")
            continue
        expected_turns = list(range(expected_turn_count))
        if sorted(observed_turns) != expected_turns:
            errors.append(f"session {conversation_id} has incomplete turns")

    _check_runtime_audits(run_root, architecture, topology, errors)
    if architecture == "pap":
        routing = _check_pap_routing(
            run_root,
            sessions=runtime_sessions,
            turns=fixed_turns,
            expected_requests=runtime_expected_requests,
            errors=errors,
        )
    else:
        routing = {
            "passed": True,
            "policy": "dynamo_kv_router",
        }

    correctness_passed = not errors
    request_throughput = _aggregate_metric(profile, "request_throughput")
    tiers = {}
    for name, limits in SLO_TIERS.items():
        good_requests = 0
        for index, record in enumerate(records):
            ttft = _metric(record, "time_to_first_token")
            itl = _metric(record, "inter_token_latency")
            if (
                index in valid_request_indices
                and ttft is not None
                and itl is not None
                and ttft <= limits["ttft_ms"]
                and itl <= limits["itl_ms"]
            ):
                good_requests += 1
        fraction = good_requests / expected_requests
        tiers[name] = {
            **limits,
            "minimum_good_request_fraction": MIN_GOOD_REQUEST_FRACTION,
            "good_requests": good_requests,
            "good_request_fraction": fraction,
            "goodput_requests_per_second": (
                request_throughput * fraction
                if request_throughput is not None
                else None
            ),
            "passed": correctness_passed and fraction >= MIN_GOOD_REQUEST_FRACTION,
        }

    run_status = _classify_run_status(
        records=records,
        expected_requests=expected_requests,
        relaxed_good_requests=tiers["relaxed"]["good_requests"],
        launcher_exit_code=launcher_exit_code,
    )
    requested_output_values = list(expected_output_lengths.values())
    fixed_output_tokens = (
        requested_output_values[0]
        if requested_output_values and len(set(requested_output_values)) == 1
        else None
    )

    return {
        "schema_version": 1,
        "architecture": architecture,
        "topology": topology,
        "concurrency": concurrency,
        "repetition": repetition,
        "run_status": run_status,
        "workload": {
            "sessions": sessions,
            "turns_per_session": fixed_turns,
            "turns_per_session_distribution": _numeric_summary(
                list(expected_turns_by_session.values())
                if expected_turns_by_session
                else [turns] * sessions
            ),
            "output_length_mode": output_length_mode,
            "output_tokens_per_turn": fixed_output_tokens,
            "requested_output_tokens": _numeric_summary(requested_output_values),
            "expected_requests": expected_requests,
        },
        "correctness": {
            "passed": correctness_passed,
            "completed_requests": len(records),
            "completed_sessions": len(turns_by_conversation),
            "errors": errors[:20],
            "error_count": len(errors),
        },
        "routing": routing,
        "metrics": {
            "ttft_ms": {
                "average": sum(ttft_values) / len(ttft_values) if ttft_values else None,
                "p95": _percentile(ttft_values, 0.95),
                "p99": _percentile(ttft_values, 0.99),
            },
            "itl_ms": {
                "average": sum(itl_values) / len(itl_values) if itl_values else None,
                "p95": _percentile(itl_values, 0.95),
                "p99": _percentile(itl_values, 0.99),
            },
            "request_throughput_per_second": request_throughput,
            "output_token_throughput_per_second": _aggregate_metric(
                profile, "output_token_throughput"
            ),
            "output_token_throughput_per_user_per_second": _aggregate_metric(
                profile, "output_token_throughput_per_user"
            ),
            "benchmark_duration_seconds": _aggregate_metric(
                profile, "benchmark_duration"
            ),
        },
        "slo": tiers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--aiperf-root", type=Path)
    parser.add_argument("--architecture", choices=("pap", "pd", "dp"), required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--sessions", type=int)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--runtime-repetitions", type=int, default=1)
    parser.add_argument("--launcher-exit-code", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher_exit_code = args.launcher_exit_code
    exit_code_path = args.run_root / "launcher_exit_code.txt"
    if launcher_exit_code is None:
        launcher_exit_code = (
            int(exit_code_path.read_text(encoding="utf-8").strip())
            if exit_code_path.is_file()
            else 0
        )
    summary = summarize_run(
        args.run_root,
        aiperf_root=args.aiperf_root,
        architecture=args.architecture,
        topology=args.topology,
        concurrency=args.concurrency,
        sessions=args.sessions or args.concurrency,
        turns=args.turns,
        output_tokens=args.output_tokens,
        dataset_file=args.dataset_file,
        repetition=args.repetition,
        runtime_repetitions=args.runtime_repetitions,
        launcher_exit_code=launcher_exit_code,
    )
    output = args.output or args.run_root / "capacity_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    eligible = summary["correctness"]["passed"]
    state = summary["run_status"]["state"]
    if eligible:
        validation = "pass"
    elif state == "completed":
        validation = "fail"
    else:
        validation = "ineligible"
    statuses = "/".join(
        ("pass" if summary["slo"][tier]["passed"] else "fail")
        if eligible
        else "ineligible"
        for tier in SLO_TIERS
    )
    print(
        f"{summary['architecture']} {summary['topology']} "
        f"C={summary['concurrency']} state={state} "
        f"validation={validation} "
        f"SLO(strict/standard/relaxed)={statuses}"
    )


if __name__ == "__main__":
    main()
