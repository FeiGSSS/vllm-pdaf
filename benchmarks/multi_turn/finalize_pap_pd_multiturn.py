"""Attach post-client lifecycle evidence to a north-star repetition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.multi_turn.pd_multiturn_reuse_metrics import (
    validate_official_streaming_one_way,
)
from benchmarks.multi_turn.pd_multiturn_load_reuse_metrics import (
    STATUS as PD_LOAD_REUSE_STATUS,
    validate_pd_multiturn_load_reuse,
)


REQUIRED_GATES = {
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
REQUIRED_ARTIFACTS = {
    "pap": frozenset(
        {
            "session_drain",
            "routing",
            "correctness_logs",
            "attention_stats",
            "run_metadata",
            "effective_config",
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
PD_REUSE_STATUS = "official_streaming_one_way_metrics_passed"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_DEFERRED_TRACE_SCOPE = "attention_process_critical_chain"
_DEFERRED_TRACE_SPAN_COUNTERS = {
    "qkv_ready_wait_gpu_ms": ("offload_exec_peer_batches",),
    "kv_append_gpu_ms": ("fast_path_hits", "fallbacks"),
    "paged_fa_gpu_ms": ("offload_exec_compute_calls",),
    "output_p2p_copy_gpu_ms": ("offload_exec_peer_batches",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
    return values


def _json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} artifact must contain a JSON object")
    return payload


def _attention_stat_payloads(
    attention: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    instances = attention.get("instances")
    if instances is None:
        payloads: Sequence[object] = (attention,)
    elif isinstance(instances, Sequence) and not isinstance(instances, (str, bytes)):
        payloads = instances
    else:
        raise ValueError("attention stats instances must be a sequence")
    stats_payloads: list[Mapping[str, Any]] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, Mapping):
            raise ValueError(f"attention stats instance {index} is invalid")
        stats = payload.get("stats", payload)
        if not isinstance(stats, Mapping):
            raise ValueError(f"attention stats instance {index} has no stats")
        stats_payloads.append(stats)
    if not stats_payloads:
        raise ValueError("attention stats contain no instances")
    return tuple(stats_payloads)


def _attention_stat_total(
    attention: Mapping[str, Any],
    key: str,
) -> float:
    values: list[float] = []
    for stats in _attention_stat_payloads(attention):
        value = stats.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"attention stats do not contain numeric {key}")
        values.append(float(value))
    return sum(values)


def _trace_count(value: object, *, field: str, instance: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"deferred CUDA trace instance {instance} has invalid {field}"
        )
    return value


def _validate_deferred_cuda_trace(
    attention: Mapping[str, Any],
) -> None:
    for instance, stats in enumerate(_attention_stat_payloads(attention)):
        trace = stats.get("deferred_cuda_trace")
        if not isinstance(trace, Mapping) or trace.get("enabled") is not True:
            raise ValueError(
                f"deferred CUDA trace instance {instance} is not enabled"
            )
        if trace.get("scope") != _DEFERRED_TRACE_SCOPE:
            raise ValueError(
                f"deferred CUDA trace instance {instance} has invalid scope"
            )
        for field in ("pending_records", "dropped_records", "error_records"):
            if _trace_count(
                trace.get(field), field=field, instance=instance
            ) != 0:
                raise ValueError(
                    f"deferred CUDA trace instance {instance} has nonzero {field}"
                )
        if _trace_count(
            trace.get("collector_count"),
            field="collector_count",
            instance=instance,
        ) <= 0:
            raise ValueError(
                f"deferred CUDA trace instance {instance} has no collectors"
            )
        spans = trace.get("spans")
        if not isinstance(spans, Mapping):
            raise ValueError(
                f"deferred CUDA trace instance {instance} has no spans"
            )
        for span_name, counter_names in _DEFERRED_TRACE_SPAN_COUNTERS.items():
            span = spans.get(span_name)
            if not isinstance(span, Mapping):
                raise ValueError(
                    "deferred CUDA trace instance "
                    f"{instance} is missing {span_name}"
                )
            span_count = _trace_count(
                span.get("count"),
                field=f"{span_name}.count",
                instance=instance,
            )
            expected_count = sum(
                _trace_count(
                    stats.get(counter_name),
                    field=counter_name,
                    instance=instance,
                )
                for counter_name in counter_names
            )
            if span_count <= 0 or span_count != expected_count:
                raise ValueError(
                    "deferred CUDA trace count mismatch for instance "
                    f"{instance} {span_name}: {span_count} != "
                    f"{'+'.join(counter_names)}={expected_count}"
                )


def _validate_clean_state(
    result: Mapping[str, object],
    artifacts: Mapping[str, Path],
) -> None:
    dirty = result.get("git_tracked_worktree_dirty")
    if not isinstance(dirty, bool):
        raise ValueError("git_tracked_worktree_dirty must be boolean")
    patch_sizes = [
        artifacts[name].stat().st_size
        for name in ("tracked_worktree_patch", "tracked_index_patch")
    ]
    if dirty and not any(patch_sizes):
        raise ValueError("dirty Git state has no tracked patch evidence")
    if not dirty and any(patch_sizes):
        raise ValueError("clean Git state has non-empty tracked patch evidence")


def _validate_correctness_artifact(path: Path, *, require_strict: bool) -> None:
    values = _env_values(path)
    if values.get("STATUS") != "passed" or values.get("MATCH_COUNT") != "0":
        raise ValueError(f"correctness log audit did not pass: {values}")
    if require_strict and values.get("STRICT") != "1":
        raise ValueError(f"correctness log audit was not strict: {values}")


def _validate_pap_evidence(
    result: Mapping[str, object],
    artifacts: Mapping[str, Path],
) -> None:
    session = _env_values(artifacts["session_drain"])
    if session.get("STATUS") != "passed" or session.get("ACTIVE_SESSIONS") != "0":
        raise ValueError(f"session drain did not pass: {session}")

    routing = _json_mapping(artifacts["routing"], "routing")
    if routing.get("status") != "passed" or routing.get("errors") != []:
        raise ValueError(f"routing audit did not pass: {routing}")

    _validate_correctness_artifact(
        artifacts["correctness_logs"],
        require_strict=True,
    )
    attention = _json_mapping(artifacts["attention_stats"], "attention stats")
    compute_calls = _attention_stat_total(attention, "offload_exec_compute_calls")
    if compute_calls <= 0:
        raise ValueError("Attention stats do not contain positive compute calls")

    metadata = _json_mapping(artifacts["run_metadata"], "run metadata")
    implementation = result.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("result implementation is missing")
    fast_key_enabled = implementation.get("unified_md_fast_key")
    if not isinstance(fast_key_enabled, bool):
        raise ValueError("implementation unified_md_fast_key must be boolean")
    fast_key_lookups = _attention_stat_total(
        attention,
        "unified_md_fast_key_lookups",
    )
    fast_key_hits = _attention_stat_total(
        attention,
        "unified_md_fast_key_hits",
    )
    full_key_scans = _attention_stat_total(
        attention,
        "unified_md_full_key_scans",
    )
    if fast_key_enabled and (fast_key_lookups <= 0 or fast_key_hits <= 0):
        raise ValueError("metadata fast key has no runtime hit evidence")
    if not fast_key_enabled and (fast_key_lookups != 0 or fast_key_hits != 0):
        raise ValueError("disabled metadata fast key recorded runtime lookups")
    if full_key_scans <= 0:
        raise ValueError("metadata cache recorded no full-key scans")
    effective_config = _env_values(artifacts["effective_config"])
    expected_fast_key = "1" if fast_key_enabled else "0"
    if effective_config.get("PAP_UNIFIED_MD_FAST_KEY") != expected_fast_key:
        raise ValueError(
            "metadata fast-key effective config mismatch: "
            f"{effective_config.get('PAP_UNIFIED_MD_FAST_KEY')} "
            f"!= {expected_fast_key}"
        )
    deferred_trace_enabled = (
        effective_config.get("PAP_DEFERRED_CUDA_TRACE", "0").lower()
        in _TRUE_VALUES
    )
    if deferred_trace_enabled:
        synchronous_trace = effective_config.get("PAP_OFFLOAD_EXEC_TRACE")
        if synchronous_trace is None or synchronous_trace.lower() not in {
            "0",
            "false",
            "no",
            "off",
        }:
            raise ValueError(
                "deferred CUDA trace requires synchronous trace to be disabled"
            )
        _validate_deferred_cuda_trace(attention)
    expected = {
        "git_commit": result.get("git_commit"),
        "git_tracked_worktree_dirty": result.get(
            "git_tracked_worktree_dirty"
        ),
        "offload_exec_transport": implementation.get(
            "offload_exec_transport"
        ),
        "direct_mailbox_output": implementation.get(
            "direct_mailbox_output"
        ),
        "unified_md_fast_key": implementation.get("unified_md_fast_key"),
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"run metadata mismatch: {actual} != {expected}")
    _validate_clean_state(result, artifacts)


def _validate_pd_evidence(
    result: Mapping[str, object],
    artifacts: Mapping[str, Path],
) -> None:
    reuse = result.get("pd_reuse_validation")
    if not isinstance(reuse, Mapping):
        raise ValueError(f"PD reuse validation did not pass: {reuse}")
    reuse_status = reuse.get("status")
    if reuse_status == PD_REUSE_STATUS:
        fresh_reuse = validate_official_streaming_one_way(
            result,
            proxy_log=artifacts["proxy_log"].read_text(encoding="utf-8"),
            prefill_metrics=artifacts["prefill_metrics"].read_text(
                encoding="utf-8"
            ),
            decode_metrics=artifacts["decode_metrics"].read_text(
                encoding="utf-8"
            ),
        )
    elif reuse_status == PD_LOAD_REUSE_STATUS:
        service_logs = tuple(
            artifacts[name].read_text(encoding="utf-8")
            for name in ("prefill_log", "decode_log")
            if name in artifacts
        )
        fresh_reuse = validate_pd_multiturn_load_reuse(
            result,
            prefill_metrics=artifacts["prefill_metrics"].read_text(
                encoding="utf-8"
            ),
            decode_metrics=artifacts["decode_metrics"].read_text(
                encoding="utf-8"
            ),
            effective_config=artifacts["effective_config"].read_text(
                encoding="utf-8"
            ),
            proxy_log=artifacts["proxy_log"].read_text(encoding="utf-8"),
            service_logs=service_logs,
        )
    else:
        raise ValueError(f"PD reuse validation did not pass: {reuse}")
    if dict(reuse) != fresh_reuse:
        raise ValueError("stored PD reuse evidence differs from fresh validation")
    _validate_correctness_artifact(
        artifacts["correctness_logs"],
        require_strict=False,
    )
    effective_config = _env_values(artifacts["effective_config"])
    if effective_config.get("GIT_COMMIT") != result.get("git_commit"):
        raise ValueError("PD effective config Git commit mismatch")
    for name in ("proxy_log", "prefill_metrics", "decode_metrics"):
        if artifacts[name].stat().st_size <= 0:
            raise ValueError(f"PD evidence artifact is empty: {name}")
    _validate_clean_state(result, artifacts)


def finalize_result(
    result: Mapping[str, object],
    *,
    architecture: str,
    passed_gates: Sequence[str],
    artifacts: Mapping[str, Path],
    artifact_root: Path | None = None,
) -> dict[str, object]:
    """Return a result carrying all required external gate evidence."""
    if architecture not in REQUIRED_GATES:
        raise ValueError(f"unsupported architecture: {architecture}")
    if result.get("architecture") != architecture:
        raise ValueError(
            "result architecture mismatch: "
            f"{result.get('architecture')} != {architecture}"
        )
    validity = result.get("validity")
    if not isinstance(validity, Mapping) or validity.get("status") != "passed":
        raise ValueError(f"client validity did not pass: {validity}")

    gate_names = set(passed_gates)
    required = REQUIRED_GATES[architecture]
    missing = sorted(required - gate_names)
    if missing:
        raise ValueError(f"missing required {architecture} gates: {missing}")
    unknown = sorted(gate_names - required)
    if unknown:
        raise ValueError(f"unknown {architecture} gates: {unknown}")

    required_artifacts = REQUIRED_ARTIFACTS[architecture]
    missing_artifacts = sorted(required_artifacts - set(artifacts))
    if missing_artifacts:
        raise ValueError(
            f"missing required {architecture} artifacts: {missing_artifacts}"
        )
    for name in sorted(required_artifacts):
        if not artifacts[name].is_file():
            raise ValueError(
                f"missing validation artifact {name}: {artifacts[name]}"
            )

    if architecture == "pap":
        _validate_pap_evidence(result, artifacts)
    else:
        _validate_pd_evidence(result, artifacts)

    if artifact_root is None:
        artifact_root = Path(
            os.path.commonpath([str(path) for path in artifacts.values()])
        )
        if artifact_root.is_file():
            artifact_root = artifact_root.parent
    artifact_root = artifact_root.resolve()

    artifact_evidence: dict[str, dict[str, str]] = {}
    for name, path in sorted(artifacts.items()):
        if not path.is_file():
            raise ValueError(f"missing validation artifact {name}: {path}")
        try:
            relative_path = path.resolve().relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                f"validation artifact is outside artifact root: {path}"
            ) from exc
        artifact_evidence[name] = {
            "path": str(relative_path),
            "sha256": _sha256(path),
        }

    finalized = dict(result)
    finalized["external_validation"] = {
        "status": "passed",
        "gates": {name: "passed" for name in sorted(required)},
        "artifacts": artifact_evidence,
    }
    return finalized


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"result root must be an object: {path}")
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    temporary.replace(path)


def _key_value(raw: str, *, value_type: type[str] | type[Path]) -> tuple[str, Any]:
    name, separator, value = raw.partition("=")
    if not separator or not name or not value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {raw!r}")
    return name, value_type(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize PAP/PD multi-turn north-star validation gates"
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--architecture", choices=("pap", "pd"), required=True)
    parser.add_argument("--passed-gate", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args()

    artifacts = dict(
        _key_value(raw, value_type=Path) for raw in args.artifact
    )
    finalized = finalize_result(
        _load_json(args.result),
        architecture=args.architecture,
        passed_gates=args.passed_gate,
        artifacts=artifacts,
        artifact_root=args.result.parent,
    )
    _atomic_write(args.result, finalized)
    print(json.dumps(finalized["external_validation"], indent=2))


if __name__ == "__main__":
    main()
