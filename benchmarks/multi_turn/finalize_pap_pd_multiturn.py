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
    compute_calls = attention.get("offload_exec_compute_calls")
    if not isinstance(compute_calls, (int, float)) or compute_calls <= 0:
        raise ValueError("Attention stats do not contain positive compute calls")

    metadata = _json_mapping(artifacts["run_metadata"], "run metadata")
    implementation = result.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("result implementation is missing")
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
    if not isinstance(reuse, Mapping) or reuse.get("status") != PD_REUSE_STATUS:
        raise ValueError(f"PD reuse validation did not pass: {reuse}")
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
