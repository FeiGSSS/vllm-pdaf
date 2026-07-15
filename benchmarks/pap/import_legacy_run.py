"""Import a legacy PAP formal-run directory without modifying raw files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[2]
EVIDENCE_GRADES = (
    "formal-clean",
    "controlled",
    "diagnostic",
    "smoke",
    "historical",
    "invalid",
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key] = value.strip().strip("'").strip('"')
    return values


def _async_decode_token_setting(
    effective: dict[str, str],
    join_audit: dict[str, str],
) -> bool | str:
    raw_value = effective.get("PAP_ASYNC_DECODE_TOKEN")
    if raw_value is not None:
        normalized = raw_value.lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        return "missing"
    if join_audit.get("DECODE_TOKEN_DELIVERY") == "async":
        return True
    legacy_value = join_audit.get("ASYNC_DECODE_TOKEN")
    if legacy_value in {"0", "1"}:
        return legacy_value == "1"
    return "missing"


def _boolean_setting(
    metadata: dict[str, Any],
    metadata_key: str,
    effective: dict[str, str],
    environment_key: str,
    *,
    fallback: bool | str = "missing",
) -> bool | str:
    value = metadata.get(metadata_key)
    if isinstance(value, bool):
        return value
    raw_value = effective.get(environment_key)
    if raw_value is None:
        return fallback
    normalized = raw_value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return "missing"


def _mps_mode(mps_audit: dict[str, str]) -> str:
    return mps_audit.get(
        "PAP_MPS_MODE",
        mps_audit.get("MPS_MODE", "missing"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _path_reference(path: Path, root_id: str, root: Path) -> dict[str, str]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{path} is outside artifact root {root}") from error
    return {
        "root_id": root_id,
        "relative_path": relative_path.as_posix(),
    }


def _external_reference(
    raw_path: object,
    roots: dict[str, Path],
) -> dict[str, str] | str:
    if not isinstance(raw_path, str) or not raw_path:
        return "missing"
    path = Path(raw_path).resolve()
    candidates = sorted(
        roots.items(),
        key=lambda item: len(str(item[1].resolve())),
        reverse=True,
    )
    for root_id, root in candidates:
        try:
            relative_path = path.relative_to(root.resolve())
        except ValueError:
            continue
        return {
            "root_id": root_id,
            "relative_path": relative_path.as_posix(),
        }
    return "missing"


def _artifact_name(repetition: int | None, path: Path) -> str:
    prefix = "aggregate" if repetition is None else f"rep{repetition}"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name)
    return f"{prefix}_{stem}"


def _artifact_kind(path: Path) -> str:
    if path.name == "aggregate.json":
        return "aggregate"
    if path.name == "result.json":
        return "result"
    if path.name == "run_metadata.json":
        return "manifest"
    if path.name == "effective_config.env":
        return "configuration"
    if "audit" in path.name or path.name == "session_drain.env":
        return "audit"
    if "stats" in path.name:
        return "metrics"
    if path.suffix == ".log":
        return "log"
    if path.suffix == ".patch":
        return "patch"
    return "other"


def _make_artifact(
    path: Path,
    repetition: int | None,
    root_id: str,
    root: Path,
) -> dict[str, Any]:
    return {
        "name": _artifact_name(repetition, path),
        "kind": _artifact_kind(path),
        "path": _path_reference(path, root_id, root),
        "sha256": _sha256(path),
    }


def _repetition_directories(source_root: Path) -> list[tuple[int, Path]]:
    repetitions: list[tuple[int, Path]] = []
    for path in source_root.iterdir():
        match = re.fullmatch(r"rep([1-9][0-9]*)", path.name)
        if path.is_dir() and match is not None:
            repetitions.append((int(match.group(1)), path))
    return sorted(repetitions)


def _required_artifact_paths(rep_dir: Path) -> list[Path]:
    paths = [
        rep_dir / "result.json",
        rep_dir / "run_metadata.json",
        rep_dir / "topology_manifest.json",
        rep_dir / "effective_config.env",
        rep_dir / "attention_fast_path_stats.json",
        rep_dir / "correctness_audit.env",
        rep_dir / "decode_token_join_audit.env",
        rep_dir / "routing_audit.json",
        rep_dir / "session_drain.env",
        rep_dir / "mps_static_audit_pa_0.env",
        rep_dir / "tracked_worktree.patch",
        rep_dir / "tracked_index.patch",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing legacy artifacts: {missing}")
    return sorted(path for path in rep_dir.rglob("*") if path.is_file())


def _status(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("status")
    return "passed" if value == "passed" else "failed"


def _gate(
    status: str,
    detail: str,
    paths: list[Path],
    root_id: str,
    root: Path,
) -> dict[str, Any]:
    return {
        "status": status,
        "detail": detail,
        "evidence": [
            _path_reference(path, root_id, root) for path in paths
        ],
    }


def _metric(
    name: str,
    scope: str,
    value: object,
) -> dict[str, Any]:
    return {
        "name": name,
        "scope": scope,
        "statistic": "median",
        "unit": "ms",
        "value": value if isinstance(value, (int, float)) else "missing",
    }


def _extract_vllm_version(rep_dir: Path) -> str:
    pattern = re.compile(r"\bversion ([^\s]+)")
    for log_path in sorted((rep_dir / "service_logs").glob("*.log")):
        match = pattern.search(log_path.read_text(encoding="utf-8"))
        if match is not None:
            return match.group(1)
    return "missing"


def _uniform(values: list[object]) -> object:
    if values and all(value == values[0] for value in values[1:]):
        return values[0]
    return "missing"


def _validate_output_path(source_root: Path, output: Path | None) -> None:
    if output is None:
        return
    try:
        output.resolve().relative_to(source_root.resolve())
    except ValueError:
        return
    raise ValueError("legacy import output must be outside the raw source tree")


def import_legacy_run(
    source_root: Path,
    *,
    experiment_id: str,
    run_id: str,
    profile_id: str,
    evidence: str,
    artifact_root_id: str,
    roots: dict[str, Path],
) -> dict[str, Any]:
    """Build a manifest by reading, but never mutating, a legacy run.

    Args:
        source_root: Legacy formal-run directory containing aggregate and reps.
        experiment_id: Stable experiment ID for the imported run.
        run_id: Stable run ID for the imported run.
        profile_id: Canonical profile ID associated with the legacy run.
        evidence: Evidence grade assigned after human review.
        artifact_root_id: Root ID used for raw artifact references.
        roots: Root ID to local path mappings used only during import.

    Returns:
        A schema-versioned run manifest.

    Raises:
        ValueError: If paths, repetitions, or required metadata are invalid.
        FileNotFoundError: If a required legacy artifact is absent.
    """
    if evidence not in EVIDENCE_GRADES:
        raise ValueError(f"unsupported evidence grade: {evidence}")
    if artifact_root_id not in roots:
        raise ValueError(f"missing artifact root mapping: {artifact_root_id}")
    artifact_root = roots[artifact_root_id]
    source_root = source_root.resolve()
    _path_reference(source_root / "aggregate.json", artifact_root_id, artifact_root)

    aggregate_path = source_root / "aggregate.json"
    aggregate = _read_json(aggregate_path)
    repetitions = _repetition_directories(source_root)
    if not repetitions:
        raise ValueError(f"no repetition directories in {source_root}")
    if aggregate.get("repetition_count") != len(repetitions):
        raise ValueError("aggregate repetition count does not match directories")

    rep_payloads: list[dict[str, Any]] = []
    rep_artifact_paths: list[list[Path]] = []
    for index, rep_dir in repetitions:
        result = _read_json(rep_dir / "result.json")
        metadata = _read_json(rep_dir / "run_metadata.json")
        effective = _read_env(rep_dir / "effective_config.env")
        topology = _read_json(rep_dir / "topology_manifest.json")
        mps = _read_env(rep_dir / "mps_static_audit_pa_0.env")
        rep_payloads.append(
            {
                "index": index,
                "directory": rep_dir,
                "result": result,
                "metadata": metadata,
                "effective": effective,
                "topology": topology,
                "mps": mps,
            }
        )
        rep_artifact_paths.append(_required_artifact_paths(rep_dir))

    artifacts = [
        _make_artifact(aggregate_path, None, artifact_root_id, artifact_root)
    ]
    for (index, _), paths in zip(repetitions, rep_artifact_paths, strict=True):
        artifacts.extend(
            _make_artifact(path, index, artifact_root_id, artifact_root)
            for path in paths
        )

    first = rep_payloads[0]
    profile = aggregate.get("profile", {})
    implementation = aggregate.get("implementation", {})
    topology = first["result"].get("topology", {})
    metadata = first["metadata"]
    topology_manifest = first["topology"]
    effective = first["effective"]
    mps = first["mps"]

    commits = [payload["metadata"].get("git_commit") for payload in rep_payloads]
    dirty_values = [
        payload["metadata"].get("git_tracked_worktree_dirty")
        for payload in rep_payloads
    ]
    worktree_patch_hashes = [
        _sha256(payload["directory"] / "tracked_worktree.patch")
        for payload in rep_payloads
    ]
    index_patch_hashes = [
        _sha256(payload["directory"] / "tracked_index.patch")
        for payload in rep_payloads
    ]

    profile_parameters = {
        key: profile.get(key, "missing")
        for key in (
            "profile_id",
            "profile_version",
            "api",
            "workload_semantics",
            "prompt_input",
            "history_rule",
            "document_tokens",
            "append_tokens_per_later_round",
            "output_tokens_per_round",
            "rounds",
            "active_conversations",
            "request_rate_per_round",
            "arrival_mode",
            "temperature",
            "seed",
            "ignore_eos",
            "return_token_ids",
            "block_size",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
        )
    }

    pa_groups = topology_manifest.get("pa_groups", [])
    projections = topology_manifest.get("projections", [])
    prefill_devices = [int(group["gpu"]) for group in pa_groups]
    projection_devices = [int(item["gpu"]) for item in projections]

    result_paths = [
        payload["directory"] / "result.json" for payload in rep_payloads
    ]
    correctness_paths = [
        payload["directory"] / "correctness_audit.env"
        for payload in rep_payloads
    ]
    join_paths = [
        payload["directory"] / "decode_token_join_audit.env"
        for payload in rep_payloads
    ]
    decode_token_setting = _async_decode_token_setting(
        effective,
        _read_env(join_paths[0]),
    )
    routing_paths = [
        payload["directory"] / "routing_audit.json"
        for payload in rep_payloads
    ]
    drain_paths = [
        payload["directory"] / "session_drain.env"
        for payload in rep_payloads
    ]
    mps_paths = [
        payload["directory"] / "mps_static_audit_pa_0.env"
        for payload in rep_payloads
    ]
    attention_paths = [
        payload["directory"] / "attention_fast_path_stats.json"
        for payload in rep_payloads
    ]

    external_gates = aggregate.get("external_validation", {}).get("gates", {})
    client_status = _status(aggregate.get("client_validation"))
    cache_status = _status(aggregate.get("cache_validation"))
    correctness_status = _status(external_gates.get("correctness_logs"))
    join_status = _status(external_gates.get("decode_token_join"))
    routing_status = _status(external_gates.get("routing"))
    drain_status = _status(external_gates.get("session_drain"))
    attention_status = _status(external_gates.get("attention_stats_capture"))
    mps_status = "passed"
    for payload in rep_payloads:
        rep_mps = payload["mps"]
        if (
            _mps_mode(rep_mps) != "static"
            or rep_mps.get("PREFILL_VISIBLE_SMS") != "64"
            or rep_mps.get("ATTENTION_VISIBLE_SMS") != "28"
        ):
            mps_status = "failed"

    hardware_signature = aggregate.get("hardware_signature", "missing")
    vllm_version = _extract_vllm_version(first["directory"])
    environment_basis = {
        "hardware_signature": hardware_signature,
        "vllm_version": vllm_version,
        "git_commit": _uniform(commits),
        "implementation_fingerprint": aggregate.get(
            "implementation_fingerprint",
            "missing",
        ),
    }

    metrics = aggregate.get("metrics", {})
    round_1 = metrics.get("round_1", {})
    steady = metrics.get("steady_rounds_2_5", {})
    completed = sum(
        payload["result"].get("overall", {}).get("completed_requests", 0)
        for payload in rep_payloads
    )
    failed = sum(
        payload["result"].get("overall", {}).get("failed_requests", 0)
        for payload in rep_payloads
    )

    audits = {
        "client": _gate(
            client_status,
            "All client request and shape checks pass.",
            result_paths,
            artifact_root_id,
            artifact_root,
        ),
        "cache": _gate(
            cache_status,
            "All exact multi-turn cache checks pass.",
            result_paths,
            artifact_root_id,
            artifact_root,
        ),
        "attention_stats": _gate(
            attention_status,
            "Runtime Attention statistics are captured for every repetition.",
            attention_paths,
            artifact_root_id,
            artifact_root,
        ),
        "correctness": _gate(
            correctness_status,
            "Strict service-log correctness audit has zero matches.",
            correctness_paths,
            artifact_root_id,
            artifact_root,
        ),
        "decode_token_join": _gate(
            join_status,
            "Async decode-token join audit reports no errors.",
            join_paths,
            artifact_root_id,
            artifact_root,
        ),
        "routing": _gate(
            routing_status,
            "All requests follow the expected 1PA1P route.",
            routing_paths,
            artifact_root_id,
            artifact_root,
        ),
        "commit": _gate(
            correctness_status,
            "Strict logs contain no decode-commit error signatures.",
            correctness_paths,
            artifact_root_id,
            artifact_root,
        ),
        "lease": _gate(
            (
                "passed"
                if correctness_status == drain_status == "passed"
                else "failed"
            ),
            "Strict logs and zero-session drain contain no lease failures.",
            correctness_paths + drain_paths,
            artifact_root_id,
            artifact_root,
        ),
        "session_drain": _gate(
            drain_status,
            "Attention drains to zero active sessions.",
            drain_paths,
            artifact_root_id,
            artifact_root,
        ),
        "mps": _gate(
            mps_status,
            "Static MPS exposes 64 Prefill and 28 Attention SMs.",
            mps_paths,
            artifact_root_id,
            artifact_root,
        ),
    }

    failure_reasons = [
        f"audit {name} did not pass"
        for name, gate in audits.items()
        if gate["status"] != "passed"
    ]
    failure_reasons.extend(str(item) for item in aggregate.get("warnings", []))
    validity_status = _status(aggregate.get("validity"))
    if failed:
        failure_reasons.append(f"{failed} benchmark requests failed")
    failure_reasons = list(dict.fromkeys(failure_reasons))

    mps_mode = _mps_mode(mps)
    mps_profile_id = effective.get("PAP_BENCH_MPS_PROFILE", "missing")
    if (
        mps_profile_id == "missing"
        and profile_id == "p17_1pa1p"
        and mps_mode == "static"
        and mps.get("PREFILL_VISIBLE_SMS") == "64"
        and mps.get("ATTENTION_VISIBLE_SMS") == "28"
    ):
        mps_profile_id = "baseline_static_64_28"
    unified_kv_fallback: bool | str = (
        True if profile_id == "p17_1pa1p" else "missing"
    )

    return {
        "schema_version": 1,
        "kind": "pap-run-manifest",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "profile_id": profile_id,
        "architecture": aggregate.get("architecture", "pap"),
        "mode": aggregate.get("mode", "historical"),
        "evidence": evidence,
        "provenance": {
            "branch": "missing",
            "commit": _uniform(commits),
            "tracked_worktree_dirty": _uniform(dirty_values),
            "tracked_worktree_patch_sha256": _uniform(
                worktree_patch_hashes
            ),
            "tracked_index_patch_sha256": _uniform(index_patch_hashes),
            "started_at": first["metadata"].get("started_at", "missing"),
        },
        "workload": {
            "model": {
                "path": _external_reference(profile.get("model"), roots),
                "dtype": profile.get("dtype", "missing"),
                "tensor_parallel_size": profile.get(
                    "tensor_parallel_size",
                    "missing",
                ),
            },
            "corpus": {
                "path": _external_reference(profile.get("corpus_path"), roots),
                "sha256": profile.get("corpus_sha256", "missing"),
            },
            "parameters": profile_parameters,
        },
        "topology": {
            "name": topology.get("name", metadata.get("topology", "missing")),
            "pa_count": topology.get("pa_count", "missing"),
            "projection_count": topology.get("projection_count", "missing"),
            "pd_prefill_count": topology.get("pd_prefill_count", 0),
            "pd_decode_count": topology.get("pd_decode_count", 0),
            "routing_policy": metadata.get("routing_policy", "missing"),
        },
        "placement": {
            "prefill_devices": prefill_devices,
            "attention_devices": prefill_devices,
            "projection_devices": projection_devices,
            "decode_devices": [],
        },
        "transport": {
            "offload_exec": metadata.get(
                "offload_exec_transport",
                "missing",
            ),
            "offload_kv": metadata.get("offload_kv_transport", "missing"),
            "same_host": True,
        },
        "mps": {
            "mode": mps_mode,
            "profile_id": mps_profile_id,
            "prefill_visible_sms": int(mps["PREFILL_VISIBLE_SMS"]),
            "attention_visible_sms": int(mps["ATTENTION_VISIBLE_SMS"]),
        },
        "runtime": {
            "profile": profile_id,
            "settings": {
                "offload_exec_transport": implementation.get(
                    "offload_exec_transport",
                    "missing",
                ),
                "direct_mailbox_output": implementation.get(
                    "direct_mailbox_output",
                    "missing",
                ),
                "unified_md_fast_key": implementation.get(
                    "unified_md_fast_key",
                    "missing",
                ),
                "prefill_kv_async": implementation.get(
                    "prefill_kv_async",
                    "missing",
                ),
                "prefill_ipc_profile": implementation.get(
                    "prefill_ipc_profile",
                    "missing",
                ),
                "kv_handoff_mode": implementation.get(
                    "kv_handoff_mode",
                    "missing",
                ),
                "async_decode_token": decode_token_setting,
                "unified_kv": _boolean_setting(
                    metadata,
                    "unified_kv",
                    effective,
                    "PAP_UNIFIED_KV",
                    fallback=unified_kv_fallback,
                ),
                "batched_route_copy": _boolean_setting(
                    metadata,
                    "batched_route_copy",
                    effective,
                    "PAP_BATCHED_ROUTE_COPY",
                ),
                "attention_dispatch_mode": metadata.get(
                    "attention_dispatch_mode",
                    effective.get("PAP_ATTENTION_DISPATCH_MODE", "missing"),
                ),
            },
        },
        "environment": {
            "hardware_signature": hardware_signature,
            "software": {
                "vllm": vllm_version,
                "cuda_driver": "missing",
                "torch": "missing",
                "nixl": "missing",
                "ucx": "missing",
            },
            "fingerprint_basis": list(environment_basis),
        },
        "fingerprints": {
            "profile": aggregate.get("profile_fingerprint", "missing"),
            "implementation": aggregate.get(
                "implementation_fingerprint",
                "missing",
            ),
            "request_shape": aggregate.get("request_shape", {}).get(
                "fingerprint",
                "missing",
            ),
            "hardware_runtime": _canonical_digest(environment_basis),
        },
        "repetitions": {
            "count": len(repetitions),
            "order": [index for index, _ in repetitions],
            "aggregation_method": aggregate.get(
                "aggregation_method",
                "missing",
            ),
            "items": [
                {
                    "index": payload["index"],
                    "repetition_id": (
                        aggregate.get("repetition_ids", [])[position]
                        if position < len(aggregate.get("repetition_ids", []))
                        else "missing"
                    ),
                    "result_artifact": _path_reference(
                        payload["directory"] / "result.json",
                        artifact_root_id,
                        artifact_root,
                    ),
                }
                for position, payload in enumerate(rep_payloads)
            ],
        },
        "audits": audits,
        "metrics": {
            "completed_requests": completed,
            "failed_requests": failed,
            "values": [
                _metric(
                    "ttft_ms",
                    "round_1",
                    round_1.get("ttft_ms", {}).get("median"),
                ),
                _metric(
                    "tpot_ms",
                    "round_1",
                    round_1.get("tpot_ms", {}).get("median"),
                ),
                _metric(
                    "ttft_ms",
                    "steady_rounds_2_5",
                    steady.get("ttft_ms", {}).get("median"),
                ),
                _metric(
                    "tpot_ms",
                    "steady_rounds_2_5",
                    steady.get("tpot_ms", {}).get("median"),
                ),
            ],
        },
        "artifacts": artifacts,
        "validity": {
            "status": validity_status,
            "reasons": failure_reasons,
        },
        "failure_reasons": failure_reasons,
    }


def _parse_root(raw: str) -> tuple[str, Path]:
    root_id, separator, path = raw.partition("=")
    if not separator or not root_id or not path:
        raise argparse.ArgumentTypeError("root must use ROOT_ID=/absolute/path")
    return root_id, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-id", default="p17_1pa1p")
    parser.add_argument("--evidence", choices=EVIDENCE_GRADES, required=True)
    parser.add_argument("--artifact-root-id", default="pap-worktree")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        type=_parse_root,
        metavar="ROOT_ID=PATH",
    )
    args = parser.parse_args()
    _validate_output_path(args.source_root, args.output)
    roots = {"pap-worktree": ROOT}
    roots.update(dict(args.root))
    manifest = import_legacy_run(
        args.source_root,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        profile_id=args.profile_id,
        evidence=args.evidence,
        artifact_root_id=args.artifact_root_id,
        roots=roots,
    )
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
