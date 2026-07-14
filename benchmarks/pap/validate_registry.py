"""Validate PAP profiles, run manifests, and experiment records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).parents[2]
PAP_ROOT = Path(__file__).parent
PROFILE_DIR = PAP_ROOT / "profiles"
SCHEMA_DIR = PAP_ROOT / "schemas"
REGISTRY_DIR = PAP_ROOT / "registry"
RUN_DIR = REGISTRY_DIR / "runs"
EXPERIMENT_DIR = REGISTRY_DIR / "experiments"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
REQUIRED_AUDITS = {
    "client",
    "cache",
    "attention_stats",
    "correctness",
    "decode_token_join",
    "routing",
    "commit",
    "lease",
    "session_drain",
    "mps",
}


@dataclass(frozen=True)
class RegistrySnapshot:
    """A validated in-memory PAP registry."""

    profiles: dict[str, dict[str, Any]]
    runs: dict[str, dict[str, Any]]
    experiments: dict[str, dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a TOML table")
    return value


def _load_keyed_documents(
    directory: Path,
    id_field: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    documents: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        document_id = document.get(id_field)
        if not isinstance(document_id, str) or not document_id:
            errors.append(f"{path}: missing {id_field}")
            continue
        if document_id in documents:
            errors.append(f"duplicate {id_field}: {document_id}")
            continue
        if path.stem != document_id:
            errors.append(
                f"{path}: filename must match {id_field} {document_id}"
            )
        documents[document_id] = document
    return documents, errors


def _load_profiles(directory: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    profiles: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            profile = _read_toml(path)
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        profile_id = profile.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            errors.append(f"{path}: missing profile_id")
            continue
        if profile_id in profiles:
            errors.append(f"duplicate profile_id: {profile_id}")
            continue
        if path.stem != profile_id:
            errors.append(
                f"{path}: filename must match profile_id {profile_id}"
            )
        profiles[profile_id] = profile
    return profiles, errors


def _schema_errors(
    document: dict[str, Any],
    schema: dict[str, Any],
    label: str,
) -> list[str]:
    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{label} {list(error.absolute_path)}: {error.message}"
        for error in errors
    ]


def _nested(profile: dict[str, Any], dotted_key: str) -> object:
    value: object = profile
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_p17_profile(profile: dict[str, Any]) -> list[str]:
    expected = {
        "schema_version": 1,
        "status": "canonical",
        "architecture": "pap",
        "release_gate": True,
        "mode": "formal",
        "repetitions": 3,
        "model.dtype": "float16",
        "model.tensor_parallel_size": 1,
        "workload.document_tokens": 16000,
        "workload.append_tokens_per_later_round": 120,
        "workload.output_tokens_per_round": 256,
        "workload.rounds": 5,
        "workload.active_conversations": 4,
        "workload.requests_per_repetition": 20,
        "workload.request_rate_per_round": 2.0,
        "workload.arrival_mode": "fixed_rate_round_barrier_closed_loop",
        "topology.name": "1pa1p",
        "topology.pa_count": 1,
        "topology.projection_count": 1,
        "transport.offload_exec": "local_fast",
        "transport.offload_kv": "cuda_ipc",
        "mps.mode": "static",
        "mps.prefill_visible_sms": 64,
        "mps.attention_visible_sms": 28,
        "runtime.decode_token_delivery": "async",
        "runtime.prefill_kv_import": "async",
        "runtime.kv_handoff": "sealed_manifest",
        "runtime.kv_ownership": "prefill_owned_unified",
        "runtime.route_copy": "batched_with_input_fallback",
        "runtime.metadata_lookup": "fast_key",
        "runtime.attention_execution": "topology_derived_direct",
        "compatibility.xpayp": "preserved-unverified",
        "compatibility.cross_host_nixl": "preserved-unverified",
    }
    errors = [
        f"p17_1pa1p {key} must be {wanted!r}, got {_nested(profile, key)!r}"
        for key, wanted in expected.items()
        if _nested(profile, key) != wanted
    ]
    for path_key in ("model", "workload.corpus"):
        reference = _nested(profile, path_key)
        if not isinstance(reference, dict):
            errors.append(f"p17_1pa1p {path_key} must be a root reference")
            continue
        relative_path = reference.get("relative_path")
        if (
            not isinstance(reference.get("root_id"), str)
            or not isinstance(relative_path, str)
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            errors.append(f"p17_1pa1p {path_key} has an invalid root reference")
    audit_gates = _nested(profile, "audit.required_gates")
    if not isinstance(audit_gates, list) or set(audit_gates) != REQUIRED_AUDITS:
        errors.append("p17_1pa1p audit gates do not match the release contract")
    for name, digest in profile.get("baseline_fingerprints", {}).items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"p17_1pa1p fingerprint {name} is not SHA-256")
    return errors


def _validate_p17_run_contract(
    run: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    parameters = run["workload"]["parameters"]
    runtime = run["runtime"]["settings"]
    expected = {
        "architecture": profile["architecture"],
        "mode": profile["mode"],
        "workload.model.path": {
            "root_id": profile["model"]["root_id"],
            "relative_path": profile["model"]["relative_path"],
        },
        "workload.model.dtype": profile["model"]["dtype"],
        "workload.model.tensor_parallel_size": profile["model"][
            "tensor_parallel_size"
        ],
        "workload.corpus.path": {
            "root_id": profile["workload"]["corpus"]["root_id"],
            "relative_path": profile["workload"]["corpus"]["relative_path"],
        },
        "workload.corpus.sha256": profile["workload"]["corpus"]["sha256"],
        "topology.name": profile["topology"]["name"],
        "topology.pa_count": profile["topology"]["pa_count"],
        "topology.projection_count": profile["topology"]["projection_count"],
        "topology.routing_policy": profile["topology"]["routing_policy"],
        "placement.prefill_devices": profile["placement"]["prefill_devices"],
        "placement.attention_devices": profile["placement"][
            "attention_devices"
        ],
        "placement.projection_devices": profile["placement"][
            "projection_devices"
        ],
        "transport.offload_exec": profile["transport"]["offload_exec"],
        "transport.offload_kv": profile["transport"]["offload_kv"],
        "transport.same_host": profile["transport"]["same_host"],
        "mps.mode": profile["mps"]["mode"],
        "mps.profile_id": profile["mps"]["profile_id"],
        "mps.prefill_visible_sms": profile["mps"]["prefill_visible_sms"],
        "mps.attention_visible_sms": profile["mps"][
            "attention_visible_sms"
        ],
        "repetitions.count": profile["repetitions"],
        "fingerprints.profile": profile["baseline_fingerprints"][
            "workload_profile"
        ],
        "fingerprints.request_shape": profile["baseline_fingerprints"][
            "request_shape"
        ],
    }
    errors = [
        f"{run['run_id']}: P17 {key} must be {wanted!r}, "
        f"got {_nested(run, key)!r}"
        for key, wanted in expected.items()
        if _nested(run, key) != wanted
    ]
    parameter_keys = (
        "document_tokens",
        "append_tokens_per_later_round",
        "output_tokens_per_round",
        "rounds",
        "active_conversations",
        "request_rate_per_round",
        "arrival_mode",
    )
    for key in parameter_keys:
        if parameters.get(key) != profile["workload"][key]:
            errors.append(f"{run['run_id']}: P17 workload parameter {key} drifted")
    runtime_expected = {
        "offload_exec_transport": profile["transport"]["offload_exec"],
        "direct_mailbox_output": profile["runtime"]["direct_mailbox_output"],
        "unified_md_fast_key": True,
        "prefill_kv_async": True,
        "prefill_ipc_profile": False,
        "kv_handoff_mode": profile["runtime"]["kv_handoff"],
        "async_decode_token": True,
        "unified_kv": True,
        "batched_route_copy": True,
    }
    for key, wanted in runtime_expected.items():
        if runtime.get(key) != wanted:
            errors.append(f"{run['run_id']}: P17 runtime setting {key} drifted")
    expected_completed = (
        profile["audit"]["completed_requests_per_repetition"]
        * profile["repetitions"]
    )
    if run["metrics"]["completed_requests"] != expected_completed:
        errors.append(f"{run['run_id']}: P17 completed request count drifted")
    if run["metrics"]["failed_requests"] != 0:
        errors.append(f"{run['run_id']}: P17 contains failed requests")
    return errors


def _validate_profiles(profiles: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    canonical_release_profiles = [
        profile_id
        for profile_id, profile in profiles.items()
        if profile.get("status") == "canonical"
        and profile.get("release_gate") is True
    ]
    if canonical_release_profiles != ["p17_1pa1p"]:
        errors.append(
            "the only canonical release profile must be p17_1pa1p: "
            f"{canonical_release_profiles}"
        )
    p17 = profiles.get("p17_1pa1p")
    if p17 is None:
        errors.append("missing p17_1pa1p profile")
    else:
        errors.extend(_validate_p17_profile(p17))
    return errors


def _artifact_key(reference: dict[str, Any]) -> tuple[str, str]:
    return reference["root_id"], reference["relative_path"]


def _validate_run_semantics(
    run: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> list[str]:
    run_id = run["run_id"]
    errors: list[str] = []
    profile = profiles.get(run["profile_id"])
    if profile is None:
        errors.append(f"{run_id}: unknown profile_id {run['profile_id']}")
    elif run["profile_id"] == "p17_1pa1p":
        errors.extend(_validate_p17_run_contract(run, profile))

    provenance = run["provenance"]
    if provenance["tracked_worktree_dirty"] is True:
        patch_hashes = {
            provenance["tracked_worktree_patch_sha256"],
            provenance["tracked_index_patch_sha256"],
        }
        if patch_hashes <= {"missing", EMPTY_SHA256}:
            errors.append(f"{run_id}: dirty run must record a non-empty patch")

    artifact_names = [item["name"] for item in run["artifacts"]]
    if len(artifact_names) != len(set(artifact_names)):
        errors.append(f"{run_id}: artifact names must be unique")
    artifact_paths = {
        _artifact_key(item["path"]) for item in run["artifacts"]
    }
    if len(artifact_paths) != len(run["artifacts"]):
        errors.append(f"{run_id}: artifact paths must be unique")

    repetitions = run["repetitions"]
    if repetitions["count"] != len(repetitions["items"]):
        errors.append(f"{run_id}: repetition count does not match items")
    item_order = [item["index"] for item in repetitions["items"]]
    if repetitions["order"] != item_order:
        errors.append(f"{run_id}: repetition order does not match items")
    for item in repetitions["items"]:
        if _artifact_key(item["result_artifact"]) not in artifact_paths:
            errors.append(
                f"{run_id}: repetition {item['index']} result is not an artifact"
            )

    if set(run["audits"]) != REQUIRED_AUDITS:
        errors.append(f"{run_id}: audit set does not match the P17 contract")
    for gate_name, gate in run["audits"].items():
        for reference in gate["evidence"]:
            if _artifact_key(reference) not in artifact_paths:
                errors.append(
                    f"{run_id}: {gate_name} evidence is not an artifact"
                )

    if run["evidence"] == "formal-clean":
        if run["mode"] != "formal":
            errors.append(f"{run_id}: formal-clean evidence requires formal mode")
        if provenance["commit"] == "missing":
            errors.append(f"{run_id}: formal-clean evidence requires a commit")
        if provenance["tracked_worktree_dirty"] is not False:
            errors.append(f"{run_id}: formal-clean evidence must be tracked-clean")
        for patch_field in (
            "tracked_worktree_patch_sha256",
            "tracked_index_patch_sha256",
        ):
            if provenance[patch_field] != EMPTY_SHA256:
                errors.append(
                    f"{run_id}: formal-clean {patch_field} must be empty"
                )
        if repetitions["count"] < 3:
            errors.append(f"{run_id}: formal-clean evidence requires 3 repetitions")
        failed_gates = [
            name
            for name, gate in run["audits"].items()
            if gate["status"] != "passed"
        ]
        if failed_gates:
            errors.append(f"{run_id}: formal-clean gates not passed: {failed_gates}")
        if run["validity"]["status"] != "passed":
            errors.append(f"{run_id}: formal-clean validity must pass")
        if run["failure_reasons"] or run["validity"]["reasons"]:
            errors.append(f"{run_id}: formal-clean run has failure reasons")
        if run["metrics"]["failed_requests"] != 0:
            errors.append(f"{run_id}: formal-clean run has failed requests")
        for name, value in run["fingerprints"].items():
            if value == "missing":
                errors.append(f"{run_id}: formal-clean {name} fingerprint is missing")
    return errors


def _validate_experiment_semantics(
    experiment: dict[str, Any],
    runs: dict[str, dict[str, Any]],
) -> list[str]:
    experiment_id = experiment["experiment_id"]
    errors: list[str] = []
    arm_run_ids: list[str] = []
    for arm_name in ("baseline", "treatment"):
        arm = experiment[arm_name]
        arm_run_ids.extend(arm["run_ids"])
        if arm["status"] == "present" and not arm["run_ids"]:
            errors.append(f"{experiment_id}: present {arm_name} has no runs")
        if arm["status"] != "present" and arm["run_ids"]:
            errors.append(f"{experiment_id}: {arm_name} must not list runs")
    if set(arm_run_ids) != set(experiment["run_ids"]):
        errors.append(f"{experiment_id}: arm runs do not match run_ids")

    for run_id in experiment["run_ids"]:
        run = runs.get(run_id)
        if run is None:
            errors.append(f"{experiment_id}: unknown run_id {run_id}")
        elif run["experiment_id"] != experiment_id:
            errors.append(
                f"{experiment_id}: run {run_id} belongs to "
                f"{run['experiment_id']}"
            )

    all_run_artifacts = {
        (
            artifact["path"]["root_id"],
            artifact["path"]["relative_path"],
            artifact["sha256"],
        )
        for run_id in experiment["run_ids"]
        if run_id in runs
        for artifact in runs[run_id]["artifacts"]
    }
    for artifact in experiment["raw_artifacts"]:
        key = (
            artifact["path"]["root_id"],
            artifact["path"]["relative_path"],
            artifact["sha256"],
        )
        if key not in all_run_artifacts:
            errors.append(
                f"{experiment_id}: raw artifact {artifact['name']} "
                "is not registered by a run"
            )

    for arm_name in ("baseline", "treatment"):
        arm = experiment[arm_name]
        if arm["status"] != "present" or len(arm["run_ids"]) != 1:
            continue
        run = runs.get(arm["run_ids"][0])
        if run is None:
            continue
        observations = {
            (metric["name"], metric["scope"], metric["statistic"]): metric[
                "value"
            ]
            for metric in run["metrics"]["values"]
        }
        value_field = f"{arm_name}_value"
        for metric in experiment["metrics"]:
            expected_value = observations.get(
                (metric["name"], metric["scope"], metric["statistic"])
            )
            recorded_value = metric[value_field]
            if isinstance(recorded_value, (int, float)) and (
                recorded_value != expected_value
            ):
                errors.append(
                    f"{experiment_id}: {arm_name} metric "
                    f"{metric['scope']}.{metric['name']} drifted"
                )

    successor = experiment["superseded_by"]
    if experiment["decision"] == "superseded":
        if successor in (None, "missing"):
            errors.append(f"{experiment_id}: superseded decision needs successor")
    elif successor not in (None, "missing"):
        errors.append(f"{experiment_id}: successor requires superseded decision")

    if experiment["evidence"] == "formal-clean":
        non_formal = [
            run_id
            for run_id in experiment["run_ids"]
            if run_id in runs and runs[run_id]["evidence"] != "formal-clean"
        ]
        if non_formal:
            errors.append(
                f"{experiment_id}: formal-clean record has weaker runs {non_formal}"
            )
        if experiment["validity"]["status"] != "valid":
            errors.append(f"{experiment_id}: formal-clean record must be valid")
    return errors


def _validate_supersede_graph(
    experiments: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for experiment_id, experiment in experiments.items():
        successor = experiment["superseded_by"]
        if isinstance(successor, str) and successor != "missing":
            if successor not in experiments:
                errors.append(f"{experiment_id}: unknown successor {successor}")
            elif experiment_id not in experiments[successor]["supersedes"]:
                errors.append(
                    f"{experiment_id}: successor {successor} is not reciprocal"
                )
        for predecessor in experiment["supersedes"]:
            if predecessor not in experiments:
                errors.append(f"{experiment_id}: unknown predecessor {predecessor}")
            elif experiments[predecessor]["superseded_by"] != experiment_id:
                errors.append(
                    f"{experiment_id}: predecessor {predecessor} is not reciprocal"
                )

    for start in experiments:
        seen: set[str] = set()
        current = start
        while current in experiments:
            if current in seen:
                errors.append(f"supersede graph cycle includes {current}")
                break
            seen.add(current)
            successor = experiments[current]["superseded_by"]
            if not isinstance(successor, str) or successor == "missing":
                break
            current = successor
    return sorted(set(errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(
    artifact: dict[str, Any],
    roots: dict[str, Path],
    label: str,
) -> list[str]:
    reference = artifact["path"]
    root_id = reference["root_id"]
    if root_id not in roots:
        return [f"{label}: unresolved root_id {root_id}"]
    root = roots[root_id].resolve()
    path = (root / reference["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return [f"{label}: artifact escapes root {root_id}"]
    if not path.is_file():
        return [f"{label}: artifact does not exist: {path}"]
    expected = artifact["sha256"]
    if expected != "missing" and _sha256(path) != expected:
        return [f"{label}: artifact digest mismatch: {path}"]
    return []


def validate_registry(
    *,
    profile_dir: Path = PROFILE_DIR,
    run_dir: Path = RUN_DIR,
    experiment_dir: Path = EXPERIMENT_DIR,
    verify_artifacts: bool = False,
    artifact_roots: dict[str, Path] | None = None,
) -> RegistrySnapshot:
    """Validate the complete PAP experiment registry.

    Args:
        profile_dir: Directory containing TOML profiles.
        run_dir: Directory containing run manifests.
        experiment_dir: Directory containing experiment records.
        verify_artifacts: Whether to resolve and hash raw artifact references.
        artifact_roots: Root ID to local directory mappings.

    Returns:
        The validated registry snapshot.

    Raises:
        ValueError: If any schema or cross-record invariant is violated.
    """
    profiles, errors = _load_profiles(profile_dir)
    runs, run_errors = _load_keyed_documents(run_dir, "run_id")
    experiments, experiment_errors = _load_keyed_documents(
        experiment_dir,
        "experiment_id",
    )
    errors.extend(run_errors)
    errors.extend(experiment_errors)
    errors.extend(_validate_profiles(profiles))

    run_schema = _read_json(SCHEMA_DIR / "run_manifest.schema.json")
    experiment_schema = _read_json(
        SCHEMA_DIR / "experiment_record.schema.json"
    )
    Draft202012Validator.check_schema(run_schema)
    Draft202012Validator.check_schema(experiment_schema)

    for run_id, run in runs.items():
        schema_errors = _schema_errors(run, run_schema, run_id)
        errors.extend(schema_errors)
        if not schema_errors:
            errors.extend(_validate_run_semantics(run, profiles))
    for experiment_id, experiment in experiments.items():
        schema_errors = _schema_errors(
            experiment,
            experiment_schema,
            experiment_id,
        )
        errors.extend(schema_errors)
        if not schema_errors:
            errors.extend(_validate_experiment_semantics(experiment, runs))
    if experiments:
        schema_valid_experiments = {
            experiment_id: experiment
            for experiment_id, experiment in experiments.items()
            if not _schema_errors(
                experiment,
                experiment_schema,
                experiment_id,
            )
        }
        errors.extend(_validate_supersede_graph(schema_valid_experiments))

    if verify_artifacts:
        roots = {"pap-worktree": ROOT}
        roots.update(artifact_roots or {})
        for run_id, run in runs.items():
            for artifact in run.get("artifacts", []):
                errors.extend(_verify_artifact(artifact, roots, run_id))
        for experiment_id, experiment in experiments.items():
            for artifact in experiment.get("raw_artifacts", []):
                errors.extend(_verify_artifact(artifact, roots, experiment_id))

    if errors:
        raise ValueError("PAP registry validation failed:\n" + "\n".join(errors))
    return RegistrySnapshot(profiles, runs, experiments)


def _parse_root(raw: str) -> tuple[str, Path]:
    root_id, separator, path = raw.partition("=")
    if not separator or not root_id or not path:
        raise argparse.ArgumentTypeError("root must use ROOT_ID=/absolute/path")
    return root_id, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        type=_parse_root,
        metavar="ROOT_ID=PATH",
    )
    args = parser.parse_args()
    snapshot = validate_registry(
        profile_dir=args.profile_dir,
        run_dir=args.run_dir,
        experiment_dir=args.experiment_dir,
        verify_artifacts=args.verify_artifacts,
        artifact_roots=dict(args.root),
    )
    print(
        "PAP registry valid: "
        f"{len(snapshot.profiles)} profiles, "
        f"{len(snapshot.runs)} runs, "
        f"{len(snapshot.experiments)} experiments"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
