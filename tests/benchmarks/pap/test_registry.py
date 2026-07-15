from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.pap.validate_registry import validate_registry


ROOT = Path(__file__).parents[3]
PAP_ROOT = ROOT / "benchmarks" / "pap"


def _copy_registry(tmp_path: Path) -> tuple[Path, Path, Path]:
    profile_dir = tmp_path / "profiles"
    run_dir = tmp_path / "runs"
    experiment_dir = tmp_path / "experiments"
    shutil.copytree(PAP_ROOT / "profiles", profile_dir)
    shutil.copytree(PAP_ROOT / "registry" / "runs", run_dir)
    shutil.copytree(PAP_ROOT / "registry" / "experiments", experiment_dir)
    return profile_dir, run_dir, experiment_dir


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_current_registry_validates() -> None:
    snapshot = validate_registry()

    assert set(snapshot.profiles) == {"p17_1pa1p"}
    assert set(snapshot.runs) == {
        "20260714_dd2073bcf_p17_pre_refactor_formal"
    }
    assert set(snapshot.experiments) == {
        "PAP-20260714-P17-PRE-REFACTOR"
    }
    assert len(snapshot.historical_experiments) == 43
    assert len(snapshot.negative_results) == 15


def test_formal_clean_run_rejects_dirty_worktree(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    run_path = next(run_dir.glob("*.json"))
    run = _load(run_path)
    run["provenance"]["tracked_worktree_dirty"] = True
    _write(run_path, run)

    with pytest.raises(ValueError, match="formal-clean evidence must be tracked-clean"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_run_schema_rejects_absolute_artifact_path(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    run_path = next(run_dir.glob("*.json"))
    run = _load(run_path)
    run["artifacts"][0]["path"]["relative_path"] = "/tmp/result.json"
    _write(run_path, run)

    with pytest.raises(ValueError, match="relative_path"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_run_schema_rejects_short_commit(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    run_path = next(run_dir.glob("*.json"))
    run = _load(run_path)
    run["provenance"]["commit"] = "dd2073bcf"
    _write(run_path, run)

    with pytest.raises(ValueError, match="commit"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_p17_run_rejects_profile_drift(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    run_path = next(run_dir.glob("*.json"))
    run = _load(run_path)
    run["mps"]["attention_visible_sms"] = 27
    _write(run_path, run)

    with pytest.raises(ValueError, match="P17 mps.attention_visible_sms"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_experiment_arm_must_match_run_ids(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    experiment_path = next(experiment_dir.glob("*.json"))
    experiment = _load(experiment_path)
    experiment["baseline"]["run_ids"] = []
    experiment["baseline"]["status"] = "not-applicable"
    _write(experiment_path, experiment)

    with pytest.raises(ValueError, match="arm runs do not match run_ids"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_experiment_metric_must_match_run_manifest(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    experiment_path = next(experiment_dir.glob("*.json"))
    experiment = _load(experiment_path)
    experiment["metrics"][0]["baseline_value"] = 1.0
    _write(experiment_path, experiment)

    with pytest.raises(ValueError, match="baseline metric round_1.ttft_ms drifted"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_experiment_artifact_must_be_registered_by_run(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    experiment_path = next(experiment_dir.glob("*.json"))
    experiment = _load(experiment_path)
    experiment["raw_artifacts"][0]["sha256"] = "f" * 64
    _write(experiment_path, experiment)

    with pytest.raises(ValueError, match="raw artifact aggregate"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_supersede_graph_rejects_cycle(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    experiment_path = next(experiment_dir.glob("*.json"))
    experiment = _load(experiment_path)
    experiment_id = experiment["experiment_id"]
    experiment["decision"] = "superseded"
    experiment["supersedes"] = [experiment_id]
    experiment["superseded_by"] = experiment_id
    _write(experiment_path, experiment)

    with pytest.raises(ValueError, match="supersede graph cycle"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )


def test_registry_rejects_duplicate_experiment_id(tmp_path: Path) -> None:
    profile_dir, run_dir, experiment_dir = _copy_registry(tmp_path)
    source = next(experiment_dir.glob("*.json"))
    shutil.copy2(source, experiment_dir / "PAP-20260714-DUPLICATE.json")

    with pytest.raises(ValueError, match="duplicate experiment_id"):
        validate_registry(
            profile_dir=profile_dir,
            run_dir=run_dir,
            experiment_dir=experiment_dir,
        )
