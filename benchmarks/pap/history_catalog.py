# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the compact status overlay for the reviewed PAP history tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import regex as re
import tomllib

EVIDENCE_GRADES = frozenset(
    {"formal-clean", "controlled", "diagnostic", "smoke", "historical", "invalid"}
)
DECISIONS = frozenset(
    {"accepted", "optional", "rejected", "rolled-back", "superseded", "inconclusive"}
)
_SECTION_PATTERN = re.compile(r"^### (6\.[0-9]+)\b")
_EXPERIMENT_ROW = re.compile(r"^\| `(?P<id>PAP-[0-9]{8}-[A-Z0-9]+(?:-[A-Z0-9]+)*)` \|")
_NEGATIVE_ROW = re.compile(r"^\| `(?P<id>NEG-[A-Z0-9]+(?:-[A-Z0-9]+)*)` \|")


@dataclass(frozen=True)
class HistoricalExperiment:
    """One reviewed row from section 6 plus its normalized status."""

    experiment_id: str
    section: str
    evidence: str
    decision: str
    superseded_by: str | None
    status: str = "archived"


@dataclass(frozen=True)
class NegativeResult:
    """One reviewed negative-result row from section 7."""

    result_id: str
    decision: str
    status: str = "archived"


@dataclass(frozen=True)
class HistoryCatalog:
    """Validated compact view of the legacy experiment history."""

    experiments: dict[str, HistoricalExperiment]
    negative_results: dict[str, NegativeResult]


def _parse_history_ids(path: Path) -> tuple[dict[str, str], set[str]]:
    experiments: dict[str, str] = {}
    negative_results: set[str] = set()
    section = ""
    area = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 6."):
            area = "experiments"
            continue
        if line.startswith("## 7."):
            area = "negative-results"
            continue
        if line.startswith("## 8."):
            break
        section_match = _SECTION_PATTERN.match(line)
        if section_match:
            section = section_match.group(1)
            continue
        experiment_match = _EXPERIMENT_ROW.match(line)
        negative_match = _NEGATIVE_ROW.match(line)
        if area == "experiments" and experiment_match:
            stable_id = experiment_match.group("id")
            if stable_id in experiments:
                raise ValueError(f"duplicate historical experiment {stable_id}")
            experiments[stable_id] = section
        elif area == "negative-results" and negative_match:
            stable_id = negative_match.group("id")
            if stable_id in negative_results:
                raise ValueError(f"duplicate negative result {stable_id}")
            negative_results.add(stable_id)
    if not experiments or not negative_results:
        raise ValueError("history index must contain sections 6 and 7")
    return experiments, negative_results


def _grouped_ids(
    raw_groups: object,
    *,
    allowed_groups: frozenset[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(raw_groups, dict):
        raise ValueError(f"{label} must be a TOML table")
    assignments: dict[str, str] = {}
    unknown_groups = set(raw_groups) - allowed_groups
    if unknown_groups:
        raise ValueError(f"{label} has unknown groups: {sorted(unknown_groups)}")
    for group, raw_ids in raw_groups.items():
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) for item in raw_ids
        ):
            raise ValueError(f"{label}.{group} must be an array of IDs")
        for stable_id in raw_ids:
            if stable_id in assignments:
                raise ValueError(f"{stable_id} has duplicate {label} assignments")
            assignments[stable_id] = group
    return assignments


def _require_exact_coverage(
    expected: set[str],
    actual: dict[str, str],
    label: str,
) -> None:
    missing = sorted(expected - set(actual))
    unknown = sorted(set(actual) - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} coverage mismatch: missing={missing}, unknown={unknown}"
        )


def load_history_catalog(index_path: Path, status_path: Path) -> HistoryCatalog:
    """Load and cross-check the reviewed history tables and status overlay."""
    rows, negative_rows = _parse_history_ids(index_path)
    with status_path.open("rb") as file_obj:
        status: dict[str, Any] = tomllib.load(file_obj)
    if status.get("schema_version") != 1:
        raise ValueError("history status schema_version must be 1")
    source_document = status.get("source_document")
    if not isinstance(source_document, str) or not index_path.as_posix().endswith(
        source_document
    ):
        raise ValueError("history status source_document does not match the index")

    experiment_status = status.get("experiments")
    negative_status = status.get("negative_results")
    if not isinstance(experiment_status, dict) or not isinstance(negative_status, dict):
        raise ValueError("history status is missing experiment tables")
    evidence = _grouped_ids(
        experiment_status.get("evidence"),
        allowed_groups=EVIDENCE_GRADES,
        label="historical evidence",
    )
    decisions = _grouped_ids(
        experiment_status.get("decision"),
        allowed_groups=DECISIONS,
        label="historical decision",
    )
    negative_decisions = _grouped_ids(
        negative_status.get("decision"),
        allowed_groups=DECISIONS,
        label="negative-result decision",
    )
    experiment_ids = set(rows)
    _require_exact_coverage(experiment_ids, evidence, "historical evidence")
    _require_exact_coverage(experiment_ids, decisions, "historical decision")
    _require_exact_coverage(
        set(negative_rows), negative_decisions, "negative-result decision"
    )

    raw_successors = experiment_status.get("successor", {})
    if not isinstance(raw_successors, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in raw_successors.items()
    ):
        raise ValueError("experiments.successor must map IDs to IDs")
    successors: dict[str, str] = raw_successors
    for experiment_id, decision in decisions.items():
        successor = successors.get(experiment_id)
        if decision == "superseded":
            if successor not in experiment_ids:
                raise ValueError(f"{experiment_id} needs a known successor")
        elif successor is not None:
            raise ValueError(f"{experiment_id} has a successor but is not superseded")

    experiments = {
        experiment_id: HistoricalExperiment(
            experiment_id=experiment_id,
            section=section,
            evidence=evidence[experiment_id],
            decision=decisions[experiment_id],
            superseded_by=successors.get(experiment_id),
        )
        for experiment_id, section in rows.items()
    }
    negative_results = {
        result_id: NegativeResult(
            result_id=result_id,
            decision=negative_decisions[result_id],
        )
        for result_id in negative_rows
    }
    return HistoryCatalog(experiments, negative_results)
