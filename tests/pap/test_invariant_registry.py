from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).parents[2]
REGISTRY_PATH = ROOT / "tests" / "pap" / "invariants.json"
SCHEMA_PATH = ROOT / "tests" / "pap" / "invariants.schema.json"
REQUIRED_NAMESPACES = {
    "protocol",
    "topology",
    "lifecycle",
    "kv",
    "attention",
    "transport",
    "integration",
    "launcher",
    "benchmark-validator",
}
SURVIVING_DISPOSITIONS = {"keep", "merge", "rewrite", "move"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def _registry() -> dict[str, Any]:
    return _load_json(REGISTRY_PATH)


@lru_cache
def _current_node_ids() -> set[str]:
    inventory = _registry()["inventory"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *inventory["collection_targets"],
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    excluded = set(inventory["excluded_owner_paths"])
    return {
        line
        for line in completed.stdout.splitlines()
        if line.startswith("tests/")
        and "::" in line
        and line.split("::", maxsplit=1)[0] not in excluded
    }


def test_invariant_registry_matches_schema() -> None:
    schema = _load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(_registry()),
        key=lambda error: list(error.absolute_path),
    )
    assert not errors, "\n".join(
        f"{list(error.absolute_path)}: {error.message}" for error in errors
    )


def test_invariant_registry_has_consistent_ids_and_paths() -> None:
    registry = _registry()
    assert set(registry["namespaces"]) == REQUIRED_NAMESPACES

    invariants = registry["invariants"]
    invariant_ids = [item["id"] for item in invariants]
    assert len(invariant_ids) == len(set(invariant_ids))
    known_invariants = set(invariant_ids)

    audit = registry["test_audit"]
    audit_by_node = {item["node_id"]: item for item in audit}
    node_ids = [item["node_id"] for item in audit]
    assert len(node_ids) == len(set(node_ids))

    retired = registry["retired_tests"]
    retired_node_ids = [item["node_id"] for item in retired]
    assert len(retired_node_ids) == len(set(retired_node_ids))
    current_node_ids = _current_node_ids()
    for item in retired:
        retired_audit = audit_by_node[item["node_id"]]
        assert retired_audit["disposition"] == "delete"
        assert item["node_id"] not in current_node_ids
        for replacement_node_id in item["replacement_node_ids"]:
            replacement_audit = audit_by_node[replacement_node_id]
            assert replacement_node_id in current_node_ids
            assert set(retired_audit["invariant_ids"]) & set(
                replacement_audit["invariant_ids"]
            )

    nodes_by_invariant: dict[str, set[str]] = defaultdict(set)
    for item in audit:
        assert item["node_id"].split("::", maxsplit=1)[0] == item["owner_path"]
        assert (ROOT / item["owner_path"]).is_file()
        assert (ROOT / item["source_owner"]).exists()
        assert set(item["invariant_ids"]) <= known_invariants
        if item["disposition"] in {"rewrite", "move"}:
            assert item["target_path"] is not None
        else:
            assert item["target_path"] is None
        for invariant_id in item["invariant_ids"]:
            nodes_by_invariant[invariant_id].add(item["node_id"])

    for invariant in invariants:
        for source_owner in invariant["source_owners"]:
            assert (ROOT / source_owner).exists()
        assert set(invariant["node_ids"]) == nodes_by_invariant[invariant["id"]]
        if invariant["required"]:
            assert invariant["regressions"]
            dispositions = {
                item["disposition"]
                for item in audit
                if invariant["id"] in item["invariant_ids"]
            }
            assert dispositions & SURVIVING_DISPOSITIONS


def test_invariant_registry_covers_frozen_pytest_inventory() -> None:
    registry = _registry()
    audited = {item["node_id"] for item in registry["test_audit"]}
    retired = {item["node_id"] for item in registry["retired_tests"]}
    current = _current_node_ids()
    assert not current & retired
    assert audited == current | retired
    assert len(audited) == registry["inventory"]["collected_count"]


def test_invariant_registry_summary_is_derived_from_audit() -> None:
    registry = _registry()
    audit = registry["test_audit"]
    summary = registry["audit_summary"]
    counts = Counter(item["disposition"] for item in audit)
    outcomes = Counter(item["outcome"] for item in audit)

    assert summary["total"] == len(audit)
    assert summary["by_disposition"] == {
        disposition: counts[disposition]
        for disposition in ("keep", "merge", "rewrite", "delete", "move")
    }
    assert round(sum(item["duration_ms"] for item in audit), 3) == registry[
        "inventory"
    ]["total_duration_ms"]
    assert outcomes["passed"] == registry["inventory"]["passed_count"]
    assert outcomes["skipped"] == registry["inventory"]["skipped_count"]
