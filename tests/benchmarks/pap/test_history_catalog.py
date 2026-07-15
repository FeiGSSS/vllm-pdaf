from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.pap.history_catalog import load_history_catalog


ROOT = Path(__file__).parents[3]
INDEX_PATH = ROOT / "docs/design/pap-experiment-history-index.md"
STATUS_PATH = ROOT / "benchmarks/pap/registry/history_status.toml"


def test_history_catalog_covers_every_reviewed_row() -> None:
    catalog = load_history_catalog(INDEX_PATH, STATUS_PATH)

    assert len(catalog.experiments) == 43
    assert len(catalog.negative_results) == 15
    root_cause = catalog.experiments["PAP-20260714-ASYNC-TTFT-ROOTCAUSE"]
    assert root_cause.status == "archived"
    assert root_cause.evidence == "diagnostic"
    assert root_cause.decision == "superseded"
    assert (
        root_cause.superseded_by
        == "PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC"
    )
    assert catalog.negative_results["NEG-ADAPTIVE-COALESCE"].decision == (
        "rolled-back"
    )
    assert catalog.negative_results["NEG-ADAPTIVE-COALESCE"].status == "archived"


def test_history_catalog_rejects_an_unclassified_row(tmp_path: Path) -> None:
    status = STATUS_PATH.read_text(encoding="utf-8")
    status = status.replace('  "PAP-20260522-PROTO-NIXL",\n', "", 1)
    broken_status = tmp_path / "history_status.toml"
    broken_status.write_text(status, encoding="utf-8")

    with pytest.raises(ValueError, match="historical evidence coverage mismatch"):
        load_history_catalog(INDEX_PATH, broken_status)
