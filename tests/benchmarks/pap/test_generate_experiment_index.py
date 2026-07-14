from __future__ import annotations

import pytest

from benchmarks.pap.generate_experiment_index import (
    BEGIN_MARKER,
    END_MARKER,
    render_generated_region,
    update_generated_region,
)
from benchmarks.pap.validate_registry import validate_registry


def test_generated_index_is_deterministic_and_contains_registry_record() -> None:
    snapshot = validate_registry()

    first = render_generated_region(snapshot)
    second = render_generated_region(snapshot)

    assert first == second
    assert first.startswith(BEGIN_MARKER)
    assert first.endswith(END_MARKER)
    assert "PAP-20260714-P17-PRE-REFACTOR" in first
    assert "formal-clean" in first
    assert "accepted" in first


def test_update_generated_region_preserves_manual_text() -> None:
    generated = f"{BEGIN_MARKER}\nnew\n{END_MARKER}"
    existing = f"manual before\n{BEGIN_MARKER}\nold\n{END_MARKER}\nmanual after\n"

    updated = update_generated_region(existing, generated)

    assert updated == "manual before\n" + generated + "\nmanual after\n"


def test_update_generated_region_rejects_unbalanced_markers() -> None:
    with pytest.raises(ValueError, match="invalid PAP experiment index markers"):
        update_generated_region(f"prefix\n{BEGIN_MARKER}\n", "generated")
