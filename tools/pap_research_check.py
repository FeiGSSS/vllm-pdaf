#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the persistent PAP research workflow scaffold."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "paper" / "pap"

REQUIRED_FILES = (
    ROOT / "docs" / "contributing" / "pap-research-workflow.md",
    PAPER_ROOT / "README.md",
    PAPER_ROOT / "manuscript.md",
    PAPER_ROOT / "claims.md",
    PAPER_ROOT / "state.md",
    PAPER_ROOT / "related-work.md",
    PAPER_ROOT / "references.bib",
    PAPER_ROOT / "figures" / "README.md",
    PAPER_ROOT / "tables" / "README.md",
)

REQUIRED_STATE_HEADINGS = (
    "# PAP Current Research State",
    "## Current loop",
    "## Evidence checkpoint",
    "## Paper gap queue",
    "## Next loop",
    "## Pause and recovery",
)

REQUIRED_MANUSCRIPT_HEADINGS = (
    "## Abstract",
    "## 1. Introduction",
    "## 2. Background and Motivation",
    "## 3. Design",
    "## 4. Implementation",
    "## 5. Evaluation",
    "## 6. Related Work",
    "## 7. Discussion and Limitations",
    "## 8. Conclusion",
)

PLACEHOLDERS = {
    "",
    "none",
    "not-recorded",
    "not-selected",
    "pending-alignment",
    "pending-user-start",
}

FIELD_PATTERN = re.compile(
    r"^- \*\*(?P<name>[^*]+):\*\*\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fields(markdown: str) -> dict[str, str]:
    fields = {}
    for match in FIELD_PATTERN.finditer(markdown):
        value = match.group("value").strip().strip("`")
        fields[match.group("name").strip()] = value
    return fields


def _check_line_budget(path: Path, limit: int, errors: list[str]) -> None:
    line_count = len(_read(path).splitlines())
    if line_count >= limit:
        errors.append(f"{path.relative_to(ROOT)} has {line_count} lines; "
                      f"limit is below {limit}")


def validate() -> list[str]:
    errors = []

    for path in REQUIRED_FILES:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    if errors:
        return errors

    agents = _read(ROOT / "AGENTS.md")
    if "docs/contributing/pap-research-workflow.md" not in agents:
        errors.append("AGENTS.md does not link the PAP research workflow")

    design_readme = _read(ROOT / "docs" / "design" / "pap" / "README.md")
    if "paper/pap/" not in design_readme:
        errors.append("PAP design README does not define the paper boundary")

    state = _read(PAPER_ROOT / "state.md")
    for heading in REQUIRED_STATE_HEADINGS:
        if heading not in state:
            errors.append(f"state.md is missing heading: {heading}")

    fields = _fields(state)
    for name in (
        "Research lifecycle",
        "Execution gate",
        "Active loop",
        "Loop status",
        "Baseline commit",
        "Next action",
    ):
        if name not in fields:
            errors.append(f"state.md is missing field: {name}")

    gate = fields.get("Execution gate")
    if gate not in {"closed", "open"}:
        errors.append("Execution gate must be `closed` or `open`")

    if gate == "open":
        for name in (
            "Active loop",
            "Hypothesis",
            "Falsification condition",
            "Expected paper delta",
            "Next action",
        ):
            if fields.get(name, "") in PLACEHOLDERS:
                errors.append(
                    f"state.md field must be concrete while gate is open: {name}"
                )

    if fields.get("Loop status") == "complete":
        if fields.get("Next action", "") in PLACEHOLDERS:
            errors.append("a completed loop must select the next action")

    manuscript = _read(PAPER_ROOT / "manuscript.md")
    for heading in REQUIRED_MANUSCRIPT_HEADINGS:
        if heading not in manuscript:
            errors.append(f"manuscript.md is missing heading: {heading}")

    _check_line_budget(ROOT / "AGENTS.md", 200, errors)
    _check_line_budget(
        ROOT / "docs" / "contributing" / "pap-research-workflow.md",
        300,
        errors,
    )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    state = _fields(_read(PAPER_ROOT / "state.md"))
    print("PAP research scaffold is valid.")
    print(f"Research lifecycle: {state['Research lifecycle']}")
    print(f"Execution gate: {state['Execution gate']}")
    print(f"Active loop: {state['Active loop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
