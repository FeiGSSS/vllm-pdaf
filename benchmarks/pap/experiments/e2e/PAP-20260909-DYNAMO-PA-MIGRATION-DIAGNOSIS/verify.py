# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify archived task-1 traces and reproduce their analysis."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize(summary: dict) -> dict:
    result = json.loads(json.dumps(summary))
    for run in result["runs"]:
        run.pop("root", None)
    return result


def main() -> None:
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = RESULTS / relative
        if digest(path) != expected:
            raise SystemExit(f"checksum mismatch: {path}")

    with tempfile.TemporaryDirectory(prefix="pap-task1-verify-") as directory:
        restored = Path(directory)
        archives = [
            (
                "samehost_trace_r3",
                "samehost_trace_r3.tar.gz",
                "20260909_task1_samehost_trace_r3",
            ),
            (
                "crosshost_trace_r2",
                "crosshost_trace_r2.tar.gz",
                "20260909_task1_crosshost_trace_r2",
            ),
        ]
        run_arguments = []
        for label, name, run_dir in archives:
            with tarfile.open(RESULTS / "archives" / name) as archive:
                archive.extractall(restored, filter="data")
            run_path = restored / run_dir
            if not run_path.is_dir():
                raise SystemExit(f"restored run not found for {label}: {run_path}")
            run_arguments.extend(["--run", f"{label}={run_path}"])
        reproduced = restored / "analysis"
        subprocess.run(
            [
                sys.executable,
                str(HERE / "analyze.py"),
                *run_arguments,
                "--output-dir",
                str(reproduced),
            ],
            check=True,
        )
        expected_summary = json.loads((RESULTS / "summary.json").read_text())
        actual_summary = json.loads((reproduced / "summary.json").read_text())
        if normalize(actual_summary) != normalize(expected_summary):
            raise SystemExit("reproduced summary differs from saved analysis")
        if (reproduced / "requests.csv").read_bytes() != (
            RESULTS / "requests.csv"
        ).read_bytes():
            raise SystemExit("reproduced request table differs from saved analysis")
    print("Task-1 Dynamo migration evidence verified")


if __name__ == "__main__":
    main()
