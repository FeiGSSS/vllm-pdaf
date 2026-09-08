# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify and reproduce the session-routing A/B experiment."""

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
ARCHIVES = (
    (
        "session_round_robin",
        "session_r2",
        "session_r2.tar.gz",
        "20260909_session_round_robin_r2",
    ),
    ("dynamo", "dynamo_r1", "dynamo_r1.tar.gz", "20260909_dynamo_control_r1"),
    (
        "session_round_robin",
        "session_r3",
        "session_r3.tar.gz",
        "20260909_session_round_robin_r3",
    ),
    ("dynamo", "dynamo_r2", "dynamo_r2.tar.gz", "20260909_dynamo_control_r2"),
)


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

    with tempfile.TemporaryDirectory(prefix="pap-session-routing-verify-") as tmp:
        restored = Path(tmp)
        run_arguments = []
        for policy, label, archive_name, run_dir in ARCHIVES:
            with tarfile.open(RESULTS / "archives" / archive_name) as archive:
                archive.extractall(restored, filter="data")
            run_path = restored / run_dir
            if not run_path.is_dir():
                raise SystemExit(f"restored run not found: {run_path}")
            run_arguments.extend(["--run", f"{policy}:{label}={run_path}"])

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
        expected = json.loads((RESULTS / "summary.json").read_text())
        actual = json.loads((reproduced / "summary.json").read_text())
        if normalize(actual) != normalize(expected):
            raise SystemExit("reproduced summary differs from saved analysis")
        if (reproduced / "requests.csv").read_bytes() != (
            RESULTS / "requests.csv"
        ).read_bytes():
            raise SystemExit("reproduced request table differs from saved analysis")
    print("Session-routing A/B evidence verified")


if __name__ == "__main__":
    main()
