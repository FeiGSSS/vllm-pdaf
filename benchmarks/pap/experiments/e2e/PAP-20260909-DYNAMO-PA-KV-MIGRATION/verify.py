# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify and reproduce the same-host PA KV-migration experiment."""

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
RUNS = (
    (
        "control",
        "control_r5",
        "control_r5.tar.gz",
        "20260909_dynamo_migration_control_r5",
    ),
    (
        "migration",
        "migration_r5",
        "migration_r5.tar.gz",
        "20260909_dynamo_migration_enabled_r5",
    ),
    (
        "control",
        "control_r6",
        "control_r6.tar.gz",
        "20260909_dynamo_migration_control_r6",
    ),
    (
        "migration",
        "migration_r6",
        "migration_r6.tar.gz",
        "20260909_dynamo_migration_enabled_r6",
    ),
)
VERIFY_SMOKE = (
    "migration_verify_smoke_r1.tar.gz",
    "20260909_migration_verify_smoke_r1",
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


def check_verify_smoke(root: Path) -> None:
    profile = json.loads((root / "aiperf/profile.json").read_text())
    routing = json.loads((root / "routing_audit.json").read_text())
    sources = json.loads((root / "migration_source_audit.json").read_text())
    verified = sum(
        path.read_text(errors="replace").count("PAP KV migration verified request_id=")
        for path in (root / "service_logs").glob("prefill_*.log")
    )
    if (
        profile["error_summary"]
        or profile["was_cancelled"]
        or profile["osl_mismatch_count"]["avg"] != 0
        or routing["status"] != "passed"
        or routing["kv_migration"]["exports"] != 7
        or verified != 7
        or sources["status"] != "passed"
        or any(item["migration_exports_active"] for item in sources["sources"])
    ):
        raise SystemExit("byte-verification smoke evidence is invalid")


def main() -> None:
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = RESULTS / relative
        if digest(path) != expected:
            raise SystemExit(f"checksum mismatch: {path}")

    with tempfile.TemporaryDirectory(prefix="pap-pa-kv-migration-verify-") as tmp:
        restored = Path(tmp)
        run_arguments = []
        for mode, label, archive_name, run_dir in RUNS:
            with tarfile.open(RESULTS / "archives" / archive_name) as archive:
                archive.extractall(restored, filter="data")
            run_path = restored / run_dir
            if not run_path.is_dir():
                raise SystemExit(f"restored run not found: {run_path}")
            run_arguments.extend(["--run", f"{mode}:{label}={run_path}"])

        smoke_archive, smoke_dir = VERIFY_SMOKE
        with tarfile.open(RESULTS / "archives" / smoke_archive) as archive:
            archive.extractall(restored, filter="data")
        check_verify_smoke(restored / smoke_dir)

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
    print("Same-host PA KV-migration evidence verified")


if __name__ == "__main__":
    main()
