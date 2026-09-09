#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify and reproduce the cross-host PAP hash consistency experiment."""

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
RUNS = {
    "old_replay_r4": "20260909_crosshost_replay_r4",
    "source_failure_r5": "20260909_crosshost_fence_trace_r5",
    "connector_failure_r6": "20260909_crosshost_fence_trace_r6",
    "unseeded_fence_r7": "20260909_crosshost_fence_trace_r7",
    "hash_fence_r8": "20260909_crosshost_hash_fence_trace_r8",
    "hash_only_r9": "20260909_crosshost_hash_only_trace_r9",
}
ANALYZED_RUNS = {
    "old_replay": "old_replay_r4",
    "unseeded_fence": "unseeded_fence_r7",
    "hash_fence": "hash_fence_r8",
    "hash_only": "hash_only_r9",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest(path: Path) -> dict[str, str]:
    return {
        relative: checksum
        for checksum, relative in (
            line.split(maxsplit=1) for line in path.read_text().splitlines()
        )
    }


def check_manifest(root: Path, path: Path) -> None:
    for relative, expected in manifest(path).items():
        target = root / relative
        if digest(target) != expected:
            raise SystemExit(f"checksum mismatch: {target}")


def extract_verified(
    archive: Path, expected: dict[str, str], destination: Path
) -> None:
    seen = set()
    with tarfile.open(archive) as source:
        for member in source:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"invalid archive path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected:
                raise SystemExit(f"unexpected archive member: {member.name}")
            if member.name in seen:
                raise SystemExit(f"duplicate archive member: {member.name}")
            seen.add(member.name)
            stream = source.extractfile(member)
            if stream is None:
                raise SystemExit(f"cannot read archive member: {member.name}")
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != expected[member.name]
            ):
                raise SystemExit(f"archive checksum mismatch: {member.name}")
        if seen != expected.keys():
            raise SystemExit(f"archive inventory mismatch: {archive}")
        source.extractall(destination, filter="data")


def check_failed_start(root: Path, needle: str) -> None:
    cleanup = json.loads((root / "cleanup_audit.json").read_text())
    if cleanup["status"] != "passed":
        raise SystemExit(f"failed run did not clean up: {root}")
    log = (root / "remote_logs/prefill_0.log").read_text(errors="replace")
    if needle not in log:
        raise SystemExit(f"expected failure absent from {root}")


def main() -> None:
    check_manifest(RESULTS, RESULTS / "SHA256SUMS")
    with tempfile.TemporaryDirectory(prefix="pap-cross-host-hash-verify-") as temp:
        restored = Path(temp)
        roots = {}
        for label, directory in RUNS.items():
            expected = manifest(RESULTS / f"{label}_SHA256SUMS")
            extract_verified(
                RESULTS / "archives" / f"{label}.tar.gz",
                expected,
                restored,
            )
            roots[label] = restored / directory

        check_failed_start(
            roots["source_failure_r5"], "PAP KV migration requires a model executor"
        )
        check_failed_start(
            roots["connector_failure_r6"],
            "PAPPrefillConnector' object has no attribute 'migration_catalog'",
        )

        reproduced = restored / "analysis"
        command = [sys.executable, str(HERE / "analyze.py")]
        for label, archive_label in ANALYZED_RUNS.items():
            command.extend(["--run", f"{label}={roots[archive_label]}"])
        command.extend(["--output-dir", str(reproduced)])
        subprocess.run(command, check=True)
        for name in ("summary.json", "requests.csv"):
            if (reproduced / name).read_bytes() != (RESULTS / name).read_bytes():
                raise SystemExit(f"reproduced {name} mismatch")
    print("Cross-host PAP hash consistency experiment verified")


if __name__ == "__main__":
    main()
