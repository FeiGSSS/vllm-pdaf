#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the same-host and cross-host PAP migration comparison."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SAME_RESULTS = HERE.parent / "PAP-20260909-KV-SNAPSHOT-EXPORT-C30" / "results"
RUNS = {
    "same_r12": (
        SAME_RESULTS,
        "snapshot_r12",
        "20260909_snapshot_native32k_r12",
    ),
    "same_r14": (
        SAME_RESULTS,
        "snapshot_r14",
        "20260909_snapshot_native32k_r14",
    ),
    "cross_r11": (
        RESULTS,
        "cross_r11",
        "20260909_crosshost_snapshot_migration_r11",
    ),
    "cross_r12": (
        RESULTS,
        "cross_r12",
        "20260909_crosshost_snapshot_migration_r12",
    ),
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


def main() -> None:
    check_manifest(RESULTS, RESULTS / "SHA256SUMS")
    with tempfile.TemporaryDirectory(prefix="pap-cross-host-migration-verify-") as temp:
        restored = Path(temp)
        roots = {}
        for label, (results, archive_label, directory) in RUNS.items():
            expected = manifest(results / f"{archive_label}_SHA256SUMS")
            extract_verified(
                results / "archives" / f"{archive_label}.tar.gz",
                expected,
                restored,
            )
            roots[label] = restored / directory
        reproduced_summary = restored / "summary.json"
        reproduced_requests = restored / "requests.csv"
        command = [sys.executable, str(HERE / "analyze.py")]
        for side, label in (
            ("same", "same_r12"),
            ("same", "same_r14"),
            ("cross", "cross_r11"),
            ("cross", "cross_r12"),
        ):
            command.extend(["--run", f"{side}:{label}={roots[label]}"])
        command.extend(
            [
                "--output",
                str(reproduced_summary),
                "--requests-output",
                str(reproduced_requests),
            ]
        )
        subprocess.run(command, check=True)
        for reproduced, retained in (
            (reproduced_summary, RESULTS / "summary.json"),
            (reproduced_requests, RESULTS / "requests.csv"),
        ):
            if reproduced.read_bytes() != retained.read_bytes():
                raise SystemExit(f"reproduced artifact mismatch: {retained.name}")
    print("Cross-host PAP KV migration comparison verified")


if __name__ == "__main__":
    main()
