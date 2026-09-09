#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify archives and reproduce the PAP KV Snapshot export result."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = {
    "engine_r11": ("engine_core", "20260909_engine_core_native32k_r11"),
    "snapshot_r12": ("snapshot", "20260909_snapshot_native32k_r12"),
    "engine_r13": ("engine_core", "20260909_engine_core_native32k_r13"),
    "snapshot_r14": ("snapshot", "20260909_snapshot_native32k_r14"),
    "snapshot_verify_r3": ("verify", "20260909_snapshot_verify_c30_r3"),
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
    with tempfile.TemporaryDirectory(prefix="pap-kv-snapshot-verify-") as temp:
        restored = Path(temp)
        command = [sys.executable, str(HERE / "analyze.py")]
        for label, (mode, directory) in RUNS.items():
            expected = manifest(RESULTS / f"{label}_SHA256SUMS")
            extract_verified(
                RESULTS / "archives" / f"{label}.tar.gz",
                expected,
                restored,
            )
            command.extend(["--run", f"{mode}:{label}={restored / directory}"])
        reproduced = restored / "summary.json"
        command.extend(["--output", str(reproduced)])
        subprocess.run(command, check=True)
        if reproduced.read_bytes() != (RESULTS / "summary.json").read_bytes():
            raise SystemExit("summary mismatch")
    print("PAP KV Snapshot export experiment verified")


if __name__ == "__main__":
    main()
