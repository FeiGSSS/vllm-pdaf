#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify and reproduce the aligned DP, PD, and PAP comparison."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RESULTS = HERE / "results"
BASE_RESULTS = ROOT / "benchmarks/pap/experiments/e2e/PAP-20260907-ALIGNED-C30/results"
PD7_RESULTS = (
    ROOT / "benchmarks/pap/experiments/e2e/PAP-20260907-ALIGNED-C30-7P1D/results"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_sha_file(root: Path, manifest: Path) -> None:
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if digest(path) != expected:
            raise SystemExit(f"checksum mismatch: {path}")


def extract_verified(
    archive: Path,
    expected: dict[str, str],
    destination: Path,
) -> None:
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"invalid archive member: {member.name}")
            if member.isdir():
                continue
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in seen
            ):
                raise SystemExit(f"invalid archive member: {member.name}")
            seen.add(member.name)
            with tar.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected[member.name]:
                raise SystemExit(f"archive member checksum: {member.name}")
        if seen != expected.keys():
            raise SystemExit(f"archive inventory mismatch: {archive}")
        tar.extractall(destination, filter="data")


def sha_manifest(path: Path) -> dict[str, str]:
    return {
        relative: checksum
        for checksum, relative in (
            line.split(maxsplit=1) for line in path.read_text().splitlines()
        )
    }


def main() -> None:
    check_sha_file(RESULTS, RESULTS / "SHA256SUMS")
    check_sha_file(BASE_RESULTS, BASE_RESULTS / "SHA256SUMS")
    check_sha_file(PD7_RESULTS, PD7_RESULTS / "SHA256SUMS")
    base_manifest = json.loads((BASE_RESULTS / "manifest.json").read_text())
    base_archives = {entry["path"]: entry for entry in base_manifest["archives"]}

    with tempfile.TemporaryDirectory(prefix="pap-kv-architecture-verify-") as tmp:
        restored = Path(tmp)
        run_specs = []
        for architecture in ("dp8", "6p2d", "4p4d", "2p6d"):
            for repetition in ("r1", "r2"):
                name = f"{architecture}_{repetition}"
                relative = f"archives/{name}.tar.gz"
                entry = base_archives[relative]
                expected = {item["path"]: item["sha256"] for item in entry["files"]}
                extract_verified(BASE_RESULTS / relative, expected, restored)
                run_specs.append(
                    (
                        "dp8" if architecture == "dp8" else f"pd{architecture}",
                        name
                        if architecture == "dp8"
                        else f"pd{architecture}_{repetition}",
                        restored / "runs_aligned" / name,
                    )
                )

        for repetition in (1, 2):
            run_name = f"run_{repetition}"
            expected = sha_manifest(PD7_RESULTS / f"{run_name}_SHA256SUMS")
            extract_verified(
                PD7_RESULTS / "archives" / f"{run_name}.tar.gz",
                expected,
                restored,
            )
            run_specs.append(("pd7p1d", f"pd7p1d_r{repetition}", restored / run_name))

        for repetition in (7, 8):
            archive_name = f"pap_native32k_r{repetition}.tar.gz"
            manifest_name = f"pap_native32k_r{repetition}_SHA256SUMS"
            expected = sha_manifest(RESULTS / manifest_name)
            extract_verified(
                RESULTS / "archives" / archive_name,
                expected,
                restored,
            )
            run_specs.append(
                (
                    "pap7pa1p_kv_migration",
                    f"pap_kv_r{repetition}",
                    restored / f"20260909_dynamo_migration_native32k_r{repetition}",
                )
            )

        output = restored / "analysis"
        command = [sys.executable, str(HERE / "analyze.py")]
        for architecture, label, path in run_specs:
            command.extend(["--run", f"{architecture}:{label}={path}"])
        command.extend(["--output-dir", str(output)])
        subprocess.run(command, check=True)
        if (output / "summary.json").read_bytes() != (
            RESULTS / "summary.json"
        ).read_bytes():
            raise SystemExit("summary mismatch")
        if (output / "requests.csv").read_bytes() != (
            RESULTS / "requests.csv"
        ).read_bytes():
            raise SystemExit("request table mismatch")
    print("Aligned DP/PD/PAP KV-migration comparison verified")


if __name__ == "__main__":
    main()
