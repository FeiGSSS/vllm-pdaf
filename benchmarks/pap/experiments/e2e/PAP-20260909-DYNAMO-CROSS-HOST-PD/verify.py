#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify and reproduce the cross-host Dynamo 7P1D experiment."""

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
SAME_HOST_RESULTS = (
    ROOT / "benchmarks/pap/experiments/e2e/PAP-20260907-ALIGNED-C30-7P1D/results"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_manifest(root: Path, manifest: Path) -> None:
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if digest(path) != expected:
            raise SystemExit(f"checksum mismatch: {path}")


def extract_verified(archive: Path, manifest: Path, destination: Path) -> Path:
    expected = {}
    for line in manifest.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        expected[name] = checksum
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"unsafe archive member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected:
                raise SystemExit(f"unexpected archive member: {member.name}")
            if member.name in seen:
                raise SystemExit(f"duplicate archive member: {member.name}")
            seen.add(member.name)
            with tar.extractfile(member) as stream:
                payload = stream.read()
            if hashlib.sha256(payload).hexdigest() != expected[member.name]:
                raise SystemExit(f"archive checksum mismatch: {member.name}")
        if seen != expected.keys():
            raise SystemExit(f"archive inventory mismatch: {archive}")
        tar.extractall(destination, filter="data")
    return destination / Path(next(iter(expected))).parts[0]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check_accepted(root: Path) -> None:
    profile = json.loads((root / "aiperf/profile.json").read_text())
    startup = json.loads((root / "startup_audit.json").read_text())
    cleanup = json.loads((root / "cleanup_audit.json").read_text())
    analysis = json.loads((root / "dynamo_ttft_analysis.json").read_text())
    config = (root / "effective_config.env").read_text()
    require(profile["request_count"]["avg"] == 180, "accepted request count")
    require(not profile["error_summary"], "accepted run has errors")
    require(not profile["was_cancelled"], "accepted run was cancelled")
    require(profile["osl_mismatch_count"]["avg"] == 0, "output mismatch")
    require(startup["status"] == "passed", "startup audit failed")
    require(startup["registered_workers"] == 8, "worker count mismatch")
    require(startup["local_decode"]["gpu_processes"] == 1, "Decode mapping")
    require(
        startup["remote_prefills"]["gpu_processes_per_gpu"] == [1] * 7,
        "Prefill mapping",
    )
    require(cleanup["status"] == "passed", "cleanup audit failed")
    require(analysis["joined_requests"] == 180, "timeline join mismatch")
    require(analysis["request_metadata_matches"] == 180, "metadata join mismatch")
    require(analysis["session_routing"]["migrations"] == 1, "migration count")
    require(analysis["kv_transfer"]["transfers"] == 180, "transfer count")
    require("ETCD_LEASE_TTL=3600\n" in config, "etcd lease config")
    require("KV_LEASE_DURATION=3600\n" in config, "NIXL lease config")
    forbidden = (
        "Keep-alive lease expired",
        "Unable to maintain lease",
        "Releasing expired KV blocks",
        "NIXL_ERR",
        "EngineDeadError",
        "Traceback",
        "CUDA out of memory",
    )
    timed_logs = []
    for path in (root / "service_logs").glob("*.log"):
        payload = path.read_text(errors="replace")
        boundaries = [
            payload.find(marker)
            for marker in (
                "Received shutdown signal",
                "trigger received signal=SIGTERM",
            )
            if marker in payload
        ]
        timed_logs.append(payload[: min(boundaries)] if boundaries else payload)
    logs = "".join(timed_logs)
    require(not any(item in logs for item in forbidden), "accepted timed log failure")


def check_failure(root: Path) -> None:
    profile = json.loads((root / "aiperf/profile.json").read_text())
    require(profile["request_count"]["avg"] == 60, "failure success count")
    require(
        len(profile["error_summary"]) == 1
        and profile["error_summary"][0]["count"] == 120
        and profile["error_summary"][0]["error_details"]["code"] == 503,
        "failure HTTP evidence",
    )
    lease_failures = sum(
        "Keep-alive lease expired" in path.read_text(errors="replace")
        for path in (root / "service_logs").glob("prefill_*.log")
    )
    require(lease_failures == 7, "expected all seven Prefill leases to expire")
    cleanup = json.loads((root / "cleanup_audit.json").read_text())
    require(cleanup["status"] == "passed", "failure cleanup audit")


def normalize(summary: dict) -> dict:
    output = json.loads(json.dumps(summary))
    for run in output["runs"]:
        run.pop("path", None)
    return output


def main() -> None:
    check_manifest(RESULTS, RESULTS / "SHA256SUMS")
    check_manifest(SAME_HOST_RESULTS, SAME_HOST_RESULTS / "SHA256SUMS")
    with tempfile.TemporaryDirectory(prefix="pap-cross-host-pd-verify-") as tmp:
        restored = Path(tmp)
        same_runs = []
        for run_name in ("run_1", "run_2"):
            root = extract_verified(
                SAME_HOST_RESULTS / "archives" / f"{run_name}.tar.gz",
                SAME_HOST_RESULTS / f"{run_name}_SHA256SUMS",
                restored,
            )
            same_runs.append(root)
        accepted = extract_verified(
            RESULTS / "archives/cross_host_full4.tar.gz",
            RESULTS / "cross_host_full4_SHA256SUMS",
            restored,
        )
        failure = extract_verified(
            RESULTS / "archives/cross_host_full2_failure.tar.gz",
            RESULTS / "cross_host_full2_failure_SHA256SUMS",
            restored,
        )
        check_accepted(accepted)
        check_failure(failure)
        reproduced = restored / "analysis"
        command = [sys.executable, str(HERE / "analyze.py")]
        for path in same_runs:
            command.extend(["--same-host-run", str(path)])
        command.extend(
            [
                "--cross-host-run",
                str(accepted),
                "--output-dir",
                str(reproduced),
            ]
        )
        subprocess.run(command, check=True)
        expected = json.loads((RESULTS / "summary.json").read_text())
        actual = json.loads((reproduced / "summary.json").read_text())
        require(normalize(actual) == normalize(expected), "summary mismatch")
        require(
            (reproduced / "requests.csv").read_bytes()
            == (RESULTS / "requests.csv").read_bytes(),
            "request table mismatch",
        )
    print("Cross-host Dynamo 7P1D evidence verified")


if __name__ == "__main__":
    main()
