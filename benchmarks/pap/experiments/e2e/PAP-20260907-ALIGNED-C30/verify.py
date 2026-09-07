# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify retained bytes and recompute client metrics without a serving runtime."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    results = Path(__file__).resolve().parent / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    comparison = json.loads((results / "comparison.json").read_text())
    destination = args.extract_to
    if destination:
        destination.mkdir(parents=True, exist_ok=False)
    for line in (results / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        with (results / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == expected, f"Checksum mismatch: {name}")

    grouped = {}
    reference = None
    total_files = 0
    for archive in manifest["archives"]:
        expected = {row["path"]: row for row in archive["files"]}
        seen = set()
        with tarfile.open(results / archive["path"], "r:gz") as tar:
            for member in tar:
                path = Path(member.name)
                require(
                    member.isfile()
                    and not path.is_absolute()
                    and ".." not in path.parts,
                    f"Unsafe member: {member.name}",
                )
                require(member.name not in seen, f"Duplicate: {member.name}")
                seen.add(member.name)
                entry = expected[member.name]
                with tar.extractfile(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                require(
                    member.size == entry["bytes"] and digest == entry["sha256"],
                    f"Member checksum mismatch: {member.name}",
                )
                if member.name.endswith("/aiperf/profile.jsonl"):
                    with tar.extractfile(member) as stream:
                        rows = [json.loads(line) for line in stream]
                    require(len(rows) == 180, f"Request count: {member.name}")
                    lengths = {
                        (
                            r["metadata"]["conversation_id"],
                            r["metadata"]["turn_index"],
                        ): r["metrics"]["output_sequence_length"]
                        for r in rows
                    }
                    require(len(lengths) == 180, "Duplicate request identity")
                    if reference is None:
                        reference = lengths
                    require(lengths == reference, "Output lengths differ across runs")
                    run = path.parts[1]
                    arch = (
                        "pap7pa1p" if run.startswith("run_pap") else run.split("_")[0]
                    )
                    grouped.setdefault(arch, []).extend(rows)
                if destination:
                    tar.extract(member, destination, filter="data")
        require(seen == expected.keys(), f"Member inventory: {archive['path']}")
        total_files += len(seen)

    require(grouped.keys() == comparison["architectures"].keys(), "Architecture set")
    for arch, rows in grouped.items():
        require(len(rows) == 360, f"Two repetitions required: {arch}")
        values = []
        for metric in ("time_to_first_token", "inter_token_latency"):
            samples = sorted(r["metrics"][metric]["value"] for r in rows)
            observed = {
                "mean_ms": statistics.mean(samples),
                "p50_ms": statistics.median(samples),
                "p95_ms": samples[int(0.95 * len(samples))],
            }
            for key, value in observed.items():
                require(
                    math.isclose(value, comparison["architectures"][arch][metric][key]),
                    f"Metric mismatch: {arch}/{metric}/{key}",
                )
            values.append(observed["mean_ms"])
        print(f"{arch}: TTFT {values[0] / 1000:.3f} s; ITL {values[1]:.3f} ms")
    print(f"Verified {total_files} archived files, 10 runs, 1800 requests.")


if __name__ == "__main__":
    main()
