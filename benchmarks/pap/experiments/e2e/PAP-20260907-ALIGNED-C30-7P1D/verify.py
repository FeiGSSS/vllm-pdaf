# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the 7P1D archives and recompute their reported metrics."""

import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    results = Path(__file__).resolve().parent / "results"
    summary = json.loads((results / "comparison.json").read_text())
    for line in (results / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        with (results / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == expected, f"Checksum mismatch: {name}")

    all_rows = []
    all_telemetry = []
    output_lengths = None
    for run_name in ("run_1", "run_2"):
        expected = {}
        manifest = results / f"{run_name}_SHA256SUMS"
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            expected[name] = digest
        seen = set()
        rows = []
        telemetry = []
        archive = results / "archives" / f"{run_name}.tar.gz"
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                path = Path(member.name)
                require(
                    not path.is_absolute() and ".." not in path.parts,
                    f"Unsafe archive member: {member.name}",
                )
                if member.isdir():
                    continue
                require(member.isfile(), f"Non-regular member: {member.name}")
                require(member.name in expected, f"Unexpected member: {member.name}")
                require(member.name not in seen, f"Duplicate member: {member.name}")
                seen.add(member.name)
                with tar.extractfile(member) as stream:
                    payload = stream.read()
                require(
                    hashlib.sha256(payload).hexdigest() == expected[member.name],
                    f"Member checksum mismatch: {member.name}",
                )
                if member.name.endswith("/aiperf/profile.jsonl"):
                    rows = [json.loads(line) for line in payload.splitlines()]
                if member.name.endswith("/aiperf/profile_gpu_telemetry.jsonl"):
                    telemetry = [json.loads(line) for line in payload.splitlines()]
        require(seen == expected.keys(), f"Archive inventory mismatch: {run_name}")
        require(len(rows) == 180, f"Request count mismatch: {run_name}")
        identities = {
            (row["metadata"]["conversation_id"], row["metadata"]["turn_index"]): row[
                "metrics"
            ]["output_sequence_length"]
            for row in rows
        }
        require(len(identities) == 180, f"Request identity mismatch: {run_name}")
        if output_lengths is None:
            output_lengths = identities
        require(identities == output_lengths, "Output lengths differ between runs")
        start = min(row["metadata"]["request_start_ns"] for row in rows)
        end = max(row["metadata"]["request_end_ns"] for row in rows)
        all_telemetry.extend(
            row for row in telemetry if start <= row["timestamp_ns"] <= end
        )
        all_rows.extend(rows)

    require(len(all_rows) == 360, "Combined request count mismatch")
    for metric in (
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
    ):
        samples = [row["metrics"][metric]["value"] for row in all_rows]
        observed = {
            "mean_ms": statistics.mean(samples),
            "p50_ms": statistics.median(samples),
            "p95_ms": percentile(samples, 0.95),
            "p99_ms": percentile(samples, 0.99),
        }
        for key, value in observed.items():
            require(
                math.isclose(value, summary["combined"][metric][key]),
                f"Metric mismatch: {metric}/{key}",
            )

    gpu = summary["gpu_telemetry_request_window"]
    require(len(all_telemetry) == gpu["samples"], "GPU sample count mismatch")
    require(
        sorted({row["gpu_index"] for row in all_telemetry}) == gpu["gpu_indices"],
        "GPU index mismatch",
    )
    telemetry_metrics = {
        "gpu_utilization_mean_percent": "gpu_utilization",
        "memory_utilization_mean_percent": "mem_utilization",
        "power_mean_watts_per_gpu": "gpu_power_usage",
        "temperature_mean_celsius_per_gpu": "gpu_temperature",
        "dram_active_mean_percent": "dram_active",
        "gr_engine_active_mean_percent": "gr_engine_active",
        "tensor_active_mean_percent": "tensor_active",
    }
    for reported, raw in telemetry_metrics.items():
        value = statistics.mean(row["telemetry_data"][raw] for row in all_telemetry)
        require(math.isclose(value, gpu[reported]), f"GPU metric mismatch: {raw}")
    print("Verified 2 runs, 360 requests, all raw files, latency, and GPU metrics.")


if __name__ == "__main__":
    main()
