#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strictly merge measured PAP Attention matrix shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()

    documents = [(path, read_json(path)) for path in args.shards]
    first = documents[0][1]
    invariant_keys = ("groups", "kernel_set", "candidate_count", "shard_count")
    for path, document in documents:
        if document.get("status") != "completed":
            raise ValueError(f"incomplete shard: {path}")
        for key in invariant_keys:
            if document.get(key) != first.get(key):
                raise ValueError(f"shard {path} disagrees on {key}")

    shard_count = int(first["shard_count"])
    shard_indices = [int(document["shard_index"]) for _, document in documents]
    if sorted(shard_indices) != list(range(shard_count)):
        raise ValueError(f"shard indices are incomplete or duplicated: {shard_indices}")

    results = []
    best_by_shape = []
    shapes = set()
    measurements = set()
    candidate_ids: set[str] | None = None
    for path, document in documents:
        local_candidates = {
            result["kernel"]["config_id"] for result in document["results"]
        }
        if len(local_candidates) != int(document["candidate_count"]):
            raise ValueError(f"shard has an incomplete candidate set: {path}")
        if candidate_ids is None:
            candidate_ids = local_candidates
        elif local_candidates != candidate_ids:
            raise ValueError(f"candidate IDs differ in shard: {path}")
        for result in document["results"]:
            shape_id = result["shape"]["shape_id"]
            measurement = (shape_id, result["kernel"]["config_id"])
            if measurement in measurements:
                raise ValueError(f"duplicate measurement: {measurement}")
            if result.get("status") != "completed":
                raise ValueError(f"failed measurement: {measurement}")
            if not result["correctness"]["allclose"]:
                raise ValueError(f"numerical mismatch: {measurement}")
            measurements.add(measurement)
            shapes.add(shape_id)
            results.append(result)
        best_by_shape.extend(document["best_by_shape"])

    expected_shape_count = int(first["total_shape_count"])
    candidate_count = int(first["candidate_count"])
    if len(shapes) != expected_shape_count:
        raise ValueError(
            f"merged shape count is {len(shapes)}, expected {expected_shape_count}"
        )
    if len(measurements) != expected_shape_count * candidate_count:
        raise ValueError("merged workload/config matrix is incomplete")
    if len(best_by_shape) != expected_shape_count:
        raise ValueError("best-by-shape rows are incomplete")

    output = {
        "schema_version": first["schema_version"],
        "kind": first["kind"],
        "status": "completed",
        "groups": first["groups"],
        "kernel_set": first["kernel_set"],
        "candidate_count": candidate_count,
        "shape_count": expected_shape_count,
        "source_shard_count": shard_count,
        "source_shards": [str(path.resolve()) for path, _ in documents],
        "results": sorted(
            results,
            key=lambda result: (
                result["shape"]["shape_id"],
                result["kernel"]["config_id"],
            ),
        ),
        "best_by_shape": sorted(best_by_shape, key=lambda row: row["shape_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
