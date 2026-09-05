# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive unsalted multi-turn fixtures without modifying registered source data."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def derive(source: Path, destination: Path) -> dict:
    source_manifest = json.loads((source / "manifest.json").read_text())
    outputs = {}
    removed = {}
    for name, expected in source_manifest["files"].items():
        raw = (source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"registered source checksum mismatch: {name}")
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        shared = copy.deepcopy(records)
        count = 0
        for session in shared:
            for turn in session["turns"]:
                if "cache_salt" in turn:
                    raise ValueError("expected salt only in turn.extra")
                extra = turn.get("extra", {})
                if "cache_salt" in extra:
                    del extra["cache_salt"]
                    count += 1
        if not count:
            raise ValueError(f"source has no salt to remove: {name}")
        outputs[name] = (
            "\n".join(json.dumps(row, ensure_ascii=False) for row in shared) + "\n"
        ).encode()
        removed[name] = count
    manifest = {
        "dataset_id": source_manifest["dataset_id"] + "-shared-prefix",
        "format": "multi-turn",
        "prefix_cache_policy": "shared_across_sessions",
        "source_dataset_id": source_manifest["dataset_id"],
        "source_files": source_manifest["files"],
        "transformation": "remove turn.extra.cache_salt only",
        "removed_salts": removed,
        "files": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in outputs.items()
        },
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in outputs.items():
        (destination / name).write_bytes(raw)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(derive(args.source, args.destination), indent=2))
