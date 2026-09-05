# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze an aligned PA/Projection window before drain overwrites trace rings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
from merge_detailed_projection_trace import IncompleteTraceWindow, merge


def capture(projection: Path, output: Path, pa_count: int, timeout: float) -> None:
    if output.exists():
        raise ValueError(f"capture already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [projection] + [
        projection.with_name(f"attention_pa_{index}_kernel_trace.pt")
        for index in range(pa_count)
    ]
    deadline = time.monotonic() + timeout
    last_status = "waiting for trace files"
    while time.monotonic() < deadline:
        errors = list(projection.parent.glob("*.error.txt"))
        if errors:
            raise RuntimeError(f"trace exporter failed: {errors}")
        if all(path.is_file() for path in sources):
            with tempfile.TemporaryDirectory(
                prefix="trace-candidate-", dir=output.parent
            ) as temporary:
                candidate = Path(temporary)
                for source in sources:
                    shutil.copyfile(source, candidate / source.name)
                try:
                    merge(candidate / projection.name)
                except IncompleteTraceWindow as exc:
                    last_status = str(exc)
                else:
                    candidate.rename(output)
                    # Reload frozen files so provenance paths name the final
                    # capture rather than a deleted candidate directory.
                    payload = merge(output / projection.name)
                    torch.save(payload, output / "merged.pt")
                    hashes = {}
                    for source in sources:
                        with (output / source.name).open("rb") as stream:
                            hashes[source.name] = hashlib.file_digest(
                                stream, "sha256"
                            ).hexdigest()
                    report = {
                        "status": "passed",
                        "projection_source": str(projection),
                        "pa_latency_shape": list(payload["latency_ns"].shape),
                        "attention_latency_shape": list(
                            payload["attention_kernel_latency_ns"].shape
                        ),
                        "projection_latency_shape": list(
                            payload["projection_latency_ns"].shape
                        ),
                        "first_step": int(payload["step_id"][0]),
                        "last_step": int(payload["step_id"][-1]),
                        "raw_file_sha256": hashes,
                    }
                    (output / "capture.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    (output / "COMPLETE").write_text("passed\n")
                    print(json.dumps(report), flush=True)
                    return
        time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"no aligned trace window within {timeout}s: {last_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pa-count", required=True, type=int)
    parser.add_argument("--timeout", default=1800.0, type=float)
    args = parser.parse_args()
    if args.pa_count < 1 or not 0 < args.timeout < float("inf"):
        parser.error("PA count and finite timeout must be positive")
    capture(args.projection, args.output, args.pa_count, args.timeout)


if __name__ == "__main__":
    main()
