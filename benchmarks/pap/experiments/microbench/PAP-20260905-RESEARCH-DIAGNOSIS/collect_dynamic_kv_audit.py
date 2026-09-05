# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot PAP's allocator and Attention growth counters during an E2E run."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def _get(url: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--attention-port", type=int, default=8300)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    records = []
    for index in range(args.count):
        prefill = _get(
            f"http://127.0.0.1:{args.prefill_port + index}/v1/pap/prefill/kv-load"
        )
        attention = _get(
            f"http://127.0.0.1:{args.attention_port + index}/v1/pap/attention/stats"
        )
        records.append(
            {
                "pa": index,
                "prefill": {
                    key: prefill[key]
                    for key in (
                        "decode_allocation_requests",
                        "decode_allocation_blocks",
                        "decode_allocation_failures",
                        "prefill_revocations",
                        "free_kv_blocks",
                        "total_kv_blocks",
                    )
                },
                "attention": {
                    key: attention[key]
                    for key in (
                        "decode_capacity_requests",
                        "decode_capacity_installs",
                        "decode_capacity_blocks_added",
                        "slot_topology_mismatches",
                    )
                },
            }
        )
    payload = {
        "captured_unix_ns": time.time_ns(),
        "scope": "sequential mid-run control-plane snapshots",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
