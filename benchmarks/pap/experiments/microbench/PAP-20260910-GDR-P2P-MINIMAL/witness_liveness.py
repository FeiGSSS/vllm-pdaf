#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent, bounded liveness record for the sink host.

Read-only: opens TCP connections to the sink host's SSH port once per second
and records the outcome. Never controls power. Runs on the surviving peer so
the timeline outlives a sink-host reset.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ports", default="22")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=3600.0)
    args = parser.parse_args()

    ports = [int(item) for item in args.ports.split(",") if item]
    out = Path(args.out)
    deadline = time.monotonic() + args.seconds
    unreachable = 0

    while time.monotonic() < deadline:
        record = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "monotonic": time.monotonic(),
        }
        reachable = False
        for port in ports:
            try:
                with socket.create_connection((args.target, port), timeout=1):
                    record[str(port)] = "reachable"
                    reachable = True
            except OSError as exc:
                record[str(port)] = str(exc)
        unreachable = 0 if reachable else unreachable + 1
        record["unreachable_streak"] = unreachable
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
