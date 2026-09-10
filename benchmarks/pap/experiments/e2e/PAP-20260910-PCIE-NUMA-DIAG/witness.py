# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Independent, bounded remote liveness record; never controls power."""

import datetime
import json
import os
import socket
import sys
import time
from pathlib import Path

folder = Path(sys.argv[1])
for _ in range(1200):
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
    }
    for port in (22, 19400):
        try:
            with socket.create_connection(("10.128.7.24", port), timeout=1):
                result[str(port)] = "reachable"
        except OSError as exc:
            result[str(port)] = str(exc)
    with (folder / "witness.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()
        os.fsync(f.fileno())
    time.sleep(1)
