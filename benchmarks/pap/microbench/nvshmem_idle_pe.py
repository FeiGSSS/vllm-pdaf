#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep an otherwise idle PE in a balanced cross-host NVSHMEM world."""

from __future__ import annotations

import json
import os
import signal
import socket

from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit
from vllm.pap.transport.nvshmem.world import get_pap_nvshmem_world


def main() -> None:
    device_index = int(os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0"))
    buffer_bytes = int(os.environ["PAP_NVSHMEM_BUFFER_BYTES"])
    world = get_pap_nvshmem_world(
        device_index=device_index,
        buffer_bytes=buffer_bytes,
    )
    world.wait_ready()
    write_runtime_cuda_context_audit(role="nvshmem_idle")
    print(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "rank": world.rank,
                "state": "ready",
                "world_size": world.world_size,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
