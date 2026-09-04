# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify device launch of a graph captured by torch.cuda.CUDAGraph."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    library = ctypes.CDLL(str(args.library))
    run = library.pap_torch_device_graph_smoke
    run.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    run.restype = ctypes.c_int

    static_input = torch.arange(4096, dtype=torch.float32, device="cuda")
    static_output = torch.empty_like(static_input)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        torch.add(static_input, 1.0, out=static_output)
    capture_stream.synchronize()

    elapsed_ms = ctypes.c_float()
    raw_graph = graph.raw_cuda_graph()
    stream = capture_stream.cuda_stream
    status = run(
        raw_graph,
        stream,
        args.iterations,
        ctypes.byref(elapsed_ms),
    )
    if status != 0:
        raise RuntimeError(f"device graph bridge returned CUDA status {status}")
    torch.testing.assert_close(static_output, static_input + 1.0)
    print(
        json.dumps(
            {
                "status": "passed",
                "iterations": args.iterations,
                "gpu_ms": elapsed_ms.value,
                "us_per_device_graph": (elapsed_ms.value * 1000.0 / args.iterations),
            }
        )
    )


if __name__ == "__main__":
    main()
