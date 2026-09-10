#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal cross-host GPUDirect P2P RDMA probe (no vLLM / Dynamo / etcd).

Reproduces the decode-side KV ingest primitive with the production transport
stack: one NIXL agent publishes a registered buffer, the peer agent registers
a destination buffer and issues repeated NIXL READ transfers into it.

The destination memory is the single variable of interest. Landing the peer
DMA in GPU VRAM (``--where cuda``, GPUDirect) is the suspected ingredient;
``--where cpu`` is the host-DRAM control.

This probe deliberately does not decode models or schedule requests. It is a
transport primitive probe under ``experiments/microbench/``.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import signal
import sys
import threading
import time

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

RUNNING = True


def _stop(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_heartbeat(path: str | None, **record) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"wall": time.time(), **record}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def alloc(nbytes: int, where: str, gpu: int) -> torch.Tensor:
    if where == "cuda":
        return torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{gpu}")
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def make_descs(base: int, length: int, count: int, dev_id: int) -> np.ndarray:
    """Build an Nx3 uint64 descriptor array: [address, length, device_id]."""
    if count <= 1:
        addrs = np.array([base], dtype=np.uint64)
        lens = np.array([length], dtype=np.uint64)
    else:
        step = np.uint64(length // count)
        addrs = np.uint64(base) + np.arange(count, dtype=np.uint64) * step
        lens = np.full(count, step, dtype=np.uint64)
    descs = np.empty((addrs.shape[0], 3), dtype=np.uint64)
    descs[:, 0] = addrs
    descs[:, 1] = lens
    descs[:, 2] = np.uint64(dev_id)
    return descs


def start_compute_load(gpu: int, gemm_n: int, h2d_mb: float) -> tuple:
    """Keep the receiving GPU busy like a decode engine while peer DMA lands.

    The modelled loop is small host-to-device input, sustained on-device
    compute, then a small result copy back. That is what makes the GPU both the
    target of inbound peer writes and the requester of its own PCIe traffic.

    With ``h2d_mb <= 0`` the host-side copies are dropped entirely, leaving a
    pure on-device GEMM loop. That isolates whether the request whose completion
    is lost is the bulk host read or something the driver issues on its own.
    """
    stop = threading.Event()
    stats = {"iters": 0}
    host_pipeline = h2d_mb > 0

    def worker() -> None:
        device = f"cuda:{gpu}"
        torch.cuda.set_device(gpu)
        weight = torch.randn(gemm_n, gemm_n, dtype=torch.float16, device=device)
        out = torch.empty_like(weight)
        h_in = h_out = inp = None
        if host_pipeline:
            side = max(1, int((h2d_mb * 1024 * 1024 / 2) ** 0.5))
            h_in = torch.randn(side, side, dtype=torch.float16, pin_memory=True)
            h_out = torch.empty(side, side, dtype=torch.float16, pin_memory=True)
            inp = torch.empty(side, side, dtype=torch.float16, device=device)
        while not stop.is_set():
            if host_pipeline:
                inp.copy_(h_in, non_blocking=False)
            torch.mm(weight, weight, out=out)
            if host_pipeline:
                h_out.copy_(inp, non_blocking=False)
            stats["iters"] += 1

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop, stats


def run_source(args, mem_type: str) -> int:
    buf = alloc(int(args.chunk_mb * 1024**2), args.where, args.gpu)
    dev_id = args.gpu if args.where == "cuda" else 0

    agent = nixl_agent(args.agent_name, nixl_agent_config(backends=["UCX"]))
    reg = agent.get_reg_descs([(buf.data_ptr(), buf.numel(), dev_id, "")], mem_type)
    agent.register_memory(reg, backends=["UCX"])

    payload = {
        "agent_name": args.agent_name,
        "metadata_b64": base64.b64encode(agent.get_agent_metadata()).decode(),
        "base_addr": buf.data_ptr(),
        "chunk_bytes": buf.numel(),
        "dev_id": dev_id,
        "where": args.where,
        "gpu": args.gpu,
    }
    tmp = f"{args.meta_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp, args.meta_file)

    log(
        f"source ready: {args.where} dev={dev_id} "
        f"buffer={buf.numel() / 1e9:.3f} GB meta={args.meta_file}"
    )

    deadline = time.monotonic() + args.duration_s
    while RUNNING and time.monotonic() < deadline:
        time.sleep(1.0)
        write_heartbeat(args.heartbeat_file, role="source", alive=True)
    log("source exiting")
    # NIXL/UCX teardown has been observed to abort at interpreter exit
    # (double free in tcache) even after a clean run; do not run it.
    sys.stdout.flush()
    os._exit(0)


def run_sink(args, mem_type: str) -> int:
    meta_deadline = time.monotonic() + args.meta_timeout_s
    while RUNNING and not os.path.exists(args.meta_file):
        if time.monotonic() > meta_deadline:
            raise SystemExit(f"remote metadata not found: {args.meta_file}")
        time.sleep(0.5)

    with open(args.meta_file, encoding="utf-8") as handle:
        remote = json.load(handle)
    chunk_bytes = int(remote["chunk_bytes"])

    buf = alloc(chunk_bytes, args.where, args.gpu)
    dev_id = args.gpu if args.where == "cuda" else 0

    agent = nixl_agent(args.agent_name, nixl_agent_config(backends=["UCX"]))
    reg = agent.get_reg_descs([(buf.data_ptr(), buf.numel(), dev_id, "")], mem_type)
    agent.register_memory(reg, backends=["UCX"])
    remote_name = agent.add_remote_agent(base64.b64decode(remote["metadata_b64"]))

    count = max(1, args.num_descs)
    local_descs = agent.get_xfer_descs(
        make_descs(buf.data_ptr(), buf.numel(), count, dev_id), mem_type
    )
    local_handle = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
    # Remote descriptors describe the *remote* agent's memory, so they must use
    # the remote segment type, not the sink's own.
    remote_mem_type = "VRAM" if remote.get("where") == "cuda" else "DRAM"
    remote_descs = agent.get_xfer_descs(
        make_descs(int(remote["base_addr"]), chunk_bytes, count, int(remote["dev_id"])),
        remote_mem_type,
    )
    remote_handle = agent.prep_xfer_dlist(remote_name, remote_descs)
    indices = list(range(count))

    log(
        f"sink ready: dst={args.where} dev={dev_id} "
        f"chunk={chunk_bytes / 1e9:.3f} GB descs={count} "
        f"inflight={args.max_inflight} remote={remote_name} "
        f"remote_mem={remote_mem_type}"
    )

    compute_stop = None
    compute_stats = None
    if args.compute_gemm > 0:
        compute_stop, compute_stats = start_compute_load(
            args.gpu, args.compute_gemm, args.compute_h2d_mb
        )
        log(
            f"compute load on: gemm={args.compute_gemm} "
            f"h2d={args.compute_h2d_mb} MB per iteration"
        )

    inflight: collections.deque = collections.deque()
    total_bytes = 0
    paced_bytes = 0
    completed = 0
    backends: set[str] = set()
    start = time.monotonic()
    deadline = start + args.duration_s
    next_heartbeat = start + 1.0
    exit_code = 0
    target_bytes_per_s = max(0.0, args.target_gbps) * 1e9

    try:
        while RUNNING and time.monotonic() < deadline:
            while len(inflight) < args.max_inflight:
                if target_bytes_per_s > 0.0:
                    allowance = chunk_bytes + target_bytes_per_s * (
                        time.monotonic() - start
                    )
                    if paced_bytes >= allowance:
                        break
                handle = agent.make_prepped_xfer(
                    "READ", local_handle, indices, remote_handle, indices
                )
                if agent.transfer(handle) == "ERR":
                    raise SystemExit("transfer post failed")
                inflight.append(handle)
                paced_bytes += chunk_bytes

            if not inflight:
                time.sleep(0.002)
                continue

            handle = inflight.popleft()
            while RUNNING and agent.check_xfer_state(handle) == "PROC":
                time.sleep(0.0002)
            state = agent.check_xfer_state(handle)
            if state != "DONE":
                log(f"transfer finished with state {state}")
                raise SystemExit(1)
            try:
                backends.add(agent.query_xfer_backend(handle))
            except Exception:
                pass
            agent.release_xfer_handle(handle)

            completed += 1
            total_bytes += chunk_bytes
            now = time.monotonic()
            if now >= next_heartbeat:
                elapsed = now - start
                rate = total_bytes / elapsed / 1e9
                log(
                    f"completed={completed} volume={total_bytes / 1e9:.2f} GB "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} GB/s"
                )
                write_heartbeat(
                    args.heartbeat_file,
                    role="sink",
                    completed=completed,
                    volume_gb=round(total_bytes / 1e9, 3),
                    rate_gbps=round(rate, 3),
                    where=args.where,
                    backends=sorted(backends),
                    compute_iters=(compute_stats["iters"] if compute_stats else None),
                )
                next_heartbeat = now + 1.0
    except SystemExit as exc:
        exit_code = int(exc.code or 1)
        raise
    finally:
        # Reap posted-but-unreaped transfers first: releasing an agent that
        # still owns live handles faults in the NIXL/UCX destructor path.
        for handle in list(inflight):
            while RUNNING and agent.check_xfer_state(handle) == "PROC":
                time.sleep(0.001)
            agent.release_xfer_handle(handle)
        inflight.clear()
        if compute_stop is not None:
            compute_stop.set()
        elapsed = max(time.monotonic() - start, 1e-9)
        log(
            f"sink finished: completed={completed} "
            f"volume={total_bytes / 1e9:.2f} GB elapsed={elapsed:.1f}s "
            f"mean_rate={total_bytes / elapsed / 1e9:.2f} GB/s "
            f"backends={sorted(backends)} "
            f"compute_iters={compute_stats['iters'] if compute_stats else 0}"
        )
        write_heartbeat(
            args.heartbeat_file,
            role="sink",
            finished=True,
            completed=completed,
            volume_gb=round(total_bytes / 1e9, 3),
            mean_rate_gbps=round(total_bytes / elapsed / 1e9, 3),
            exit_code=exit_code,
            where=args.where,
            compute_iters=compute_stats["iters"] if compute_stats else 0,
        )
        agent.release_dlist_handle(local_handle)
        agent.release_dlist_handle(remote_handle)
        agent.deregister_memory(reg, backends=["UCX"])
        # Exit without running NIXL/UCX teardown; see the source path above.
        sys.stdout.flush()
        os._exit(exit_code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["source", "sink"], required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--where", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--chunk-mb", type=float, default=1400.0)
    parser.add_argument("--num-descs", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--meta-timeout-s", type=float, default=120.0)
    parser.add_argument("--meta-file", required=True)
    parser.add_argument("--heartbeat-file", default=None)
    parser.add_argument("--compute-gemm", type=int, default=0)
    parser.add_argument("--compute-h2d-mb", type=float, default=8.0)
    parser.add_argument(
        "--target-gbps",
        type=float,
        default=0.0,
        help=(
            "Pace issuance so the average ingest rate approaches this value, "
            "without changing the chunk size. A NIC cannot write slower, so "
            "this shortens the duty cycle rather than the instantaneous rate."
        ),
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    mem_type = "VRAM" if args.where == "cuda" else "DRAM"
    if args.role == "source":
        return run_source(args, mem_type)
    return run_sink(args, mem_type)


if __name__ == "__main__":
    raise SystemExit(main())
