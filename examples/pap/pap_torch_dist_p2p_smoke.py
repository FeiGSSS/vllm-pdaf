#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test torch.distributed NCCL tensor roundtrip for PAP OFFLOAD_EXEC."""

from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import queue
import time

import torch
import torch.distributed as dist


def _exchange_all_reduce(tensor: torch.Tensor, src: int) -> torch.Tensor:
    if dist.get_rank() != src:
        tensor.zero_()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _worker_main(
    *,
    rank: int,
    local_rank: int,
    visible_gpus: str,
    init_method: str,
    q: mp.Queue,
    timeout: float,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=timeout),
    )
    q.put((f"rank{rank}_ready", local_rank))
    try:
        if rank == 0:
            qkv = torch.empty((1, 12), dtype=torch.float32, device="cuda")
            _exchange_all_reduce(qkv, src=1)
            q.put(("attention_recv_qkv", tuple(qkv.shape), float(qkv.sum().item())))
            output = qkv[:, :4].contiguous() + 1
            _exchange_all_reduce(output, src=0)
            q.put(("attention_sent_output", tuple(output.shape), float(output.sum().item())))
        else:
            qkv = torch.arange(12, dtype=torch.float32, device="cuda").reshape(1, 12)
            _exchange_all_reduce(qkv, src=1)
            q.put(("projection_sent_qkv", tuple(qkv.shape), float(qkv.sum().item())))
            output = torch.empty((1, 4), dtype=torch.float32, device="cuda")
            _exchange_all_reduce(output, src=0)
            q.put(("projection_recv_output", tuple(output.shape), float(output.sum().item())))
    finally:
        dist.destroy_process_group()


def _wait_for_event(
    q: mp.Queue,
    pending_events: list[tuple],
    name: str,
    timeout: float,
) -> tuple:
    for index, event in enumerate(pending_events):
        if event[0] == name:
            return pending_events.pop(index)

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {name}")
        try:
            event = q.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError(f"timed out waiting for {name}") from exc
        print(event, flush=True)
        if event[0] == name:
            return event
        pending_events.append(event)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12500)
    parser.add_argument("--visible-gpus", default="0,1")
    parser.add_argument("--attention-local-rank", type=int, default=0)
    parser.add_argument("--projection-local-rank", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for torch.distributed NCCL smoke")

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    init_method = f"tcp://{args.host}:{args.port}"
    attention = ctx.Process(
        target=_worker_main,
        kwargs={
            "rank": 0,
            "local_rank": args.attention_local_rank,
            "visible_gpus": args.visible_gpus,
            "init_method": init_method,
            "q": q,
            "timeout": args.timeout,
        },
    )
    projection = ctx.Process(
        target=_worker_main,
        kwargs={
            "rank": 1,
            "local_rank": args.projection_local_rank,
            "visible_gpus": args.visible_gpus,
            "init_method": init_method,
            "q": q,
            "timeout": args.timeout,
        },
    )
    attention.start()
    projection.start()
    pending_events: list[tuple] = []
    try:
        _wait_for_event(q, pending_events, "rank0_ready", args.timeout)
        _wait_for_event(q, pending_events, "rank1_ready", args.timeout)
        _wait_for_event(q, pending_events, "projection_sent_qkv", args.timeout)
        _wait_for_event(q, pending_events, "attention_recv_qkv", args.timeout)
        _wait_for_event(q, pending_events, "attention_sent_output", args.timeout)
        output_event = _wait_for_event(
            q, pending_events, "projection_recv_output", args.timeout
        )
        if output_event[1] != (1, 4) or abs(output_event[2] - 10.0) > 1e-5:
            raise RuntimeError(f"unexpected output event: {output_event}")
    finally:
        for proc in (projection, attention):
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join()
    if attention.exitcode != 0:
        raise RuntimeError(f"attention process failed exitcode={attention.exitcode}")
    if projection.exitcode != 0:
        raise RuntimeError(f"projection process failed exitcode={projection.exitcode}")


if __name__ == "__main__":
    main()
