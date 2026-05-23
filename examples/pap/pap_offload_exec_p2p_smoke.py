#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test PAP OFFLOAD_EXEC P2P NCCL tensor roundtrip.

This bypasses vLLM serving and PAP HTTP control flow. It starts one Attention
role and one Projection role, sends a packed QKV tensor Projection->Attention,
then sends an output tensor Attention->Projection.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time

import torch

from vllm.pap.data_plane import (
    PAPOffloadExecDescriptor,
    build_p2p_nccl_offload_exec_transport,
)


def _set_common_env(send_type: str) -> None:
    os.environ.setdefault("PAP_OFFLOAD_EXEC_SEND_TYPE", send_type)
    os.environ.setdefault("PAP_OFFLOAD_EXEC_NCCL_NUM_CHANNELS", "1")


def _attention_main(
    *,
    host: str,
    attention_port: int,
    projection_address: str,
    visible_gpus: str,
    local_rank: int,
    q: mp.Queue,
    output_ready: mp.Event,
    send_type: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    _set_common_env(send_type)
    torch.cuda.set_device(local_rank)
    transport = build_p2p_nccl_offload_exec_transport(
        local_rank=local_rank,
        kv_port=attention_port,
        hostname=host,
    )
    descriptor = PAPOffloadExecDescriptor(
        request_id="smoke",
        layer_name="layer0",
        step=1,
        scale=1.0,
    )
    q.put(("attention_ready", f"{host}:{attention_port}"))
    qkv = transport.recv_qkv(descriptor, remote_address=projection_address)
    q.put(("attention_recv_qkv", tuple(qkv.shape), float(qkv.sum().item())))
    output = qkv[:, :4].contiguous() + 1
    transport.send_output(descriptor, output, remote_address=projection_address)
    output_ready.set()
    q.put(("attention_sent_output", tuple(output.shape), float(output.sum().item())))


def _projection_main(
    *,
    host: str,
    projection_port: int,
    attention_address: str,
    visible_gpus: str,
    local_rank: int,
    q: mp.Queue,
    output_ready: mp.Event,
    send_type: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    _set_common_env(send_type)
    torch.cuda.set_device(local_rank)
    transport = build_p2p_nccl_offload_exec_transport(
        local_rank=local_rank,
        kv_port=projection_port,
        hostname=host,
    )
    descriptor = PAPOffloadExecDescriptor(
        request_id="smoke",
        layer_name="layer0",
        step=1,
        scale=1.0,
    )
    q.put(("projection_ready", f"{host}:{projection_port}"))
    qkv = torch.arange(12, dtype=torch.float32, device="cuda").reshape(1, 12)
    transport.send_qkv(descriptor, qkv, remote_address=attention_address)
    q.put(("projection_sent_qkv", tuple(qkv.shape), float(qkv.sum().item())))
    if not output_ready.wait(timeout=30):
        raise TimeoutError("timed out waiting for Attention output publication")
    output = transport.recv_output(descriptor, remote_address=attention_address)
    q.put(("projection_recv_output", tuple(output.shape), float(output.sum().item())))


def _wait_for_event(q: mp.Queue, name: str, timeout: float) -> tuple:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--attention-port", type=int, default=12300)
    parser.add_argument("--projection-port", type=int, default=12310)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--send-type", default="GET", choices=["GET", "PUT", "PUT_ASYNC"]
    )
    parser.add_argument(
        "--visible-gpus",
        default="0,1",
        help="CUDA_VISIBLE_DEVICES value shared by both child processes.",
    )
    parser.add_argument("--attention-local-rank", type=int, default=0)
    parser.add_argument("--projection-local-rank", type=int, default=1)
    parser.add_argument(
        "--attention-gpu",
        default=None,
        help="Deprecated alias: use --visible-gpus with --attention-local-rank.",
    )
    parser.add_argument(
        "--projection-gpu",
        default=None,
        help="Deprecated alias: use --visible-gpus with --projection-local-rank.",
    )
    parser.add_argument(
        "--separate-visible-gpus",
        action="store_true",
        help="Expose only --attention-gpu/--projection-gpu in each child.",
    )
    args = parser.parse_args()

    if args.attention_gpu is not None or args.projection_gpu is not None:
        attention_gpu = args.attention_gpu or "0"
        projection_gpu = args.projection_gpu or "1"
        if args.separate_visible_gpus:
            attention_visible_gpus = attention_gpu
            projection_visible_gpus = projection_gpu
            args.attention_local_rank = 0
            args.projection_local_rank = 0
        else:
            args.visible_gpus = f"{attention_gpu},{projection_gpu}"
            attention_visible_gpus = args.visible_gpus
            projection_visible_gpus = args.visible_gpus
            args.attention_local_rank = 0
            args.projection_local_rank = 1
    else:
        attention_visible_gpus = args.visible_gpus
        projection_visible_gpus = args.visible_gpus

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PAP OFFLOAD_EXEC P2P smoke")

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    output_ready = ctx.Event()
    attention_address = f"{args.host}:{args.attention_port}"
    projection_address = f"{args.host}:{args.projection_port}"
    attention = ctx.Process(
        target=_attention_main,
        kwargs={
            "host": args.host,
            "attention_port": args.attention_port,
            "projection_address": projection_address,
            "visible_gpus": attention_visible_gpus,
            "local_rank": args.attention_local_rank,
            "q": q,
            "output_ready": output_ready,
            "send_type": args.send_type,
        },
    )
    projection = ctx.Process(
        target=_projection_main,
        kwargs={
            "host": args.host,
            "projection_port": args.projection_port,
            "attention_address": attention_address,
            "visible_gpus": projection_visible_gpus,
            "local_rank": args.projection_local_rank,
            "q": q,
            "output_ready": output_ready,
            "send_type": args.send_type,
        },
    )
    attention.start()
    try:
        _wait_for_event(q, "attention_ready", args.timeout)
        projection.start()
        _wait_for_event(q, "projection_ready", args.timeout)
        _wait_for_event(q, "projection_sent_qkv", args.timeout)
        _wait_for_event(q, "attention_recv_qkv", args.timeout)
        _wait_for_event(q, "attention_sent_output", args.timeout)
        output_event = _wait_for_event(q, "projection_recv_output", args.timeout)
        if output_event[1] != (1, 4) or abs(output_event[2] - 10.0) > 1e-5:
            raise RuntimeError(f"unexpected output event: {output_event}")
    finally:
        for proc in (projection, attention):
            if proc.pid is None:
                continue
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join()
    if output_event[1] == (1, 4) and abs(output_event[2] - 10.0) <= 1e-5:
        return
    if attention.exitcode != 0:
        raise RuntimeError(f"attention process failed exitcode={attention.exitcode}")
    if projection.exitcode != 0:
        raise RuntimeError(f"projection process failed exitcode={projection.exitcode}")


if __name__ == "__main__":
    main()
