#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the PAP device-side NVSHMEM Graph protocol across two hosts."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch

from vllm.pap.protocol import PAPOffloadExecBatchDescriptor
from vllm.pap.transport.nvshmem.transport import PAPNVSHMEMTransport

_READY_QKV = 0
_READY_OUTPUT = 1
_GRAPH_ABORT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=16 * 1024)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def _descriptor(generation: int) -> PAPOffloadExecBatchDescriptor:
    return PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.0.self_attn.attn",
        items=(),
        batch_id_suffix=f"control-{generation}",
        metadata_template={
            "r": ("request-0",),
            "s": (generation,),
            "a": (0.125,),
        },
    )


def main() -> None:
    args = parse_args()
    if args.bytes <= 0 or args.iterations <= 0:
        raise ValueError("bytes and iterations must be positive")
    rank = int(os.environ["PAP_NVSHMEM_RANK"])
    world_size = int(os.environ["PAP_NVSHMEM_WORLD_SIZE"])
    if world_size != 2 or rank not in {0, 1}:
        raise RuntimeError("cross-host Graph smoke requires exactly two PEs")

    torch.accelerator.set_device_index(0)
    transport = PAPNVSHMEMTransport(
        actor_id=f"cross-host-graph-{rank}",
        device=torch.device("cuda", 0),
        buffer_bytes=args.bytes,
    )
    peer_metadata = json.loads(transport.local_agent_metadata.decode("utf-8"))
    peer_metadata["hostname"] = "remote-peer"
    peer_metadata["rank"] = 1 - rank
    transport.bind_peer(
        json.dumps(peer_metadata, separators=(",", ":")).encode("utf-8")
    )
    world = transport.world
    if (
        world.data is None
        or world.staging is None
        or world.graph_signals is None
        or world.graph_epochs is None
    ):
        raise RuntimeError("cross-host Graph smoke has incomplete NVSHMEM state")

    if rank == 0:
        for generation in range(2):
            transport.send_step_prepare(
                _descriptor(generation),
                dtype=torch.float16,
                qkv_width=16,
                layer_count=1,
            )
    else:
        for generation in range(2):
            descriptor, qkv, _context = transport.recv_graph_step_plan()
            if descriptor.batch_id_suffix != f"control-{generation}":
                raise RuntimeError("cross-host NVSHMEM control plan mismatch")
            if qkv.shape != (1, 16) or qkv.dtype is not torch.float16:
                raise RuntimeError("cross-host NVSHMEM control shape mismatch")
    transport.step_prepare_stream().synchronize()
    world.runtime.barrier()

    peer = 1 - rank
    data = world.data.tensor.narrow(
        0,
        world.data_slot_offset(peer),
        args.bytes,
    )
    staging = world.staging.tensor.narrow(
        0,
        world.staging_data_slot_offset(peer),
        args.bytes,
    )
    epoch = world.graph_epochs.narrow(0, peer, 1)
    source = (
        torch.arange(args.bytes, dtype=torch.int32, device="cuda")
        .remainder_(251)
        .to(torch.uint8)
    )
    result = torch.zeros_like(source)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, stream=stream):
        world.runtime.graph_advance_epoch(epoch=epoch, stream=stream)
        if rank == 0:
            staging.copy_(source)
            world.runtime.graph_put_signal(
                destination=world.data,
                destination_offset=world.data_slot_offset(rank),
                source=staging,
                num_bytes=args.bytes,
                signal=world.graph_signals,
                signal_offset=world.signal_offset(_READY_QKV, rank),
                abort_signal_offset=world.signal_offset(_GRAPH_ABORT, rank),
                epoch=epoch,
                layer_count=1,
                layer_index=0,
                peer=peer,
                stream=stream,
            )
            world.runtime.graph_wait_signal(
                signal=world.graph_signals,
                signal_offset=world.signal_offset(_READY_OUTPUT, peer),
                epoch=epoch,
                layer_count=1,
                layer_index=0,
                generation_delta=0,
                stream=stream,
            )
            result.copy_(data)
        else:
            world.runtime.graph_wait_signal(
                signal=world.graph_signals,
                signal_offset=world.signal_offset(_READY_QKV, peer),
                epoch=epoch,
                layer_count=1,
                layer_index=0,
                generation_delta=0,
                stream=stream,
            )
            staging.copy_(data)
            world.runtime.graph_put_signal(
                destination=world.data,
                destination_offset=world.data_slot_offset(rank),
                source=staging,
                num_bytes=args.bytes,
                signal=world.graph_signals,
                signal_offset=world.signal_offset(_READY_OUTPUT, rank),
                abort_signal_offset=world.signal_offset(_GRAPH_ABORT, rank),
                epoch=epoch,
                layer_count=1,
                layer_index=0,
                peer=peer,
                stream=stream,
            )

    world.runtime.barrier()
    started = time.perf_counter()
    for _ in range(args.iterations):
        graph.replay()
    torch.accelerator.synchronize(0)
    elapsed_s = time.perf_counter() - started
    world.runtime.barrier()
    valid = rank != 0 or bool(torch.equal(result, source))
    print(
        json.dumps(
            {
                "bytes": args.bytes,
                "elapsed_s": elapsed_s,
                "hostname": socket.gethostname(),
                "iterations": args.iterations,
                "mode": world.config.mode,
                "rank": rank,
                "round_trip_us": (
                    elapsed_s * 1_000_000 / args.iterations if rank == 0 else None
                ),
                "valid": valid,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    graph.reset()
    world.runtime.barrier()
    transport.close()
    world.runtime.finalize()
    if not valid:
        raise RuntimeError("cross-host NVSHMEM Graph payload validation failed")


if __name__ == "__main__":
    main()
