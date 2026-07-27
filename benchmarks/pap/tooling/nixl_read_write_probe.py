# SPDX-License-Identifier: Apache-2.0
"""Compare NIXL READ and WRITE for a paged GPU-to-GPU transfer."""

import argparse
import multiprocessing as mp
import os
import time
import uuid

import torch

from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config


def _make_wrapper(name: str) -> NixlWrapper:
    return NixlWrapper(
        name,
        nixl_agent_config(num_threads=4, capture_telemetry=True),
    )


def _allocate_regions(
    device: int,
    regions: int,
    blocks_per_region: int,
    segment_bytes: int,
) -> list[torch.Tensor]:
    torch.cuda.set_device(device)
    region_bytes = blocks_per_region * segment_bytes * 2
    return [
        torch.empty(region_bytes, dtype=torch.uint8, device=f"cuda:{device}")
        for _ in range(regions)
    ]


def _registration_rows(
    tensors: list[torch.Tensor],
    device: int,
) -> list[tuple[int, int, int, str]]:
    return [(int(tensor.data_ptr()), tensor.numel(), device, "") for tensor in tensors]


def _selected_rows(
    tensors: list[torch.Tensor],
    device: int,
    blocks: int,
    segment_bytes: int,
) -> list[tuple[int, int, int]]:
    stride = segment_bytes * 2
    return [
        (int(tensor.data_ptr()) + block * stride, segment_bytes, device)
        for tensor in tensors
        for block in range(blocks)
    ]


def _initiator(
    device: int,
    regions: int,
    blocks: int,
    segment_bytes: int,
    repetitions: int,
    operation: str,
    connection: mp.connection.Connection,
) -> None:
    tensors = _allocate_regions(device, regions, blocks, segment_bytes)
    wrapper = _make_wrapper(f"nixl-initiator-{os.getpid()}-{uuid.uuid4()}")
    registration = wrapper.get_reg_descs(
        _registration_rows(tensors, device),
        "VRAM",
    )
    wrapper.register_memory(registration, backends=["UCX"])
    connection.send(
        (wrapper.get_agent_metadata(), [int(t.data_ptr()) for t in tensors])
    )
    peer_metadata, peer_bases, peer_device = connection.recv()
    peer_name = wrapper.add_remote_agent(peer_metadata)

    local_descs = wrapper.get_xfer_descs(
        _selected_rows(tensors, device, blocks, segment_bytes),
        "VRAM",
    )
    peer_rows = [
        (base + block * segment_bytes * 2, segment_bytes, peer_device)
        for base in peer_bases
        for block in range(blocks)
    ]
    peer_descs = wrapper.get_xfer_descs(peer_rows, "VRAM")
    local_handle = wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", local_descs)
    peer_handle = wrapper.prep_xfer_dlist(peer_name, peer_descs)
    indices = list(range(regions * blocks))

    started = time.perf_counter()
    transfer_handle = wrapper.make_prepped_xfer(
        operation,
        local_handle,
        indices,
        peer_handle,
        indices,
    )
    prepare_seconds = time.perf_counter() - started
    durations: list[float] = []
    telemetry_durations: list[float] = []
    try:
        for _ in range(repetitions + 1):
            started = time.perf_counter()
            wrapper.transfer(transfer_handle)
            while wrapper.check_xfer_state(transfer_handle) != "DONE":
                pass
            durations.append(time.perf_counter() - started)
            telemetry = wrapper.get_xfer_telemetry(transfer_handle)
            telemetry_durations.append(telemetry.xferDuration / 1e6)
        connection.send((prepare_seconds, durations[1:], telemetry_durations[1:]))
    finally:
        wrapper.release_xfer_handle(transfer_handle)
        wrapper.release_dlist_handle(local_handle)
        wrapper.release_dlist_handle(peer_handle)
        wrapper.deregister_memory(registration)


def _peer(
    device: int,
    regions: int,
    blocks: int,
    segment_bytes: int,
    connection: mp.connection.Connection,
) -> None:
    tensors = _allocate_regions(device, regions, blocks, segment_bytes)
    wrapper = _make_wrapper(f"nixl-peer-{os.getpid()}-{uuid.uuid4()}")
    registration = wrapper.get_reg_descs(
        _registration_rows(tensors, device),
        "VRAM",
    )
    wrapper.register_memory(registration, backends=["UCX"])
    connection.send(
        (
            wrapper.get_agent_metadata(),
            [int(t.data_ptr()) for t in tensors],
            device,
        )
    )
    connection.recv()
    connection.recv()
    wrapper.deregister_memory(registration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initiator", type=int, default=0)
    parser.add_argument("--peer", type=int, default=1)
    parser.add_argument("--operation", choices=("READ", "WRITE"), required=True)
    parser.add_argument("--regions", type=int, default=72)
    parser.add_argument("--total-mib", type=int, default=1024)
    parser.add_argument("--segment-kib", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    segment_bytes = args.segment_kib * 1024
    total_bytes = args.total_mib * 1024 * 1024
    total_blocks = total_bytes // segment_bytes
    blocks = total_blocks // args.regions
    transferred_bytes = blocks * args.regions * segment_bytes

    context = mp.get_context("spawn")
    initiator_parent, initiator_child = context.Pipe()
    peer_parent, peer_child = context.Pipe()
    initiator = context.Process(
        target=_initiator,
        args=(
            args.initiator,
            args.regions,
            blocks,
            segment_bytes,
            args.repetitions,
            args.operation,
            initiator_child,
        ),
    )
    peer = context.Process(
        target=_peer,
        args=(
            args.peer,
            args.regions,
            blocks,
            segment_bytes,
            peer_child,
        ),
    )
    initiator.start()
    peer.start()
    initiator_metadata, initiator_bases = initiator_parent.recv()
    peer_metadata, peer_bases, peer_device = peer_parent.recv()
    initiator_parent.send((peer_metadata, peer_bases, peer_device))
    peer_parent.send((initiator_metadata, initiator_bases, args.initiator))
    prepare_seconds, durations, telemetry_durations = initiator_parent.recv()
    peer_parent.send("done")
    initiator.join()
    peer.join()
    if initiator.exitcode != 0 or peer.exitcode != 0:
        raise RuntimeError(
            f"probe failed: initiator={initiator.exitcode}, peer={peer.exitcode}"
        )

    average = sum(durations) / len(durations)
    telemetry_average = sum(telemetry_durations) / len(telemetry_durations)
    source = args.initiator if args.operation == "WRITE" else args.peer
    destination = args.peer if args.operation == "WRITE" else args.initiator
    print(
        f"operation={args.operation} direction={source}->{destination} "
        f"regions={args.regions} descriptors={args.regions * blocks} "
        f"prepare_ms={prepare_seconds * 1000:.3f} "
        f"transfer_ms={average * 1000:.3f} "
        f"telemetry_ms={telemetry_average * 1000:.3f} "
        f"GB/s={transferred_bytes / average / 1e9:.3f}"
    )


if __name__ == "__main__":
    main()
