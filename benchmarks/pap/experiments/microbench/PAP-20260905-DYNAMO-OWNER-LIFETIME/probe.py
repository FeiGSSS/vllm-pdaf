# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-clock reservation lifetime and cross-session prefix reuse, without GPUs."""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import msgspec
import zmq

from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

ROOT = Path(__file__).resolve().parents[5]


async def run(runtime: str, hold_seconds: float) -> None:
    module = "pap_dynamo_router" if runtime == "pap" else "dynamo.llm"
    location = (
        ".local/pap-dynamo-router"
        if runtime == "pap"
        else ".venv-dynamo/lib/python3.12/site-packages"
    )
    sys.path.insert(0, str(ROOT / location))
    os.environ.update(
        DYN_USE_KV_EVENTS="true",
        DYN_ROUTER_TRACK_ACTIVE_BLOCKS="true",
        DYN_ROUTER_TRACK_PREFILL_TOKENS="true",
        DYN_ROUTER_TRACK_OUTPUT_BLOCKS="false",
        DYN_ROUTER_ASSUME_KV_REUSE="true",
        PYTHONHASHSEED="123",
    )
    init_none_hash(sha256)
    binding = importlib.import_module(module)
    service = binding.SelectionService(indexer_threads=1)
    library = (
        Path(binding.__file__)
        if runtime == "pap"
        else ROOT / location / "dynamo/_core.abi3.so"
    )
    print(
        json.dumps(
            {
                "runtime": runtime,
                "library": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )
    context = zmq.Context()
    socket = context.socket(zmq.XPUB)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    start = time.monotonic()

    def snapshot(phase):
        loads = service.loads(model_name="probe")
        print(
            json.dumps(
                {
                    "runtime": runtime,
                    "phase": phase,
                    "elapsed_s": time.monotonic() - start,
                    "loads": loads,
                }
            ),
            flush=True,
        )
        return sum(row["active_requests"] for model in loads for row in model["loads"])

    try:
        await service.upsert_worker(
            {
                "worker_id": 0,
                "model_name": "probe",
                "endpoint": f"http://127.0.0.1:{port}",
                "kv_events_endpoint": f"tcp://127.0.0.1:{port}",
                "block_size": 16,
                "max_num_batched_tokens": 2048,
                "total_kv_blocks": 1000,
            }
        )
        assert (await asyncio.to_thread(socket.recv)).startswith(b"\x01")
        tokens = list(range(32))
        parent = None
        for chunk in range(2):
            current = tokens[chunk * 16 : (chunk + 1) * 16]
            digest = hash_block_tokens(sha256, parent, current, None)
            external = lambda value: int.from_bytes(value, "big") & ((1 << 64) - 1)
            event = BlockStored(
                block_hashes=[external(digest)],
                parent_block_hash=external(parent) if parent else None,
                token_ids=current,
                block_size=16,
                lora_id=None,
                medium="GPU",
                lora_name=None,
                extra_keys=[None],
                group_idx=0,
            )
            socket.send_multipart(
                [
                    b"probe",
                    chunk.to_bytes(8, "big"),
                    msgspec.msgpack.encode(
                        KVEventBatch(
                            ts=time.time(), events=[event], data_parallel_rank=0
                        )
                    ),
                ]
            )
            parent = digest
        for _ in range(100):
            scores = await service.overlap_scores(
                {"model_name": "probe", "token_ids": tokens}
            )
            if scores["workers"][0]["device_blocks"] == 2:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError(f"prefix events not indexed: {scores}")
        for index, request in enumerate(("session-a-turn-1", "session-b-turn-1")):
            selected = await service.select_and_reserve(
                {
                    "model_name": "probe",
                    "token_ids": list(range(32 + index * 16)),
                    "selection_id": request,
                    "expected_output_tokens": 16,
                }
            )
            print(json.dumps({"request": request, "selection": selected}), flush=True)
            assert selected["effective_prefill_tokens"] == index * 16
        assert snapshot("two_sessions_share_prefix") == 2
        await service.prefill_complete("session-a-turn-1")
        # Keep one request in prefill, the other in decode beyond the old expiry.
        for elapsed in (30, 310, hold_seconds):
            await asyncio.sleep(max(0, start + elapsed - time.monotonic()))
            count = snapshot(f"hold_{elapsed}")
        assert count == (2 if runtime == "pap" else 0)
        if runtime == "pap":
            await service.select_and_reserve(
                {
                    "model_name": "probe",
                    "token_ids": tokens,
                    "selection_id": "session-c-turn-1",
                    "expected_output_tokens": 16,
                }
            )
            assert snapshot("new_booking_after_old_expiry") == 3
            for request in ("session-a-turn-1", "session-b-turn-1", "session-c-turn-1"):
                await service.free_reservation(request)
            assert snapshot("explicit_release") == 0
        print(json.dumps({"runtime": runtime, "passed": True}), flush=True)
    finally:
        service.shutdown()
        socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["pap", "official"], required=True)
    parser.add_argument("--hold-seconds", type=float, default=370)
    args = parser.parse_args()
    asyncio.run(run(args.runtime, args.hold_seconds))
