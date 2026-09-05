# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only A/B: identical tokens and seeds, equal versus different cache salts."""

import argparse
import asyncio
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


def external(value: bytes) -> int:
    return int.from_bytes(value, "big") & ((1 << 64) - 1)


async def run(different_salts: bool) -> None:
    sys.path.append(str(ROOT / ".venv-dynamo/lib/python3.12/site-packages"))
    from dynamo.llm import SelectionService

    os.environ["PYTHONHASHSEED"] = "123"
    init_none_hash(sha256)
    service = SelectionService(indexer_threads=4)
    context = zmq.Context()
    sockets = []
    try:
        for worker in range(2):
            socket = context.socket(zmq.XPUB)
            socket.setsockopt(zmq.RCVTIMEO, 5000)
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            sockets.append(socket)
            await service.upsert_worker(
                {
                    "worker_id": worker,
                    "model_name": "salt-probe",
                    "endpoint": f"http://127.0.0.1:{port}",
                    "kv_events_endpoint": f"tcp://127.0.0.1:{port}",
                    "block_size": 16,
                    "max_num_batched_tokens": 2048,
                    "total_kv_blocks": 1000,
                }
            )
            # Do not publish before the subscriber has actually subscribed.
            assert (await asyncio.to_thread(socket.recv)).startswith(b"\x01")
        parents = [None, None]
        observations = []
        for chunk in range(2):
            for worker, socket in enumerate(sockets):
                salt = f"salt-{worker if different_salts else 0}"
                tokens = list(range(chunk * 16, (chunk + 1) * 16))
                extra = (salt,) if chunk == 0 else None
                digest = hash_block_tokens(sha256, parents[worker], tokens, extra)
                event = BlockStored(
                    block_hashes=[external(digest)],
                    parent_block_hash=(
                        external(parents[worker])
                        if parents[worker] is not None
                        else None
                    ),
                    token_ids=tokens,
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                    extra_keys=[extra],
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
                parents[worker] = digest
            # Query repeatedly to expose both initial and settled async state.
            for _ in range(5):
                await asyncio.sleep(0.1)
                observations.append(
                    {
                        "chunk": chunk,
                        "scores": await service.overlap_scores(
                            {"model_name": "salt-probe", "token_ids": list(range(32))}
                        ),
                    }
                )
        print(
            json.dumps(
                {"different_salts": different_salts, "observations": observations}
            )
        )
    finally:
        service.shutdown()
        for socket in sockets:
            socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--different-salts", action="store_true")
    asyncio.run(run(parser.parse_args().different_salts))
