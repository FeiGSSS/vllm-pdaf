# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only source-contract probe; no model forward or KV publication."""

from __future__ import annotations

import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS"] = "64"

import torch

import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

# The scheduler is CPU-only; no accelerator is visible inside this sandbox.
vllm.platforms._current_platform = CpuPlatform()

from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm.pap.model.prefill import (  # noqa: E402
    _pap_block_ids_from_block_table,
)
from vllm.pap.protocol import PAPPrefillKVSessionManifest  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

KVConnectorFactory.register_connector(
    "PAPPrefillConnector", "vllm.pap.kv_connector", "PAPPrefillConnector"
)

for budget in (128, 32):
    scheduler = create_scheduler(
        model="/data/ssd1/llm-models/Qwen3-8B",
        max_num_batched_tokens=budget,
        max_model_len=256,
        block_size=16,
        num_blocks=128,
        skip_tokenizer_init=True,
        async_scheduling=True,
        use_kv_connector="PAPPrefillConnector",
        use_v2_model_runner=True,
    )
    request = create_requests(
        num_requests=1,
        num_tokens=64,
        max_tokens=1,
        req_ids=[f"pap-reservation-probe-{budget}"],
    )[0]
    request.kv_transfer_params = {
        "pap_import_prefill_kv_to_attention": True,
        "pap_decode_capacity_tokens": 64,
        "pap_prefill_kv_handle": "probe-session",
        "pap_attention_tcp_endpoint": "tcp://127.0.0.1:1",
    }
    scheduler.add_request(request)
    for iteration in range(2):
        output = scheduler.schedule()
        blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
        decode_capacity = scheduler.pap_scheduler.decode_capacity_tokens(request)
        print(
            json.dumps(
                {
                    "token_budget": budget,
                    "iteration": iteration,
                    "prompt_tokens": request.num_prompt_tokens,
                    "advertised_decode_capacity": 64,
                    "adapter_decode_capacity": decode_capacity,
                    "scheduler_lookahead": scheduler.num_lookahead_tokens,
                    "scheduler_v2_branch": scheduler.use_v2_model_runner,
                    "scheduler_class": type(scheduler).__name__,
                    "scheduled_tokens": output.num_scheduled_tokens[request.request_id],
                    "optimistic_computed_tokens": request.num_computed_tokens,
                    "allocated_block_ids": blocks,
                    "allocated_token_capacity": len(blocks) * 16,
                    "final_prefill_chunk": request.num_computed_tokens >= 64,
                }
            ),
            flush=True,
        )
        if request.num_computed_tokens >= 64:
            # Model a fresh V2 input row: zero-initialized, with only allocated
            # IDs copied by the gather kernel. This is not a GPU execution.
            padded_table = torch.zeros((1, 16), dtype=torch.int32)
            padded_table[0, : len(blocks)] = torch.tensor(blocks)
            exported = _pap_block_ids_from_block_table(
                block_table=padded_table,
                seq_len=128,
                block_size=16,
            )
            manifest = PAPPrefillKVSessionManifest(
                request_id=request.request_id,
                session_handle="probe-session",
                catalog_id="probe-catalog",
                prefix_len=64,
                block_ids=tuple(exported),
                block_size=16,
                expected_layer_count=1,
                lease_id="probe-lease",
                leased_block_ids=tuple(exported),
                lease_capacity_tokens=128,
                writable_start_token=64,
                writable_end_token=128,
                ready_event_handle=None,
            )
            print(
                json.dumps(
                    {
                        "token_budget": budget,
                        "kind": "source_modeled_fresh_v2_row_export",
                        "allocated_block_ids": blocks,
                        "exported_block_ids": list(manifest.block_ids),
                        "manifest_validation": "accepted",
                        "gpu_execution": False,
                    }
                ),
                flush=True,
            )
            break
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[]],
            ),
        )
    scheduler.shutdown()
