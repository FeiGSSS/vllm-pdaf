# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from examples.pap.pd_payloads import (
    build_decode_payload,
    build_prefill_payload,
    enrich_prefill_kv_params,
)


def test_build_prefill_payload_sets_native_nixl_pd_flags() -> None:
    payload = build_prefill_payload(
        {
            "model": "qwen",
            "prompt": "hello",
            "stream": True,
            "max_tokens": 32,
            "max_completion_tokens": 32,
            "stream_options": {"include_usage": True},
        }
    )

    assert payload["stream"] is False
    assert payload["max_tokens"] == 1
    assert payload["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    assert "max_completion_tokens" not in payload
    assert "stream_options" not in payload


def test_build_decode_payload_attaches_prefill_kv_params_without_aliasing() -> None:
    kv_params = {
        "remote_engine_id": "prefill-0",
        "remote_block_ids": [[4, 5]],
    }

    payload = build_decode_payload({"model": "qwen", "prompt": "hello"}, kv_params)
    kv_params["remote_engine_id"] = "mutated"

    assert payload["kv_transfer_params"]["remote_engine_id"] == "prefill-0"
    assert payload["kv_transfer_params"]["remote_block_ids"] == [[4, 5]]


def test_enrich_prefill_kv_params_fills_missing_nixl_endpoint() -> None:
    kv_params = enrich_prefill_kv_params(
        {"remote_engine_id": "prefill-0", "remote_host": "host-from-prefill"},
        prefill_host="127.0.0.1",
        prefill_nixl_port=5559,
    )

    assert kv_params["remote_host"] == "host-from-prefill"
    assert kv_params["remote_port"] == 5559
