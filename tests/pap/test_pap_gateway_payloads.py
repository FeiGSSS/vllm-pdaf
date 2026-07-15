# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway request-payload tests."""

from vllm.pap.gateway.payloads import (
    build_decode_payload,
    build_prefill_payload,
    build_projection_kv_unaware_payload,
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


def test_prefill_payload_marks_attention_kv_import() -> None:
    from vllm.pap.gateway.payloads import attach_pap_prefill_attention_params

    payload = attach_pap_prefill_attention_params(
        build_prefill_payload({"model": "qwen", "prompt": "hello"}),
        pap_attention_endpoint="http://127.0.0.1:8300",
        pap_offload_exec_zmq_endpoint="127.0.0.1:10300",
        pap_prefill_kv_handle="req-7",
        pap_mode="pap",
    )

    assert (
        payload["kv_transfer_params"]["pap_import_prefill_kv_to_attention"] is True
    )
    assert (
        payload["kv_transfer_params"]["pap_offload_exec_zmq_endpoint"]
        == "127.0.0.1:10300"
    )


def test_build_decode_payload_can_mark_attention_kv_installed_after_prefill() -> None:
    payload = build_decode_payload(
        {"model": "qwen", "prompt": "hello"},
        {"remote_engine_id": "prefill-0"},
        pap_attention_kv_installed=True,
    )

    assert payload["kv_transfer_params"]["pap_attention_kv_installed"] is True


def test_build_projection_kv_unaware_payload_strips_remote_kv_transport() -> None:
    payload = build_projection_kv_unaware_payload(
        {"model": "qwen", "prompt": "hello"},
        {
            "remote_engine_id": "prefill-0",
            "remote_request_id": "prefill-req",
            "remote_block_ids": [[4, 5]],
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
            "remote_num_tokens": 17,
            "tp_size": 1,
        },
        pap_attention_endpoint="http://127.0.0.1:8300",
        pap_attention_tcp_endpoint="tcp://127.0.0.1:9300",
        pap_offload_exec_zmq_endpoint="127.0.0.1:10300",
        pap_prefill_kv_handle="req-7",
        pap_attention_kv_installed=True,
    )

    kv_params = payload["kv_transfer_params"]
    assert kv_params == {
        "pap_projection_kv_unaware": True,
        "pap_remote_prefix_len": 17,
        "pap_attention_endpoint": "http://127.0.0.1:8300",
        "pap_attention_tcp_endpoint": "tcp://127.0.0.1:9300",
        "pap_offload_exec_zmq_endpoint": "127.0.0.1:10300",
        "pap_prefill_kv_handle": "req-7",
        "pap_attention_kv_installed": True,
    }


def test_enrich_prefill_kv_params_fills_missing_nixl_endpoint() -> None:
    kv_params = enrich_prefill_kv_params(
        {"remote_engine_id": "prefill-0", "remote_host": "host-from-prefill"},
        prefill_host="127.0.0.1",
        prefill_nixl_port=5559,
    )

    assert kv_params["remote_host"] == "host-from-prefill"
    assert kv_params["remote_port"] == 5559
