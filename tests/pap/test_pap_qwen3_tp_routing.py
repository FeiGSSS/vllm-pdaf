# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3 import (
    _pap_decode_token_rows_for_indices,
    _pap_endpoint_for_tp_rank,
    _pap_offload_exec_step_groups,
    _pap_offload_exec_session_request_id,
)


def test_pap_endpoint_for_tp_rank_keeps_scalar_endpoint() -> None:
    assert _pap_endpoint_for_tp_rank("127.0.0.1:10300", tp_rank=1) == (
        "127.0.0.1:10300"
    )


def test_pap_endpoint_for_tp_rank_selects_csv_entry() -> None:
    assert (
        _pap_endpoint_for_tp_rank(
            "127.0.0.1:10300,127.0.0.1:10301",
            tp_rank=1,
        )
        == "127.0.0.1:10301"
    )


def test_pap_endpoint_for_tp_rank_selects_sequence_entry() -> None:
    assert (
        _pap_endpoint_for_tp_rank(
            ("http://127.0.0.1:8300", "http://127.0.0.1:8301"),
            tp_rank=1,
        )
        == "http://127.0.0.1:8301"
    )


def test_pap_decode_token_rows_for_indices_selects_request_tokens() -> None:
    assert _pap_decode_token_rows_for_indices((11, 22, 33), (2, 0)) == (
        (33,),
        (11,),
    )


def test_pap_decode_token_rows_for_indices_uses_empty_tuple_when_missing() -> None:
    assert _pap_decode_token_rows_for_indices((11,), (0, 2)) == ((11,), ())


def test_pap_offload_exec_session_request_id_prefers_prefill_handle() -> None:
    assert (
        _pap_offload_exec_session_request_id(
            "cmpl-projection-0-deadbeef",
            "cmpl-prefill-0-cafebabe",
        )
        == "cmpl-prefill-0-cafebabe"
    )


def test_pap_offload_exec_session_request_id_falls_back_to_local_id() -> None:
    assert (
        _pap_offload_exec_session_request_id("cmpl-projection-0-deadbeef", None)
        == "cmpl-projection-0-deadbeef"
    )


def test_pap_offload_exec_step_groups_are_built_once_per_forward(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0")
    request_ids = (
        "cmpl-projection-0-deadbeef",
        "cmpl-projection-1-deadbeef",
    )
    kwargs = {
        "pap_request_ids": request_ids,
        "pap_input_token_ids": (11, 22),
        "pap_prefill_kv_handle_by_request": {
            request_ids[0]: "cmpl-prefill-0-cafebabe",
            request_ids[1]: "cmpl-prefill-1-cafebabe",
        },
        "pap_attention_kv_installed_by_request": request_ids,
        "pap_prefill_prefix_len_by_request": {
            request_ids[0]: 50,
            request_ids[1]: 50,
        },
        "pap_offload_exec_route_groups": (
            {
                "attention_endpoint": "http://127.0.0.1:8300",
                "offload_exec_zmq_endpoint": "127.0.0.1:10300",
                "req_indices": (0, 1),
                "request_ids": request_ids,
                "steps": (51, 52),
            },
        ),
    }

    first = _pap_offload_exec_step_groups(kwargs, num_reqs=2, scaling=0.125)
    kwargs["pap_offload_exec_route_groups"] = ()
    second = _pap_offload_exec_step_groups(kwargs, num_reqs=2, scaling=0.125)

    assert second is first
    assert first[0].req_indices == (0, 1)
    assert first[0].batch_id_suffix == (
        "cmpl-prefill-0-cafebabe@51,cmpl-prefill-1-cafebabe@52"
    )
    assert first[0].metadata_template == {
        "r": (
            "cmpl-prefill-0-cafebabe",
            "cmpl-prefill-1-cafebabe",
        ),
        "s": (51, 52),
        "a": (0.125, 0.125),
        "t": ((11,), (22,)),
    }
