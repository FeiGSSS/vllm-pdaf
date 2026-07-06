# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3 import (
    _pap_decode_token_rows_for_indices,
    _pap_endpoint_for_tp_rank,
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
