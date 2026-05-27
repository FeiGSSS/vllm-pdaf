# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.models.qwen3 import _pap_endpoint_for_tp_rank


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
