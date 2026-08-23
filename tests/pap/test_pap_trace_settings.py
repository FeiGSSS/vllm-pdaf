# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.pap.integration.settings import pap_offload_exec_trace_enabled


def test_offload_exec_trace_layer_filter() -> None:
    environ = {
        "PAP_OFFLOAD_EXEC_TRACE": "1",
        "PAP_OFFLOAD_EXEC_TRACE_LAYER": "18",
    }

    assert pap_offload_exec_trace_enabled(18, environ)
    assert pap_offload_exec_trace_enabled(
        "model.layers.18.self_attn.attn",
        environ,
    )
    assert not pap_offload_exec_trace_enabled(17, environ)
    assert not pap_offload_exec_trace_enabled(
        "model.layers.19.self_attn.attn",
        environ,
    )


def test_offload_exec_trace_layer_filter_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-negative layer index"):
        pap_offload_exec_trace_enabled(
            18,
            {
                "PAP_OFFLOAD_EXEC_TRACE": "1",
                "PAP_OFFLOAD_EXEC_TRACE_LAYER": "middle",
            },
        )
