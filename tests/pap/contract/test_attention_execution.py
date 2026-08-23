# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.attention import attention as attention_module
from vllm.model_executor.layers.attention.execution import (
    register_attention_execution_factory,
    resolve_attention_execution,
)


def test_generic_attention_execution_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    layer = SimpleNamespace(
        execution_override=lambda *args, **kwargs: calls.append((*args, kwargs)),
        impl=SimpleNamespace(
            forward=lambda *_args, **_kwargs: pytest.fail(
                "the local Attention backend must be bypassed"
            )
        ),
    )
    kv_cache = object()
    metadata = object()
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda _name: (metadata, layer, kv_cache, None),
    )
    query = object()
    key = object()
    value = object()
    output = object()

    attention_module.unified_attention_with_output(
        query,  # type: ignore[arg-type]
        key,  # type: ignore[arg-type]
        value,  # type: ignore[arg-type]
        output,  # type: ignore[arg-type]
        "layer",
    )

    assert calls == [
        (
            layer,
            query,
            key,
            value,
            output,
            kv_cache,
            metadata,
            {"output_scale": None, "output_block_scale": None},
        )
    ]


def test_attention_execution_factory_is_model_independent() -> None:
    selected_attention = object()
    execution = object()

    def factory(attention, _vllm_config):
        return execution if attention is selected_attention else None

    register_attention_execution_factory("pap-test-model-independent", factory)

    assert resolve_attention_execution(selected_attention, object()) is execution
    assert resolve_attention_execution(object(), object()) is None
