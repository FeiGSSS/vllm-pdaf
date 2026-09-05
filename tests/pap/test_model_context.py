# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.pap.config import PAPConfigError, pap_model_hooks_enabled
from vllm.pap.model import context
from vllm.pap.model.projection import (
    PAPProjectionAttentionAdapter,
    PAPProjectionAttentionExecution,
)


class _CountedString:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0

    def __str__(self) -> str:
        self.calls += 1
        return self.value


@pytest.mark.parametrize("name", ["PAP_MODEL_HOOKS", "PAP_PROJECTION_KV_UNAWARE"])
def test_invalid_model_activation_is_not_silently_disabled(name):
    with pytest.raises(PAPConfigError, match=name):
        pap_model_hooks_enabled({name: "typo"})


def test_model_activation_uses_shared_boolean_values():
    assert pap_model_hooks_enabled({"PAP_MODEL_HOOKS": " yes "})
    assert not pap_model_hooks_enabled(
        {"PAP_MODEL_HOOKS": "false", "PAP_TOPOLOGY": "7pa1p"}
    )


def test_projection_debug_flag_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("PAP_DEBUG_DECISION", "typo")
    with pytest.raises(PAPConfigError, match="PAP_DEBUG_DECISION"):
        PAPProjectionAttentionAdapter("layers.0.attn", 32, 8, 128, 1.0)


def test_projection_execution_consumes_tensor_without_message_lifetime():
    query = torch.arange(6).reshape(1, 6)
    output = torch.empty((2, 3), dtype=query.dtype)
    adapter = SimpleNamespace(
        should_execute=lambda: True,
        execute=lambda q, k, v: q,
    )
    execution = PAPProjectionAttentionExecution(adapter)
    execution(None, query, query, query, output, None, None)
    assert torch.equal(output, query.reshape_as(output))


def test_forward_batch_base_is_normalized_once_per_model_forward(
    monkeypatch,
) -> None:
    request_id = _CountedString("request-0")
    layer_0_metadata = object()
    layer_1_metadata = object()
    forward_context = SimpleNamespace(
        additional_kwargs={
            "pap_enabled": True,
            "pap_request_ids": [request_id],
            "pap_num_scheduled_tokens": [1],
            "pap_num_reqs": 1,
            "pap_num_actual_tokens": 1,
        },
        attn_metadata={
            "layers.0.attn": layer_0_metadata,
            "layers.1.attn": layer_1_metadata,
        },
    )
    monkeypatch.setattr(context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(context, "get_forward_context", lambda: forward_context)

    layer_0 = context.PAPModelForwardBatch.current("layers.0.attn")
    layer_1 = context.PAPModelForwardBatch.current("layers.1.attn")

    assert layer_0 is not None
    assert layer_1 is not None
    assert layer_0.request_ids is layer_1.request_ids
    assert layer_0.num_scheduled_tokens is layer_1.num_scheduled_tokens
    assert layer_0.attention_metadata is layer_0_metadata
    assert layer_1.attention_metadata is layer_1_metadata
    assert request_id.calls == 1
