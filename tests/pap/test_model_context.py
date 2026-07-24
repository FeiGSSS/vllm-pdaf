from types import SimpleNamespace

from vllm.pap.model import context


class _CountedString:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0

    def __str__(self) -> str:
        self.calls += 1
        return self.value


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
