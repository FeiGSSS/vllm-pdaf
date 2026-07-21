from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CompilationConfig
from vllm.pap.model import cudagraph


class _ReleaseMessage:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def test_pap_cudagraph_role_is_explicit() -> None:
    assert cudagraph.pap_cudagraph_role({}) is None
    assert (
        cudagraph.pap_cudagraph_role(
            {
                "PAP_CUDAGRAPH_COMPATIBLE": "1",
                "PAP_CUDAGRAPH_ROLE": "projection",
            }
        )
        == "projection"
    )
    with pytest.raises(RuntimeError, match="PAP_CUDAGRAPH_ROLE"):
        cudagraph.pap_cudagraph_role(
            {
                "PAP_CUDAGRAPH_COMPATIBLE": "1",
                "PAP_CUDAGRAPH_ROLE": "decode",
            }
        )


def test_pap_model_hooks_are_disabled_for_normal_vllm() -> None:
    assert not cudagraph.pap_model_hooks_enabled({})
    assert cudagraph.pap_model_hooks_enabled({"PAP_MODEL_HOOKS": "1"})
    assert not cudagraph.pap_model_hooks_enabled({"PAP_MODEL_HOOKS": "0"})
    assert cudagraph.pap_model_hooks_enabled({"PAP_TOPOLOGY": "3pa1p"})


def test_projection_boundary_copies_output_and_releases_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _ReleaseMessage()

    class Adapter:
        def should_execute(self) -> bool:
            return True

        def execute(self, query: torch.Tensor, *_args, **_kwargs):
            return query + 1, [message]

    attention = SimpleNamespace(_pap_cudagraph_projection_adapter=Adapter())
    monkeypatch.setattr(cudagraph, "_pap_attention_layer", lambda _name: attention)
    monkeypatch.setattr(
        cudagraph.PAPModelForwardBatch,
        "current",
        classmethod(lambda _cls, _name: SimpleNamespace(enabled=True)),
    )
    query = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    output = torch.empty_like(query)

    cudagraph._pap_projection_attention_with_output_impl(
        query,
        query,
        query,
        output,
        None,
        "layer",
    )

    torch.testing.assert_close(output, query + 1)
    assert message.released


def test_projection_boundary_uses_shape_only_output_for_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = SimpleNamespace(_pap_cudagraph_projection_adapter=object())
    monkeypatch.setattr(cudagraph, "_pap_attention_layer", lambda _name: attention)
    monkeypatch.setattr(
        cudagraph.PAPModelForwardBatch,
        "current",
        classmethod(lambda _cls, _name: None),
    )
    value = torch.ones(2, 3)

    cudagraph._pap_projection_attention_with_output_impl(
        value,
        value,
        value,
        value,
        None,
        "layer",
    )

    assert torch.count_nonzero(value) == 0


def test_prefill_publisher_boundary_is_registered_and_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = []
    publisher = SimpleNamespace(publish=lambda attention: published.append(attention))
    attention = SimpleNamespace(_pap_cudagraph_prefill_publisher=publisher)
    monkeypatch.setattr(cudagraph, "_pap_attention_layer", lambda _name: attention)

    cudagraph._pap_publish_prefill_kv_impl(torch.ones(1), "layer")

    assert published == [attention]
    assert hasattr(torch.ops.vllm, "pap_projection_attention_with_output")
    assert hasattr(torch.ops.vllm, "pap_publish_prefill_kv")
    assert "vllm::pap_projection_attention_with_output" in (
        CompilationConfig._attention_ops
    )
    assert "vllm::pap_publish_prefill_kv" in CompilationConfig._attention_ops
