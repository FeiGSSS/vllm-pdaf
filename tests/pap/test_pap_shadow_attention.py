import json
import os
from unittest.mock import patch

import torch

from vllm.pap.shadow_attention import (
    _attention_endpoint,
    _enabled,
    build_layer_event_payload,
    maybe_report_qkv_boundary,
)


def test_build_layer_event_payload_uses_forward_context_request_ids() -> None:
    query = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    key = torch.empty((1, 8, 128), dtype=torch.bfloat16)
    value = torch.empty((1, 8, 128), dtype=torch.bfloat16)

    payload = build_layer_event_payload(
        layer_name="model.layers.0.self_attn.attn",
        query=query,
        key=key,
        value=value,
        request_ids=["cmpl-abc-0-deadbeef"],
        num_scheduled_tokens=[1],
        num_reqs=1,
        num_actual_tokens=1,
        max_seq_len=9,
    )

    assert payload["request_id"] == "cmpl-abc-0-deadbeef"
    assert payload["query_shape"] == [1, 32, 128]
    assert payload["key_shape"] == [1, 8, 128]
    assert payload["value_shape"] == [1, 8, 128]
    assert payload["dtype"] == "torch.bfloat16"
    assert payload["device"] == "cpu"
    assert payload["is_decode"] is True
    assert payload["num_reqs"] == 1
    assert payload["num_actual_tokens"] == 1
    assert payload["max_seq_len"] == 9


def test_build_layer_event_payload_skips_warmup_request_ids() -> None:
    query = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    key = torch.empty((1, 8, 128), dtype=torch.bfloat16)
    value = torch.empty((1, 8, 128), dtype=torch.bfloat16)

    payload = build_layer_event_payload(
        layer_name="model.layers.0.self_attn.attn",
        query=query,
        key=key,
        value=value,
        request_ids=["req_0_warmup"],
        num_scheduled_tokens=[1],
        num_reqs=1,
        num_actual_tokens=1,
        max_seq_len=9,
    )

    assert payload is None


def test_maybe_report_qkv_boundary_posts_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PAP_SHADOW_ATTENTION", "1")
    monkeypatch.setenv("PAP_ATTENTION_ENDPOINT", "http://127.0.0.1:8300")
    query = torch.empty((1, 32, 128))
    key = torch.empty((1, 8, 128))
    value = torch.empty((1, 8, 128))

    with patch("vllm.pap.shadow_attention.urlopen") as urlopen:
        maybe_report_qkv_boundary(
            layer_name="model.layers.0.self_attn.attn",
            query=query,
            key=key,
            value=value,
            request_ids=["cmpl-abc-0-deadbeef"],
            num_scheduled_tokens=[1],
            num_reqs=1,
            num_actual_tokens=1,
            max_seq_len=9,
        )

    assert urlopen.call_count == 1
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:8300/v1/pap/attention/layer-event"
    assert request.get_method() == "POST"


def test_maybe_report_qkv_boundary_uses_forward_context_config() -> None:
    query = torch.empty((1, 32, 128))
    key = torch.empty((1, 8, 128))
    value = torch.empty((1, 8, 128))

    with patch("vllm.pap.shadow_attention.urlopen") as urlopen:
        maybe_report_qkv_boundary(
            layer_name="model.layers.0.self_attn.attn",
            query=query,
            key=key,
            value=value,
            request_ids=["cmpl-abc-0-deadbeef"],
            num_scheduled_tokens=[1],
            num_reqs=1,
            num_actual_tokens=1,
            max_seq_len=9,
            enabled=True,
            endpoint="http://pa0:8300",
        )

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://pa0:8300/v1/pap/attention/layer-event"


def test_maybe_report_qkv_boundary_is_disabled_by_default() -> None:
    os.environ.pop("PAP_SHADOW_ATTENTION", None)
    query = torch.empty((1, 32, 128))
    key = torch.empty((1, 8, 128))
    value = torch.empty((1, 8, 128))

    with patch("vllm.pap.shadow_attention.urlopen") as urlopen:
        maybe_report_qkv_boundary(
            layer_name="model.layers.0.self_attn.attn",
            query=query,
            key=key,
            value=value,
            request_ids=["cmpl-abc-0-deadbeef"],
            num_scheduled_tokens=[1],
            num_reqs=1,
            num_actual_tokens=1,
            max_seq_len=9,
        )

    urlopen.assert_not_called()


def test_shadow_config_can_come_from_current_vllm_config(monkeypatch) -> None:
    class FakeKVTransferConfig:
        kv_connector_extra_config = {
            "pap_shadow_attention": True,
            "pap_attention_endpoint": "http://pa0:8300",
        }

    class FakeVllmConfig:
        kv_transfer_config = FakeKVTransferConfig()

    monkeypatch.delenv("PAP_SHADOW_ATTENTION", raising=False)
    monkeypatch.delenv("PAP_ATTENTION_ENDPOINT", raising=False)
    with patch(
        "vllm.config.get_current_vllm_config_or_none",
        return_value=FakeVllmConfig(),
    ):
        assert _enabled() is True
        assert _attention_endpoint() == "http://pa0:8300"
        assert _enabled(False) is False
        assert _attention_endpoint("http://ctx:8300") == "http://ctx:8300"


def test_build_remote_attention_request_gathers_kv_cache() -> None:
    import torch

    from vllm.pap.remote_attention import deserialize_tensor
    from vllm.pap.shadow_attention import build_remote_attention_request

    query = torch.ones((1, 4, 2), dtype=torch.float32)
    kv_cache = torch.zeros((2, 2, 4, 2, 2), dtype=torch.float32)
    kv_cache[0, 0, 0] = 11
    kv_cache[1, 0, 0] = 12
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)

    payload = build_remote_attention_request(
        request_id="cmpl-abc-0-deadbeef",
        layer_name="model.layers.0.self_attn.attn",
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_len=1,
        num_kv_heads=2,
        scale=0.5,
        layout="NHD",
    )

    assert payload["request_id"] == "cmpl-abc-0-deadbeef"
    assert payload["scale"] == 0.5
    assert torch.equal(deserialize_tensor(payload["query"]), query.cpu())
    assert torch.equal(deserialize_tensor(payload["key"]), torch.full((1, 2, 2), 11.0))
    assert torch.equal(
        deserialize_tensor(payload["value"]), torch.full((1, 2, 2), 12.0)
    )


def test_compute_remote_attention_output_posts_and_deserializes() -> None:
    from vllm.pap.remote_attention import serialize_attention_result
    from vllm.pap.shadow_attention import compute_remote_attention_output

    query = torch.ones((1, 4, 2), dtype=torch.float32)
    kv_cache = torch.zeros((2, 2, 4, 2, 2), dtype=torch.float32)
    kv_cache[0, 0, 0] = 11
    kv_cache[1, 0, 0] = 12
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    expected = torch.full((1, 4, 2), 0.25, dtype=torch.float32)
    captured = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "request_id": "cmpl-abc-0-deadbeef",
                    "layer_name": "model.layers.0.self_attn.attn",
                    "output": serialize_attention_result(expected),
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    with patch("vllm.pap.shadow_attention.urlopen", side_effect=fake_urlopen):
        output = compute_remote_attention_output(
            request_id="cmpl-abc-0-deadbeef",
            layer_name="model.layers.0.self_attn.attn",
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_len=1,
            num_kv_heads=2,
            scale=0.5,
            layout="NHD",
            endpoint="http://pa0:8300",
            timeout=1.25,
        )

    assert captured["url"] == "http://pa0:8300/v1/pap/attention/compute"
    assert captured["body"]["request_id"] == "cmpl-abc-0-deadbeef"
    assert captured["timeout"] == 1.25
    assert torch.equal(output, expected)
