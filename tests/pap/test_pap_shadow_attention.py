import os
from unittest.mock import patch

import torch

from vllm.pap.shadow_attention import (
    _attention_endpoint,
    _enabled,
    build_layer_event_payload,
    maybe_report_qkv_boundary,
    select_attention_endpoint_for_request,
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


def test_select_attention_endpoint_for_request_prefers_per_request_route() -> None:
    endpoint = select_attention_endpoint_for_request(
        "cmpl-a-0-deadbeef",
        default_endpoint="http://fallback:8300",
        endpoint_by_request={
            "cmpl-a-0-deadbeef": "http://pa0:8300",
            "cmpl-b-0-deadbeef": "http://pa1:8301",
        },
    )

    assert endpoint == "http://pa0:8300"


def test_select_attention_endpoint_for_request_falls_back_to_static_endpoint() -> None:
    endpoint = select_attention_endpoint_for_request(
        "cmpl-missing-0-deadbeef",
        default_endpoint="http://fallback:8300",
        endpoint_by_request={"cmpl-a-0-deadbeef": "http://pa0:8300"},
    )

    assert endpoint == "http://fallback:8300"


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

    def fake_post_json(*, endpoint, path, payload, timeout):
        captured["endpoint"] = endpoint
        captured["path"] = path
        captured["body"] = payload
        captured["timeout"] = timeout
        return {"output": serialize_attention_result(expected)}

    with patch("vllm.pap.shadow_attention._post_json", side_effect=fake_post_json):
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

    assert captured["endpoint"] == "http://pa0:8300"
    assert captured["path"] == "/v1/pap/attention/compute"
    assert captured["body"]["request_id"] == "cmpl-abc-0-deadbeef"
    assert captured["timeout"] == 1.25
    assert torch.equal(output, expected)


def test_compute_stateful_remote_attention_output_posts_current_qkv_only(
    monkeypatch,
) -> None:
    import torch

    from vllm.pap.remote_attention import serialize_tensor
    from vllm.pap.shadow_attention import compute_stateful_remote_attention_output

    calls = []
    output_payload = serialize_tensor(torch.tensor([[[3.0, 5.0]]]))
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_BINARY", "0")

    def fake_post_json(*, endpoint, path, payload, timeout):
        calls.append((endpoint, path, payload, timeout))
        return {"output": output_payload}

    with patch("vllm.pap.shadow_attention._post_json", side_effect=fake_post_json):
        output = compute_stateful_remote_attention_output(
            request_id="cmpl-stateful",
            layer_name="model.layers.0.self_attn.attn",
            query=torch.tensor([[[1.0, 0.0]]]),
            key=torch.tensor([[[1.0, 0.0]]]),
            value=torch.tensor([[[2.0, 0.0]]]),
            scale=1.0,
            block_id=4,
            slot=64,
            seq_len=1,
            endpoint="http://attention:8300",
        )

    assert torch.equal(output, torch.tensor([[[3.0, 5.0]]]))
    assert len(calls) == 1
    endpoint, path, payload, timeout = calls[0]
    assert endpoint == "http://attention:8300"
    assert path == "/v1/pap/attention/append-and-compute"
    assert timeout == 5.0
    assert set(payload) == {
        "request_id",
        "layer_name",
        "query",
        "key",
        "value",
        "scale",
        "block_id",
        "slot",
        "seq_len",
    }
    assert payload["request_id"] == "cmpl-stateful"
    assert payload["block_id"] == 4
    assert payload["slot"] == 64
    assert payload["seq_len"] == 1


def test_compute_stateful_remote_attention_output_uses_binary_path(
    monkeypatch,
) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import compute_stateful_remote_attention_output

    captured = {}
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_BINARY", "1")

    def fake_post_bytes(*, endpoint, path, payload, timeout):
        metadata, tensors = deserialize_tensor_bundle(payload)
        captured["endpoint"] = endpoint
        captured["path"] = path
        captured["metadata"] = metadata
        captured["tensors"] = tensors
        captured["timeout"] = timeout
        return serialize_tensor_bundle(
            {"request_id": metadata["request_id"]},
            {"output": torch.tensor([[[3.0, 5.0]]])},
        )

    with patch("vllm.pap.shadow_attention._post_bytes", side_effect=fake_post_bytes):
        output = compute_stateful_remote_attention_output(
            request_id="cmpl-stateful",
            layer_name="model.layers.0.self_attn.attn",
            query=torch.tensor([[[1.0, 0.0]]]),
            key=torch.tensor([[[1.0, 0.0]]]),
            value=torch.tensor([[[2.0, 0.0]]]),
            scale=1.0,
            block_id=4,
            slot=64,
            seq_len=1,
            endpoint="http://attention:8300",
        )

    assert torch.equal(output, torch.tensor([[[3.0, 5.0]]]))
    assert captured["endpoint"] == "http://attention:8300"
    assert captured["path"] == "/v1/pap/attention/append-and-compute-binary"
    assert captured["metadata"]["request_id"] == "cmpl-stateful"
    assert captured["metadata"]["block_id"] == 4
    assert torch.equal(captured["tensors"]["query"], torch.tensor([[[1.0, 0.0]]]))


def test_compute_stateful_remote_attention_outputs_batch_uses_binary_path(
    monkeypatch,
) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import (
        compute_stateful_remote_attention_outputs_batch,
    )

    captured = {}
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_BINARY", "1")

    def fake_post_bytes(*, endpoint, path, payload, timeout):
        metadata, tensors = deserialize_tensor_bundle(payload)
        captured["endpoint"] = endpoint
        captured["path"] = path
        captured["metadata"] = metadata
        captured["tensors"] = tensors
        captured["timeout"] = timeout
        return serialize_tensor_bundle(
            {
                "items": [
                    {"request_id": metadata["items"][0]["request_id"]},
                    {"request_id": metadata["items"][1]["request_id"]},
                ]
            },
            {
                "output_0": torch.tensor([[[3.0, 5.0]]]),
                "output_1": torch.tensor([[[7.0, 11.0]]]),
            },
        )

    calls = [
        {
            "request_id": "cmpl-stateful-0",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": torch.tensor([[[1.0, 0.0]]]),
            "key": torch.tensor([[[1.0, 0.0]]]),
            "value": torch.tensor([[[2.0, 0.0]]]),
            "scale": 1.0,
            "block_id": 4,
            "slot": 64,
            "seq_len": 1,
        },
        {
            "request_id": "cmpl-stateful-1",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": torch.tensor([[[0.0, 1.0]]]),
            "key": torch.tensor([[[0.0, 1.0]]]),
            "value": torch.tensor([[[0.0, 4.0]]]),
            "scale": 1.0,
            "block_id": 5,
            "slot": 80,
            "seq_len": 1,
        },
    ]

    with patch("vllm.pap.shadow_attention._post_bytes", side_effect=fake_post_bytes):
        outputs = compute_stateful_remote_attention_outputs_batch(
            calls=calls,
            endpoint="http://attention:8300",
        )

    assert torch.equal(outputs[0], torch.tensor([[[3.0, 5.0]]]))
    assert torch.equal(outputs[1], torch.tensor([[[7.0, 11.0]]]))
    assert captured["endpoint"] == "http://attention:8300"
    assert captured["path"] == "/v1/pap/attention/append-and-compute-batch-binary"
    assert captured["metadata"]["items"][0]["request_id"] == "cmpl-stateful-0"
    assert captured["metadata"]["items"][1]["block_id"] == 5
    assert torch.equal(
        captured["tensors"]["qkv_1"],
        torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 4.0]]),
    )


def test_compute_stateful_remote_attention_outputs_batch_uses_tcp_transport(
    monkeypatch,
) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import (
        compute_stateful_remote_attention_outputs_batch,
    )

    captured = {}
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_BINARY", "1")
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_TRANSPORT", "tcp")

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        metadata, tensors = deserialize_tensor_bundle(payload)
        captured["endpoint"] = endpoint
        captured["metadata"] = metadata
        captured["tensors"] = tensors
        captured["timeout"] = timeout
        return serialize_tensor_bundle(
            {"items": [{"request_id": metadata["items"][0]["request_id"]}]},
            {"output_0": torch.tensor([[[13.0, 17.0]]])},
        )

    calls = [
        {
            "request_id": "cmpl-stateful-0",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": torch.tensor([[[1.0, 0.0]]]),
            "key": torch.tensor([[[1.0, 0.0]]]),
            "value": torch.tensor([[[2.0, 0.0]]]),
            "scale": 1.0,
            "block_id": 4,
            "slot": 64,
            "seq_len": 1,
        },
    ]

    with patch(
        "vllm.pap.shadow_attention._post_bytes_tcp",
        side_effect=fake_post_bytes_tcp,
    ):
        outputs = compute_stateful_remote_attention_outputs_batch(
            calls=calls,
            endpoint="http://attention:8300",
            tcp_endpoint="tcp://attention:9300",
            timeout=1.25,
        )

    assert torch.equal(outputs[0], torch.tensor([[[13.0, 17.0]]]))
    assert captured["endpoint"] == "tcp://attention:9300"
    assert captured["timeout"] == 1.25
    assert captured["metadata"]["items"][0]["request_id"] == "cmpl-stateful-0"
    assert torch.equal(
        captured["tensors"]["qkv_0"],
        torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
    )


def test_compute_stateful_remote_attention_outputs_batch_uses_compact_tcp(
    monkeypatch,
) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_compact_attention_batch,
        serialize_compact_attention_response,
    )
    from vllm.pap.shadow_attention import (
        compute_stateful_remote_attention_outputs_batch,
    )

    captured = {}
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_BINARY", "1")
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_TRANSPORT", "tcp")
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_COMPACT_TCP", "1")

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        items, qkv_tensors = deserialize_compact_attention_batch(payload)
        captured["endpoint"] = endpoint
        captured["items"] = items
        captured["qkv_tensors"] = qkv_tensors
        captured["timeout"] = timeout
        return serialize_compact_attention_response(
            [torch.tensor([[[13.0, 17.0]]])]
        )

    calls = [
        {
            "request_id": "cmpl-stateful-0",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": torch.tensor([[[1.0, 0.0]]]),
            "key": torch.tensor([[[1.0, 0.0]]]),
            "value": torch.tensor([[[2.0, 0.0]]]),
            "scale": 1.0,
            "block_id": 4,
            "slot": 64,
            "seq_len": 1,
        },
    ]

    with patch(
        "vllm.pap.shadow_attention._post_bytes_tcp",
        side_effect=fake_post_bytes_tcp,
    ):
        outputs = compute_stateful_remote_attention_outputs_batch(
            calls=calls,
            endpoint="http://attention:8300",
            tcp_endpoint="tcp://attention:9300",
            timeout=1.25,
        )

    assert torch.equal(outputs[0], torch.tensor([[[13.0, 17.0]]]))
    assert captured["endpoint"] == "tcp://attention:9300"
    assert captured["timeout"] == 1.25
    assert captured["items"][0]["request_id"] == "cmpl-stateful-0"
    assert captured["items"][0]["num_heads"] == 1
    assert torch.equal(
        captured["qkv_tensors"][0],
        torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
    )


def test_import_prefill_kv_posts_prompt_tensors() -> None:
    import torch

    from vllm.pap.remote_attention import deserialize_tensor
    from vllm.pap.shadow_attention import import_prefill_kv

    calls = []

    def fake_post_json(*, endpoint, path, payload, timeout):
        calls.append((endpoint, path, payload, timeout))
        return {"seq_len": 2}

    with (
        patch("vllm.pap.shadow_attention._post_json", side_effect=fake_post_json),
        patch.dict(
            "os.environ",
            {"PAP_REMOTE_ATTENTION_BINARY": "0"},
        ),
    ):
        seq_len = import_prefill_kv(
            request_id="cmpl-prefix",
            layer_name="model.layers.0.self_attn.attn",
            key=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            value=torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]]),
            seq_len=2,
            block_ids=[4],
            endpoint="http://attention:8300",
            timeout=1.25,
        )

    assert seq_len == 2
    assert len(calls) == 1
    endpoint, path, payload, timeout = calls[0]
    assert endpoint == "http://attention:8300"
    assert path == "/v1/pap/attention/import-prefill-kv"
    assert torch.equal(
        deserialize_tensor(payload["key"]),
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
    )


def test_import_prefill_kv_uses_tcp_binary_when_configured(monkeypatch) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import import_prefill_kv

    captured = {}

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        captured["endpoint"] = endpoint
        captured["timeout"] = timeout
        metadata, tensors = deserialize_tensor_bundle(payload)
        captured["metadata"] = metadata
        captured["tensors"] = tensors
        return serialize_tensor_bundle({"seq_len": metadata["seq_len"]}, {})

    def fail_post_json(**_kwargs):
        raise AssertionError("HTTP JSON path should not be used")

    monkeypatch.setenv("PAP_REMOTE_ATTENTION_TRANSPORT", "tcp")
    with (
        patch(
            "vllm.pap.shadow_attention._post_bytes_tcp",
            side_effect=fake_post_bytes_tcp,
        ),
        patch("vllm.pap.shadow_attention._post_json", side_effect=fail_post_json),
    ):
        seq_len = import_prefill_kv(
            request_id="cmpl-prefix",
            layer_name="model.layers.0.self_attn.attn",
            key=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            value=torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]]),
            seq_len=2,
            block_ids=[4],
            endpoint="http://attention:8300",
            tcp_endpoint="tcp://attention:9300",
            timeout=1.25,
        )

    assert seq_len == 2
    assert captured["endpoint"] == "tcp://attention:9300"
    assert captured["timeout"] == 1.25
    assert captured["metadata"] == {
        "command": "import_prefill_kv",
        "request_id": "cmpl-prefix",
        "layer_name": "model.layers.0.self_attn.attn",
        "seq_len": 2,
        "block_ids": [4],
    }
    assert torch.equal(
        captured["tensors"]["key"],
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
    )


def test_trigger_offload_exec_attention_posts_control_only() -> None:
    from unittest.mock import patch

    from vllm.pap.shadow_attention import trigger_offload_exec_attention

    captured = {}

    def fake_post_json(*, endpoint, path, payload, timeout):
        captured.update(
            {
                "endpoint": endpoint,
                "path": path,
                "payload": payload,
                "timeout": timeout,
            }
        )
        return {"ok": True}

    with patch("vllm.pap.shadow_attention._post_json", side_effect=fake_post_json):
        trigger_offload_exec_attention(
            endpoint="http://127.0.0.1:8300",
            request_id="req-1",
            layer_name="layer0",
            step=9,
            scale=0.5,
            remote_address="127.0.0.1:11300",
            timeout=1.25,
        )

    assert captured["endpoint"] == "http://127.0.0.1:8300"
    assert captured["path"] == "/v1/pap/attention/offload-exec"
    assert captured["payload"] == {
        "request_id": "req-1",
        "layer_name": "layer0",
        "step": 9,
        "scale": 0.5,
        "remote_address": "127.0.0.1:11300",
    }
    assert captured["timeout"] == 1.25


def test_trigger_offload_exec_attention_uses_tcp_binary_when_configured(
    monkeypatch,
) -> None:
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import trigger_offload_exec_attention

    captured = {}

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        captured["endpoint"] = endpoint
        captured["timeout"] = timeout
        metadata, tensors = deserialize_tensor_bundle(payload)
        captured["metadata"] = metadata
        captured["tensors"] = tensors
        return serialize_tensor_bundle({"ok": True}, {})

    def fail_post_json(**_kwargs):
        raise AssertionError("HTTP JSON path should not be used")

    monkeypatch.setenv("PAP_REMOTE_ATTENTION_TRANSPORT", "tcp")
    with (
        patch(
            "vllm.pap.shadow_attention._post_bytes_tcp",
            side_effect=fake_post_bytes_tcp,
        ),
        patch("vllm.pap.shadow_attention._post_json", side_effect=fail_post_json),
    ):
        trigger_offload_exec_attention(
            endpoint="http://127.0.0.1:8300",
            tcp_endpoint="tcp://127.0.0.1:9300",
            request_id="req-1",
            layer_name="layer0",
            step=9,
            scale=0.5,
            remote_address="127.0.0.1:11300",
            timeout=1.25,
        )

    assert captured["endpoint"] == "tcp://127.0.0.1:9300"
    assert captured["timeout"] == 1.25
    assert captured["tensors"] == {}
    assert captured["metadata"] == {
        "command": "offload_exec",
        "request_id": "req-1",
        "layer_name": "layer0",
        "step": 9,
        "scale": 0.5,
        "remote_address": "127.0.0.1:11300",
    }


def test_import_prefill_kv_from_paged_cache_posts_blocks() -> None:
    import torch

    from vllm.pap.remote_attention import deserialize_tensor
    from vllm.pap.shadow_attention import import_prefill_kv_from_paged_cache

    captured = {}

    def fake_post_json(*, endpoint, path, payload, timeout):
        captured["body"] = payload
        captured["timeout"] = timeout
        return {"seq_len": 5}

    kv_cache = torch.zeros((2, 8, 4, 2, 2), dtype=torch.float32)
    for block in (4, 7):
        for offset in range(4):
            kv_cache[0, block, offset] = block * 100 + offset * 10 + 1
            kv_cache[1, block, offset] = block * 100 + offset * 10 + 2

    with (
        patch("vllm.pap.shadow_attention._post_json", side_effect=fake_post_json),
        patch.dict(
            "os.environ",
            {"PAP_REMOTE_ATTENTION_BINARY": "0"},
        ),
    ):
        seq_len = import_prefill_kv_from_paged_cache(
            request_id="cmpl-prefix",
            layer_name="model.layers.0.self_attn.attn",
            kv_cache=kv_cache,
            block_table=torch.tensor([[4, 7]], dtype=torch.int32),
            seq_len=5,
            block_size=4,
            num_kv_heads=2,
            layout="NHD",
            endpoint="http://attention:8300",
        )

    assert seq_len == 5
    assert captured["timeout"] == 5.0
    assert captured["body"]["block_ids"] == [4, 7]
    assert torch.equal(
        deserialize_tensor(captured["body"]["key"])[4],
        torch.full((2, 2), 701.0),
    )


def test_compute_stateful_remote_attention_output_omits_descriptor_when_absent(
    monkeypatch,
) -> None:
    import torch

    from vllm.pap.remote_attention import serialize_tensor
    from vllm.pap.shadow_attention import compute_stateful_remote_attention_output

    captured = {}
    output_payload = serialize_tensor(torch.tensor([[[3.0, 5.0]]]))
    monkeypatch.setenv("PAP_REMOTE_ATTENTION_BINARY", "0")

    def fake_post_json(*, endpoint, path, payload, timeout):
        captured["body"] = payload
        return {"output": output_payload}

    with patch("vllm.pap.shadow_attention._post_json", side_effect=fake_post_json):
        compute_stateful_remote_attention_output(
            request_id="cmpl-stateful",
            layer_name="model.layers.0.self_attn.attn",
            query=torch.tensor([[[1.0, 0.0]]]),
            key=torch.tensor([[[1.0, 0.0]]]),
            value=torch.tensor([[[2.0, 0.0]]]),
            scale=1.0,
            endpoint="http://attention:8300",
        )

    assert "block_id" not in captured["body"]
    assert "slot" not in captured["body"]
    assert "seq_len" not in captured["body"]
