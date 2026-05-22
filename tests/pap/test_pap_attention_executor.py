from examples.pap.pap_attention_executor import (
    PAPAttentionRegistration,
    PAPAttentionRegistry,
)


def test_attention_registry_records_prefill_kv_handle() -> None:
    registry = PAPAttentionRegistry()
    registration = PAPAttentionRegistration(
        request_id="req-1",
        conversation_id="conv-1",
        prefill_endpoint="http://localhost:8100",
        kv_transfer_params={
            "remote_engine_id": "prefill-0",
            "remote_block_ids": [[1, 2, 3]],
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
        },
        prefix_len=17,
    )

    snapshot = registry.register_prefill_kv(registration)

    assert snapshot.request_id == "req-1"
    assert snapshot.conversation_id == "conv-1"
    assert snapshot.prefill_endpoint == "http://localhost:8100"
    assert snapshot.kv_transfer_params["remote_block_ids"] == [[1, 2, 3]]
    assert snapshot.prefix_len == 17
    assert snapshot.role == "attention"


def test_attention_registry_returns_registered_snapshot_copy() -> None:
    registry = PAPAttentionRegistry()
    registration = PAPAttentionRegistration(
        request_id="req-2",
        conversation_id="conv-2",
        prefill_endpoint="http://localhost:8100",
        kv_transfer_params={"remote_engine_id": "prefill-0"},
        prefix_len=None,
    )

    registry.register_prefill_kv(registration)
    snapshot = registry.get_session("req-2")

    assert snapshot is not None
    assert snapshot.request_id == "req-2"
    assert snapshot.kv_transfer_params == {"remote_engine_id": "prefill-0"}

    snapshot.kv_transfer_params["remote_engine_id"] = "mutated"
    stored = registry.get_session("req-2")
    assert stored is not None
    assert stored.kv_transfer_params == {"remote_engine_id": "prefill-0"}


def test_attention_registry_records_layer_event_without_payload_tensors() -> None:
    registry = PAPAttentionRegistry()
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-4",
            conversation_id="conv-4",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={"remote_engine_id": "prefill-0"},
        )
    )

    event = registry.record_layer_event(
        request_id="cmpl-req-4-0-deadbeef",
        layer_name="model.layers.0.self_attn.attn",
        query_shape=[1, 32, 128],
        key_shape=[1, 8, 128],
        value_shape=[1, 8, 128],
        dtype="torch.bfloat16",
        device="cuda:0",
        is_decode=True,
        num_reqs=1,
        num_actual_tokens=1,
        max_seq_len=9,
    )

    assert event.request_id == "cmpl-req-4-0-deadbeef"
    assert event.session_request_id == "req-4"
    assert event.layer_name == "model.layers.0.self_attn.attn"
    assert event.query_shape == [1, 32, 128]
    assert event.is_decode is True

    stored = registry.get_layer_events("req-4")
    assert len(stored) == 1
    assert stored[0].max_seq_len == 9


def test_attention_registry_matches_wrapped_vllm_request_ids() -> None:
    registry = PAPAttentionRegistry()
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="abcd",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
        )
    )

    assert registry.resolve_session_request_id("cmpl-abcd-0-deadbeef") == "abcd"
    assert registry.resolve_session_request_id("chatcmpl-abcd-1-deadbeef") == "abcd"
    assert registry.resolve_session_request_id("abcd") == "abcd"
    assert registry.resolve_session_request_id("unknown") is None


def test_attention_executor_compute_route_returns_attention_output() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = TestClient(app)
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    response = client.post(
        "/v1/pap/attention/compute",
        json={
            "request_id": "cmpl-unit-0-deadbeef",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(query),
            "key": serialize_tensor(key),
            "value": serialize_tensor(value),
            "scale": 1.0,
        },
    )

    assert response.status_code == 200
    output = deserialize_attention_result(response.json()["output"])
    assert output.shape == (1, 2, 2)
    assert output.dtype == torch.float32
