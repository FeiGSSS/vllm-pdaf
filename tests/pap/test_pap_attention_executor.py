from pathlib import Path
from threading import Thread

from examples.pap.pap_attention_executor import (
    PAPAttentionRegistration,
    PAPAttentionRegistry,
    compute_binary_attention_response,
    compute_offload_exec_output,
    create_app,
    maybe_start_offload_exec_transport,
    run_offload_exec_once,
)
from vllm.pap.data_plane import (
    PAPCudaIPCTensorHandle,
    PAPOffloadExecDescriptor,
    PAPOffloadKVIPCDescriptor,
)

ROOT = Path(__file__).resolve().parents[2]


def test_attention_executor_declares_offload_exec_zmq_port() -> None:
    text = (ROOT / "examples" / "pap" / "pap_attention_executor.py").read_text()

    assert "--offload-exec-zmq-port" in text
    assert "OFFLOAD_EXEC NCCL/P2P data plane listening" in text


def test_attention_executor_skips_offload_exec_when_zmq_port_is_none(
    monkeypatch,
) -> None:
    app = create_app()
    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=None)
    assert app.state.offload_exec_transport is None


def test_attention_executor_starts_offload_exec_transport(monkeypatch) -> None:
    app = create_app()
    fake_transport = object()

    def fake_build_transport(**kwargs):
        fake_build_transport.kwargs = kwargs
        return fake_transport

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "2")
    monkeypatch.setattr(
        "examples.pap.pap_attention_executor.build_p2p_nccl_offload_exec_transport",
        fake_build_transport,
    )

    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=10300)

    assert app.state.offload_exec_transport is fake_transport
    assert fake_build_transport.kwargs == {
        "local_rank": 2,
        "kv_port": 10300,
        "hostname": "127.0.0.1",
    }


def test_compute_offload_exec_output_from_packed_qkv(monkeypatch) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_Q_SIZE", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_KV_SIZE", "2")

    output = compute_offload_exec_output(
        registry=registry,
        request_id="req-offload",
        layer_name="layer0",
        qkv=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
        scale=1.0,
        step=1,
    )

    assert output.shape == (1, 2)
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_compute_offload_exec_output_uses_step_block_descriptor(monkeypatch) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
            block_size=16,
            prefix_len=497,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    registry.import_prefill_kv(
        request_id="req-offload",
        layer_name="layer0",
        key=torch.zeros(497, 1, 2),
        value=torch.zeros(497, 1, 2),
        seq_len=497,
        block_ids=list(range(32)),
    )

    output = compute_offload_exec_output(
        registry=registry,
        request_id="req-offload",
        layer_name="layer0",
        qkv=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
        scale=1.0,
        step=498,
    )

    session = registry.get_session("req-offload")
    assert output.shape == (1, 2)
    assert session is not None
    assert session.block_ids[-1] == 31
    assert session.decode_seq_lens["layer0"] == 498


def test_run_offload_exec_once_receives_qkv_and_sends_output(monkeypatch) -> None:
    import torch

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert remote_address == "127.0.0.1:11300"
            assert descriptor.request_id == "req-offload"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    descriptor = PAPOffloadExecDescriptor(
        request_id="req-offload",
        layer_name="layer0",
        step=1,
        scale=1.0,
    )

    run_offload_exec_once(
        registry=registry,
        transport=transport,
        remote_address="127.0.0.1:11300",
        descriptor=descriptor,
    )

    assert len(transport.sent) == 1
    _, output, remote_address = transport.sent[0]
    assert remote_address == "127.0.0.1:11300"
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_run_offload_exec_once_resolves_wrapped_request_id(monkeypatch) -> None:
    import torch

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "cmpl-req-offload-0-deadbeef"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    descriptor = PAPOffloadExecDescriptor(
        request_id="cmpl-req-offload-0-deadbeef",
        layer_name="layer0",
        step=1,
        scale=1.0,
    )

    run_offload_exec_once(
        registry=registry,
        transport=transport,
        remote_address="127.0.0.1:11300",
        descriptor=descriptor,
    )

    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_offload_exec_endpoint_triggers_transport(monkeypatch) -> None:
    import torch
    from fastapi.testclient import TestClient

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "req-offload"
            assert descriptor.step == 1
            assert remote_address == "127.0.0.1:11300"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    app = create_app(registry)
    transport = FakeTransport()
    app.state.offload_exec_transport = transport
    client = TestClient(app)

    response = client.post(
        "/v1/pap/attention/offload-exec",
        json={
            "request_id": "req-offload",
            "layer_name": "layer0",
            "step": 1,
            "scale": 1.0,
            "remote_address": "127.0.0.1:11300",
        },
    )

    assert response.status_code == 200
    assert response.json()["remote_address"] == "127.0.0.1:11300"
    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_offload_exec_binary_command_triggers_transport(monkeypatch) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "req-offload"
            assert descriptor.step == 1
            assert remote_address == "127.0.0.1:11300"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    class TrackingLock:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, exc_type, exc, traceback):
            return False

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    lock = TrackingLock()

    response = compute_binary_attention_response(
        registry,
        serialize_tensor_bundle(
            {
                "command": "offload_exec",
                "request_id": "req-offload",
                "layer_name": "layer0",
                "step": 1,
                "scale": 1.0,
                "remote_address": "127.0.0.1:11300",
            },
            {},
        ),
        offload_exec_transport=transport,
        offload_exec_lock=lock,
    )

    metadata, tensors = deserialize_tensor_bundle(response)
    assert metadata["request_id"] == "req-offload"
    assert tensors == {}
    assert lock.entered
    assert len(transport.sent) == 1


def test_offload_exec_compact_command_triggers_transport(monkeypatch) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_compact_offload_exec_ack,
        serialize_compact_offload_exec_command,
    )

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "req-offload"
            assert descriptor.step == 1
            assert remote_address == "127.0.0.1:11300"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    class TrackingLock:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, exc_type, exc, traceback):
            return False

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    lock = TrackingLock()

    response = compute_binary_attention_response(
        registry,
        serialize_compact_offload_exec_command(
            request_id="req-offload",
            layer_name="layer0",
            step=1,
            scale=1.0,
            remote_address="127.0.0.1:11300",
        ),
        offload_exec_transport=transport,
        offload_exec_lock=lock,
    )

    deserialize_compact_offload_exec_ack(response)
    assert lock.entered
    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


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
    assert snapshot.prefill_kv_handle == "req-1"
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


def test_attention_registry_can_pin_state_to_configured_device() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-device",
            conversation_id="conv-device",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
            prefix_len=1,
        )
    )

    registry.import_prefill_kv(
        request_id="req-device",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[1.0, 0.0]]]),
        value=torch.tensor([[[2.0, 0.0]]]),
        seq_len=1,
    )
    segments, seq_len = registry.append_decode_kv(
        request_id="req-device",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[0.0, 1.0]]]),
        value=torch.tensor([[[0.0, 4.0]]]),
    )

    assert seq_len == 2
    assert registry.storage_device.type == "cpu"
    assert all(
        tensor.device.type == "cpu" for segment in segments for tensor in segment
    )


def test_attention_registry_waits_for_layer_prefill_before_decode(monkeypatch) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-wait",
            conversation_id="conv-wait",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
        )
    )
    monkeypatch.setenv("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "2.0")

    result = {}

    def append_decode() -> None:
        segments, seq_len = registry.append_decode_kv(
            request_id="req-wait",
            layer_name="model.layers.0.self_attn.attn",
            key=torch.tensor([[[0.0, 1.0]]]),
            value=torch.tensor([[[0.0, 4.0]]]),
            block_id=0,
            slot=1,
            seq_len=2,
        )
        result["seq_len"] = seq_len
        result["segments"] = segments

    thread = Thread(target=append_decode)
    thread.start()
    registry.import_prefill_kv(
        request_id="req-wait",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[1.0, 0.0]]]),
        value=torch.tensor([[[2.0, 0.0]]]),
        seq_len=1,
    )
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result["seq_len"] == 2
    assert len(result["segments"]) == 2


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


def test_attention_executor_append_and_compute_keeps_stateful_kv() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-stateful",
            "conversation_id": "conv-stateful",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 0,
        },
    )

    first = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-stateful",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
        },
    )
    assert first.status_code == 200
    first_output = deserialize_attention_result(first.json()["output"])
    assert torch.allclose(first_output, torch.tensor([[[2.0, 0.0]]]))

    second = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-stateful",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "key": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[0.0, 4.0]]])),
            "scale": 1.0,
        },
    )
    assert second.status_code == 200
    second_output = deserialize_attention_result(second.json()["output"])

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(second_output, expected)

    session = client.get("/v1/pap/attention/sessions/req-stateful").json()
    assert session["decode_seq_lens"] == {"model.layers.0.self_attn.attn": 2}


def test_attention_executor_binary_append_and_compute_keeps_stateful_kv() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-binary-stateful",
            "conversation_id": "conv-binary-stateful",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute-binary",
        content=serialize_tensor_bundle(
            {
                "request_id": "req-binary-stateful",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
            },
            {
                "query": torch.tensor([[[1.0, 0.0]]]),
                "key": torch.tensor([[[1.0, 0.0]]]),
                "value": torch.tensor([[[2.0, 0.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(response.content)
    assert metadata["request_id"] == "req-binary-stateful"
    assert metadata["seq_len"] == 1
    assert torch.allclose(tensors["output"], torch.tensor([[[2.0, 0.0]]]))


def test_attention_executor_batch_binary_append_and_compute_keeps_stateful_kv() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    app = create_app()
    client = TestClient(app)
    for request_id in ("req-batch-0", "req-batch-1"):
        client.post(
            "/v1/pap/attention/register",
            json={
                "request_id": request_id,
                "conversation_id": "conv-batch",
                "prefill_endpoint": "http://localhost:8100",
                "kv_transfer_params": {},
            },
        )

    response = client.post(
        "/v1/pap/attention/append-and-compute-batch-binary",
        content=serialize_tensor_bundle(
            {
                "items": [
                    {
                        "request_id": "req-batch-0",
                        "layer_name": "model.layers.0.self_attn.attn",
                        "scale": 1.0,
                    },
                    {
                        "request_id": "req-batch-1",
                        "layer_name": "model.layers.0.self_attn.attn",
                        "scale": 1.0,
                    },
                ]
            },
            {
                "query_0": torch.tensor([[[1.0, 0.0]]]),
                "key_0": torch.tensor([[[1.0, 0.0]]]),
                "value_0": torch.tensor([[[2.0, 0.0]]]),
                "query_1": torch.tensor([[[0.0, 1.0]]]),
                "key_1": torch.tensor([[[0.0, 1.0]]]),
                "value_1": torch.tensor([[[0.0, 4.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(response.content)
    assert [item["request_id"] for item in metadata["items"]] == [
        "req-batch-0",
        "req-batch-1",
    ]
    assert torch.allclose(tensors["output_0"], torch.tensor([[[2.0, 0.0]]]))
    assert torch.allclose(tensors["output_1"], torch.tensor([[[0.0, 4.0]]]))


def test_attention_executor_batch_binary_accepts_packed_qkv(monkeypatch) -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_Q_SIZE", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_KV_SIZE", "2")
    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-packed",
            "conversation_id": "conv-packed",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute-batch-binary",
        content=serialize_tensor_bundle(
            {
                "items": [
                    {
                        "request_id": "req-packed",
                        "layer_name": "model.layers.0.self_attn.attn",
                        "scale": 1.0,
                    },
                ]
            },
            {"qkv_0": torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    _metadata, tensors = deserialize_tensor_bundle(response.content)
    assert torch.allclose(tensors["output_0"], torch.tensor([[[2.0, 0.0]]]))


def test_attention_executor_compact_tcp_payload_keeps_stateful_kv() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
        compute_binary_attention_response,
    )
    from vllm.pap.remote_attention import (
        deserialize_compact_attention_response,
        serialize_compact_attention_batch,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-compact",
            conversation_id="conv-compact",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    payload = serialize_compact_attention_batch(
        [
            {
                "request_id": "req-compact",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
                "block_id": 0,
                "slot": 0,
                "seq_len": 1,
                "q_size": 2,
                "kv_size": 2,
                "num_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 2,
            },
        ],
        [torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])],
    )

    response = compute_binary_attention_response(registry, payload)
    outputs = deserialize_compact_attention_response(response)

    assert torch.allclose(outputs[0], torch.tensor([[[2.0, 0.0]]]))



def test_attention_executor_rejects_decode_when_prompt_kv_is_missing() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-missing",
            "conversation_id": "conv-prefix-missing",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-prefix-missing",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
        },
    )

    assert response.status_code == 409
    assert "prefill KV" in response.json()["detail"]


def test_attention_executor_imports_prefill_kv_before_stateful_decode() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-imported",
            "conversation_id": "conv-prefix-imported",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv",
        json={
            "request_id": "req-prefix-imported",
            "layer_name": "model.layers.0.self_attn.attn",
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])),
            "seq_len": 2,
        },
    )
    assert imported.status_code == 200
    assert imported.json()["seq_len"] == 2

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-prefix-imported",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "key": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[0.0, 8.0]]])),
            "scale": 1.0,
        },
    )

    assert response.status_code == 200
    output = deserialize_attention_result(response.json()["output"])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]], [[0.0, 8.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(output, expected)

    session = client.get("/v1/pap/attention/sessions/req-prefix-imported").json()
    assert session["prefill_seq_lens"] == {"model.layers.0.self_attn.attn": 2}
    assert session["decode_seq_lens"] == {"model.layers.0.self_attn.attn": 3}


def test_attention_executor_binary_imports_prefill_kv_before_decode() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-binary",
            "conversation_id": "conv-prefix-binary",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_kv",
                "request_id": "req-prefix-binary",
                "layer_name": "model.layers.0.self_attn.attn",
                "seq_len": 2,
                "block_ids": [4],
            },
            {
                "key": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                "value": torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 2
    assert tensors == {}

    response = client.post(
        "/v1/pap/attention/append-and-compute-binary",
        content=serialize_tensor_bundle(
            {
                "request_id": "req-prefix-binary",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
                "block_id": 0,
                "slot": 2,
                "seq_len": 3,
            },
            {
                "query": torch.tensor([[[0.0, 1.0]]]),
                "key": torch.tensor([[[0.0, 1.0]]]),
                "value": torch.tensor([[[0.0, 8.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(response.content)
    assert metadata["seq_len"] == 3
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]], [[0.0, 8.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(tensors["output"], expected)


def test_attention_executor_binary_imports_prefill_kv_ipc_descriptor(
    monkeypatch,
) -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap import pap_attention_executor
    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    def fake_open_ipc_prefill_kv(descriptor):
        assert descriptor.request_id == "req-prefix-ipc"
        assert descriptor.seq_len == 2
        return key, value

    monkeypatch.setattr(
        pap_attention_executor,
        "open_ipc_prefill_kv",
        fake_open_ipc_prefill_kv,
    )

    descriptor = PAPOffloadKVIPCDescriptor(
        request_id="req-prefix-ipc",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=2,
        block_ids=(4,),
        key=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(key.shape),
            ipc_handle={"GPU-test": ("key", 1, 2, 3, 4, 5, 0)},
        ),
        value=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(value.shape),
            ipc_handle={"GPU-test": ("value", 1, 2, 3, 4, 5, 0)},
        ),
    )
    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-ipc",
            "conversation_id": "conv-prefix-ipc",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_kv_ipc",
                "descriptor": descriptor.to_dict(),
            },
            {},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 2
    assert tensors == {}
    session = client.get("/v1/pap/attention/sessions/req-prefix-ipc").json()
    assert session["prefill_seq_lens"] == {"model.layers.0.self_attn.attn": 2}


def test_attention_executor_compute_existing_prefill_token_does_not_append() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-existing",
            "conversation_id": "conv-prefix-existing",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
            "block_size": 16,
        },
    )
    client.post(
        "/v1/pap/attention/import-prefill-kv",
        json={
            "request_id": "req-prefix-existing",
            "layer_name": "model.layers.0.self_attn.attn",
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])),
            "seq_len": 2,
            "block_ids": [4],
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-prefix-existing",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "key": serialize_tensor(torch.tensor([[[9.0, 9.0]]])),
            "value": serialize_tensor(torch.tensor([[[9.0, 9.0]]])),
            "scale": 1.0,
            "block_id": 4,
            "slot": 4 * 16 + 1,
            "seq_len": 2,
        },
    )

    assert response.status_code == 200
    output = deserialize_attention_result(response.json()["output"])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(output, expected)

    session = client.get("/v1/pap/attention/sessions/req-prefix-existing").json()
    assert session["block_ids"] == [4]
    assert session["seq_len"] == 2
    assert session["decode_seq_lens"] == {"model.layers.0.self_attn.attn": 2}


def test_attention_executor_tracks_decode_progress_per_layer() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-layer-progress",
            "conversation_id": "conv-layer-progress",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
            "block_size": 16,
        },
    )
    for layer_name in (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ):
        imported = client.post(
            "/v1/pap/attention/import-prefill-kv",
            json={
                "request_id": "req-layer-progress",
                "layer_name": layer_name,
                "key": serialize_tensor(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
                "value": serialize_tensor(torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])),
                "seq_len": 2,
                "block_ids": [4],
            },
        )
        assert imported.status_code == 200

    for layer_name in (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ):
        response = client.post(
            "/v1/pap/attention/append-and-compute",
            json={
                "request_id": "req-layer-progress",
                "layer_name": layer_name,
                "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
                "key": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
                "value": serialize_tensor(torch.tensor([[[0.0, 8.0]]])),
                "scale": 1.0,
                "block_id": 99,
                "slot": 99 * 16 + 2,
                "seq_len": 3,
            },
        )
        assert response.status_code == 200
        assert response.json()["seq_len"] == 3

    session = client.get("/v1/pap/attention/sessions/req-layer-progress").json()
    assert session["seq_len"] == 3
    assert session["block_ids"] == [4, 99]
    assert session["decode_seq_lens"] == {
        "model.layers.0.self_attn.attn": 3,
        "model.layers.1.self_attn.attn": 3,
    }


def test_attention_executor_records_scheduler_descriptor_state() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = TestClient(app)
    registered = client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-descriptor",
            "conversation_id": "conv-descriptor",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 0,
            "block_size": 16,
            "max_seq_len": 64,
        },
    )
    assert registered.status_code == 200
    assert registered.json()["block_size"] == 16
    assert registered.json()["max_seq_len"] == 64

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-descriptor",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
            "block_id": 4,
            "slot": 4 * 16,
            "seq_len": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["seq_len"] == 1
    session = client.get("/v1/pap/attention/sessions/req-descriptor").json()
    assert session["block_ids"] == [4]
    assert session["seq_len"] == 1


def test_attention_executor_rejects_bad_scheduler_descriptor() -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-bad-descriptor",
            "conversation_id": "conv-bad-descriptor",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 0,
            "block_size": 16,
            "max_seq_len": 64,
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-bad-descriptor",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
            "block_id": 4,
            "slot": 4 * 16 + 1,
            "seq_len": 1,
        },
    )

    assert response.status_code == 400
    assert "slot" in response.json()["detail"]
    session = client.get("/v1/pap/attention/sessions/req-bad-descriptor").json()
    assert session["block_ids"] == []
    assert session["seq_len"] == 0
