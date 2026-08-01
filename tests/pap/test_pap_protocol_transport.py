# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP protocol and transport contract tests."""

import pytest
import torch

from vllm.pap.protocol import (
    PAPCudaIPCTensorHandle,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
    PAPStepPlannedOffloadExecTransport,
)
from vllm.pap.protocol.offload_exec import (
    _offload_exec_batch_descriptor_from_metadata,
    _offload_exec_batch_descriptor_to_metadata,
)
from vllm.pap.transport.local.transport import PAPLocalFastTransport
from vllm.pap.transport.nixl.offload import (
    PAPNixlMailboxOffloadExecTransport,
)


def test_step_planned_transport_capability_is_local_only() -> None:
    local_transport = object.__new__(PAPLocalFastTransport)
    nixl_transport = PAPNixlMailboxOffloadExecTransport(object())

    assert isinstance(local_transport, PAPStepPlannedOffloadExecTransport)
    assert not isinstance(nixl_transport, PAPStepPlannedOffloadExecTransport)


def test_nixl_mailbox_sends_direct_qkv_batch_without_copy_payload() -> None:
    class ReservedPayload:
        def __init__(self, tensor: torch.Tensor, slot_id: int) -> None:
            self.tensor = tensor
            self.slot_id = slot_id

    class FakeEndpoint:
        def __init__(self) -> None:
            self.sent = []
            self.reserved = []

        def reserve_direct_send_tensor(self, msg_id, shape, dtype):
            tensor = torch.empty(tuple(shape), dtype=dtype)
            self.reserved.append((msg_id, tuple(shape), dtype, tensor))
            return ReservedPayload(tensor=tensor, slot_id=0)

        def send(self, message) -> None:
            self.sent.append(message)

    endpoint = FakeEndpoint()
    transport = PAPNixlMailboxOffloadExecTransport(endpoint)
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125),),
    )
    qkv = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    transport.send_qkv_batch_direct(
        descriptor,
        qkv,
        remote_address="ignored",
    )

    assert len(endpoint.sent) == 1
    message = endpoint.sent[0]
    assert message.msg_id == descriptor.qkv_tensor_id
    assert message.kind == "attention_task_batch"
    assert endpoint.reserved[0][:3] == (descriptor.qkv_tensor_id, (1, 4), qkv.dtype)
    assert message.tensor is endpoint.reserved[0][3]
    torch.testing.assert_close(message.tensor, qkv)
    assert message.direct_payload is True
    assert message.payload_slot_id == 0
    assert message.payload_shape == tuple(qkv.shape)


def test_nixl_mailbox_direct_qkv_batch_supports_inference_mode_slot() -> None:
    class ReservedPayload:
        def __init__(self, tensor: torch.Tensor, slot_id: int) -> None:
            self.tensor = tensor
            self.slot_id = slot_id

    class FakeEndpoint:
        def __init__(self) -> None:
            self.sent = []

        def reserve_direct_send_tensor(self, msg_id, shape, dtype):
            tensor = torch.empty(tuple(shape), dtype=dtype)
            return ReservedPayload(tensor=tensor, slot_id=0)

        def send(self, message) -> None:
            self.sent.append(message)

    endpoint = FakeEndpoint()
    transport = PAPNixlMailboxOffloadExecTransport(endpoint)
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125),),
    )
    qkv = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    with torch.inference_mode():
        transport.send_qkv_batch_direct(
            descriptor,
            qkv,
            remote_address="ignored",
        )

    torch.testing.assert_close(endpoint.sent[0].tensor, qkv)


def test_nixl_mailbox_qkv_batch_uses_plan_ref_after_first_layer_by_default() -> None:
    class FakeEndpoint:
        def __init__(self) -> None:
            self.sent = []

        def send(self, message) -> None:
            self.sent.append(message)

    endpoint = FakeEndpoint()
    transport = PAPNixlMailboxOffloadExecTransport(endpoint)
    template = {
        "r": ("req-a", "req-b"),
        "s": (7, 8),
        "a": (0.125, 0.125),
    }
    first = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template=template,
    )
    second = PAPOffloadExecBatchDescriptor(
        layer_name="layer1",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template=template,
    )
    qkv = torch.zeros((2, 4), dtype=torch.float32)

    transport.send_qkv_batch(first, qkv, remote_address="ignored")
    transport.send_qkv_batch(second, qkv, remote_address="ignored")

    assert endpoint.sent[0].metadata["v"] == 4
    assert endpoint.sent[0].metadata["l"] == "layer0"
    assert endpoint.sent[0].metadata["r"] == ["req-a", "req-b"]
    assert "t" not in endpoint.sent[0].metadata
    plan_id = endpoint.sent[0].metadata["p"]
    assert endpoint.sent[1].metadata == {
        "v": 5,
        "l": "layer1",
        "p": plan_id,
    }


def test_nixl_mailbox_qkv_batch_plan_ref_roundtrips_on_receiver_by_default() -> None:
    class FakeEndpoint:
        def __init__(self) -> None:
            self.messages = []

        def send(self, message) -> None:
            self.messages.append(message)

        def recv(self, msg_id=None):
            assert msg_id is None
            return self.messages.pop(0)

    endpoint = FakeEndpoint()
    sender = PAPNixlMailboxOffloadExecTransport(endpoint)
    receiver = PAPNixlMailboxOffloadExecTransport(endpoint)
    template = {
        "r": ("req-a", "req-b"),
        "s": (7, 8),
        "a": (0.125, 0.125),
    }
    first = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template=template,
    )
    second = PAPOffloadExecBatchDescriptor(
        layer_name="layer1",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template=template,
    )
    qkv = torch.zeros((2, 4), dtype=torch.float32)
    sender.send_qkv_batch(first, qkv, remote_address="ignored")
    sender.send_qkv_batch(second, qkv, remote_address="ignored")

    restored_first, first_message = receiver.recv_next_qkv_batch_message()
    restored_second, second_message = receiver.recv_next_qkv_batch_message()

    assert first_message.metadata["v"] == 4
    assert second_message.metadata["v"] == 5
    assert restored_first.layer_name == "layer0"
    assert restored_second.layer_name == "layer1"
    assert [item.request_id for item in restored_second.items] == ["req-a", "req-b"]
    assert [item.step for item in restored_second.items] == [7, 8]
    assert [item.scale for item in restored_second.items] == [0.125, 0.125]
    assert restored_second.output_tensor_id == "layer1#req-a@7,req-b@8#attn_out_batch"


def test_batch_plan_ref_can_restore_template_only_descriptor() -> None:
    plan_cache = {}
    full = {
        "v": 4,
        "l": "layer0",
        "p": "plan-a",
        "b": "req-a@7,req-b@8",
        "r": ["req-a", "req-b"],
        "s": [7, 8],
        "a": [0.125, 0.125],
    }

    first = _offload_exec_batch_descriptor_from_metadata(
        full,
        plan_cache=plan_cache,
        template_only=True,
    )
    second = _offload_exec_batch_descriptor_from_metadata(
        {"v": 5, "l": "layer1", "p": "plan-a"},
        plan_cache=plan_cache,
        template_only=True,
    )

    assert first.items == ()
    assert second.items == ()
    assert second.item_count == 2
    assert second.batch_id_suffix == "req-a@7,req-b@8"
    assert second.metadata_template == {
        "r": ("req-a", "req-b"),
        "s": (7, 8),
        "a": (0.125, 0.125),
    }


def test_offload_exec_batch_metadata_uses_compact_arrays() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125),
            PAPOffloadExecDescriptor("req-b", "layer0", 8, 0.25),
        ),
    )

    metadata = _offload_exec_batch_descriptor_to_metadata(descriptor)

    assert metadata == {
        "v": 2,
        "l": "layer0",
        "r": ["req-a", "req-b"],
        "s": [7, 8],
        "a": [0.125, 0.25],
    }
    assert "items" not in metadata


def test_offload_exec_batch_metadata_compact_roundtrip() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125),
            PAPOffloadExecDescriptor("req-b", "layer0", 8, 0.25),
        ),
    )

    restored = _offload_exec_batch_descriptor_from_metadata(
        _offload_exec_batch_descriptor_to_metadata(descriptor)
    )

    assert restored == descriptor


def test_prefill_kv_catalog_descriptor_roundtrip() -> None:
    descriptor = PAPPrefillKVCacheCatalogDescriptor(
        catalog_id="prefill-42-r0",
        layer_name="model.layers.0.self_attn.attn",
        block_size=16,
        num_kv_heads=8,
        layout="NHD",
        kv_cache=_paged_ipc_handle(),
    )

    assert (
        PAPPrefillKVCacheCatalogDescriptor.from_dict(descriptor.to_dict()) == descriptor
    )


def test_prefill_kv_session_manifest_roundtrip() -> None:
    manifest = PAPPrefillKVSessionManifest(
        request_id="req-1",
        session_handle="req-1@pap-session-1",
        catalog_id="prefill-42-r0",
        prefix_len=16,
        block_ids=(2, 3, 4),
        block_size=16,
        expected_layer_count=2,
        lease_id="lease-1",
        leased_block_ids=(2, 3, 4),
        lease_capacity_tokens=48,
        writable_start_token=16,
        writable_end_token=48,
        ready_event_handle=b"cuda-event-handle",
    )

    assert PAPPrefillKVSessionManifest.from_dict(manifest.to_dict()) == manifest


def test_prefill_kv_session_manifest_rejects_shared_writable_prefix() -> None:
    with pytest.raises(ValueError, match="writable_start_token"):
        PAPPrefillKVSessionManifest(
            request_id="req-1",
            session_handle="req-1@pap-session-1",
            catalog_id="prefill-42-r0",
            prefix_len=16,
            block_ids=(2, 3, 4),
            block_size=16,
            expected_layer_count=2,
            lease_id="lease-1",
            leased_block_ids=(2, 3, 4),
            lease_capacity_tokens=48,
            writable_start_token=15,
            writable_end_token=48,
        )


def _paged_ipc_handle() -> PAPCudaIPCTensorHandle:
    return PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(2, 4, 1, 8),
        ipc_handle={"GPU-test": ("storage", 1, 2, 3, 4, 5, 0)},
    )


def _session_manifest(**overrides) -> PAPPrefillKVSessionManifest:
    values = {
        "request_id": "req-1",
        "session_handle": "req-1@pap-session-1",
        "catalog_id": "prefill-test",
        "prefix_len": 4,
        "block_ids": (0, 1),
        "block_size": 4,
        "expected_layer_count": 1,
        "lease_id": "lease-1",
        "leased_block_ids": (0, 1),
        "lease_capacity_tokens": 8,
        "writable_start_token": 4,
        "writable_end_token": 8,
    }
    values.update(overrides)
    return PAPPrefillKVSessionManifest(**values)


def test_prefill_kv_session_manifest_requires_session_handle() -> None:
    with pytest.raises(ValueError, match="session_handle"):
        _session_manifest(session_handle="")


def test_prefill_kv_session_manifest_requires_lease() -> None:
    with pytest.raises(ValueError, match="lease_id"):
        _session_manifest(lease_id="")


def test_prefill_kv_session_manifest_requires_capacity_for_writable_end() -> None:
    with pytest.raises(ValueError, match="lease capacity"):
        _session_manifest(lease_capacity_tokens=4)


def test_prefill_kv_session_manifest_requires_blocks_covering_writable_end() -> None:
    with pytest.raises(ValueError, match="block_ids"):
        _session_manifest(block_ids=(0,), leased_block_ids=(0,))


@pytest.mark.parametrize(
    ("decode_capacity_tokens", "expected_writable_end"),
    [(None, 8), (2, 7)],
)
def test_sealed_prefill_kv_handoff_posts_catalog_and_manifest_without_sync(
    monkeypatch, decode_capacity_tokens, expected_writable_end
) -> None:
    from vllm.pap.kv.handoff import (
        publish_prefill_kv_session_manifest,
        register_prefill_kv_catalog,
    )
    from vllm.pap.lifecycle.lease import reset_global_kv_lease_registry
    from vllm.pap.protocol.wire import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    reset_global_kv_lease_registry()
    monkeypatch.setenv("PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS", "3")
    posted_metadata = []

    def fake_reduce_tensor(tensor):
        return object(), ("storage", 1, 2, 3, 4, 5, 0)

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        assert endpoint == "127.0.0.1:8300"
        metadata, tensors = deserialize_tensor_bundle(payload)
        assert tensors == {}
        posted_metadata.append(metadata)
        if metadata["command"] == "register_prefill_kv_catalog":
            return serialize_tensor_bundle({"status": "registered"}, {})
        assert metadata["command"] == "publish_prefill_kv_manifest"
        return serialize_tensor_bundle(
            {"status": "ready", "prefix_len": 5},
            {},
        )

    monkeypatch.setattr(
        "vllm.pap.kv.handoff.reduce_tensor",
        fake_reduce_tensor,
    )
    monkeypatch.setattr(
        "vllm.pap.kv.handoff._post_bytes_tcp",
        fake_post_bytes_tcp,
    )

    try:
        kv_cache = torch.zeros(2, 2, 4, 1, 2)
        assert (
            register_prefill_kv_catalog(
                catalog_id="prefill-test",
                layer_name="layer0",
                kv_cache=kv_cache,
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
                tcp_endpoint="127.0.0.1:8300",
            )
            == "registered"
        )
        assert (
            publish_prefill_kv_session_manifest(
                request_id="cmpl-sealed",
                session_handle="cmpl-sealed@pap-session-1",
                catalog_id="prefill-test",
                block_ids=(0, 1),
                prefix_len=5,
                block_size=4,
                expected_layer_count=1,
                ready_event_handle=b"event-handle",
                tcp_endpoint="127.0.0.1:8300",
                decode_capacity_tokens=decode_capacity_tokens,
            )
            == 5
        )
    finally:
        reset_global_kv_lease_registry()

    assert [item["command"] for item in posted_metadata] == [
        "register_prefill_kv_catalog",
        "publish_prefill_kv_manifest",
    ]
    catalog = posted_metadata[0]["descriptor"]
    assert catalog["catalog_id"] == "prefill-test"
    assert catalog["layer_name"] == "layer0"
    manifest = posted_metadata[1]["manifest"]
    assert manifest["session_handle"] == "cmpl-sealed@pap-session-1"
    assert manifest["prefix_len"] == 5
    assert manifest["writable_start_token"] == 5
    assert manifest["writable_end_token"] == expected_writable_end
    assert manifest["ready_event_handle"] is not None
