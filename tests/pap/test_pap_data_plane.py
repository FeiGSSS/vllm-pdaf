import pytest
import torch

from vllm.pap.data_plane import (
    PAPCudaIPCTensorHandle,
    PAPNixlMailboxOffloadExecTransport,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPOffloadKVPagedIPCDescriptor,
    PAPTensorTransport,
    _offload_exec_batch_descriptor_from_metadata,
    _offload_exec_batch_descriptor_to_metadata,
)
from vllm.pap.nixl_mailbox import PAPMailboxMessage


def test_offload_exec_descriptor_uses_stable_tensor_ids() -> None:
    descriptor = PAPOffloadExecDescriptor(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        step=7,
        scale=0.125,
    )

    assert descriptor.qkv_tensor_id == "cmpl-1#model.layers.0.self_attn.attn#7#qkv"
    assert (
        descriptor.output_tensor_id
        == "cmpl-1#model.layers.0.self_attn.attn#7#attn_out"
    )


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


def test_nixl_mailbox_transport_naked_next_qkv_clones_and_releases() -> None:
    class FakeEndpoint:
        def __init__(self, message) -> None:
            self.message = message

        def recv(self, msg_id=None):
            assert msg_id is None
            return self.message

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125),),
    )
    released = []
    tensor = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    message = PAPMailboxMessage(
        msg_id=descriptor.qkv_tensor_id,
        kind="attention_task_batch",
        metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
        tensor=tensor,
        release_callback=lambda: released.append(True),
    )
    transport = PAPNixlMailboxOffloadExecTransport(FakeEndpoint(message))

    restored_descriptor, restored_tensor = transport.recv_next_qkv_batch()

    assert restored_descriptor == descriptor
    torch.testing.assert_close(restored_tensor, tensor)
    assert restored_tensor.data_ptr() != tensor.data_ptr()
    assert released == [True]


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


def test_offload_exec_batch_metadata_roundtrips_decode_token_ids() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                "req-a", "layer0", 7, 0.125, decode_token_ids=(42,)
            ),
            PAPOffloadExecDescriptor(
                "req-b", "layer0", 8, 0.25, decode_token_ids=(99,)
            ),
        ),
    )

    metadata = _offload_exec_batch_descriptor_to_metadata(descriptor)
    restored = _offload_exec_batch_descriptor_from_metadata(metadata)

    assert metadata["v"] == 3
    assert metadata["t"] == [[42], [99]]
    assert restored.items[0].decode_token_ids == (42,)
    assert restored.items[1].decode_token_ids == (99,)


def test_offload_exec_batch_metadata_v2_remains_backward_compatible() -> None:
    metadata = {
        "v": 2,
        "l": "layer0",
        "r": ["req-a"],
        "s": [7],
        "a": [0.125],
    }

    restored = _offload_exec_batch_descriptor_from_metadata(metadata)

    assert restored.items[0].decode_token_ids == ()


def test_offload_exec_batch_metadata_accepts_legacy_items() -> None:
    metadata = {
        "layer_name": "layer0",
        "items": [
            {
                "request_id": "req-a",
                "layer_name": "layer0",
                "step": 7,
                "scale": 0.125,
            },
            {
                "request_id": "req-b",
                "layer_name": "layer0",
                "step": 8,
                "scale": 0.25,
            },
        ],
    }

    restored = _offload_exec_batch_descriptor_from_metadata(metadata)

    assert restored == PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor("req-a", "layer0", 7, 0.125),
            PAPOffloadExecDescriptor("req-b", "layer0", 8, 0.25),
        ),
    )


def test_offload_kv_ipc_descriptor_roundtrip() -> None:
    from vllm.pap.data_plane import (
        PAPCudaIPCTensorHandle,
        PAPOffloadKVIPCDescriptor,
    )

    key_handle = PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(8, 2, 16),
        ipc_handle={"GPU-abc": ("storage", 1, 2, 3, 4, 5, 0)},
    )
    value_handle = PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(8, 2, 16),
        ipc_handle={"GPU-abc": ("storage", 1, 2, 3, 4, 5, 0)},
    )
    descriptor = PAPOffloadKVIPCDescriptor(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=8,
        block_ids=(3, 4),
        key=key_handle,
        value=value_handle,
    )

    restored = PAPOffloadKVIPCDescriptor.from_dict(descriptor.to_dict())

    assert restored == descriptor
    assert restored.transport is PAPTensorTransport.CUDA_IPC


def test_offload_kv_paged_ipc_descriptor_roundtrip() -> None:
    from vllm.pap.data_plane import (
        PAPCudaIPCTensorHandle,
        PAPOffloadKVPagedIPCDescriptor,
    )

    kv_cache_handle = PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(2, 8, 16, 2, 64),
        ipc_handle={"GPU-abc": ("storage", 1, 2, 3, 4, 5, 0)},
    )
    descriptor = PAPOffloadKVPagedIPCDescriptor(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=19,
        block_ids=(3, 4),
        block_size=16,
        num_kv_heads=2,
        layout="NHD",
        kv_cache=kv_cache_handle,
    )

    restored = PAPOffloadKVPagedIPCDescriptor.from_dict(descriptor.to_dict())

    assert restored == descriptor
    assert restored.transport is PAPTensorTransport.CUDA_IPC


def _paged_ipc_handle() -> PAPCudaIPCTensorHandle:
    return PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(2, 4, 1, 8),
        ipc_handle={"GPU-test": ("storage", 1, 2, 3, 4, 5, 0)},
    )


def test_unified_paged_ipc_descriptor_requires_lease() -> None:
    with pytest.raises(ValueError, match="requires lease_id"):
        PAPOffloadKVPagedIPCDescriptor(
            request_id="req-1",
            layer_name="layer0",
            seq_len=4,
            block_ids=(0,),
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            kv_cache=_paged_ipc_handle(),
            unified_kv_mode=True,
            prefix_len=4,
            writable_start_token=4,
            writable_end_token=8,
        )


def test_unified_paged_ipc_descriptor_requires_capacity_for_writable_end() -> None:
    with pytest.raises(ValueError, match="lease_capacity_tokens"):
        PAPOffloadKVPagedIPCDescriptor(
            request_id="req-1",
            layer_name="layer0",
            seq_len=4,
            block_ids=(0, 1),
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            kv_cache=_paged_ipc_handle(),
            lease_id="lease-1",
            leased_block_ids=(0, 1),
            lease_capacity_tokens=4,
            unified_kv_mode=True,
            prefix_len=4,
            writable_start_token=4,
            writable_end_token=8,
        )


def test_unified_paged_ipc_descriptor_requires_blocks_covering_writable_end() -> None:
    with pytest.raises(ValueError, match="block_ids"):
        PAPOffloadKVPagedIPCDescriptor(
            request_id="req-1",
            layer_name="layer0",
            seq_len=4,
            block_ids=(0,),
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            kv_cache=_paged_ipc_handle(),
            lease_id="lease-1",
            leased_block_ids=(0,),
            lease_capacity_tokens=8,
            unified_kv_mode=True,
            prefix_len=4,
            writable_start_token=4,
            writable_end_token=8,
        )


def test_import_prefill_kv_cuda_ipc_posts_descriptor_without_tensors(
    monkeypatch,
) -> None:
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import import_prefill_kv

    posted_payloads: list[bytes] = []

    def fake_reduce_tensor(tensor):
        return object(), ("storage", 1, 2, 3, 4, 5, 0)

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        assert endpoint == "127.0.0.1:8300"
        posted_payloads.append(payload)
        return serialize_tensor_bundle({"seq_len": 2}, {})

    monkeypatch.setattr(
        "vllm.pap.shadow_attention.reduce_tensor",
        fake_reduce_tensor,
    )
    monkeypatch.setattr(
        "vllm.pap.shadow_attention._post_bytes_tcp",
        fake_post_bytes_tcp,
    )

    seq_len = import_prefill_kv(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.zeros(2, 1, 2),
        value=torch.ones(2, 1, 2),
        seq_len=2,
        block_ids=[4],
        tcp_endpoint="127.0.0.1:8300",
        transport=PAPTensorTransport.CUDA_IPC,
    )

    assert seq_len == 2
    assert len(posted_payloads) == 1
    metadata, tensors = deserialize_tensor_bundle(posted_payloads[0])
    assert tensors == {}
    assert metadata["command"] == "import_prefill_kv_ipc"
    descriptor = metadata["descriptor"]
    assert descriptor["request_id"] == "cmpl-1"
    assert descriptor["layer_name"] == "model.layers.0.self_attn.attn"
    assert descriptor["seq_len"] == 2
    assert descriptor["block_ids"] == [4]
    assert descriptor["key"]["shape"] == [2, 1, 2]
    assert "ipc_handle_pickled" in descriptor["key"]
    assert descriptor["value"]["shape"] == [2, 1, 2]
    assert "ipc_handle_pickled" in descriptor["value"]


def test_import_prefill_paged_kv_cuda_ipc_posts_descriptor_without_tensors(
    monkeypatch,
) -> None:
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import import_prefill_paged_kv

    posted_payloads: list[bytes] = []

    def fake_reduce_tensor(tensor):
        return object(), ("storage", 1, 2, 3, 4, 5, 0)

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        assert endpoint == "127.0.0.1:8300"
        posted_payloads.append(payload)
        return serialize_tensor_bundle({"seq_len": 5}, {})

    monkeypatch.setattr(
        "vllm.pap.shadow_attention.reduce_tensor",
        fake_reduce_tensor,
    )
    monkeypatch.setattr(
        "vllm.pap.shadow_attention._post_bytes_tcp",
        fake_post_bytes_tcp,
    )

    kv_cache = torch.zeros(2, 2, 4, 1, 2)
    seq_len = import_prefill_paged_kv(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        block_ids=[0, 1],
        seq_len=5,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
        tcp_endpoint="127.0.0.1:8300",
    )

    assert seq_len == 5
    assert len(posted_payloads) == 1
    metadata, tensors = deserialize_tensor_bundle(posted_payloads[0])
    assert tensors == {}
    assert metadata["command"] == "import_prefill_paged_kv_ipc"
    descriptor = metadata["descriptor"]
    assert descriptor["request_id"] == "cmpl-1"
    assert descriptor["layer_name"] == "model.layers.0.self_attn.attn"
    assert descriptor["seq_len"] == 5
    assert descriptor["block_ids"] == [0, 1]
    assert descriptor["block_size"] == 4
    assert descriptor["num_kv_heads"] == 1
    assert descriptor["layout"] == "NHD"
    assert descriptor["kv_cache"]["shape"] == [2, 2, 4, 1, 2]
    assert "ipc_handle_pickled" in descriptor["kv_cache"]


def test_import_prefill_paged_kv_unified_reuses_active_lease(
    monkeypatch,
) -> None:
    from vllm.pap.kv_lease import reset_global_kv_lease_registry
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )
    from vllm.pap.shadow_attention import import_prefill_paged_kv

    reset_global_kv_lease_registry()
    monkeypatch.setenv("PAP_UNIFIED_KV", "1")
    monkeypatch.setenv("PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS", "3")
    descriptors = []

    def fake_reduce_tensor(tensor):
        return object(), ("storage", 1, 2, 3, 4, 5, 0)

    def fake_post_bytes_tcp(*, endpoint, payload, timeout):
        metadata, tensors = deserialize_tensor_bundle(payload)
        assert tensors == {}
        descriptors.append(metadata["descriptor"])
        return serialize_tensor_bundle({"seq_len": 5}, {})

    monkeypatch.setattr(
        "vllm.pap.shadow_attention.reduce_tensor",
        fake_reduce_tensor,
    )
    monkeypatch.setattr(
        "vllm.pap.shadow_attention._post_bytes_tcp",
        fake_post_bytes_tcp,
    )

    try:
        kv_cache = torch.zeros(2, 2, 4, 1, 2)
        for layer_name in ("layer0", "layer1"):
            assert import_prefill_paged_kv(
                request_id="cmpl-lease",
                layer_name=layer_name,
                kv_cache=kv_cache,
                block_ids=[0, 1],
                seq_len=5,
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
                tcp_endpoint="127.0.0.1:8300",
            ) == 5
    finally:
        reset_global_kv_lease_registry()

    assert len(descriptors) == 2
    assert descriptors[0]["lease_id"]
    assert descriptors[1]["lease_id"] == descriptors[0]["lease_id"]
    assert descriptors[1]["leased_block_ids"] == [0, 1]
    assert descriptors[1]["lease_capacity_tokens"] == 8
