from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
)
from vllm.pap.protocol.offload_exec import (
    _offload_exec_batch_descriptor_to_metadata,
)
from vllm.pap.topology.routing import (
    build_offload_exec_route_groups,
    filter_offload_exec_route_groups_for_request_slice,
)


def test_build_offload_exec_route_groups_groups_by_attention_mailbox_pair() -> None:
    groups = build_offload_exec_route_groups(
        ("cmpl-a", "cmpl-b", "cmpl-c", "cmpl-d"),
        attention_endpoint_by_request={
            "cmpl-a": "http://a0:8300",
            "cmpl-b": "http://a1:8300",
            "cmpl-c": "http://a0:8300",
        },
        offload_exec_zmq_endpoint_by_request={
            "cmpl-a": "nixl://a0-r0",
            "cmpl-b": "nixl://a1-r0",
            "cmpl-c": "nixl://a0-r0",
            "cmpl-d": "nixl://a2-r0",
        },
        steps_by_request={
            "cmpl-a": 17,
            "cmpl-b": 18,
            "cmpl-c": 19,
            "cmpl-d": 20,
        },
    )

    assert groups == (
        {
            "attention_endpoint": "http://a0:8300",
            "offload_exec_zmq_endpoint": "nixl://a0-r0",
            "req_indices": (0, 2),
            "request_ids": ("cmpl-a", "cmpl-c"),
            "steps": (17, 19),
            "batch_id_suffix": "cmpl-a@17,cmpl-c@19",
        },
        {
            "attention_endpoint": "http://a1:8300",
            "offload_exec_zmq_endpoint": "nixl://a1-r0",
            "req_indices": (1,),
            "request_ids": ("cmpl-b",),
            "steps": (18,),
            "batch_id_suffix": "cmpl-b@18",
        },
    )


def test_filter_offload_exec_route_groups_rebases_to_ubatch_indices() -> None:
    groups = (
        {
            "attention_endpoint": "http://a0:8300",
            "offload_exec_zmq_endpoint": "nixl://a0-r0",
            "req_indices": (0, 2, 3),
            "request_ids": ("cmpl-a", "cmpl-c", "cmpl-d"),
            "steps": (17, 19, 20),
            "batch_id_suffix": "cmpl-a@17,cmpl-c@19,cmpl-d@20",
        },
        {
            "attention_endpoint": "http://a1:8300",
            "offload_exec_zmq_endpoint": "nixl://a1-r0",
            "req_indices": (1,),
            "request_ids": ("cmpl-b",),
            "steps": (18,),
            "batch_id_suffix": "cmpl-b@18",
        },
    )

    assert filter_offload_exec_route_groups_for_request_slice(
        groups,
        slice(2, 4),
    ) == (
        {
            "attention_endpoint": "http://a0:8300",
            "offload_exec_zmq_endpoint": "nixl://a0-r0",
            "req_indices": (0, 1),
            "request_ids": ("cmpl-c", "cmpl-d"),
            "steps": (19, 20),
            "batch_id_suffix": "cmpl-c@19,cmpl-d@20",
        },
    )


def test_batch_descriptor_uses_precomputed_route_template() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor("cmpl-a", "layer0", 17, 0.125),
            PAPOffloadExecDescriptor("cmpl-c", "layer0", 19, 0.125),
        ),
        batch_id_suffix="cmpl-a@17,cmpl-c@19",
        metadata_template={
            "r": ("cmpl-a", "cmpl-c"),
            "s": (17, 19),
        },
    )

    assert descriptor.batch_id == "layer0#cmpl-a@17,cmpl-c@19"
    assert descriptor.qkv_tensor_id == "layer0#cmpl-a@17,cmpl-c@19#qkv_batch"
    assert descriptor.output_tensor_id == ("layer0#cmpl-a@17,cmpl-c@19#attn_out_batch")
    assert _offload_exec_batch_descriptor_to_metadata(descriptor) == {
        "v": 2,
        "l": "layer0",
        "r": ["cmpl-a", "cmpl-c"],
        "s": [17, 19],
        "a": [0.125, 0.125],
    }


def test_batch_descriptor_allows_template_without_item_descriptors() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(),
        batch_id_suffix="cmpl-a@17,cmpl-c@19",
        metadata_template={
            "r": ("cmpl-a", "cmpl-c"),
            "s": (17, 19),
            "a": (0.125, 0.125),
        },
    )

    assert descriptor.item_count == 2
    assert descriptor.batch_id == "layer0#cmpl-a@17,cmpl-c@19"
    assert _offload_exec_batch_descriptor_to_metadata(descriptor) == {
        "v": 2,
        "l": "layer0",
        "r": ["cmpl-a", "cmpl-c"],
        "s": [17, 19],
        "a": [0.125, 0.125],
    }
