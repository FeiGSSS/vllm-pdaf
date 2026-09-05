# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.pap.integration.projection import build_offload_exec_route_groups
from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
)


def test_build_offload_exec_route_groups_groups_by_attention_endpoint() -> None:
    groups = build_offload_exec_route_groups(
        ("cmpl-a", "cmpl-b", "cmpl-c", "cmpl-d"),
        attention_endpoint_by_request={
            "cmpl-a": "http://a0:8300",
            "cmpl-b": "http://a1:8300",
            "cmpl-c": "http://a0:8300",
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
            "req_indices": (0, 2),
            "request_ids": ("cmpl-a", "cmpl-c"),
            "steps": (17, 19),
            "batch_id_suffix": "cmpl-a@17,cmpl-c@19",
        },
        {
            "attention_endpoint": "http://a1:8300",
            "req_indices": (1,),
            "request_ids": ("cmpl-b",),
            "steps": (18,),
            "batch_id_suffix": "cmpl-b@18",
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
