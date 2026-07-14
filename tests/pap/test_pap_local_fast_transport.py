import os
import threading
from collections import OrderedDict

import pytest
import torch

from vllm.pap.cuda_stream_memops import (
    _signal_address,
    stream_wait_value32,
    stream_write_value32,
)
from vllm.pap.transport.local_fast import (
    _LocalFastMessage,
    PAPLocalFastTransport,
)
from vllm.pap.transport.local_fast_endpoint import _open_or_create_doorbell
from vllm.pap.transport.local_fast_protocol import (
    DTYPE_CODE_BFLOAT16,
    DIR_OUTPUT,
    DIR_QKV,
    RECORD_FLAG_FIXED_TENSOR,
    RECORD_FLAG_OUTPUT_DESCRIPTORLESS,
    RECORD_FLAG_PLAN_FULL,
    RECORD_FLAG_PLAN_REF,
    _doorbell_ack,
    _doorbell_bytes,
    _doorbell_read_header,
    _doorbell_read_record,
    _doorbell_read_metadata,
    _doorbell_record_offset,
    _doorbell_write,
    _payload_metadata,
    _signal_index,
)
from vllm.pap.protocol import PAPOffloadExecBatchDescriptor


def test_local_fast_doorbell_slots_are_independent(tmp_path) -> None:
    slot_count = 2
    path = tmp_path / "doorbell"
    fd, mm = _open_or_create_doorbell(str(path), _doorbell_bytes(slot_count))
    try:
        qkv_offset = _doorbell_record_offset(DIR_QKV, 0, slot_count)
        output_offset = _doorbell_record_offset(DIR_OUTPUT, 1, slot_count)
        _doorbell_write(
            mm,
            qkv_offset,
            seq=1,
            nbytes=128,
            offset=0,
            metadata={"kind": "qkv"},
        )
        _doorbell_write(
            mm,
            output_offset,
            seq=2,
            nbytes=64,
            offset=256,
            metadata={"kind": "output"},
        )

        assert _doorbell_read_header(mm, qkv_offset)[:4] == (1, 128, 0, 14)
        assert _doorbell_read_header(mm, output_offset)[:4] == (
            2,
            64,
            256,
            17,
        )
        assert _doorbell_read_metadata(mm, qkv_offset, 14) == {"kind": "qkv"}
        assert _doorbell_read_metadata(mm, output_offset, 17) == {"kind": "output"}

        _doorbell_ack(mm, qkv_offset, 1)
        assert _doorbell_read_header(mm, qkv_offset)[4] == 1
        assert _doorbell_read_header(mm, output_offset)[4] == 0
    finally:
        mm.close()
        os.close(fd)


def test_local_fast_signal_layout_is_unique() -> None:
    indices = {
        _signal_index(direction, slot, 2, release=release)
        for direction in (DIR_QKV, DIR_OUTPUT)
        for slot in range(2)
        for release in (False, True)
    }

    assert indices == set(range(8))


def test_local_fast_message_releases_once() -> None:
    releases = []
    message = _LocalFastMessage(
        msg_id="msg",
        kind="attention_result_batch",
        tensor=torch.zeros(1),
        metadata={},
        release_callback=lambda: releases.append(True),
    )

    message.release()
    message.release()

    assert releases == [True]


def test_local_fast_step_plan_is_built_once_and_output_is_descriptorless() -> None:
    transport = object.__new__(PAPLocalFastTransport)
    transport.close = lambda: None
    transport._batch_plan_enabled = True
    transport._step_plan_cache_limit = 8
    transport._sent_step_plans = OrderedDict()
    transport._recv_plan_ids_by_key = {}
    transport._step_plan_builds = 0
    transport._step_plan_refs = 0
    transport._output_descriptor_elisions = 0
    transport._binary_qkv_refs = 0
    transport._binary_outputs = 0
    transport._json_records = 0
    template = {
        "r": ("req-a", "req-b"),
        "s": (7, 8),
        "a": (0.125, 0.125),
    }
    first = PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.0.self_attn.attn",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template=template,
    )
    second = PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.1.self_attn.attn",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template=template,
    )

    tensor = torch.zeros((2, 4), dtype=torch.bfloat16)
    first_wire = transport._wire_metadata(DIR_QKV, first, tensor)
    second_wire = transport._wire_metadata(DIR_QKV, second, tensor)
    output_wire = transport._wire_metadata(DIR_OUTPUT, second, tensor)

    assert first_wire.metadata is not None
    first_metadata = first_wire.metadata["descriptor"]
    assert first_metadata["v"] == 4
    assert first_wire.flags == RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_FULL
    assert second_wire.metadata is None
    assert second_wire.flags == RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_REF
    assert output_wire.metadata is None
    assert output_wire.flags == (
        RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_OUTPUT_DESCRIPTORLESS
    )
    assert second_wire.plan_id == first_wire.plan_id
    assert output_wire.plan_id == first_wire.plan_id
    assert second_wire.shape == (2, 4)
    assert second_wire.dtype_code == DTYPE_CODE_BFLOAT16
    assert second_wire.layer_index == 1
    assert {
        "v": 5,
        "l": second.layer_name,
        "p": first_metadata["p"],
    } == {
        "v": 5,
        "l": second.layer_name,
        "p": f"{second_wire.plan_id:016x}",
    }
    assert transport._step_plan_builds == 1
    assert transport._step_plan_refs == 1
    assert transport._output_descriptor_elisions == 1
    assert transport._binary_qkv_refs == 1
    assert transport._binary_outputs == 1
    assert transport._json_records == 1


def test_local_fast_payload_omits_empty_descriptor_metadata() -> None:
    tensor = torch.zeros((2, 4), dtype=torch.bfloat16)

    metadata = _payload_metadata({}, tensor)

    assert metadata == {"shape": [2, 4], "dtype": "bfloat16"}


def test_local_fast_fixed_doorbell_record_needs_no_json(tmp_path) -> None:
    path = tmp_path / "doorbell-fixed"
    fd, mm = _open_or_create_doorbell(str(path), _doorbell_bytes(1))
    try:
        offset = _doorbell_record_offset(DIR_QKV, 0, 1)
        _doorbell_write(
            mm,
            offset,
            seq=3,
            nbytes=64,
            offset=0,
            metadata=None,
            plan_id=0x1234,
            shape=(2, 16),
            layer_index=7,
            dtype_code=DTYPE_CODE_BFLOAT16,
            flags=RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_REF,
        )

        record = _doorbell_read_record(mm, offset)

        assert record.seq == 3
        assert record.metadata_len == 0
        assert record.plan_id == 0x1234
        assert (record.dim0, record.dim1) == (2, 16)
        assert record.layer_index == 7
        assert record.dtype_code == DTYPE_CODE_BFLOAT16
        assert record.flags == RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_REF
        assert _doorbell_read_metadata(mm, offset, record.metadata_len) == {}
    finally:
        mm.close()
        os.close(fd)


def test_cuda_stream_signal_requires_cuda_tensor() -> None:
    with pytest.raises(ValueError, match="must be on CUDA"):
        _signal_address(torch.zeros(1, dtype=torch.int32), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_stream_signal_roundtrip() -> None:
    signal = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream()

    stream_write_value32(signal, 0, 7, stream)
    stream_wait_value32(signal, 0, 7, stream)
    stream.synchronize()

    assert signal.item() == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_stream_signal_roundtrip_from_background_thread() -> None:
    signal = torch.zeros(1, dtype=torch.int32, device="cuda")
    errors = []

    def run() -> None:
        try:
            with torch.cuda.device(signal.device):
                stream = torch.cuda.current_stream(signal.device)
                stream_write_value32(signal, 0, 9, stream)
                stream_wait_value32(signal, 0, 9, stream)
                stream.synchronize()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()

    assert errors == []
    assert signal.item() == 9
