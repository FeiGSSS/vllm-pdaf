import os
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm.pap.cuda_stream_memops import (
    _signal_address,
    stream_wait_value32,
    stream_write_value32,
)
from vllm.pap.protocol import PAPOffloadExecBatchDescriptor
from vllm.pap.transport.local import io as local_fast_io
from vllm.pap.transport.local.endpoint import _open_or_create_doorbell
from vllm.pap.transport.local.io import _LocalFastMessage
from vllm.pap.transport.local.protocol import (
    DOORBELL_BYTES,
    DTYPE_CODE_BFLOAT16,
    DIR_OUTPUT,
    DIR_QKV,
    RECORD_FLAG_FIXED_TENSOR,
    RECORD_FLAG_OUTPUT_DESCRIPTORLESS,
    RECORD_FLAG_PLAN_FULL,
    RECORD_FLAG_PLAN_REF,
    RECORD_FLAG_STEP_PREPARE,
    _doorbell_ack,
    _doorbell_read_header,
    _doorbell_read_metadata,
    _doorbell_read_record,
    _doorbell_record_offset,
    _doorbell_write,
    _signal_index,
)
from vllm.pap.transport.local.transport import (
    PAPLocalFastTransport,
)


def test_local_fast_doorbell_directions_are_independent(tmp_path) -> None:
    path = tmp_path / "doorbell"
    fd, mm = _open_or_create_doorbell(str(path), DOORBELL_BYTES)
    try:
        qkv_offset = _doorbell_record_offset(DIR_QKV)
        output_offset = _doorbell_record_offset(DIR_OUTPUT)
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
        _signal_index(direction, release=release)
        for direction in (DIR_QKV, DIR_OUTPUT)
        for release in (False, True)
    }

    assert indices == set(range(4))


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


def test_local_fast_fanout_uses_peer_stream() -> None:
    calls = []
    fanout_stream = object()
    transport = object.__new__(PAPLocalFastTransport)
    transport.close = lambda: None
    transport._qkv_fanout_stream = fanout_stream
    transport._send_to_peer = lambda **kwargs: calls.append(kwargs)
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.0.self_attn.attn",
        items=(),
        batch_id_suffix="req-a@7",
        metadata_template={
            "r": ("req-a",),
            "s": (7,),
            "a": (0.125,),
        },
    )
    qkv = torch.zeros((1, 4), dtype=torch.bfloat16)

    transport.send_qkv_batch_fanout(
        descriptor,
        qkv,
        remote_address="unused",
    )

    assert calls == [
        {
            "direction": DIR_QKV,
            "descriptor": descriptor,
            "tensor": qkv,
            "stream": fanout_stream,
        }
    ]


def test_local_fast_step_plan_is_built_once_and_output_is_descriptorless() -> None:
    transport = object.__new__(PAPLocalFastTransport)
    transport.close = lambda: None
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


def test_local_fast_consumes_step_prepare_before_qkv() -> None:
    transport = object.__new__(PAPLocalFastTransport)
    transport.close = lambda: None
    prepared = []
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.0.self_attn.attn",
        items=(),
        batch_id_suffix="req-a@7",
        metadata_template={
            "r": ("req-a",),
            "s": (7,),
            "a": (0.125,),
        },
    )
    records = iter(
        (
            (
                1,
                0,
                0,
                {
                    "dtype": "bfloat16",
                    "_fixed_flags": RECORD_FLAG_STEP_PREPARE,
                },
            ),
            (2, 8, 0, {"dtype": "bfloat16", "_fixed_flags": 0}),
        )
    )
    transport._recv_from_peer = lambda **_kwargs: next(records)
    transport._decode_qkv_descriptor = lambda _metadata: (
        descriptor,
        {"descriptor": True},
    )
    tensor = torch.zeros((1, 4), dtype=torch.bfloat16)
    transport._materialize_recv = lambda **_kwargs: tensor
    transport._step_prepare_handler = (
        lambda received, dtype: prepared.append((received, dtype))
    )

    received, message = transport.recv_next_qkv_batch_message()

    assert prepared == [(descriptor, torch.bfloat16)]
    assert received is descriptor
    assert message.tensor is tensor
    assert message.metadata == {"descriptor": True}


def test_local_fast_fixed_doorbell_record_needs_no_json(tmp_path) -> None:
    path = tmp_path / "doorbell-fixed"
    fd, mm = _open_or_create_doorbell(str(path), DOORBELL_BYTES)
    try:
        offset = _doorbell_record_offset(DIR_QKV)
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

        _doorbell_ack(mm, offset, 5)
        _doorbell_write(
            mm,
            offset,
            seq=5,
            nbytes=64,
            offset=0,
            metadata=None,
            plan_id=0x5678,
            shape=(2, 16),
            layer_index=8,
            dtype_code=DTYPE_CODE_BFLOAT16,
            flags=RECORD_FLAG_FIXED_TENSOR | RECORD_FLAG_PLAN_REF,
        )
        assert _doorbell_read_record(mm, offset).ack == 5
    finally:
        mm.close()
        os.close(fd)


def test_local_fast_prepares_descriptorless_output_gpu_wait(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "doorbell-output"
    fd, mm = _open_or_create_doorbell(str(path), DOORBELL_BYTES)
    waits = []
    stream = object()
    monkeypatch.setattr(
        local_fast_io.torch.cuda,
        "current_stream",
        lambda _device: stream,
    )
    monkeypatch.setattr(
        local_fast_io,
        "stream_wait_value32",
        lambda signal, index, seq, current_stream: waits.append(
            (signal, index, seq, current_stream)
        ),
    )

    transport = object.__new__(PAPLocalFastTransport)
    transport.close = lambda: None
    transport.device = torch.device("cuda:0")
    transport.buffer_bytes = 128
    transport._recv_buffer = torch.zeros(128, dtype=torch.uint8)
    transport._signal_buffer = torch.zeros(4, dtype=torch.int32)
    transport._doorbell_mm = mm
    transport._deferred_cuda_trace = False
    transport._descriptorless_output_receives = 0
    transport._peer = SimpleNamespace(expected_output_seq=1)
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.0.self_attn.attn",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template={
            "r": ("req-a", "req-b"),
            "s": (7, 8),
            "a": (0.125, 0.125),
        },
    )
    try:
        message = transport.prepare_output_batch_message(
            descriptor,
            shape=(2, 4),
            dtype=torch.bfloat16,
            remote_address="",
        )

        assert message is not None
        assert tuple(message.tensor.shape) == (2, 4)
        assert message.tensor.dtype == torch.bfloat16
        assert transport._peer.expected_output_seq == 2
        assert transport._descriptorless_output_receives == 1
        assert message.msg_id == descriptor.layer_name
        assert message.metadata == {}
        record_offset = _doorbell_record_offset(DIR_OUTPUT)
        assert _doorbell_read_header(mm, record_offset)[4] == 0
        assert len(waits) == 1
        signal, signal_index, seq, current_stream = waits[0]
        assert signal is transport._signal_buffer
        assert signal_index == _signal_index(
            DIR_OUTPUT,
            release=False,
        )
        assert seq == 1
        assert current_stream is stream
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
