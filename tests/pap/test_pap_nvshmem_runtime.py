# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.pap.cuda_stream_memops import cuda_stream_handle
from vllm.pap.protocol import PAPOffloadExecBatchDescriptor
from vllm.pap.transport.nvshmem.protocol import decode_step_plan, encode_step_plan
from vllm.pap.transport.nvshmem.runtime import (
    PAPNVSHMEMAllocation,
    PAPNVSHMEMError,
    PAPNVSHMEMRuntime,
)
from vllm.pap.transport.nvshmem.world import PAPNVSHMEMWorld, PAPNVSHMEMWorldConfig


def test_nvshmem_allocation_checks_pointer_offsets() -> None:
    allocation = PAPNVSHMEMAllocation(
        pointer=1000,
        num_bytes=64,
        tensor=torch.empty(64, dtype=torch.uint8),
    )

    assert allocation.pointer_at() == 1000
    assert allocation.pointer_at(64) == 1064
    with pytest.raises(PAPNVSHMEMError, match="out of range"):
        allocation.pointer_at(65)


def test_nvshmem_world_config_reads_launcher_metadata(monkeypatch) -> None:
    monkeypatch.setenv("PAP_NVSHMEM_RANK", "3")
    monkeypatch.setenv("PAP_NVSHMEM_WORLD_SIZE", "8")
    monkeypatch.setenv("PAP_NVSHMEM_UID_FILE", "/tmp/pap-test.uid")

    config = PAPNVSHMEMWorldConfig.from_env(
        device_index=0,
        buffer_bytes=4096,
    )

    assert config.rank == 3
    assert config.world_size == 8
    assert config.device_index == 0
    assert config.buffer_bytes == 4096
    assert config.control_bytes == 64 * 1024
    assert config.uid_path == Path("/tmp/pap-test.uid")


def test_nvshmem_world_config_requires_absolute_uid_path(monkeypatch) -> None:
    monkeypatch.setenv("PAP_NVSHMEM_RANK", "0")
    monkeypatch.setenv("PAP_NVSHMEM_WORLD_SIZE", "2")
    monkeypatch.setenv("PAP_NVSHMEM_UID_FILE", "relative.uid")

    with pytest.raises(PAPNVSHMEMError, match="must be absolute"):
        PAPNVSHMEMWorldConfig.from_env(device_index=0, buffer_bytes=4096)


def test_nvshmem_world_config_requires_uid_file(monkeypatch) -> None:
    monkeypatch.setenv("PAP_NVSHMEM_RANK", "1")
    monkeypatch.setenv("PAP_NVSHMEM_WORLD_SIZE", "2")
    monkeypatch.delenv("PAP_NVSHMEM_UID_FILE", raising=False)
    with pytest.raises(PAPNVSHMEMError, match="UID file"):
        PAPNVSHMEMWorldConfig.from_env(device_index=0, buffer_bytes=4096)


@pytest.mark.parametrize("timeout", ["nan", "inf", "-1", "0"])
def test_nvshmem_world_rejects_invalid_init_timeout_before_start(monkeypatch, timeout):
    monkeypatch.setenv("PAP_NVSHMEM_RANK", "0")
    monkeypatch.setenv("PAP_NVSHMEM_WORLD_SIZE", "2")
    monkeypatch.setenv("PAP_NVSHMEM_UID_FILE", "/tmp/pap-test.uid")
    monkeypatch.setenv("PAP_NVSHMEM_INIT_TIMEOUT", timeout)
    with pytest.raises(PAPNVSHMEMError, match="timeout|PAP_NVSHMEM_INIT_TIMEOUT"):
        PAPNVSHMEMWorldConfig.from_env(device_index=0, buffer_bytes=4096)


def test_nvshmem_wait_uses_configured_timeout_snapshot(monkeypatch):
    waits: list[float] = []

    def ready(timeout: float) -> bool:
        waits.append(timeout)
        return True

    world = object.__new__(PAPNVSHMEMWorld)
    world.config = SimpleNamespace(init_timeout_s=7.0, rank=0)
    world._ready = SimpleNamespace(wait=ready)
    world._error = None
    world.data = world.control = world.signals = world.graph_signals = object()
    monkeypatch.setenv("PAP_NVSHMEM_INIT_TIMEOUT", "nan")
    world.wait_ready()
    assert waits == [7.0]
    with pytest.raises(PAPNVSHMEMError, match="finite"):
        world.wait_ready(float("nan"))
    assert waits == [7.0]


def test_nvshmem_binary_step_plan_round_trip() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="model.layers.0.self_attn.attn",
        items=(),
        batch_id_suffix="session-a@17,session-b@29",
        metadata_template={
            "r": ("session-a", "session-b"),
            "s": (17, 29),
            "a": (0.125, 0.125),
        },
    )

    encoded = encode_step_plan(
        descriptor,
        dtype=torch.float16,
        qkv_width=6144,
        layer_count=36,
    )
    control = torch.zeros(4096, dtype=torch.uint8)
    control[: len(encoded)].copy_(
        torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    )
    decoded, dtype, qkv_width, layer_count = decode_step_plan(
        control,
        capacity=control.numel(),
    )

    assert decoded.layer_name == descriptor.layer_name
    assert decoded.batch_id_suffix == descriptor.batch_id_suffix
    assert decoded.metadata_template == descriptor.metadata_template
    assert dtype is torch.float16
    assert qkv_width == 6144
    assert layer_count == 36


def test_cuda_stream_handle_accepts_only_cuda_streams() -> None:
    stream = SimpleNamespace(device=torch.device("cuda", 0), cuda_stream=123)
    assert cuda_stream_handle(stream) == 123
    assert cuda_stream_handle(stream, expected_device_index=0) == 123

    with pytest.raises(ValueError, match="requires a CUDA stream"):
        cuda_stream_handle(SimpleNamespace(device=torch.device("cpu"), cuda_stream=123))
    with pytest.raises(TypeError, match="incompatible stream"):
        cuda_stream_handle(SimpleNamespace(device=torch.device("cuda", 0)))
    with pytest.raises(ValueError, match="expected 1"):
        cuda_stream_handle(stream, expected_device_index=1)


def test_nvshmem_runtime_validates_stream_device() -> None:
    runtime = object.__new__(PAPNVSHMEMRuntime)
    runtime.device_index = 0

    with pytest.raises(PAPNVSHMEMError, match="incompatible CUDA stream"):
        runtime._cuda_stream_handle(
            SimpleNamespace(device=torch.device("cuda", 1), cuda_stream=123)
        )


def test_nvshmem_bridge_call_enters_runtime_device(monkeypatch) -> None:
    runtime = object.__new__(PAPNVSHMEMRuntime)
    runtime.device_index = 3
    entered = []

    @contextmanager
    def fake_device_index(device_index):
        entered.append(device_index)
        yield

    monkeypatch.setattr("torch.accelerator.device_index", fake_device_index)

    assert runtime._call_cuda_bridge(lambda value: value + 1, 4) == 5
    assert entered == [3]
