# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA stream-ordered 32-bit memory operations for PAP IPC handoff."""

from __future__ import annotations

import threading

import torch

_thread_state = threading.local()


def _driver():
    from cuda.bindings import driver

    return driver


def _runtime():
    from cuda.bindings import runtime

    return runtime


def _check(result: tuple[object, ...], operation: str) -> None:
    driver = _driver()
    error = result[0]
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")


def _ensure_cuda_context(device: torch.device) -> None:
    """Make PyTorch's primary CUDA context current on this thread."""

    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    if getattr(_thread_state, "device_index", None) == device_index:
        return

    runtime = _runtime()
    result = runtime.cudaSetDevice(device_index)
    if result[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice failed: {result[0]}")
    _thread_state.device_index = device_index


def cuda_stream_handle(
    stream: torch.Stream,
    *,
    expected_device_index: int | None = None,
) -> int:
    """Return a checked CUDA stream handle for low-level driver bridges."""
    device = torch.device(stream.device)
    if device.type != "cuda":
        raise ValueError("PAP CUDA bridge requires a CUDA stream")
    device_index = (
        int(torch.accelerator.current_device_index())
        if device.index is None
        else int(device.index)
    )
    if expected_device_index is not None and device_index != expected_device_index:
        raise ValueError(
            "PAP CUDA stream uses device "
            f"{device_index}, expected {expected_device_index}"
        )
    handle = getattr(stream, "cuda_stream", None)
    if handle is None:
        raise TypeError("PAP CUDA bridge received an incompatible stream")
    return int(handle)


def _stream_device_index(stream: torch.Stream) -> int:
    device = torch.device(stream.device)
    if device.type != "cuda":
        raise ValueError("PAP CUDA stream memory operation requires CUDA")
    if device.index is None:
        return int(torch.accelerator.current_device_index())
    return int(device.index)


def _signal_address(signal: torch.Tensor, index: int) -> int:
    if not signal.is_cuda:
        raise ValueError("CUDA stream memory signal must be on CUDA")
    if signal.dtype is not torch.int32:
        raise ValueError("CUDA stream memory signal must use torch.int32")
    if not signal.is_contiguous():
        raise ValueError("CUDA stream memory signal must be contiguous")
    if index < 0 or index >= signal.numel():
        raise IndexError(f"CUDA stream memory signal index out of range: {index}")
    return int(signal.data_ptr()) + int(index) * signal.element_size()


def _validate_signal_stream_device(
    signal: torch.Tensor,
    stream: torch.Stream,
) -> int:
    stream_device_index = _stream_device_index(stream)
    signal_device_index = signal.device.index
    if signal_device_index is None:
        signal_device_index = int(torch.accelerator.current_device_index())
    if signal_device_index != stream_device_index:
        raise ValueError("PAP CUDA signal and stream must use the same device")
    return stream_device_index


def stream_wait_value32(
    signal: torch.Tensor,
    index: int,
    value: int,
    stream: torch.Stream,
    *,
    flush_remote_writes: bool = False,
) -> None:
    """Make ``stream`` wait until ``signal[index]`` reaches ``value``."""

    driver = _driver()
    address = _signal_address(signal, index)
    stream_device_index = _validate_signal_stream_device(signal, stream)
    with torch.accelerator.device_index(stream_device_index):
        _ensure_cuda_context(torch.device(stream.device))
        flags = driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
        if flush_remote_writes:
            flags = driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_FLUSH
        result = driver.cuStreamWaitValue32(
            driver.CUstream(cuda_stream_handle(stream)),
            driver.CUdeviceptr(address),
            int(value) & 0xFFFFFFFF,
            flags,
        )
    _check(result, "cuStreamWaitValue32")
