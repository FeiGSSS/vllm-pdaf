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
        device_index = torch.cuda.current_device()
    if getattr(_thread_state, "device_index", None) == device_index:
        return

    runtime = _runtime()
    result = runtime.cudaSetDevice(device_index)
    if result[0] != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice failed: {result[0]}")
    _thread_state.device_index = device_index


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


def stream_wait_value32(
    signal: torch.Tensor,
    index: int,
    value: int,
    stream: torch.cuda.Stream,
) -> None:
    """Make ``stream`` wait until ``signal[index]`` reaches ``value``."""

    driver = _driver()
    address = _signal_address(signal, index)
    with torch.cuda.device(stream.device):
        _ensure_cuda_context(torch.device(stream.device))
        result = driver.cuStreamWaitValue32(
            driver.CUstream(stream.cuda_stream),
            driver.CUdeviceptr(address),
            int(value) & 0xFFFFFFFF,
            driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
        )
    _check(result, "cuStreamWaitValue32")


def stream_write_value32(
    signal: torch.Tensor,
    index: int,
    value: int,
    stream: torch.cuda.Stream,
) -> None:
    """Write ``value`` to ``signal[index]`` in ``stream`` order."""

    driver = _driver()
    address = _signal_address(signal, index)
    with torch.cuda.device(stream.device):
        _ensure_cuda_context(torch.device(stream.device))
        result = driver.cuStreamWriteValue32(
            driver.CUstream(stream.cuda_stream),
            driver.CUdeviceptr(address),
            int(value) & 0xFFFFFFFF,
            driver.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    _check(result, "cuStreamWriteValue32")


def probe_stream_mem_ops(device: torch.device) -> bool:
    """Return whether 32-bit stream wait/write operations work on ``device``."""

    try:
        with torch.cuda.device(device):
            signal = torch.zeros(1, dtype=torch.int32, device=device)
            stream = torch.cuda.current_stream(device)
            stream_write_value32(signal, 0, 1, stream)
            stream_wait_value32(signal, 0, 1, stream)
            stream.synchronize()
        return True
    except (ImportError, RuntimeError):
        return False
