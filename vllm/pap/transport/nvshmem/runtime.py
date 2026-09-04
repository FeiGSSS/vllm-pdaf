# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin ctypes bridge for the project-owned NVSHMEM runtime."""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.pap.cuda_stream_memops import cuda_stream_handle, stream_wait_value32

_NVSHMEM_VERSION = "3.3.24"
_UNIQUE_ID_BYTES = 128
_SIGNAL_SET = 9


class PAPNVSHMEMError(RuntimeError):
    """Raised when the NVSHMEM runtime contract is not satisfied."""


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(_DLManagedTensor),
)
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

_LIBC = ctypes.CDLL(None)
_LIBC.malloc.argtypes = [ctypes.c_size_t]
_LIBC.malloc.restype = ctypes.c_void_p
_LIBC.free.argtypes = [ctypes.c_void_p]
_RELEASE_DLPACK = ctypes.cast(_LIBC.free, _DLManagedTensorDeleter)


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_New.restype = ctypes.py_object


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_prefix() -> Path:
    configured = os.environ.get("PAP_NVSHMEM_PREFIX")
    if configured:
        return Path(configured)
    return _repo_root() / ".local" / "nvshmem-3.3.24-cuda13"


def _tensor_from_device_pointer(
    pointer: int,
    num_bytes: int,
    device_index: int,
) -> torch.Tensor:
    """Create a non-owning uint8 CUDA tensor for an NVSHMEM allocation."""
    if pointer <= 0 or num_bytes <= 0:
        raise PAPNVSHMEMError("NVSHMEM tensor view requires a valid allocation")
    managed_bytes = ctypes.sizeof(_DLManagedTensor)
    owner = int(_LIBC.malloc(managed_bytes + ctypes.sizeof(ctypes.c_int64)) or 0)
    if owner == 0:
        raise PAPNVSHMEMError("failed to allocate the DLPack tensor descriptor")
    managed = ctypes.cast(owner, ctypes.POINTER(_DLManagedTensor))
    shape = ctypes.cast(
        owner + managed_bytes,
        ctypes.POINTER(ctypes.c_int64),
    )
    shape[0] = num_bytes
    managed[0] = _DLManagedTensor(
        dl_tensor=_DLTensor(
            data=ctypes.c_void_p(pointer),
            device=_DLDevice(device_type=2, device_id=device_index),
            ndim=1,
            dtype=_DLDataType(code=1, bits=8, lanes=1),
            shape=shape,
            strides=None,
            byte_offset=0,
        ),
        manager_ctx=None,
        deleter=_RELEASE_DLPACK,
    )
    capsule = _PyCapsule_New(owner, b"dltensor", None)
    try:
        tensor = torch.utils.dlpack.from_dlpack(capsule)
    except BaseException:
        _LIBC.free(owner)
        raise
    if tensor.device != torch.device("cuda", device_index):
        raise PAPNVSHMEMError("NVSHMEM DLPack tensor uses the wrong CUDA device")
    if tensor.dtype is not torch.uint8 or tensor.numel() != num_bytes:
        raise PAPNVSHMEMError("NVSHMEM DLPack tensor has an invalid layout")
    return tensor


class _NVSHMEMBindings:
    """Typed function table for the NVSHMEM Graph bridge."""

    def __init__(self, prefix: Path) -> None:
        device_library_path = prefix / "lib" / "libpap_nvshmem_device.so"
        if not device_library_path.is_file():
            raise PAPNVSHMEMError(
                f"PAP NVSHMEM Graph bridge is missing: {device_library_path}"
            )
        self.device_library = ctypes.CDLL(
            str(device_library_path),
            mode=ctypes.RTLD_GLOBAL,
        )
        self._bind()

    def _bind(self) -> None:
        device = self.device_library
        device.pap_nvshmem_device_bridge_version.argtypes = []
        device.pap_nvshmem_device_bridge_version.restype = ctypes.c_int
        if int(device.pap_nvshmem_device_bridge_version()) != 10:
            raise PAPNVSHMEMError("PAP NVSHMEM GPU graph bridge version mismatch")
        device.pap_cuda_host_get_device_pointer.argtypes = [ctypes.c_void_p]
        device.pap_cuda_host_get_device_pointer.restype = ctypes.c_void_p
        device.pap_cuda_graph_probe_device_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        device.pap_cuda_graph_probe_device_launch.restype = ctypes.c_int
        device.pap_cuda_graph_create_device_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        device.pap_cuda_graph_create_device_launch.restype = ctypes.c_int
        device.pap_cuda_graph_create_resident_dispatcher.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        device.pap_cuda_graph_create_resident_dispatcher.restype = ctypes.c_int
        device.pap_cuda_graph_resident_run.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        device.pap_cuda_graph_resident_run.restype = ctypes.c_int
        device.pap_cuda_graph_destroy_resident_dispatcher.argtypes = [ctypes.c_void_p]
        device.pap_cuda_graph_destroy_resident_dispatcher.restype = ctypes.c_int
        device.pap_cuda_graph_launch_from_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        device.pap_cuda_graph_launch_from_device.restype = ctypes.c_int
        device.pap_cuda_graph_destroy_device_launch.argtypes = [ctypes.c_void_p]
        device.pap_cuda_graph_destroy_device_launch.restype = ctypes.c_int
        device.pap_nvshmem_device_bridge_get_unique_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        device.pap_nvshmem_device_bridge_get_unique_id.restype = ctypes.c_int
        device.pap_nvshmem_device_bridge_init_uid.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
        ]
        device.pap_nvshmem_device_bridge_init_uid.restype = ctypes.c_int
        device.pap_nvshmem_device_bridge_finalize.argtypes = []
        device.pap_nvshmem_device_bridge_finalize.restype = None
        device.pap_nvshmem_device_bridge_my_pe.argtypes = []
        device.pap_nvshmem_device_bridge_my_pe.restype = ctypes.c_int
        device.pap_nvshmem_device_bridge_n_pes.argtypes = []
        device.pap_nvshmem_device_bridge_n_pes.restype = ctypes.c_int
        device.pap_nvshmem_device_bridge_malloc.argtypes = [ctypes.c_size_t]
        device.pap_nvshmem_device_bridge_malloc.restype = ctypes.c_void_p
        device.pap_nvshmem_device_bridge_free.argtypes = [ctypes.c_void_p]
        device.pap_nvshmem_device_bridge_free.restype = None
        device.pap_nvshmem_device_bridge_barrier.argtypes = []
        device.pap_nvshmem_device_bridge_barrier.restype = None
        device.pap_nvshmem_device_bridge_put_signal_on_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_device_bridge_put_signal_on_stream.restype = None
        device.pap_nvshmem_device_bridge_signal_on_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_device_bridge_signal_on_stream.restype = None
        device.pap_nvshmem_graph_advance_epoch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_graph_advance_epoch.restype = ctypes.c_int
        device.pap_nvshmem_graph_wait_signal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_graph_wait_signal.restype = ctypes.c_int
        device.pap_nvshmem_graph_put_signal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_graph_put_signal.restype = ctypes.c_int
        device.pap_nvshmem_graph_dispatch_qkv.argtypes = [
            ctypes.c_void_p,  # symmetric_data
            ctypes.c_size_t,  # data_slot_bytes
            ctypes.c_void_p,  # source
            ctypes.c_void_p,  # packed
            ctypes.c_void_p,  # route_indices
            ctypes.c_void_p,  # route_counts
            ctypes.c_void_p,  # peer_ranks
            ctypes.c_int,  # peer_count
            ctypes.c_int,  # batch_rows
            ctypes.c_int,  # row_bytes
            ctypes.c_void_p,  # signals
            ctypes.c_void_p,  # epochs
            ctypes.c_int,  # world_size
            ctypes.c_int,  # local_rank
            ctypes.c_int,  # layer_count
            ctypes.c_int,  # layer_index
            ctypes.c_void_p,  # trace_start_ns
            ctypes.c_void_p,  # trace_step_ids
            ctypes.c_void_p,  # trace_route_counts
            ctypes.c_void_p,  # trace_peer_epochs
            ctypes.c_void_p,  # trace_step_counter
            ctypes.c_void_p,  # trace_current_step
            ctypes.c_void_p,  # trace_host_completion
            ctypes.c_int,  # trace_steps
            ctypes.c_int,  # trace_layers
            ctypes.c_void_p,  # stream
        ]
        device.pap_nvshmem_graph_dispatch_qkv.restype = ctypes.c_int
        device.pap_nvshmem_graph_gather_output.argtypes = [
            ctypes.c_void_p,  # symmetric_data
            ctypes.c_size_t,  # data_slot_bytes
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # route_indices
            ctypes.c_void_p,  # route_counts
            ctypes.c_void_p,  # peer_ranks
            ctypes.c_int,  # peer_count
            ctypes.c_int,  # batch_rows
            ctypes.c_int,  # row_bytes
            ctypes.c_void_p,  # signals
            ctypes.c_void_p,  # epochs
            ctypes.c_int,  # world_size
            ctypes.c_int,  # layer_count
            ctypes.c_int,  # layer_index
            ctypes.c_void_p,  # trace_end_ns
            ctypes.c_void_p,  # trace_current_step
            ctypes.c_int,  # trace_steps
            ctypes.c_int,  # trace_layers
            ctypes.c_void_p,  # stream
        ]
        device.pap_nvshmem_graph_gather_output.restype = ctypes.c_int
        device.pap_trace_projection_dispatch_done.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_trace_projection_dispatch_done.restype = ctypes.c_int
        device.pap_trace_projection_gather_done.argtypes = [
            *([ctypes.c_void_p] * 16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_trace_projection_gather_done.restype = ctypes.c_int
        device.pap_trace_attention_marker.argtypes = [
            *([ctypes.c_void_p] * 12),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_trace_attention_marker.restype = ctypes.c_int


@dataclass(frozen=True)
class PAPNVSHMEMAllocation:
    """One process-lifetime allocation from the symmetric heap."""

    pointer: int
    num_bytes: int
    tensor: torch.Tensor
    int32_tensor: torch.Tensor | None = None

    def pointer_at(self, offset: int = 0) -> int:
        """Return a checked byte offset into the symmetric allocation."""
        if offset < 0 or offset > self.num_bytes:
            raise PAPNVSHMEMError("NVSHMEM allocation offset is out of range")
        return self.pointer + offset


class PAPNVSHMEMRuntime:
    """Process-global NVSHMEM lifecycle and stream operation owner."""

    def __init__(self, prefix: str | Path | None = None) -> None:
        self.prefix = Path(prefix) if prefix is not None else _default_prefix()
        self._bindings = _NVSHMEMBindings(self.prefix)
        self._lock = threading.Lock()
        self._initialized = False
        self._finalized = False
        self._allocations: list[PAPNVSHMEMAllocation] = []
        self.rank = -1
        self.world_size = -1
        self.device_index = -1

    def get_unique_id(self) -> bytes:
        """Create the root bootstrap identifier before collective init."""
        unique_id = (ctypes.c_ubyte * _UNIQUE_ID_BYTES)()
        status = self._bindings.device_library.pap_nvshmem_device_bridge_get_unique_id(
            ctypes.byref(unique_id),
            len(unique_id),
        )
        if status != 0:
            raise PAPNVSHMEMError(f"NVSHMEM unique ID failed: {status}")
        return bytes(unique_id)

    def initialize_uid(
        self,
        *,
        unique_id: bytes,
        rank: int,
        world_size: int,
        device_index: int,
    ) -> None:
        """Collectively initialize one PE using control-plane UID metadata."""
        if len(unique_id) != _UNIQUE_ID_BYTES:
            raise PAPNVSHMEMError("NVSHMEM unique ID must be 128 bytes")
        if rank < 0 or rank >= world_size or world_size <= 0:
            raise PAPNVSHMEMError("NVSHMEM rank configuration is invalid")
        with self._lock:
            if self._initialized:
                if (self.rank, self.world_size, self.device_index) != (
                    rank,
                    world_size,
                    device_index,
                ):
                    raise PAPNVSHMEMError("NVSHMEM runtime was already initialized")
                return
            if self._finalized:
                raise PAPNVSHMEMError("NVSHMEM runtime was already finalized")

            device_library = self._bindings.device_library
            uid = (ctypes.c_ubyte * _UNIQUE_ID_BYTES).from_buffer_copy(unique_id)
            status = device_library.pap_nvshmem_device_bridge_init_uid(
                ctypes.byref(uid),
                len(uid),
                rank,
                world_size,
            )
            if status != 0:
                raise PAPNVSHMEMError(
                    f"PAP NVSHMEM Graph bridge initialization failed: {status}"
                )
            actual_rank = int(device_library.pap_nvshmem_device_bridge_my_pe())
            actual_world_size = int(device_library.pap_nvshmem_device_bridge_n_pes())
            if (actual_rank, actual_world_size) != (rank, world_size):
                raise PAPNVSHMEMError(
                    "NVSHMEM initialized with unexpected PE coordinates "
                    f"expected=({rank}, {world_size}) "
                    f"actual=({actual_rank}, {actual_world_size})"
                )
            self.rank = rank
            self.world_size = world_size
            self.device_index = device_index
            self._initialized = True

    def allocate(self, num_bytes: int) -> PAPNVSHMEMAllocation:
        """Collectively allocate and expose one symmetric CUDA byte tensor."""
        self._require_initialized()
        if num_bytes <= 0:
            raise PAPNVSHMEMError("NVSHMEM allocation size must be positive")
        pointer = int(
            self._bindings.device_library.pap_nvshmem_device_bridge_malloc(num_bytes)
            or 0
        )
        if pointer == 0:
            raise PAPNVSHMEMError(f"NVSHMEM allocation failed: {num_bytes} bytes")
        tensor = _tensor_from_device_pointer(
            pointer,
            num_bytes,
            self.device_index,
        )
        int32_tensor = tensor.view(torch.int32) if num_bytes % 4 == 0 else None
        allocation = PAPNVSHMEMAllocation(
            pointer,
            num_bytes,
            tensor,
            int32_tensor,
        )
        self._allocations.append(allocation)
        return allocation

    def host_device_pointer(self, tensor: torch.Tensor) -> int:
        """Return the CUDA UVA address for a pinned host tensor."""
        if tensor.device.type != "cpu" or not tensor.is_pinned():
            raise PAPNVSHMEMError("PAP trace host tensors must use pinned memory")
        pointer = int(
            self._bindings.device_library.pap_cuda_host_get_device_pointer(
                ctypes.c_void_p(tensor.data_ptr())
            )
            or 0
        )
        if pointer == 0:
            raise PAPNVSHMEMError("failed to map a PAP trace host tensor")
        return pointer

    def probe_device_graph_launch(
        self,
        *,
        graph_handle: int,
        stream: torch.Stream,
    ) -> None:
        """Validate that a captured graph supports CUDA device launch."""
        self._require_initialized()
        if graph_handle <= 0:
            raise PAPNVSHMEMError("PAP CUDA Graph handle is invalid")
        result = ctypes.c_int(-1)
        node_type = ctypes.c_int(-1)
        node_name = ctypes.create_string_buffer(512)
        status = self._call_cuda_bridge(
            self._graph_library().pap_cuda_graph_probe_device_launch,
            ctypes.c_void_p(graph_handle),
            ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ctypes.byref(result),
            ctypes.byref(node_type),
            ctypes.cast(node_name, ctypes.c_void_p),
            ctypes.sizeof(node_name),
        )
        if int(status) != 0:
            name = node_name.value.decode("utf-8", errors="replace") or "unknown"
            raise PAPNVSHMEMError(
                "PAP CUDA Graph device-launch probe failed: "
                f"CUDA error {status}, instantiate result {result.value}, "
                f"node type {node_type.value}, node name {name}"
            )

    def create_device_graph_launch(
        self,
        *,
        graph_handle: int,
        stream: torch.Stream,
    ) -> int:
        """Instantiate and upload a graph for launch from a GPU kernel."""
        self._require_initialized()
        executable = ctypes.c_void_p()
        result = ctypes.c_int(-1)
        node_type = ctypes.c_int(-1)
        node_name = ctypes.create_string_buffer(512)
        status = self._call_cuda_bridge(
            self._graph_library().pap_cuda_graph_create_device_launch,
            ctypes.c_void_p(graph_handle),
            ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ctypes.byref(executable),
            ctypes.byref(result),
            ctypes.byref(node_type),
            ctypes.cast(node_name, ctypes.c_void_p),
            ctypes.sizeof(node_name),
        )
        if int(status) != 0 or executable.value is None:
            name = node_name.value.decode("utf-8", errors="replace") or "unknown"
            raise PAPNVSHMEMError(
                "PAP CUDA Graph device-launch creation failed: "
                f"CUDA error {status}, instantiate result {result.value}, "
                f"node type {node_type.value}, node name {name}"
            )
        return int(executable.value)

    def create_resident_graph_dispatcher(
        self,
        *,
        stream: torch.Stream,
        window_size: int,
    ) -> int:
        """Create a hardware-wait dispatcher with a prequeued launch window."""
        self._require_initialized()
        if window_size <= 0:
            raise PAPNVSHMEMError("PAP resident dispatch window must be positive")
        dispatcher = ctypes.c_void_p()
        status = self._call_cuda_bridge(
            self._graph_library().pap_cuda_graph_create_resident_dispatcher,
            ctypes.c_void_p(self._cuda_stream_handle(stream)),
            window_size,
            ctypes.byref(dispatcher),
        )
        if int(status) != 0 or dispatcher.value is None:
            raise PAPNVSHMEMError(
                f"PAP resident graph dispatcher creation failed: CUDA error {status}"
            )
        return int(dispatcher.value)

    def run_resident_device_graph(
        self,
        *,
        dispatcher_handle: int,
        executable_handle: int,
        generation: int,
    ) -> None:
        """Publish one dynamic graph choice and await GPU completion."""
        self._check_graph_launch(
            "resident_dispatch",
            self._call_cuda_bridge(
                self._graph_library().pap_cuda_graph_resident_run,
                ctypes.c_void_p(dispatcher_handle),
                ctypes.c_void_p(executable_handle),
                generation,
            ),
        )

    def destroy_resident_graph_dispatcher(self, dispatcher_handle: int) -> None:
        """Stop a resident dispatcher and release its mapped state."""
        self._check_graph_launch(
            "destroy_resident_dispatcher",
            self._call_cuda_bridge(
                self._graph_library().pap_cuda_graph_destroy_resident_dispatcher,
                ctypes.c_void_p(dispatcher_handle),
            ),
        )

    def launch_device_graph(
        self,
        *,
        executable_handle: int,
        stream: torch.Stream,
    ) -> None:
        """Launch one device-launch executable from a GPU kernel."""
        self._check_graph_launch(
            "launch_from_device",
            self._call_cuda_bridge(
                self._graph_library().pap_cuda_graph_launch_from_device,
                ctypes.c_void_p(executable_handle),
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def destroy_device_graph_launch(self, executable_handle: int) -> None:
        """Destroy one device-launch executable after its stream is idle."""
        self._check_graph_launch(
            "destroy_device_launch",
            self._call_cuda_bridge(
                self._graph_library().pap_cuda_graph_destroy_device_launch,
                ctypes.c_void_p(executable_handle),
            ),
        )

    def graph_advance_epoch(
        self,
        *,
        epoch: torch.Tensor,
        stream: torch.Stream,
    ) -> None:
        """Enqueue the device-owned step generation increment."""
        self._require_graph_tensor(epoch)
        self._check_graph_launch(
            "advance_epoch",
            self._call_cuda_bridge(
                self._graph_library().pap_nvshmem_graph_advance_epoch,
                ctypes.c_void_p(epoch.data_ptr()),
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def graph_wait_signal(
        self,
        *,
        signal: PAPNVSHMEMAllocation,
        signal_offset: int,
        epoch: torch.Tensor,
        layer_count: int,
        layer_index: int,
        generation_delta: int,
        stream: torch.Stream,
    ) -> None:
        """Capture a device-side wait for a step/layer generation."""
        self._require_graph_tensor(epoch)
        self._validate_graph_layer(layer_count, layer_index)
        self._validate_range(signal, signal_offset, ctypes.sizeof(ctypes.c_uint64))
        self._check_graph_launch(
            "wait_signal",
            self._call_cuda_bridge(
                self._graph_library().pap_nvshmem_graph_wait_signal,
                ctypes.c_void_p(signal.pointer_at(signal_offset)),
                ctypes.c_void_p(epoch.data_ptr()),
                layer_count,
                layer_index,
                generation_delta,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def graph_put_signal(
        self,
        *,
        destination: PAPNVSHMEMAllocation,
        destination_offset: int,
        source: torch.Tensor,
        num_bytes: int,
        signal: PAPNVSHMEMAllocation,
        signal_offset: int,
        abort_signal_offset: int,
        epoch: torch.Tensor,
        layer_count: int,
        layer_index: int,
        peer: int,
        stream: torch.Stream,
    ) -> None:
        """Capture a device-side remote put followed by a ready signal."""
        self._require_graph_tensor(epoch)
        self._validate_graph_layer(layer_count, layer_index)
        self._validate_peer(peer)
        self._validate_range(destination, destination_offset, num_bytes)
        self._validate_range(signal, signal_offset, ctypes.sizeof(ctypes.c_uint64))
        self._validate_range(
            signal,
            abort_signal_offset,
            ctypes.sizeof(ctypes.c_uint64),
        )
        if (
            source.device != torch.device("cuda", self.device_index)
            or not source.is_contiguous()
            or num_bytes <= 0
            or num_bytes > source.numel() * source.element_size()
        ):
            raise PAPNVSHMEMError(
                "PAP NVSHMEM graph source must be local, contiguous, and non-empty"
            )
        self._check_graph_launch(
            "put_signal",
            self._call_cuda_bridge(
                self._graph_library().pap_nvshmem_graph_put_signal,
                ctypes.c_void_p(destination.pointer_at(destination_offset)),
                ctypes.c_void_p(source.data_ptr()),
                num_bytes,
                ctypes.c_void_p(signal.pointer_at(signal_offset)),
                ctypes.c_void_p(signal.pointer_at(abort_signal_offset)),
                ctypes.c_void_p(epoch.data_ptr()),
                layer_count,
                layer_index,
                peer,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def graph_dispatch_qkv(
        self,
        *,
        data: PAPNVSHMEMAllocation,
        data_slot_bytes: int,
        source: torch.Tensor,
        packed: torch.Tensor,
        route_indices: torch.Tensor,
        route_counts: torch.Tensor,
        peer_ranks: torch.Tensor,
        signals: PAPNVSHMEMAllocation,
        epochs: torch.Tensor,
        layer_count: int,
        layer_index: int,
        trace_start_ns: torch.Tensor | None,
        trace_step_ids: torch.Tensor | None,
        trace_route_counts: torch.Tensor | None,
        trace_peer_epochs: torch.Tensor | None,
        trace_step_counter: torch.Tensor | None,
        trace_current_step: torch.Tensor | None,
        trace_host_completion_pointer: int,
        trace_steps: int,
        trace_layers: int,
        stream: torch.Stream,
    ) -> None:
        """Capture one dynamic multi-peer QKV pack and dispatch kernel."""
        peer_count, batch_rows, row_bytes = self._validate_routed_graph_tensors(
            payload=source,
            route_indices=route_indices,
            route_counts=route_counts,
            peer_ranks=peer_ranks,
            epochs=epochs,
        )
        self._validate_graph_layer(layer_count, layer_index)
        if batch_rows * row_bytes > data_slot_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM routed QKV exceeds data slot")
        if (
            packed.device != torch.device("cuda", self.device_index)
            or not packed.is_contiguous()
        ):
            raise PAPNVSHMEMError("PAP NVSHMEM routed QKV scratch is incompatible")
        if packed.numel() * packed.element_size() < (
            peer_count * batch_rows * row_bytes
        ):
            raise PAPNVSHMEMError("PAP NVSHMEM routed QKV scratch is too small")
        self._check_graph_launch(
            "dispatch_qkv",
            self._call_cuda_bridge(
                self._graph_library().pap_nvshmem_graph_dispatch_qkv,
                ctypes.c_void_p(data.pointer),
                data_slot_bytes,
                ctypes.c_void_p(source.data_ptr()),
                ctypes.c_void_p(packed.data_ptr()),
                ctypes.c_void_p(route_indices.data_ptr()),
                ctypes.c_void_p(route_counts.data_ptr()),
                ctypes.c_void_p(peer_ranks.data_ptr()),
                peer_count,
                batch_rows,
                row_bytes,
                ctypes.c_void_p(signals.pointer),
                ctypes.c_void_p(epochs.data_ptr()),
                self.world_size,
                self.rank,
                layer_count,
                layer_index,
                ctypes.c_void_p(
                    trace_start_ns.data_ptr() if trace_start_ns is not None else 0
                ),
                ctypes.c_void_p(
                    trace_step_ids.data_ptr() if trace_step_ids is not None else 0
                ),
                ctypes.c_void_p(
                    trace_route_counts.data_ptr()
                    if trace_route_counts is not None
                    else 0
                ),
                ctypes.c_void_p(
                    trace_peer_epochs.data_ptr() if trace_peer_epochs is not None else 0
                ),
                ctypes.c_void_p(
                    trace_step_counter.data_ptr()
                    if trace_step_counter is not None
                    else 0
                ),
                ctypes.c_void_p(
                    trace_current_step.data_ptr()
                    if trace_current_step is not None
                    else 0
                ),
                ctypes.c_void_p(trace_host_completion_pointer),
                trace_steps,
                trace_layers,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def graph_gather_output(
        self,
        *,
        data: PAPNVSHMEMAllocation,
        data_slot_bytes: int,
        output: torch.Tensor,
        route_indices: torch.Tensor,
        route_counts: torch.Tensor,
        peer_ranks: torch.Tensor,
        signals: PAPNVSHMEMAllocation,
        epochs: torch.Tensor,
        layer_count: int,
        layer_index: int,
        trace_end_ns: torch.Tensor | None,
        trace_current_step: torch.Tensor | None,
        trace_steps: int,
        trace_layers: int,
        stream: torch.Stream,
    ) -> None:
        """Capture one dynamic multi-peer output barrier and scatter kernel."""
        _peer_count, batch_rows, row_bytes = self._validate_routed_graph_tensors(
            payload=output,
            route_indices=route_indices,
            route_counts=route_counts,
            peer_ranks=peer_ranks,
            epochs=epochs,
        )
        self._validate_graph_layer(layer_count, layer_index)
        if batch_rows * row_bytes > data_slot_bytes:
            raise PAPNVSHMEMError("PAP NVSHMEM routed output exceeds data slot")
        self._check_graph_launch(
            "gather_output",
            self._call_cuda_bridge(
                self._graph_library().pap_nvshmem_graph_gather_output,
                ctypes.c_void_p(data.pointer),
                data_slot_bytes,
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_void_p(route_indices.data_ptr()),
                ctypes.c_void_p(route_counts.data_ptr()),
                ctypes.c_void_p(peer_ranks.data_ptr()),
                int(peer_ranks.numel()),
                int(route_indices.shape[1]),
                int(output.shape[1] * output.element_size()),
                ctypes.c_void_p(signals.pointer),
                ctypes.c_void_p(epochs.data_ptr()),
                self.world_size,
                layer_count,
                layer_index,
                ctypes.c_void_p(
                    trace_end_ns.data_ptr() if trace_end_ns is not None else 0
                ),
                ctypes.c_void_p(
                    trace_current_step.data_ptr()
                    if trace_current_step is not None
                    else 0
                ),
                trace_steps,
                trace_layers,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def trace_projection_dispatch_done(
        self,
        *,
        current_step: torch.Tensor,
        dispatch_done_ns: torch.Tensor,
        trace_steps: int,
        trace_layers: int,
        layer_index: int,
        stream: torch.Stream,
    ) -> None:
        """Record completion of one Projection QKV dispatch kernel."""
        self._check_graph_launch(
            "trace_projection_dispatch_done",
            self._call_cuda_bridge(
                self._graph_library().pap_trace_projection_dispatch_done,
                ctypes.c_void_p(current_step.data_ptr()),
                ctypes.c_void_p(dispatch_done_ns.data_ptr()),
                trace_steps,
                trace_layers,
                layer_index,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def trace_projection_gather_done(
        self,
        *,
        current_step: torch.Tensor,
        start_ns: torch.Tensor,
        end_ns: torch.Tensor,
        step_ids: torch.Tensor,
        route_counts: torch.Tensor,
        peer_epochs: torch.Tensor,
        dispatch_done_ns: torch.Tensor,
        gather_done_ns: torch.Tensor,
        host_start_pointer: int,
        host_end_pointer: int,
        host_step_ids_pointer: int,
        host_route_counts_pointer: int,
        host_peer_epochs_pointer: int,
        host_dispatch_done_pointer: int,
        host_gather_done_pointer: int,
        host_completion_pointer: int,
        trace_steps: int,
        trace_layers: int,
        layer_index: int,
        stream: torch.Stream,
    ) -> None:
        """Record gather completion and mirror a complete Projection step."""
        pointers = (
            current_step.data_ptr(),
            start_ns.data_ptr(),
            end_ns.data_ptr(),
            step_ids.data_ptr(),
            route_counts.data_ptr(),
            peer_epochs.data_ptr(),
            dispatch_done_ns.data_ptr(),
            gather_done_ns.data_ptr(),
            host_start_pointer,
            host_end_pointer,
            host_step_ids_pointer,
            host_route_counts_pointer,
            host_peer_epochs_pointer,
            host_dispatch_done_pointer,
            host_gather_done_pointer,
            host_completion_pointer,
        )
        self._check_graph_launch(
            "trace_projection_gather_done",
            self._call_cuda_bridge(
                self._graph_library().pap_trace_projection_gather_done,
                *(ctypes.c_void_p(pointer) for pointer in pointers),
                trace_steps,
                trace_layers,
                self.world_size,
                layer_index,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def trace_attention_marker(
        self,
        *,
        epoch: torch.Tensor,
        replay_start_ns: torch.Tensor,
        step_start_ns: torch.Tensor,
        start_ns: torch.Tensor,
        end_ns: torch.Tensor,
        step_ids: torch.Tensor,
        host_replay_start_pointer: int,
        host_step_start_pointer: int,
        host_start_pointer: int,
        host_end_pointer: int,
        host_step_ids_pointer: int,
        host_completion_pointer: int,
        trace_steps: int,
        trace_layers: int,
        layer_index: int,
        marker_kind: int,
        stream: torch.Stream,
    ) -> None:
        """Record an Attention-kernel boundary inside the whole-step Graph."""
        pointers = (
            epoch.data_ptr(),
            replay_start_ns.data_ptr(),
            step_start_ns.data_ptr(),
            start_ns.data_ptr(),
            end_ns.data_ptr(),
            step_ids.data_ptr(),
            host_replay_start_pointer,
            host_step_start_pointer,
            host_start_pointer,
            host_end_pointer,
            host_step_ids_pointer,
            host_completion_pointer,
        )
        self._check_graph_launch(
            "trace_attention_marker",
            self._call_cuda_bridge(
                self._graph_library().pap_trace_attention_marker,
                *(ctypes.c_void_p(pointer) for pointer in pointers),
                trace_steps,
                trace_layers,
                layer_index,
                marker_kind,
                ctypes.c_void_p(self._cuda_stream_handle(stream)),
            ),
        )

    def put_signal_on_stream(
        self,
        *,
        destination: PAPNVSHMEMAllocation,
        destination_offset: int,
        source: torch.Tensor,
        num_bytes: int,
        signal: PAPNVSHMEMAllocation,
        signal_offset: int,
        generation: int,
        peer: int,
        stream: torch.Stream,
    ) -> None:
        """Enqueue one remote put and completion signal on a CUDA stream."""
        self._require_initialized()
        if source.device != torch.device("cuda", self.device_index):
            raise PAPNVSHMEMError("NVSHMEM source uses the wrong CUDA device")
        if not source.is_contiguous() or num_bytes <= 0:
            raise PAPNVSHMEMError("NVSHMEM source must be non-empty and contiguous")
        if num_bytes > source.numel() * source.element_size():
            raise PAPNVSHMEMError("NVSHMEM put exceeds the source tensor")
        self._validate_peer(peer)
        self._validate_range(destination, destination_offset, num_bytes)
        self._validate_range(signal, signal_offset, ctypes.sizeof(ctypes.c_uint64))
        self._call_cuda_bridge(
            self._bindings.device_library.pap_nvshmem_device_bridge_put_signal_on_stream,
            ctypes.c_void_p(destination.pointer_at(destination_offset)),
            ctypes.c_void_p(source.data_ptr()),
            num_bytes,
            ctypes.c_void_p(signal.pointer_at(signal_offset)),
            generation,
            _SIGNAL_SET,
            peer,
            ctypes.c_void_p(self._cuda_stream_handle(stream)),
        )

    def wait_signal_on_stream(
        self,
        *,
        signal: PAPNVSHMEMAllocation,
        signal_offset: int,
        generation: int,
        stream: torch.Stream,
    ) -> None:
        """Wait for a local signal with a CUDA stream memory operation."""
        self._require_initialized()
        self._validate_range(signal, signal_offset, ctypes.sizeof(ctypes.c_uint64))
        if signal_offset % ctypes.sizeof(ctypes.c_uint64) != 0:
            raise PAPNVSHMEMError("NVSHMEM signal offset must be uint64 aligned")
        if generation < 0 or generation >= 1 << 31:
            raise PAPNVSHMEMError("NVSHMEM signal generation exceeds memop range")
        signal_i32 = signal.int32_tensor
        if signal_i32 is None:
            raise PAPNVSHMEMError("NVSHMEM signal allocation is not int32 aligned")
        self._cuda_stream_handle(stream)
        stream_wait_value32(
            signal_i32,
            signal_offset // ctypes.sizeof(ctypes.c_int32),
            generation,
            stream,
            flush_remote_writes=False,
        )

    def signal_on_stream(
        self,
        *,
        signal: PAPNVSHMEMAllocation,
        signal_offset: int,
        generation: int,
        peer: int,
        stream: torch.Stream,
    ) -> None:
        """Enqueue a generation update in a remote symmetric signal."""
        self._require_initialized()
        self._validate_peer(peer)
        self._validate_range(signal, signal_offset, ctypes.sizeof(ctypes.c_uint64))
        self._call_cuda_bridge(
            self._bindings.device_library.pap_nvshmem_device_bridge_signal_on_stream,
            ctypes.c_void_p(signal.pointer_at(signal_offset)),
            generation,
            _SIGNAL_SET,
            peer,
            ctypes.c_void_p(self._cuda_stream_handle(stream)),
        )

    def barrier(self) -> None:
        """Synchronize all PEs outside the per-layer data path."""
        self._require_initialized()
        self._bindings.device_library.pap_nvshmem_device_bridge_barrier()

    def finalize(self) -> None:
        """Collectively release allocations and finalize the runtime."""
        with self._lock:
            if not self._initialized or self._finalized:
                return
            for allocation in reversed(self._allocations):
                self._bindings.device_library.pap_nvshmem_device_bridge_free(
                    ctypes.c_void_p(allocation.pointer)
                )
            self._allocations.clear()
            self._bindings.device_library.pap_nvshmem_device_bridge_finalize()
            self._finalized = True
            self._initialized = False

    def _require_initialized(self) -> None:
        if not self._initialized or self._finalized:
            raise PAPNVSHMEMError("NVSHMEM runtime is not initialized")

    def _validate_peer(self, peer: int) -> None:
        if peer < 0 or peer >= self.world_size or peer == self.rank:
            raise PAPNVSHMEMError(f"invalid NVSHMEM peer: {peer}")

    def _graph_library(self) -> ctypes.CDLL:
        return self._bindings.device_library

    def _cuda_stream_handle(self, stream: torch.Stream) -> int:
        try:
            return cuda_stream_handle(
                stream,
                expected_device_index=self.device_index,
            )
        except (TypeError, ValueError) as exc:
            raise PAPNVSHMEMError(
                "NVSHMEM received an incompatible CUDA stream"
            ) from exc

    def _call_cuda_bridge(self, function: Any, *args: object) -> Any:
        with torch.accelerator.device_index(self.device_index):
            return function(*args)

    def _require_graph_tensor(self, epoch: torch.Tensor) -> None:
        self._require_initialized()
        if (
            epoch.device != torch.device("cuda", self.device_index)
            or epoch.dtype is not torch.uint64
            or epoch.numel() != 1
            or not epoch.is_contiguous()
        ):
            raise PAPNVSHMEMError("PAP NVSHMEM graph epoch tensor is incompatible")

    def _validate_routed_graph_tensors(
        self,
        *,
        payload: torch.Tensor,
        route_indices: torch.Tensor,
        route_counts: torch.Tensor,
        peer_ranks: torch.Tensor,
        epochs: torch.Tensor,
    ) -> tuple[int, int, int]:
        self._require_initialized()
        expected_device = torch.device("cuda", self.device_index)
        tensors = (payload, route_indices, route_counts, peer_ranks, epochs)
        if any(tensor.device != expected_device for tensor in tensors):
            raise PAPNVSHMEMError("PAP NVSHMEM routed graph uses another device")
        if payload.ndim != 2 or not payload.is_contiguous():
            raise PAPNVSHMEMError("PAP NVSHMEM routed payload must be rank two")
        if (
            route_indices.ndim != 2
            or route_indices.dtype is not torch.int64
            or not route_indices.is_contiguous()
        ):
            raise PAPNVSHMEMError("PAP NVSHMEM route indices are incompatible")
        peer_count, batch_rows = map(int, route_indices.shape)
        if batch_rows != int(payload.shape[0]) or peer_count <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM routed graph shape mismatch")
        if (
            tuple(route_counts.shape) != (peer_count,)
            or route_counts.dtype is not torch.int32
            or tuple(peer_ranks.shape) != (peer_count,)
            or peer_ranks.dtype is not torch.int32
            or not route_counts.is_contiguous()
            or not peer_ranks.is_contiguous()
        ):
            raise PAPNVSHMEMError("PAP NVSHMEM route vectors are incompatible")
        if (
            epochs.dtype is not torch.uint64
            or tuple(epochs.shape) != (self.world_size,)
            or not epochs.is_contiguous()
        ):
            raise PAPNVSHMEMError("PAP NVSHMEM routed epochs are incompatible")
        row_bytes = int(payload.shape[1] * payload.element_size())
        if row_bytes <= 0:
            raise PAPNVSHMEMError("PAP NVSHMEM routed row is empty")
        return peer_count, batch_rows, row_bytes

    @staticmethod
    def _validate_graph_layer(layer_count: int, layer_index: int) -> None:
        if layer_count <= 0 or layer_index < 0 or layer_index >= layer_count:
            raise PAPNVSHMEMError("PAP NVSHMEM graph layer is out of range")

    @staticmethod
    def _check_graph_launch(operation: str, status: int) -> None:
        if int(status) != 0:
            raise PAPNVSHMEMError(
                f"PAP NVSHMEM graph {operation} launch failed: CUDA error {status}"
            )

    @staticmethod
    def _validate_range(
        allocation: PAPNVSHMEMAllocation,
        offset: int,
        num_bytes: int,
    ) -> None:
        if offset < 0 or num_bytes < 0 or offset + num_bytes > allocation.num_bytes:
            raise PAPNVSHMEMError("NVSHMEM allocation range is out of bounds")
