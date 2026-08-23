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

_NVSHMEM_VERSION = "3.2.5"
_UNIQUE_ID_BYTES = 128
_INIT_WITH_UNIQUE_ID = 1 << 3
_SIGNAL_SET = 9
_VERSION_BASE = 1 << 16


class PAPNVSHMEMError(RuntimeError):
    """Raised when the NVSHMEM runtime contract is not satisfied."""


class _UniqueID(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("internal", ctypes.c_char * 124),
    ]


class _UniqueIDArgs(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("unique_id", ctypes.POINTER(_UniqueID)),
        ("rank", ctypes.c_int),
        ("world_size", ctypes.c_int),
    ]


class _InitArgs(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("unique_id_args", _UniqueIDArgs),
        ("content", ctypes.c_char * 96),
    ]


class _InitAttr(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_int),
        ("mpi_comm", ctypes.c_void_p),
        ("args", _InitArgs),
    ]


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
    return _repo_root() / ".local" / "nvshmem-3.2.5-cuda12"


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
        library_path = prefix / "lib" / "libnvshmem_host.so.3"
        if not library_path.is_file():
            raise PAPNVSHMEMError(f"NVSHMEM library is missing: {library_path}")
        self.library = ctypes.CDLL(
            str(library_path),
            mode=ctypes.RTLD_GLOBAL,
        )
        self._bind()

    def _bind(self) -> None:
        library = self.library
        library.nvshmemx_get_uniqueid.argtypes = [ctypes.POINTER(_UniqueID)]
        library.nvshmemx_get_uniqueid.restype = ctypes.c_int
        library.nvshmemx_set_attr_uniqueid_args.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_UniqueID),
            ctypes.POINTER(_InitAttr),
        ]
        library.nvshmemx_set_attr_uniqueid_args.restype = ctypes.c_int
        device = self.device_library
        device.pap_nvshmem_device_bridge_version.argtypes = []
        device.pap_nvshmem_device_bridge_version.restype = ctypes.c_int
        if int(device.pap_nvshmem_device_bridge_version()) != 3:
            raise PAPNVSHMEMError("PAP NVSHMEM GPU graph bridge version mismatch")
        device.pap_nvshmem_device_bridge_init.argtypes = [
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_device_bridge_init.restype = ctypes.c_int
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
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_graph_put_signal.restype = ctypes.c_int
        device.pap_nvshmem_graph_dispatch_qkv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_graph_dispatch_qkv.restype = ctypes.c_int
        device.pap_nvshmem_graph_gather_output.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        device.pap_nvshmem_graph_gather_output.restype = ctypes.c_int


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
        unique_id = _UniqueID()
        status = self._bindings.library.nvshmemx_get_uniqueid(ctypes.byref(unique_id))
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

            torch.accelerator.set_device_index(device_index)
            uid = _UniqueID.from_buffer_copy(unique_id)
            uid_args = _UniqueIDArgs(
                version=_VERSION_BASE + ctypes.sizeof(_UniqueIDArgs),
                unique_id=ctypes.pointer(uid),
                rank=rank,
                world_size=world_size,
            )
            init_args = _InitArgs(
                version=_VERSION_BASE + ctypes.sizeof(_InitArgs),
                unique_id_args=uid_args,
            )
            attr = _InitAttr(
                version=_VERSION_BASE + ctypes.sizeof(_InitAttr),
                mpi_comm=None,
                args=init_args,
            )
            status = self._bindings.library.nvshmemx_set_attr_uniqueid_args(
                rank,
                world_size,
                ctypes.byref(uid),
                ctypes.byref(attr),
            )
            if status != 0:
                raise PAPNVSHMEMError(f"NVSHMEM UID attributes failed: {status}")
            device_library = self._bindings.device_library
            status = device_library.pap_nvshmem_device_bridge_init(
                _INIT_WITH_UNIQUE_ID,
                ctypes.byref(attr),
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


assert ctypes.sizeof(_UniqueID) == 128
assert ctypes.sizeof(_UniqueIDArgs) == 24
assert ctypes.sizeof(_InitArgs) == 128
assert ctypes.sizeof(_InitAttr) == 144
