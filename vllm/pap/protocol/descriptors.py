# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stable PAP wire and transport contracts."""

from __future__ import annotations

import base64
import hashlib
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch


class PAPDataPlaneRole(str, Enum):
    PREFILL = "prefill"
    ATTENTION = "attention"
    PROJECTION = "projection"


class PAPDataPlaneChannel(str, Enum):
    OFFLOAD_KV = "offload_kv"
    OFFLOAD_EXEC = "offload_exec"


def pap_offload_exec_trace_id(value: str) -> str:
    """Short stable id for correlating Projection and Attention trace lines."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


class PAPTensorTransport(str, Enum):
    CUDA_IPC = "cuda_ipc"
    NIXL_MAILBOX = "nixl_mailbox"


@dataclass(frozen=True)
class PAPOffloadExecDescriptor:
    """Control metadata for one Projection<->Attention decode attention call."""

    request_id: str
    layer_name: str
    step: int
    scale: float


@dataclass(frozen=True)
class PAPOffloadExecBatchDescriptor:
    """Control metadata for one batched Projection<->Attention attention call."""

    layer_name: str
    items: tuple[PAPOffloadExecDescriptor, ...]
    batch_id_suffix: str | None = None
    metadata_template: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "items", tuple(self.items))
        if self.batch_id_suffix is not None:
            object.__setattr__(self, "batch_id_suffix", str(self.batch_id_suffix))
        if self.metadata_template is not None:
            template = self.metadata_template
            try:
                request_ids = tuple(str(request_id) for request_id in template["r"])
                steps = tuple(int(step) for step in template["s"])
            except KeyError as exc:
                raise ValueError(
                    "PAP OFFLOAD_EXEC metadata template requires r and s"
                ) from exc
            normalized_template: dict[str, Any] = {
                "r": request_ids,
                "s": steps,
            }
            if "a" in template:
                normalized_template["a"] = tuple(
                    float(scale) for scale in template["a"]
                )
            if "t" in template:
                raise ValueError(
                    "PAP OFFLOAD_EXEC decode-token metadata was removed; "
                    "use asynchronous decode-token delivery"
                )
            lengths = {len(request_ids), len(steps)}
            if "a" in normalized_template:
                lengths.add(len(normalized_template["a"]))
            if len(lengths) != 1:
                raise ValueError("PAP OFFLOAD_EXEC metadata template length mismatch")
            if not request_ids:
                raise ValueError(
                    "PAP OFFLOAD_EXEC metadata template requires at least one item"
                )
            object.__setattr__(self, "metadata_template", normalized_template)
        if not self.items:
            if self.metadata_template is None:
                raise ValueError("PAP OFFLOAD_EXEC batch requires at least one item")
            if "a" not in self.metadata_template:
                raise ValueError("template-only PAP OFFLOAD_EXEC batch requires scales")
            return
        if self.metadata_template is not None and len(
            self.metadata_template["r"]
        ) != len(self.items):
            raise ValueError("PAP OFFLOAD_EXEC batch template length mismatch")
        for item in self.items:
            if item.layer_name != self.layer_name:
                raise ValueError("all PAP OFFLOAD_EXEC batch items must share layer")

    @property
    def item_count(self) -> int:
        if self.items:
            return len(self.items)
        if self.metadata_template is None:
            return 0
        return len(self.metadata_template["r"])

    @property
    def batch_id(self) -> str:
        if self.batch_id_suffix is not None:
            return f"{self.layer_name}#{self.batch_id_suffix}"
        entries = ",".join(f"{item.request_id}@{item.step}" for item in self.items)
        return f"{self.layer_name}#{entries}"

    @property
    def qkv_tensor_id(self) -> str:
        return f"{self.batch_id}#qkv_batch"

    @property
    def output_tensor_id(self) -> str:
        return f"{self.batch_id}#attn_out_batch"


@dataclass(frozen=True)
class PAPCudaIPCTensorHandle:
    """Serializable CUDA IPC tensor metadata for one PAP OFFLOAD_KV tensor."""

    dtype: str
    shape: tuple[int, ...]
    ipc_handle: dict[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if any(dim < 0 for dim in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if not self.ipc_handle:
            raise ValueError("ipc_handle must not be empty")
        object.__setattr__(
            self,
            "ipc_handle",
            {
                str(gpu_uuid): tuple(ipc_args)
                for gpu_uuid, ipc_args in self.ipc_handle.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "ipc_handle_pickled": base64.b64encode(
                pickle.dumps(self.ipc_handle)
            ).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PAPCudaIPCTensorHandle:
        ipc_handle = data.get("ipc_handle")
        if ipc_handle is None:
            ipc_handle = pickle.loads(
                base64.b64decode(str(data["ipc_handle_pickled"]).encode("ascii"))
            )
        return cls(
            dtype=str(data["dtype"]),
            shape=tuple(int(dim) for dim in data["shape"]),
            ipc_handle={
                str(gpu_uuid): tuple(ipc_args)
                for gpu_uuid, ipc_args in ipc_handle.items()
            },
        )


@dataclass(frozen=True)
class PAPPrefillKVCacheCatalogDescriptor:
    """Process-lifetime CUDA IPC metadata for one Prefill KV-cache layer."""

    catalog_id: str
    layer_name: str
    block_size: int
    num_kv_heads: int
    layout: str
    kv_cache: PAPCudaIPCTensorHandle

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog_id", str(self.catalog_id))
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "num_kv_heads", int(self.num_kv_heads))
        object.__setattr__(self, "layout", str(self.layout))
        if not self.catalog_id:
            raise ValueError("catalog_id must not be empty")
        if not self.layer_name:
            raise ValueError("layer_name must not be empty")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.layout not in {"NHD", "HND"}:
            raise ValueError(f"unsupported KV cache layout: {self.layout}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "layer_name": self.layer_name,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "layout": self.layout,
            "kv_cache": self.kv_cache.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PAPPrefillKVCacheCatalogDescriptor:
        return cls(
            catalog_id=str(data["catalog_id"]),
            layer_name=str(data["layer_name"]),
            block_size=int(data["block_size"]),
            num_kv_heads=int(data["num_kv_heads"]),
            layout=str(data["layout"]),
            kv_cache=PAPCudaIPCTensorHandle.from_dict(data["kv_cache"]),
        )


@dataclass(frozen=True)
class PAPPrefillKVSessionManifest:
    """Request-level block layout published after a Prefill chunk."""

    request_id: str
    session_handle: str
    catalog_id: str
    prefix_len: int
    block_ids: tuple[int, ...]
    block_size: int
    expected_layer_count: int
    lease_id: str
    leased_block_ids: tuple[int, ...]
    lease_capacity_tokens: int
    writable_start_token: int
    writable_end_token: int
    ready_event_handle: bytes | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "session_handle", str(self.session_handle))
        object.__setattr__(self, "catalog_id", str(self.catalog_id))
        object.__setattr__(self, "prefix_len", int(self.prefix_len))
        object.__setattr__(
            self, "block_ids", tuple(int(block_id) for block_id in self.block_ids)
        )
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "expected_layer_count", int(self.expected_layer_count))
        object.__setattr__(self, "lease_id", str(self.lease_id))
        object.__setattr__(
            self,
            "leased_block_ids",
            tuple(int(block_id) for block_id in self.leased_block_ids),
        )
        object.__setattr__(
            self, "lease_capacity_tokens", int(self.lease_capacity_tokens)
        )
        object.__setattr__(self, "writable_start_token", int(self.writable_start_token))
        object.__setattr__(self, "writable_end_token", int(self.writable_end_token))
        if self.ready_event_handle is not None:
            object.__setattr__(
                self,
                "ready_event_handle",
                bytes(self.ready_event_handle),
            )
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.session_handle:
            raise ValueError("session_handle must not be empty")
        if not self.catalog_id:
            raise ValueError("catalog_id must not be empty")
        if self.prefix_len <= 0:
            raise ValueError("prefix_len must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.expected_layer_count <= 0:
            raise ValueError("expected_layer_count must be positive")
        if not self.lease_id:
            raise ValueError("lease_id must not be empty")
        if not self.leased_block_ids:
            raise ValueError("leased_block_ids must not be empty")
        if self.writable_start_token != self.prefix_len:
            raise ValueError("writable_start_token must equal prefix_len")
        if self.writable_end_token < self.writable_start_token:
            raise ValueError("writable_end_token must cover prefix_len")
        if self.lease_capacity_tokens < self.writable_end_token:
            raise ValueError("lease capacity must cover writable_end_token")
        required_blocks = (
            self.writable_end_token + self.block_size - 1
        ) // self.block_size
        if len(self.block_ids) < required_blocks:
            raise ValueError("block_ids must cover writable_end_token")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_handle": self.session_handle,
            "catalog_id": self.catalog_id,
            "prefix_len": self.prefix_len,
            "block_ids": list(self.block_ids),
            "block_size": self.block_size,
            "expected_layer_count": self.expected_layer_count,
            "lease_id": self.lease_id,
            "leased_block_ids": list(self.leased_block_ids),
            "lease_capacity_tokens": self.lease_capacity_tokens,
            "writable_start_token": self.writable_start_token,
            "writable_end_token": self.writable_end_token,
            "ready_event_handle": (
                None
                if self.ready_event_handle is None
                else base64.b64encode(self.ready_event_handle).decode("ascii")
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PAPPrefillKVSessionManifest:
        event_handle = data.get("ready_event_handle")
        return cls(
            request_id=str(data["request_id"]),
            session_handle=str(data["session_handle"]),
            catalog_id=str(data["catalog_id"]),
            prefix_len=int(data["prefix_len"]),
            block_ids=tuple(int(block_id) for block_id in data["block_ids"]),
            block_size=int(data["block_size"]),
            expected_layer_count=int(data["expected_layer_count"]),
            lease_id=str(data["lease_id"]),
            leased_block_ids=tuple(
                int(block_id) for block_id in data["leased_block_ids"]
            ),
            lease_capacity_tokens=int(data["lease_capacity_tokens"]),
            writable_start_token=int(data["writable_start_token"]),
            writable_end_token=int(data["writable_end_token"]),
            ready_event_handle=(
                None
                if event_handle is None
                else base64.b64decode(str(event_handle).encode("ascii"))
            ),
        )


class PAPOffloadExecMessage(Protocol):
    """Transport-owned tensor view with explicit slot release."""

    tensor: torch.Tensor

    def release(self) -> None: ...


class PAPOffloadExecTransportClosed(RuntimeError):
    """Raised when a transport receive loop is intentionally stopped."""


class PAPOffloadExecTransport(Protocol):
    """Projection<->Attention QKV/O data-plane transport."""

    transport: PAPTensorTransport
    requires_tcp_trigger: bool

    @property
    def local_agent_metadata(self) -> bytes: ...

    def bind_peer(self, peer_agent_metadata: bytes) -> None: ...

    def send_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None: ...

    def send_qkv_batch_direct(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None: ...

    def recv_next_qkv_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, PAPOffloadExecMessage]: ...

    def send_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None: ...

    def prepare_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        shape: tuple[int, int],
        dtype: torch.dtype,
        remote_address: str,
    ) -> PAPOffloadExecMessage | None: ...

    def recv_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> PAPOffloadExecMessage: ...

    def stop_receiving(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class PAPStepPlannedOffloadExecTransport(Protocol):
    """Optional step-planned execution capability.

    The same-host local transport implements this capability. General mailbox
    transports remain on :class:`PAPOffloadExecTransport` and must not claim
    step-planned fan-out or prepared receive semantics.
    """

    def send_step_prepare(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        dtype: torch.dtype,
        remote_address: str,
        descriptorless_qkv: bool = False,
        qkv_width: int = 0,
        layer_count: int = 0,
    ) -> None: ...

    def reserve_qkv_fanout(self, *, num_layers: int) -> dict[str, int]: ...

    def send_qkv_batch_fanout(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None: ...

    def set_step_prepare_handler(self, handler: Any) -> None: ...

    def output_receive_stream(self) -> torch.cuda.Stream: ...

    def step_prepare_stream(self) -> torch.cuda.Stream: ...
