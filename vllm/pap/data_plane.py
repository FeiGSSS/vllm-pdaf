# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP tensor data-plane contracts.

PAP keeps control-plane routing separate from tensor movement:

* OFFLOAD_KV installs Prefill KV into the colocated Attention executor.
* OFFLOAD_EXEC exchanges per-decode-step Q/K/V and O between Projection and
  Attention.

Control metadata travels over ZMQ/TCP. Tensor payloads use GPU-direct NCCL/P2P.
"""

from __future__ import annotations

import base64
import os
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch


class PAPDataPlaneRole(str, Enum):
    PREFILL = "prefill"
    ATTENTION = "attention"
    PROJECTION = "projection"


class PAPDataPlaneChannel(str, Enum):
    OFFLOAD_KV = "offload_kv"
    OFFLOAD_EXEC = "offload_exec"


class PAPTensorTransport(str, Enum):
    NCCL_P2P = "nccl_p2p"
    CUDA_IPC = "cuda_ipc"


@dataclass(frozen=True)
class PAPOffloadExecDescriptor:
    """Control metadata for one Projection<->Attention decode attention call."""

    request_id: str
    layer_name: str
    step: int
    scale: float

    @property
    def qkv_tensor_id(self) -> str:
        return f"{self.request_id}#{self.layer_name}#{self.step}#qkv"

    @property
    def output_tensor_id(self) -> str:
        return f"{self.request_id}#{self.layer_name}#{self.step}#attn_out"


@dataclass(frozen=True)
class PAPOffloadKVDescriptor:
    """Control metadata for installing Prefill KV in the Attention role."""

    request_id: str
    layer_name: str
    seq_len: int
    block_ids: tuple[int, ...]
    transport: PAPTensorTransport

    def __post_init__(self) -> None:
        if self.seq_len < 0:
            raise ValueError("seq_len must be non-negative")


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
    def from_dict(cls, data: dict[str, Any]) -> "PAPCudaIPCTensorHandle":
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
class PAPOffloadKVIPCDescriptor:
    """CUDA IPC metadata for installing Prefill KV in Attention."""

    request_id: str
    layer_name: str
    seq_len: int
    block_ids: tuple[int, ...]
    key: PAPCudaIPCTensorHandle
    value: PAPCudaIPCTensorHandle
    transport: PAPTensorTransport = PAPTensorTransport.CUDA_IPC

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "seq_len", int(self.seq_len))
        object.__setattr__(
            self, "block_ids", tuple(int(block_id) for block_id in self.block_ids)
        )
        object.__setattr__(self, "transport", PAPTensorTransport(self.transport))
        if self.seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if self.transport is not PAPTensorTransport.CUDA_IPC:
            raise ValueError("PAP OFFLOAD_KV IPC descriptor requires cuda_ipc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_name": self.layer_name,
            "seq_len": self.seq_len,
            "block_ids": list(self.block_ids),
            "transport": self.transport.value,
            "key": self.key.to_dict(),
            "value": self.value.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PAPOffloadKVIPCDescriptor":
        return cls(
            request_id=str(data["request_id"]),
            layer_name=str(data["layer_name"]),
            seq_len=int(data["seq_len"]),
            block_ids=tuple(int(block_id) for block_id in data.get("block_ids", [])),
            transport=PAPTensorTransport(
                data.get("transport", PAPTensorTransport.CUDA_IPC)
            ),
            key=PAPCudaIPCTensorHandle.from_dict(data["key"]),
            value=PAPCudaIPCTensorHandle.from_dict(data["value"]),
        )


@dataclass(frozen=True)
class PAPOffloadKVPagedIPCDescriptor:
    """CUDA IPC metadata for Prefill-owned paged KV cache backing storage."""

    request_id: str
    layer_name: str
    seq_len: int
    block_ids: tuple[int, ...]
    block_size: int
    num_kv_heads: int
    layout: str
    kv_cache: PAPCudaIPCTensorHandle
    transport: PAPTensorTransport = PAPTensorTransport.CUDA_IPC

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "seq_len", int(self.seq_len))
        object.__setattr__(
            self, "block_ids", tuple(int(block_id) for block_id in self.block_ids)
        )
        object.__setattr__(self, "block_size", int(self.block_size))
        object.__setattr__(self, "num_kv_heads", int(self.num_kv_heads))
        object.__setattr__(self, "layout", str(self.layout))
        object.__setattr__(self, "transport", PAPTensorTransport(self.transport))
        if self.seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.layout not in {"NHD", "HND"}:
            raise ValueError(f"unsupported KV cache layout: {self.layout}")
        if self.transport is not PAPTensorTransport.CUDA_IPC:
            raise ValueError("PAP OFFLOAD_KV paged IPC descriptor requires cuda_ipc")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_name": self.layer_name,
            "seq_len": self.seq_len,
            "block_ids": list(self.block_ids),
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "layout": self.layout,
            "transport": self.transport.value,
            "kv_cache": self.kv_cache.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PAPOffloadKVPagedIPCDescriptor":
        return cls(
            request_id=str(data["request_id"]),
            layer_name=str(data["layer_name"]),
            seq_len=int(data["seq_len"]),
            block_ids=tuple(int(block_id) for block_id in data.get("block_ids", [])),
            block_size=int(data["block_size"]),
            num_kv_heads=int(data["num_kv_heads"]),
            layout=str(data["layout"]),
            transport=PAPTensorTransport(
                data.get("transport", PAPTensorTransport.CUDA_IPC)
            ),
            kv_cache=PAPCudaIPCTensorHandle.from_dict(data["kv_cache"]),
        )


class PAPOffloadExecTransport(Protocol):
    """Projection<->Attention QKV/O data-plane transport."""

    transport: PAPTensorTransport

    def send_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        ...

    def recv_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        ...

    def send_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        ...

    def recv_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        ...


class PAPP2PNCCLOffloadExecTransport:
    """Adapter around vLLM's P2pNcclEngine for PAP OFFLOAD_EXEC tensors."""

    transport = PAPTensorTransport.NCCL_P2P

    def __init__(self, engine: object) -> None:
        self.engine = engine

    def send_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send(descriptor.qkv_tensor_id, qkv, remote_address)

    def recv_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        return self._recv(descriptor.qkv_tensor_id, remote_address)

    def send_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send(descriptor.output_tensor_id, output, remote_address)

    def recv_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        return self._recv(descriptor.output_tensor_id, remote_address)

    def _send(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str,
    ) -> None:
        ok = self.engine.send_tensor(tensor_id, tensor, remote_address)
        if not ok:
            raise RuntimeError(
                f"PAP OFFLOAD_EXEC NCCL send failed tensor_id={tensor_id} "
                f"remote_address={remote_address}"
            )

    def _recv(self, tensor_id: str, remote_address: str) -> torch.Tensor:
        deadline = time.monotonic() + float(
            os.environ.get("PAP_OFFLOAD_EXEC_RECV_TIMEOUT", "30")
        )
        poll_seconds = float(
            os.environ.get("PAP_OFFLOAD_EXEC_RECV_POLL_SECONDS", "0.01")
        )
        while True:
            tensor = self.engine.recv_tensor(tensor_id, remote_address)
            if tensor is not None:
                return tensor
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"PAP OFFLOAD_EXEC NCCL recv failed tensor_id={tensor_id} "
                    f"remote_address={remote_address}"
                )
            time.sleep(poll_seconds)


def build_p2p_nccl_offload_exec_transport(
    *,
    local_rank: int,
    kv_port: int,
    hostname: str = "",
    port_offset: int = 0,
    kv_buffer_size: float | None = None,
    extra_config: dict[str, Any] | None = None,
    engine_cls: type | None = None,
) -> PAPP2PNCCLOffloadExecTransport:
    """Create the PAP OFFLOAD_EXEC NCCL/P2P transport.

    This reuses vLLM's P2pNcclEngine, which already separates ZMQ control
    messages from NCCL tensor transfer.
    """

    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
        P2pNcclEngine,
    )

    p2p_disable = os.environ.get("PAP_OFFLOAD_EXEC_NCCL_P2P_DISABLE", "1")
    if p2p_disable != "":
        os.environ.setdefault("NCCL_P2P_DISABLE", p2p_disable)

    config = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_both",
        kv_port=int(kv_port),
        kv_buffer_size=(
            float(kv_buffer_size)
            if kv_buffer_size is not None
            else float(os.environ.get("PAP_OFFLOAD_EXEC_BUFFER_SIZE", "1000000000"))
        ),
        kv_connector_extra_config={
            "send_type": os.environ.get("PAP_OFFLOAD_EXEC_SEND_TYPE", "GET"),
            "nccl_num_channels": os.environ.get(
                "PAP_OFFLOAD_EXEC_NCCL_NUM_CHANNELS", "8"
            ),
            **(extra_config or {}),
        },
    )
    engine_type = P2pNcclEngine if engine_cls is None else engine_cls
    engine = engine_type(
        local_rank=int(local_rank),
        config=config,
        hostname=hostname,
        port_offset=int(port_offset),
    )
    return PAPP2PNCCLOffloadExecTransport(engine)
