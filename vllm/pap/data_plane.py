# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP tensor data-plane contracts.

PAP keeps control-plane routing separate from tensor movement:

* OFFLOAD_KV installs Prefill KV into the colocated Attention executor.
* OFFLOAD_EXEC exchanges per-decode-step Q/K/V and O between Projection and
  Attention.

Control metadata travels over ZMQ/TCP. OFFLOAD_EXEC tensor payloads use the
PAP NIXL mailbox transport.
"""

from __future__ import annotations

import base64
import hashlib
import os
import pickle
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

    @property
    def qkv_tensor_id(self) -> str:
        return f"{self.request_id}#{self.layer_name}#{self.step}#qkv"

    @property
    def output_tensor_id(self) -> str:
        return f"{self.request_id}#{self.layer_name}#{self.step}#attn_out"


@dataclass(frozen=True)
class PAPOffloadExecBatchDescriptor:
    """Control metadata for one batched Projection<->Attention attention call."""

    layer_name: str
    items: tuple[PAPOffloadExecDescriptor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "items", tuple(self.items))
        if not self.items:
            raise ValueError("PAP OFFLOAD_EXEC batch requires at least one item")
        for item in self.items:
            if item.layer_name != self.layer_name:
                raise ValueError("all PAP OFFLOAD_EXEC batch items must share layer")

    @property
    def batch_id(self) -> str:
        entries = ",".join(
            f"{item.request_id}@{item.step}" for item in self.items
        )
        return f"{self.layer_name}#{entries}"

    @property
    def qkv_tensor_id(self) -> str:
        return f"{self.batch_id}#qkv_batch"

    @property
    def query_tensor_id(self) -> str:
        return f"{self.batch_id}#query_batch"

    @property
    def kv_tensor_id(self) -> str:
        return f"{self.batch_id}#kv_batch"

    @property
    def output_tensor_id(self) -> str:
        return f"{self.batch_id}#attn_out_batch"


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
    def from_dict(cls, data: dict[str, Any]) -> PAPOffloadKVIPCDescriptor:
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
    def from_dict(cls, data: dict[str, Any]) -> PAPOffloadKVPagedIPCDescriptor:
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

    def send_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        ...

    def recv_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        ...

    def send_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        ...

    def recv_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        ...


def _clone_and_release_mailbox_message(message: Any) -> torch.Tensor:
    tensor = message.tensor
    if getattr(message, "release_callback", None) is not None:
        tensor = tensor.clone()
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
    message.release()
    return tensor


class PAPNixlMailboxOffloadExecTransport:
    """OFFLOAD_EXEC transport backed by the PAP NIXL mailbox runtime."""

    transport = PAPTensorTransport.NIXL_MAILBOX
    requires_tcp_trigger = False

    def __init__(self, endpoint: object) -> None:
        self.endpoint = endpoint

    @property
    def local_agent_metadata(self) -> bytes:
        return self.endpoint.local_agent_metadata

    @property
    def supports_query_first_kv_later(self) -> bool:
        return bool(
            getattr(self.endpoint, "_async_send_slots_enabled", False)
            and int(getattr(self.endpoint, "_slot_count", 1)) >= 2
        )

    def bind_peer(self, peer_agent_metadata: bytes) -> None:
        self.endpoint.bind_peer(peer_agent_metadata)
        self.endpoint.start()

    def send_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_message(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task",
            metadata=_offload_exec_descriptor_to_metadata(descriptor),
            tensor=qkv,
        )

    def recv_qkv(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        return _clone_and_release_mailbox_message(
            self.endpoint.recv(descriptor.qkv_tensor_id)
        )

    def send_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_message(
            msg_id=descriptor.output_tensor_id,
            kind="attention_result",
            metadata=_offload_exec_descriptor_to_metadata(descriptor),
            tensor=output,
        )

    def recv_output(
        self,
        descriptor: PAPOffloadExecDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        return _clone_and_release_mailbox_message(
            self.endpoint.recv(descriptor.output_tensor_id)
        )

    def send_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_message(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task_batch",
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
            tensor=qkv,
        )

    def send_qkv_batch_segments(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        segments: tuple[torch.Tensor, ...],
        *,
        payload_shape: tuple[int, ...],
        remote_address: str,
    ) -> None:
        from vllm.pap.nixl_mailbox import PAPMailboxMessage

        self.endpoint.send(
            PAPMailboxMessage(
                msg_id=descriptor.qkv_tensor_id,
                kind="attention_task_batch",
                metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
                tensor=segments[0],
                payload_segments=tuple(segments),
                payload_shape=tuple(payload_shape),
            )
        )

    def send_query_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        query: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_message(
            msg_id=descriptor.query_tensor_id,
            kind="attention_query_batch",
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
            tensor=query,
        )

    def send_kv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        kv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_message(
            msg_id=descriptor.kv_tensor_id,
            kind="attention_kv_batch",
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
            tensor=kv,
        )

    def recv_qkv_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        return _clone_and_release_mailbox_message(
            self.recv_qkv_batch_message(
                descriptor,
                remote_address=remote_address,
            )
        )

    def recv_qkv_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> Any:
        return self.endpoint.recv(descriptor.qkv_tensor_id)

    def recv_next_qkv_batch(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, torch.Tensor]:
        descriptor, message = self.recv_next_qkv_batch_message()
        return descriptor, _clone_and_release_mailbox_message(message)

    def recv_next_qkv_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        descriptor, message = self.recv_next_attention_batch_message()
        if message.kind == "attention_task_batch":
            return descriptor, message
        if message.kind == "attention_query_batch":
            return self._recv_query_first_qkv_batch_message(message)
        message.release()
        raise RuntimeError(
            f"unexpected PAP mailbox message kind: {message.kind}"
        )

    def recv_next_attention_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        message = self.endpoint.recv()
        if message.kind in {"attention_task_batch", "attention_query_batch"}:
            descriptor = _offload_exec_batch_descriptor_from_metadata(
                message.metadata
            )
            return descriptor, message
        message.release()
        raise RuntimeError(
            f"unexpected PAP mailbox message kind: {message.kind}"
        )

    def recv_kv_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
    ) -> Any:
        kv_message = self.endpoint.recv(descriptor.kv_tensor_id)
        if kv_message.kind != "attention_kv_batch":
            kv_message.release()
            raise RuntimeError(
                f"unexpected PAP mailbox KV message kind: {kv_message.kind}"
            )
        kv_descriptor = _offload_exec_batch_descriptor_from_metadata(
            kv_message.metadata
        )
        if kv_descriptor != descriptor:
            kv_message.release()
            raise RuntimeError("PAP mailbox query/KV descriptors do not match")
        return kv_message

    def _recv_query_first_qkv_batch_message(
        self,
        query_message: Any,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        from vllm.pap.nixl_mailbox import PAPMailboxMessage

        descriptor = _offload_exec_batch_descriptor_from_metadata(
            query_message.metadata
        )
        try:
            kv_message = self.recv_kv_batch_message(descriptor)
        except Exception:
            query_message.release()
            raise
        qkv = torch.cat((query_message.tensor, kv_message.tensor), dim=-1)

        def release_inputs() -> None:
            query_message.release()
            kv_message.release()

        return descriptor, PAPMailboxMessage(
            msg_id=descriptor.qkv_tensor_id,
            kind="attention_task_batch",
            metadata=query_message.metadata,
            tensor=qkv,
            release_callback=release_inputs,
        )

    def send_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        output: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        self._send_message(
            msg_id=descriptor.output_tensor_id,
            kind="attention_result_batch",
            metadata=_offload_exec_batch_descriptor_to_metadata(descriptor),
            tensor=output,
        )

    def recv_output_batch(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> torch.Tensor:
        return _clone_and_release_mailbox_message(
            self.recv_output_batch_message(
                descriptor,
                remote_address=remote_address,
            )
        )

    def recv_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> Any:
        return self.endpoint.recv(descriptor.output_tensor_id)

    def _send_message(
        self,
        *,
        msg_id: str,
        kind: str,
        metadata: dict[str, Any],
        tensor: torch.Tensor,
    ) -> None:
        from vllm.pap.nixl_mailbox import PAPMailboxMessage

        self.endpoint.send(
            PAPMailboxMessage(
                msg_id=msg_id,
                kind=kind,
                metadata=metadata,
                tensor=tensor,
            )
        )


def _offload_exec_descriptor_to_metadata(
    descriptor: PAPOffloadExecDescriptor,
) -> dict[str, Any]:
    return {
        "request_id": descriptor.request_id,
        "layer_name": descriptor.layer_name,
        "step": descriptor.step,
        "scale": descriptor.scale,
    }


def _offload_exec_batch_descriptor_to_metadata(
    descriptor: PAPOffloadExecBatchDescriptor,
) -> dict[str, Any]:
    return {
        "v": 2,
        "l": descriptor.layer_name,
        "r": [item.request_id for item in descriptor.items],
        "s": [int(item.step) for item in descriptor.items],
        "a": [float(item.scale) for item in descriptor.items],
    }


def _offload_exec_batch_descriptor_from_metadata(
    metadata: dict[str, Any],
) -> PAPOffloadExecBatchDescriptor:
    if metadata.get("v") == 2:
        layer_name = str(metadata["l"])
        request_ids = list(metadata["r"])
        steps = list(metadata["s"])
        scales = list(metadata["a"])
        if not (len(request_ids) == len(steps) == len(scales)):
            raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
        return PAPOffloadExecBatchDescriptor(
            layer_name=layer_name,
            items=tuple(
                PAPOffloadExecDescriptor(
                    request_id=str(request_id),
                    layer_name=layer_name,
                    step=int(step),
                    scale=float(scale),
                )
                for request_id, step, scale in zip(request_ids, steps, scales)
            ),
        )

    layer_name = str(metadata["layer_name"])
    return PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=tuple(
            PAPOffloadExecDescriptor(
                request_id=str(item["request_id"]),
                layer_name=layer_name,
                step=int(item["step"]),
                scale=float(item["scale"]),
            )
            for item in metadata["items"]
        ),
    )


def build_nixl_mailbox_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
) -> PAPNixlMailboxOffloadExecTransport:
    from vllm.pap.nixl_mailbox import PAPNixlMailboxEndpoint

    endpoint = PAPNixlMailboxEndpoint(
        actor_id=actor_id,
        device=torch.device(f"cuda:{int(local_rank)}"),
        buffer_bytes=(
            int(buffer_bytes)
            if buffer_bytes is not None
            else int(os.environ.get("PAP_NIXL_MAILBOX_BUFFER_BYTES", "16777216"))
        ),
    )
    return PAPNixlMailboxOffloadExecTransport(endpoint)
