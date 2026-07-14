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

import hashlib
import os
from dataclasses import dataclass
from typing import Any

import torch

from vllm.pap.protocol import (
    PAPCudaIPCTensorHandle,
    PAPDataPlaneChannel,
    PAPDataPlaneRole,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPOffloadExecTransport,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
    PAPTensorTransport,
    pap_offload_exec_trace_id,
)
from vllm.pap.topology.routing import (
    build_offload_exec_route_groups,
    filter_offload_exec_route_groups_for_request_slice,
)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _data_plane_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return value.lower() in _TRUE_ENV_VALUES


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
    lease_id: str | None = None
    leased_block_ids: tuple[int, ...] | None = None
    lease_seq_len: int | None = None
    lease_capacity_tokens: int | None = None
    unified_kv_mode: bool = False
    prefix_len: int | None = None
    writable_start_token: int | None = None
    writable_end_token: int | None = None

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
        if self.lease_id is not None:
            object.__setattr__(self, "lease_id", str(self.lease_id))
            leased = self.leased_block_ids
            if leased is not None:
                leased = tuple(int(b) for b in leased)
                object.__setattr__(self, "leased_block_ids", leased)
            if self.lease_seq_len is not None:
                object.__setattr__(
                    self, "lease_seq_len", int(self.lease_seq_len)
                )
            if self.lease_capacity_tokens is not None:
                object.__setattr__(
                    self,
                    "lease_capacity_tokens",
                    int(self.lease_capacity_tokens),
                )
        if self.unified_kv_mode:
            if self.prefix_len is None:
                raise ValueError("unified_kv_mode requires prefix_len")
            object.__setattr__(self, "prefix_len", int(self.prefix_len))
            if self.writable_start_token is None:
                object.__setattr__(
                    self, "writable_start_token", int(self.prefix_len)
                )
            else:
                object.__setattr__(
                    self, "writable_start_token", int(self.writable_start_token)
                )
            if self.writable_end_token is None:
                object.__setattr__(
                    self, "writable_end_token", int(self.prefix_len)
                )
            else:
                object.__setattr__(
                    self, "writable_end_token", int(self.writable_end_token)
                )
            if int(self.writable_start_token) > int(self.writable_end_token):
                raise ValueError(
                    "unified_kv_mode writable_start_token > writable_end_token"
                )
            if self.lease_id is None:
                raise ValueError("unified_kv_mode requires lease_id")
            if self.leased_block_ids is None or not self.leased_block_ids:
                raise ValueError("unified_kv_mode requires leased_block_ids")
            if self.lease_capacity_tokens is None:
                raise ValueError("unified_kv_mode requires lease_capacity_tokens")
            if int(self.lease_capacity_tokens) < int(self.writable_end_token):
                raise ValueError(
                    "lease_capacity_tokens must cover writable_end_token"
                )
            required_blocks = (
                int(self.writable_end_token) + self.block_size - 1
            ) // self.block_size
            if len(self.block_ids) < required_blocks:
                raise ValueError("block_ids must cover writable_end_token")

    def to_dict(self) -> dict[str, Any]:
        d = {
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
        if self.lease_id is not None:
            d["lease_id"] = self.lease_id
            if self.leased_block_ids is not None:
                d["leased_block_ids"] = list(self.leased_block_ids)
            if self.lease_seq_len is not None:
                d["lease_seq_len"] = int(self.lease_seq_len)
            if self.lease_capacity_tokens is not None:
                d["lease_capacity_tokens"] = int(self.lease_capacity_tokens)
        if self.unified_kv_mode:
            d["unified_kv_mode"] = True
            d["prefix_len"] = int(self.prefix_len) if self.prefix_len is not None else 0
            d["writable_start_token"] = int(self.writable_start_token or 0)
            d["writable_end_token"] = int(self.writable_end_token or 0)
        return d

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
            lease_id=data.get("lease_id"),
            leased_block_ids=(
                tuple(int(b) for b in data["leased_block_ids"])
                if data.get("leased_block_ids") is not None
                else None
            ),
            lease_seq_len=(
                int(data["lease_seq_len"])
                if data.get("lease_seq_len") is not None
                else None
            ),
            lease_capacity_tokens=(
                int(data["lease_capacity_tokens"])
                if data.get("lease_capacity_tokens") is not None
                else None
            ),
            unified_kv_mode=bool(data.get("unified_kv_mode", False)),
            prefix_len=(
                int(data["prefix_len"])
                if data.get("prefix_len") is not None
                else None
            ),
            writable_start_token=(
                int(data["writable_start_token"])
                if data.get("writable_start_token") is not None
                else None
            ),
            writable_end_token=(
                int(data["writable_end_token"])
                if data.get("writable_end_token") is not None
                else None
            ),
        )


def _clone_and_release_mailbox_message(message: Any) -> torch.Tensor:
    tensor = message.tensor
    if getattr(message, "release_callback", None) is not None:
        tensor = tensor.clone()
        if tensor.is_cuda:
            torch.cuda.current_stream(tensor.device).synchronize()
    message.release()
    return tensor


def _record_tensor_ready_event(tensor: torch.Tensor) -> Any | None:
    if not tensor.is_cuda:
        return None
    with torch.cuda.device(tensor.device):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(tensor.device))
    return event


class PAPNixlMailboxOffloadExecTransport:
    """OFFLOAD_EXEC transport backed by the PAP NIXL mailbox runtime."""

    transport = PAPTensorTransport.NIXL_MAILBOX
    requires_tcp_trigger = False

    def __init__(self, endpoint: object) -> None:
        self.endpoint = endpoint
        self._batch_plan_enabled = _data_plane_env_bool(
            "PAP_NIXL_MAILBOX_BATCH_PLAN",
            True,
        )
        self._sent_batch_plans: set[str] = set()
        self._recv_batch_plans: dict[str, dict[str, Any]] = {}

    @property
    def local_agent_metadata(self) -> bytes:
        return self.endpoint.local_agent_metadata

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
            metadata=self._qkv_batch_metadata(descriptor),
            tensor=qkv,
        )

    def send_qkv_batch_direct(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        qkv: torch.Tensor,
        *,
        remote_address: str,
    ) -> None:
        from vllm.pap.transport.nixl import PAPMailboxMessage

        reserve_direct_send_tensor = getattr(
            self.endpoint,
            "reserve_direct_send_tensor",
            None,
        )
        if not callable(reserve_direct_send_tensor):
            raise RuntimeError("PAP NIXL mailbox endpoint cannot reserve direct QKV")
        payload = reserve_direct_send_tensor(
            descriptor.qkv_tensor_id,
            shape=tuple(qkv.shape),
            dtype=qkv.dtype,
        )
        payload.tensor.copy_(qkv, non_blocking=True)
        self.endpoint.send(
            PAPMailboxMessage(
                msg_id=descriptor.qkv_tensor_id,
                kind="attention_task_batch",
                metadata=self._qkv_batch_metadata(descriptor),
                tensor=payload.tensor,
                payload_shape=tuple(qkv.shape),
                direct_payload=True,
                payload_slot_id=payload.slot_id,
                payload_ready_event=_record_tensor_ready_event(payload.tensor),
            )
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
        message.release()
        raise RuntimeError(f"unexpected PAP mailbox message kind: {message.kind}")

    def recv_next_attention_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        message = self.endpoint.recv()
        if message.kind == "attention_task_batch":
            descriptor = _offload_exec_batch_descriptor_from_metadata(
                message.metadata,
                plan_cache=(
                    self._recv_batch_plans if self._batch_plan_enabled else None
                ),
            )
            return descriptor, message
        message.release()
        raise RuntimeError(f"unexpected PAP mailbox message kind: {message.kind}")

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
        from vllm.pap.transport.nixl import PAPMailboxMessage

        self.endpoint.send(
            PAPMailboxMessage(
                msg_id=msg_id,
                kind=kind,
                metadata=metadata,
                tensor=tensor,
            )
        )

    def _qkv_batch_metadata(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
    ) -> dict[str, Any]:
        if not self._batch_plan_enabled:
            return _offload_exec_batch_descriptor_to_metadata(descriptor)
        return _offload_exec_batch_descriptor_to_plan_metadata(
            descriptor,
            sent_plans=self._sent_batch_plans,
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
    if descriptor.metadata_template is not None:
        if "a" in descriptor.metadata_template:
            scales = [float(scale) for scale in descriptor.metadata_template["a"]]
        else:
            scales = [float(item.scale) for item in descriptor.items]
        request_ids = list(descriptor.metadata_template["r"])
        steps = [int(step) for step in descriptor.metadata_template["s"]]
        if not (len(request_ids) == len(steps) == len(scales)):
            raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
        return {
            "v": 2,
            "l": descriptor.layer_name,
            "r": request_ids,
            "s": steps,
            "a": scales,
        }
    return {
        "v": 2,
        "l": descriptor.layer_name,
        "r": [item.request_id for item in descriptor.items],
        "s": [int(item.step) for item in descriptor.items],
        "a": [float(item.scale) for item in descriptor.items],
    }


def _offload_exec_batch_plan_payload(
    descriptor: PAPOffloadExecBatchDescriptor,
) -> dict[str, Any]:
    metadata = _offload_exec_batch_descriptor_to_metadata(descriptor)
    payload: dict[str, Any] = {
        "b": descriptor.batch_id_suffix
        or ",".join(
            f"{request_id}@{step}"
            for request_id, step in zip(metadata["r"], metadata["s"])
        ),
        "r": list(metadata["r"]),
        "s": [int(step) for step in metadata["s"]],
        "a": [float(scale) for scale in metadata["a"]],
    }
    return payload


def _offload_exec_batch_plan_id(plan_payload: dict[str, Any]) -> str:
    key = (
        str(plan_payload["b"]),
        tuple(str(request_id) for request_id in plan_payload["r"]),
        tuple(int(step) for step in plan_payload["s"]),
        tuple(float(scale) for scale in plan_payload["a"]),
    )
    return hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:16]


def _offload_exec_batch_descriptor_to_plan_metadata(
    descriptor: PAPOffloadExecBatchDescriptor,
    *,
    sent_plans: set[str],
) -> dict[str, Any]:
    plan_payload = _offload_exec_batch_plan_payload(descriptor)
    plan_id = _offload_exec_batch_plan_id(plan_payload)
    if plan_id in sent_plans:
        return {
            "v": 5,
            "l": descriptor.layer_name,
            "p": plan_id,
        }
    sent_plans.add(plan_id)
    return {
        "v": 4,
        "l": descriptor.layer_name,
        "p": plan_id,
        **plan_payload,
    }


def _offload_exec_batch_descriptor_from_plan_payload(
    layer_name: str,
    plan_payload: dict[str, Any],
    *,
    template_only: bool = False,
) -> PAPOffloadExecBatchDescriptor:
    if "t" in plan_payload:
        raise ValueError(
            "PAP OFFLOAD_EXEC decode-token metadata was removed; "
            "use asynchronous decode-token delivery"
        )
    request_ids = list(plan_payload["r"])
    steps = [int(step) for step in plan_payload["s"]]
    scales = [float(scale) for scale in plan_payload["a"]]
    if not (len(request_ids) == len(steps) == len(scales)):
        raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
    if template_only:
        return PAPOffloadExecBatchDescriptor(
            layer_name=layer_name,
            items=(),
            batch_id_suffix=str(plan_payload["b"]),
            metadata_template={
                "r": tuple(str(request_id) for request_id in request_ids),
                "s": tuple(steps),
                "a": tuple(scales),
            },
        )
    return PAPOffloadExecBatchDescriptor(
        layer_name=layer_name,
        items=tuple(
            PAPOffloadExecDescriptor(
                request_id=str(request_id),
                layer_name=layer_name,
                step=int(step),
                scale=float(scale),
            )
            for request_id, step, scale in zip(
                request_ids,
                steps,
                scales,
            )
        ),
        batch_id_suffix=str(plan_payload["b"]),
    )


def _offload_exec_batch_descriptor_from_metadata(
    metadata: dict[str, Any],
    *,
    plan_cache: dict[str, dict[str, Any]] | None = None,
    template_only: bool = False,
) -> PAPOffloadExecBatchDescriptor:
    if metadata.get("v") == 4:
        if "t" in metadata:
            raise ValueError(
                "PAP OFFLOAD_EXEC decode-token metadata was removed; "
                "use asynchronous decode-token delivery"
            )
        layer_name = str(metadata["l"])
        plan_id = str(metadata["p"])
        plan_payload: dict[str, Any] = {
            "b": str(metadata["b"]),
            "r": list(metadata["r"]),
            "s": list(metadata["s"]),
            "a": list(metadata["a"]),
        }
        if plan_cache is not None:
            plan_cache[plan_id] = plan_payload
        return _offload_exec_batch_descriptor_from_plan_payload(
            layer_name,
            plan_payload,
            template_only=template_only,
        )
    if metadata.get("v") == 5:
        if plan_cache is None:
            raise ValueError("PAP OFFLOAD_EXEC batch plan cache is required")
        layer_name = str(metadata["l"])
        plan_id = str(metadata["p"])
        try:
            plan_payload = plan_cache[plan_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown PAP OFFLOAD_EXEC batch plan id: {plan_id}"
            ) from exc
        return _offload_exec_batch_descriptor_from_plan_payload(
            layer_name,
            plan_payload,
            template_only=template_only,
        )
    if metadata.get("v") == 3 or "t" in metadata:
        raise ValueError(
            "PAP OFFLOAD_EXEC decode-token metadata was removed; "
            "use asynchronous decode-token delivery"
        )
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
    if any("decode_token_ids" in item for item in metadata["items"]):
        raise ValueError(
            "PAP OFFLOAD_EXEC decode-token metadata was removed; "
            "use asynchronous decode-token delivery"
        )
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
    from vllm.pap.transport.nixl import PAPNixlMailboxEndpoint

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


def build_local_fast_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
):
    """Construct a same-machine CUDA IPC + spin-doorbell transport.

    Activated via ``PAP_OFFLOAD_EXEC_TRANSPORT=local_fast``.  See
    ``vllm.pap.transport.local_fast`` for the design and constraints.
    """

    from vllm.pap.transport.local_fast import (
        build_local_fast_offload_exec_transport as _impl,
    )

    return _impl(
        actor_id=actor_id,
        local_rank=local_rank,
        buffer_bytes=buffer_bytes,
    )
