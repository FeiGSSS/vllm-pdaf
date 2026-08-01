# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL mailbox adapter for PAP OFFLOAD_EXEC."""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.pap.protocol import (
    PAPOffloadExecBatchDescriptor,
    PAPTensorTransport,
)
from vllm.pap.protocol.offload_exec import (
    _offload_exec_batch_descriptor_from_metadata,
    _offload_exec_batch_descriptor_to_metadata,
    _offload_exec_batch_descriptor_to_plan_metadata,
)
from vllm.pap.transport.nixl.message import PAPMailboxMessage


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
        self._sent_batch_plans: set[str] = set()
        self._recv_batch_plans: dict[str, dict[str, Any]] = {}

    @property
    def local_agent_metadata(self) -> bytes:
        return self.endpoint.local_agent_metadata

    def bind_peer(self, peer_agent_metadata: bytes) -> None:
        self.endpoint.bind_peer(peer_agent_metadata)
        self.endpoint.start()

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

    def recv_next_qkv_batch_message(
        self,
    ) -> tuple[PAPOffloadExecBatchDescriptor, Any]:
        message = self.endpoint.recv()
        if message.kind == "attention_task_batch":
            descriptor = _offload_exec_batch_descriptor_from_metadata(
                message.metadata,
                plan_cache=self._recv_batch_plans,
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

    def prepare_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        shape: tuple[int, int],
        dtype: torch.dtype,
        remote_address: str,
    ) -> None:
        del descriptor, shape, dtype, remote_address
        return None

    def recv_output_batch_message(
        self,
        descriptor: PAPOffloadExecBatchDescriptor,
        *,
        remote_address: str,
    ) -> Any:
        return self.endpoint.recv(descriptor.output_tensor_id)

    def stop_receiving(self) -> None:
        self.endpoint.stop_receiving()

    def close(self) -> None:
        self.endpoint.close()

    def _send_message(
        self,
        *,
        msg_id: str,
        kind: str,
        metadata: dict[str, Any],
        tensor: torch.Tensor,
    ) -> None:
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
        return _offload_exec_batch_descriptor_to_plan_metadata(
            descriptor,
            sent_plans=self._sent_batch_plans,
        )


def build_nixl_mailbox_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    buffer_bytes: int | None = None,
) -> PAPNixlMailboxOffloadExecTransport:
    from vllm.pap.transport.nixl.endpoint import PAPNixlMailboxEndpoint

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
