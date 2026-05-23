# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP tensor data-plane contracts.

PAP keeps control-plane routing separate from tensor movement:

* OFFLOAD_KV installs Prefill KV into the colocated Attention executor.
* OFFLOAD_EXEC exchanges per-decode-step Q/K/V and O between Projection and
  Attention.

HTTP/ZMQ may carry control metadata, but performance-mode tensor payloads must
use GPU-direct transports such as CUDA IPC or NCCL/P2P.
"""

from __future__ import annotations

import os
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
    PROTOTYPE_HTTP = "prototype_http"
    PROTOTYPE_TCP = "prototype_tcp"
    CUDA_IPC = "cuda_ipc"
    NCCL_P2P = "nccl_p2p"


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
        if self.transport in {
            PAPTensorTransport.PROTOTYPE_HTTP,
            PAPTensorTransport.PROTOTYPE_TCP,
        }:
            raise ValueError(
                "OFFLOAD_KV performance descriptor cannot use HTTP/TCP tensor "
                "transport"
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
        poll_seconds = float(os.environ.get("PAP_OFFLOAD_EXEC_RECV_POLL_SECONDS", "0.01"))
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


def performance_mode_requires_gpu_data_plane(
    *,
    pap_mode: str | None,
    prefill_attention_transport: PAPTensorTransport,
    projection_attention_transport: PAPTensorTransport,
    prefill_attention_kv_installed: bool = False,
) -> None:
    """Reject prototype tensor transports in true performance mode."""

    if pap_mode != "true_split_performance":
        return
    prototype = {PAPTensorTransport.PROTOTYPE_HTTP, PAPTensorTransport.PROTOTYPE_TCP}
    if prefill_attention_transport in prototype and not prefill_attention_kv_installed:
        raise RuntimeError(
            "PAP true_split_performance requires CUDA IPC/shared KV transport "
            "for Prefill-to-Attention KV"
        )
    if projection_attention_transport in prototype:
        raise RuntimeError(
            "PAP true_split_performance requires NCCL/P2P/NVLink transport for "
            "Projection-to-Attention QKV/O"
        )


def offload_exec_transport_from_env() -> PAPTensorTransport:
    value = os.environ.get("PAP_OFFLOAD_EXEC_TRANSPORT", "").lower()
    if value in {"nccl", "nccl_p2p", "p2p_nccl"}:
        return PAPTensorTransport.NCCL_P2P
    if value in {"tcp", "prototype_tcp"}:
        return PAPTensorTransport.PROTOTYPE_TCP
    if value in {"http", "prototype_http", ""}:
        return PAPTensorTransport.PROTOTYPE_HTTP
    raise ValueError(f"unknown PAP_OFFLOAD_EXEC_TRANSPORT={value!r}")


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
