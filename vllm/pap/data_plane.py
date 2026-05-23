# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP tensor data-plane contracts.

PAP keeps control-plane routing separate from tensor movement:

* OFFLOAD_KV installs Prefill KV into the colocated Attention executor.
* OFFLOAD_EXEC exchanges per-decode-step Q/K/V and O between Projection and
  Attention using torch.distributed scatter/gather (NCCL).

Control metadata travels over ZMQ/TCP.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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


