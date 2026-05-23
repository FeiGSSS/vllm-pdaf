# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP OFFLOAD_EXEC agent: torch.distributed scatter/gather for QKV/O."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch.distributed import Backend

from vllm.pap.offload_comm import OffloadComm


class OffloadExecAgent:
    """Per-group agent using torch.distributed scatter/gather for projection↔Attention."""

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        dist_backend: Union[str, Backend] = Backend.NCCL,
        decode_master_rank: int = -1,
    ):
        assert decode_master_rank != -1
        self.decode_master_rank = decode_master_rank
        self.data_comm = OffloadComm(group_ranks, local_rank, dist_backend)
        self.fin_comm = OffloadComm(group_ranks, local_rank, dist_backend)
        self.sche_comm = OffloadComm(group_ranks, local_rank, Backend.GLOO)
        self.sync_comm = OffloadComm(group_ranks, local_rank, Backend.GLOO)

    def destroy(self) -> None:
        for comm in (self.data_comm, self.fin_comm, self.sche_comm, self.sync_comm):
            if comm is not None:
                comm.close()

    def broadcast_schedule(
        self, objects: list
    ) -> list:
        torch.distributed.broadcast_object_list(
            objects,
            src=self.decode_master_rank,
            group=self.sche_comm.device_group,
        )
        return objects

    def broadcast_tensor(
        self, tensor: torch.Tensor, *, async_op: bool = False
    ):
        return torch.distributed.broadcast(
            tensor,
            src=self.decode_master_rank,
            group=self.fin_comm.device_group,
            async_op=async_op,
        )

    def scatter_qkv(
        self,
        *,
        output_tensor: Optional[torch.Tensor] = None,
        input_tensors: Optional[list[torch.Tensor]] = None,
        async_op: bool = False,
    ):
        """Scatter QKV from projection (rank 0) to attention instances.

        Projection side: pass input_tensors = [local_qkv, attn_0_qkv, ...]
        Attention side: pass output_tensor as a pre-allocated receive buffer.
        """
        return torch.distributed.scatter(
            output_tensor,
            input_tensors,
            src=self.data_comm.src_rank,
            group=self.data_comm.device_group,
            async_op=async_op,
        )

    def gather_attn_output(
        self,
        *,
        input_tensor: Optional[torch.Tensor] = None,
        output_tensors: Optional[list[torch.Tensor]] = None,
        async_op: bool = False,
    ):
        """Gather attention output from instances to projection (rank 0).

        Projection side: pass output_tensors = [local_buffer, attn_0_out, ...]
        Attention side: pass input_tensor as the computed attention output.
        """
        return torch.distributed.gather(
            input_tensor,
            output_tensors,
            dst=self.data_comm.src_rank,
            group=self.data_comm.device_group,
            async_op=async_op,
        )
