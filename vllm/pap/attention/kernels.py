# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP-owned paged decode-attention kernel integration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import torch

from vllm.pap.kv.metadata import PAPPagedFlashMetadata

PAP_TRITON_DECODE_LOW_RESOURCE_MAX_SMS = 20


@dataclass(frozen=True)
class PAPPagedDecodeKernelConfig:
    """Launch specialization for PAP grouped-query decode Attention."""

    num_splits: int
    block_h: int
    num_warps: int
    num_stages: int


PAP_TRITON_DECODE_DEFAULT_CONFIG = PAPPagedDecodeKernelConfig(
    num_splits=4,
    block_h=16,
    num_warps=4,
    num_stages=2,
)
PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG = PAPPagedDecodeKernelConfig(
    num_splits=8,
    block_h=4,
    num_warps=4,
    num_stages=1,
)


def paged_decode_kernel_config_for_sms(
    visible_sms: int,
) -> PAPPagedDecodeKernelConfig:
    """Select the measured low-SM specialization without changing full GPUs."""
    if 0 < int(visible_sms) <= PAP_TRITON_DECODE_LOW_RESOURCE_MAX_SMS:
        return PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
    return PAP_TRITON_DECODE_DEFAULT_CONFIG


@dataclass(frozen=True)
class PAPPagedDecodeWorkspace:
    """Step-owned scratch reused by every Attention layer."""

    output: torch.Tensor
    partial: torch.Tensor
    lse: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    batch_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    kernel_config: PAPPagedDecodeKernelConfig

    def validate(self, query: torch.Tensor) -> None:
        signature = (
            int(query.shape[0]),
            int(query.shape[1]),
            int(query.shape[2]),
            query.dtype,
            query.device,
        )
        expected = (
            self.batch_size,
            self.num_heads,
            self.head_dim,
            self.dtype,
            self.device,
        )
        if signature != expected:
            raise RuntimeError(
                "PAP paged decode workspace does not match the query shape"
            )


class PAPPagedDecodeWorkspaceCache:
    """Bounded per-peer cache for shape-stable decode scratch."""

    def __init__(self, *, max_entries: int = 16) -> None:
        if max_entries <= 0:
            raise ValueError("PAP paged decode workspace cache must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[
            tuple[int, int, int, torch.dtype, torch.device],
            PAPPagedDecodeWorkspace,
        ] = OrderedDict()
        self._lock = Lock()

    def get(self, query: torch.Tensor) -> PAPPagedDecodeWorkspace:
        """Return reusable scratch for one query shape."""
        if query.ndim != 3:
            raise ValueError("PAP paged decode query must be rank 3")
        key = (
            int(query.shape[0]),
            int(query.shape[1]),
            int(query.shape[2]),
            query.dtype,
            query.device,
        )
        with self._lock:
            workspace = self._entries.get(key)
            if workspace is None:
                workspace = build_paged_decode_workspace(query)
                self._entries[key] = workspace
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
            return workspace


class PAPAttentionStepTensorCache:
    """Bounded per-peer cache for mutable decode-step metadata tensors."""

    def __init__(self, *, max_entries: int = 64) -> None:
        if max_entries <= 0:
            raise ValueError("PAP Attention step tensor cache must be positive")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[
            tuple[str, int, torch.dtype, torch.device],
            tuple[torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._lock = Lock()

    def copy(
        self,
        *,
        kind: str,
        values: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Copy host values into one shape-stable reusable device tensor."""
        normalized_device = torch.device(device)
        key = (str(kind), len(values), dtype, normalized_device)
        with self._lock:
            buffers = self._entries.get(key)
            if buffers is None:
                host = torch.empty(
                    len(values),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=normalized_device.type == "cuda",
                )
                target = torch.empty(
                    len(values),
                    dtype=dtype,
                    device=normalized_device,
                )
                buffers = (host, target)
                self._entries[key] = buffers
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
            host, target = buffers
            host.copy_(torch.tensor(values, dtype=dtype))
            target.copy_(
                host,
                non_blocking=normalized_device.type == "cuda",
            )
            return target


def build_paged_decode_workspace(
    query: torch.Tensor,
) -> PAPPagedDecodeWorkspace:
    """Allocate the fixed PAP decode scratch once for a decode step."""

    if query.ndim != 3:
        raise ValueError("PAP paged decode query must be rank 3")
    batch_size, num_heads, head_dim = map(int, query.shape)
    visible_sms = (
        torch.cuda.get_device_properties(query.device).multi_processor_count
        if query.device.type == "cuda"
        else 0
    )
    kernel_config = paged_decode_kernel_config_for_sms(visible_sms)
    return PAPPagedDecodeWorkspace(
        output=torch.empty_like(query),
        partial=torch.empty(
            (
                batch_size,
                num_heads,
                kernel_config.num_splits,
                head_dim + 1,
            ),
            dtype=torch.float32,
            device=query.device,
        ),
        lse=torch.empty(
            (batch_size, num_heads),
            dtype=torch.float32,
            device=query.device,
        ),
        k_scale=torch.ones((), dtype=torch.float32, device=query.device),
        v_scale=torch.ones((), dtype=torch.float32, device=query.device),
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=query.dtype,
        device=query.device,
        kernel_config=kernel_config,
    )


def run_paged_decode_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: PAPPagedFlashMetadata,
    workspace: PAPPagedDecodeWorkspace,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    """Run the current Triton paged-decode kernel without a layer fallback."""

    from vllm.v1.attention.ops.triton_decode_attention import (
        decode_attention_fwd,
    )

    workspace.validate(query)
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError("PAP paged decode KV cache must be rank 4")
    if key_cache.shape != value_cache.shape:
        raise RuntimeError("PAP paged decode K/V cache shapes differ")
    if int(key_cache.shape[1]) != int(block_size):
        raise RuntimeError("PAP paged decode block size does not match KV cache")
    if int(query.shape[1]) % int(key_cache.shape[-2]) != 0:
        raise RuntimeError("PAP paged decode GQA head counts are incompatible")

    decode_attention_fwd(
        query,
        key_cache,
        value_cache,
        workspace.output,
        workspace.lse,
        metadata.block_table,
        metadata.seq_lens,
        workspace.partial,
        workspace.kernel_config.num_splits,
        float(scale),
        page_size=int(block_size),
        k_scale=workspace.k_scale,
        v_scale=workspace.v_scale,
        grouped_block_h=workspace.kernel_config.block_h,
        grouped_num_warps=workspace.kernel_config.num_warps,
        grouped_num_stages=workspace.kernel_config.num_stages,
    )
    return workspace.output


def warm_paged_decode_attention(
    *,
    kv_cache: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    """Compile the PAP paged-decode kernel before the first decode step."""
    if (
        kv_cache.device.type != "cuda"
        or kv_cache.ndim != 5
        or int(num_heads) <= 0
        or int(head_dim) <= 0
    ):
        return
    key_cache, value_cache = kv_cache.unbind(1)
    device = kv_cache.device
    query = torch.empty(
        (1, int(num_heads), int(head_dim)),
        dtype=kv_cache.dtype,
        device=device,
    )
    workspace = build_paged_decode_workspace(query)
    block_table_backing = torch.zeros(
        (1, int(kv_cache.shape[0])),
        dtype=torch.int32,
        device=device,
    )
    metadata = PAPPagedFlashMetadata(
        block_table=block_table_backing[:, :1],
        seq_lens=torch.ones(1, dtype=torch.int32, device=device),
        cu_seqlens_q=torch.arange(2, dtype=torch.int32, device=device),
        max_seq_len=1,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        run_paged_decode_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            workspace=workspace,
            scale=float(int(head_dim) ** -0.5),
            block_size=int(block_size),
        )
    stream.synchronize()


__all__ = [
    "PAPAttentionStepTensorCache",
    "PAPPagedDecodeKernelConfig",
    "PAPPagedDecodeWorkspace",
    "PAPPagedDecodeWorkspaceCache",
    "PAP_TRITON_DECODE_DEFAULT_CONFIG",
    "PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG",
    "PAP_TRITON_DECODE_LOW_RESOURCE_MAX_SMS",
    "build_paged_decode_workspace",
    "paged_decode_kernel_config_for_sms",
    "run_paged_decode_attention",
    "warm_paged_decode_attention",
]
