# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP-owned paged decode-attention kernel integration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Literal

import torch

from vllm.pap.kv.layout import split_paged_kv_cache
from vllm.pap.kv.metadata import PAPPagedFlashMetadata
from vllm.platforms import current_platform

PAP_TRITON_DECODE_LOW_RESOURCE_MAX_SMS = 20


@dataclass(frozen=True)
class PAPPagedDecodeKernelConfig:
    """Launch specialization for PAP grouped-query decode Attention."""

    num_splits: int
    block_h: int
    num_warps: int
    num_stages: int
    block_n: int = 32

    def __post_init__(self) -> None:
        if self.num_splits <= 0:
            raise ValueError("PAP decode num_splits must be positive")
        if self.block_h not in (1, 2, 4, 8, 16):
            raise ValueError("PAP decode block_h must be a supported power of two")
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError("PAP decode num_warps is unsupported")
        if self.num_stages <= 0:
            raise ValueError("PAP decode num_stages must be positive")
        if self.block_n not in (16, 32, 64, 128):
            raise ValueError("PAP decode block_n must be a supported power of two")


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
            tuple[torch.Tensor, torch.Tensor, torch.Event | None],
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
                buffers = (host, target, None)
                self._entries[key] = buffers
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(key)
            host, target, copy_done = buffers
            if copy_done is not None:
                copy_done.synchronize()
            host.copy_(torch.tensor(values, dtype=dtype))
            target.copy_(
                host,
                non_blocking=normalized_device.type == "cuda",
            )
            if normalized_device.type == "cuda":
                if copy_done is None:
                    copy_done = torch.cuda.Event()
                copy_done.record(torch.cuda.current_stream(normalized_device))
                self._entries[key] = (host, target, copy_done)
            return target


def build_paged_decode_workspace(
    query: torch.Tensor,
    kernel_config: PAPPagedDecodeKernelConfig | None = None,
) -> PAPPagedDecodeWorkspace:
    """Allocate the fixed PAP decode scratch once for a decode step."""

    if query.ndim != 3:
        raise ValueError("PAP paged decode query must be rank 3")
    batch_size, num_heads, head_dim = map(int, query.shape)
    visible_sms = (
        current_platform.num_compute_units(query.device.index or 0)
        if query.device.type == "cuda"
        else 0
    )
    kernel_config = kernel_config or paged_decode_kernel_config_for_sms(visible_sms)
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


def _run_grouped_paged_decode_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    metadata: PAPPagedFlashMetadata,
    workspace: PAPPagedDecodeWorkspace,
    scale: float,
    block_size: int,
) -> None:
    """Launch the PAP low-resource GQA specialization on vLLM's kernel."""
    from vllm.v1.attention.ops import triton_decode_attention as decode_ops

    config = workspace.kernel_config
    key_dim = int(key_cache.shape[-1])
    value_dim = int(value_cache.shape[-1])
    block_dv = 1 << (value_dim - 1).bit_length()
    block_dmodel = 1 << (key_dim - 1).bit_length()
    block_n = 16 if decode_ops.is_hip_ else config.block_n
    batch_size, num_heads = map(int, query.shape[:2])
    kv_group_count = num_heads // int(key_cache.shape[-2])
    grid = (
        batch_size,
        (num_heads + min(config.block_h, kv_group_count) - 1)
        // min(config.block_h, kv_group_count),
        config.num_splits,
    )
    extra_args = {}
    if decode_ops.is_hip_:
        extra_args = {
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        }

    decode_ops._fwd_grouped_kernel_stage1[grid](
        query,
        key_cache,
        value_cache,
        float(scale),
        metadata.block_table,
        metadata.seq_lens,
        workspace.partial,
        metadata.block_table.stride(0),
        query.stride(0),
        query.stride(1),
        decode_ops._page_stride(key_cache, int(block_size)),
        key_cache.stride(-3),
        key_cache.stride(-2),
        decode_ops._page_stride(value_cache, int(block_size)),
        value_cache.stride(-3),
        value_cache.stride(-2),
        workspace.partial.stride(0),
        workspace.partial.stride(1),
        workspace.partial.stride(2),
        workspace.k_scale,
        workspace.v_scale,
        kv_group_num=kv_group_count,
        q_head_num=num_heads,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DPE=0,
        BLOCK_DV=block_dv,
        BLOCK_N=block_n,
        BLOCK_H=config.block_h,
        NUM_KV_SPLITS=config.num_splits,
        PAGE_SIZE=int(block_size),
        logit_cap=0.0,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        Lk=key_dim,
        Lv=value_dim,
        IS_MLA=False,
        **extra_args,
    )
    decode_ops._decode_softmax_reducev_fwd(
        workspace.partial,
        query,
        workspace.output,
        workspace.lse,
        value_cache,
        metadata.seq_lens,
        config.num_splits,
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
    implementation: Literal["auto", "pap_grouped", "vllm"] = "auto",
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

    if implementation not in ("auto", "pap_grouped", "vllm"):
        raise ValueError(f"unknown PAP decode implementation: {implementation}")
    has_grouped_query_attention = int(query.shape[1]) != int(key_cache.shape[-2])
    use_low_resource_grouped_kernel = implementation == "pap_grouped" or (
        implementation == "auto"
        and workspace.kernel_config == PAP_TRITON_DECODE_LOW_RESOURCE_CONFIG
        and has_grouped_query_attention
    )
    if use_low_resource_grouped_kernel and not has_grouped_query_attention:
        raise RuntimeError("PAP grouped decode requires grouped-query attention")
    if use_low_resource_grouped_kernel:
        _run_grouped_paged_decode_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            metadata=metadata,
            workspace=workspace,
            scale=scale,
            block_size=block_size,
        )
    else:
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
        or kv_cache.ndim not in (4, 5)
        or int(num_heads) <= 0
        or int(head_dim) <= 0
    ):
        return
    key_cache, value_cache = split_paged_kv_cache(kv_cache, int(head_dim))
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
    with stream:
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
