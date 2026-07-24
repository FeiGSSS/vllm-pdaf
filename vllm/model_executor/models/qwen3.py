# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

import atexit
import json
import os
import threading
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.pap.deferred_cuda_trace import (
    begin_deferred_cuda_span,
    deferred_cuda_trace_enabled,
    deferred_trace_role,
    end_deferred_cuda_span,
)
from vllm.pap.integration.settings import (
    pap_env_enabled,
    pap_projection_critical_trace_enabled,
)
from vllm.pap.mode import pap_request_ids_are_routable
from vllm.pap.model.cudagraph import (
    bind_pap_cudagraph_adapters,
    pap_cudagraph_role,
    pap_model_hooks_enabled,
)
from vllm.pap.model.prefill import PAPPrefillKVPublisher
from vllm.pap.model.projection import PAPProjectionAttentionAdapter
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    LocalArgmaxMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _pap_projection_decode_trace_enabled() -> bool:
    if not pap_projection_critical_trace_enabled():
        return False
    if not is_forward_context_available():
        return False
    additional_kwargs = get_forward_context().additional_kwargs or {}
    if not additional_kwargs.get("pap_enabled"):
        return False
    request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
    num_scheduled_tokens = tuple(
        int(num_tokens)
        for num_tokens in additional_kwargs.get("pap_num_scheduled_tokens") or ()
    )
    num_reqs = int(additional_kwargs.get("pap_num_reqs") or len(num_scheduled_tokens))
    return bool(
        num_reqs > 0
        and len(request_ids) >= num_reqs
        and len(num_scheduled_tokens) >= num_reqs
        and pap_request_ids_are_routable(request_ids, num_reqs)
        and all(num_tokens == 1 for num_tokens in num_scheduled_tokens[:num_reqs])
    )


def _qwen3_layer_profile_enabled() -> bool:
    return os.environ.get("VLLM_QWEN3_LAYER_PROFILE", "").lower() in _TRUE_ENV_VALUES


def _qwen3_layer_profile_async_enabled() -> bool:
    return (
        os.environ.get("VLLM_QWEN3_LAYER_PROFILE_ASYNC", "").lower() in _TRUE_ENV_VALUES
    )


def _qwen3_profile_attn_metadata(layer_name: str) -> Any | None:
    if not is_forward_context_available():
        return None
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, dict):
        return metadata.get(layer_name)
    if isinstance(metadata, list) and metadata:
        return metadata[0].get(layer_name)
    return None


def _qwen3_deferred_qkv_trace_selected_role(
    *,
    trace_role: str,
    pap_attention_enabled: bool,
    max_query_len: int,
) -> str:
    """Select a bilateral QKV trace role for one real decode layer."""

    if trace_role == "projection" and pap_attention_enabled:
        return trace_role
    if trace_role == "pd_decode" and not pap_attention_enabled and max_query_len == 1:
        return trace_role
    return ""


def _qwen3_profile_decode_key(
    layer_name: str,
    hidden_states: torch.Tensor,
) -> tuple[int, int, int] | None:
    metadata = _qwen3_profile_attn_metadata(layer_name)
    configured_batch_size = int(
        os.environ.get("VLLM_QWEN3_LAYER_PROFILE_CONFIGURED_BATCH_SIZE", "0") or 0
    )
    configured_prompt_len = int(
        os.environ.get("VLLM_QWEN3_LAYER_PROFILE_PROMPT_LEN", "0") or 0
    )
    if metadata is None:
        total_tokens = int(hidden_states.shape[0])
        if configured_batch_size > 0 and total_tokens == configured_batch_size:
            return configured_batch_size, configured_prompt_len, total_tokens
        return None
    if int(getattr(metadata, "max_query_len", 0)) != 1:
        return None
    num_reqs = int(getattr(metadata, "num_reqs", 0))
    is_prefilling = getattr(metadata, "is_prefilling", None)
    if (
        is_prefilling is not None
        and num_reqs > 0
        and bool(torch.any(is_prefilling[:num_reqs]).item())
    ):
        return None
    batch_size = int(getattr(metadata, "num_actual_tokens", hidden_states.shape[0]))
    context_len = int(getattr(metadata, "max_seq_len", 0))
    total_tokens = int(hidden_states.shape[0])
    return batch_size, context_len, total_tokens


class _Qwen3LayerProfileScope:
    def __init__(
        self,
        *,
        layer_name: str,
        layer_index: int,
        stage: str,
        batch_size: int,
        context_len: int,
        total_tokens: int,
    ) -> None:
        self.layer_name = layer_name
        self.layer_index = layer_index
        self.stage = stage
        self.batch_size = batch_size
        self.context_len = context_len
        self.total_tokens = total_tokens
        self.start: torch.cuda.Event | None = None
        self.end: torch.cuda.Event | None = None

    def __enter__(self) -> None:
        if not _qwen3_layer_profile_async_enabled():
            _qwen3_profile_drain_pending()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None or self.start is None or self.end is None:
            return
        self.end.record()
        if _qwen3_layer_profile_async_enabled():
            _qwen3_profile_enqueue_pending(self)
        else:
            self.end.synchronize()
            self.write_sample()

    def write_sample(self) -> None:
        if self.start is None or self.end is None:
            return
        _qwen3_profile_write_sample(
            layer_name=self.layer_name,
            layer_index=self.layer_index,
            stage=self.stage,
            batch_size=self.batch_size,
            context_len=self.context_len,
            total_tokens=self.total_tokens,
            elapsed_ms=float(self.start.elapsed_time(self.end)),
        )


_QWEN3_PROFILE_PENDING_LOCK = threading.Lock()
_QWEN3_PROFILE_PENDING: list[_Qwen3LayerProfileScope] = []


def _qwen3_profile_enqueue_pending(scope: _Qwen3LayerProfileScope) -> None:
    flush_threshold = int(
        os.environ.get("VLLM_QWEN3_LAYER_PROFILE_ASYNC_FLUSH_THRESHOLD", "4096") or 4096
    )
    should_flush = False
    with _QWEN3_PROFILE_PENDING_LOCK:
        _QWEN3_PROFILE_PENDING.append(scope)
        should_flush = len(_QWEN3_PROFILE_PENDING) >= flush_threshold
    if should_flush:
        _qwen3_profile_drain_pending(block=True)


def _qwen3_profile_drain_pending(*, block: bool = False) -> None:
    if not _QWEN3_PROFILE_PENDING:
        return
    ready: list[_Qwen3LayerProfileScope] = []
    with _QWEN3_PROFILE_PENDING_LOCK:
        pending = list(_QWEN3_PROFILE_PENDING)
        _QWEN3_PROFILE_PENDING.clear()
        for scope in pending:
            end = scope.end
            if end is None:
                continue
            try:
                if block:
                    end.synchronize()
                    ready.append(scope)
                elif end.query():
                    ready.append(scope)
                else:
                    _QWEN3_PROFILE_PENDING.append(scope)
            except Exception:
                # CUDA may already be shutting down at process exit.
                continue
    for scope in ready:
        scope.write_sample()


atexit.register(lambda: _qwen3_profile_drain_pending(block=True))


def _qwen3_profile_scope(
    *,
    layer_name: str,
    layer_index: int,
    stage: str,
    hidden_states: torch.Tensor,
) -> _Qwen3LayerProfileScope | None:
    if not _qwen3_layer_profile_enabled():
        return None
    if not hidden_states.is_cuda:
        return None
    decode_key = _qwen3_profile_decode_key(layer_name, hidden_states)
    if decode_key is None:
        return None
    batch_size, context_len, total_tokens = decode_key
    return _Qwen3LayerProfileScope(
        layer_name=layer_name,
        layer_index=layer_index,
        stage=stage,
        batch_size=batch_size,
        context_len=context_len,
        total_tokens=total_tokens,
    )


def _qwen3_profile_context(
    *,
    layer_name: str,
    layer_index: int,
    stage: str,
    hidden_states: torch.Tensor,
) -> Any:
    scope = _qwen3_profile_scope(
        layer_name=layer_name,
        layer_index=layer_index,
        stage=stage,
        hidden_states=hidden_states,
    )
    return scope if scope is not None else nullcontext()


def _qwen3_profile_write_sample(
    *,
    layer_name: str,
    layer_index: int,
    stage: str,
    batch_size: int,
    context_len: int,
    total_tokens: int,
    elapsed_ms: float,
) -> None:
    output_dir = os.environ.get("VLLM_QWEN3_LAYER_PROFILE_DIR")
    if not output_dir:
        return
    try:
        rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
    except Exception:
        rank = int(os.environ.get("RANK", "0"))
        tp_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"samples_pid{os.getpid()}_rank{rank}.jsonl")
    sample = {
        "run_id": os.environ.get("VLLM_QWEN3_LAYER_PROFILE_RUN_ID", ""),
        "model": os.environ.get("VLLM_QWEN3_LAYER_PROFILE_MODEL", ""),
        "pid": os.getpid(),
        "rank": rank,
        "tp_size": tp_size,
        "layer_name": layer_name,
        "layer_index": layer_index,
        "stage": stage,
        "batch_size": batch_size,
        "context_len": context_len,
        "configured_prompt_len": int(
            os.environ.get("VLLM_QWEN3_LAYER_PROFILE_PROMPT_LEN", "0") or 0
        ),
        "configured_batch_size": int(
            os.environ.get("VLLM_QWEN3_LAYER_PROFILE_CONFIGURED_BATCH_SIZE", "0") or 0
        ),
        "total_tokens": total_tokens,
        "elapsed_ms": elapsed_ms,
    }
    with open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(sample, sort_keys=True) + "\n")


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        is_last_layer: bool = False,
        num_hidden_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_index = extract_layer_index(prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self._pap_projection_adapter = PAPProjectionAttentionAdapter(
            layer_name=self.attn.layer_name,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
        )
        self._pap_prefill_publisher = PAPPrefillKVPublisher(
            layer_name=self.attn.layer_name,
            num_kv_heads=self.num_kv_heads,
            is_last_layer=bool(is_last_layer),
            expected_layer_count=int(num_hidden_layers or 0),
        )
        self._pap_cudagraph_role = pap_cudagraph_role()
        self._pap_model_hooks_enabled = pap_model_hooks_enabled()
        self._pap_trace_offload_exec = pap_env_enabled("PAP_OFFLOAD_EXEC_TRACE")
        bind_pap_cudagraph_adapters(
            self.attn,
            projection_adapter=self._pap_projection_adapter,
            prefill_publisher=self._pap_prefill_publisher,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._pap_model_hooks_enabled:
            self._pap_projection_adapter.begin_step()

        layer_name = self.attn.layer_name
        if not self._pap_model_hooks_enabled:
            pap_attention_enabled = False
        elif self._pap_cudagraph_role is None:
            pap_attention_enabled = self._pap_projection_adapter.should_execute()
        else:
            pap_attention_enabled = self._pap_cudagraph_role == "projection"
        deferred_qkv_role = ""
        if deferred_cuda_trace_enabled():
            deferred_metadata = _qwen3_profile_attn_metadata(layer_name)
            deferred_qkv_role = _qwen3_deferred_qkv_trace_selected_role(
                trace_role=deferred_trace_role(),
                pap_attention_enabled=pap_attention_enabled,
                max_query_len=int(getattr(deferred_metadata, "max_query_len", 0)),
            )
        qkv_trace = None
        if deferred_qkv_role:
            qkv_trace = begin_deferred_cuda_span(
                "qkv_norm_rope_gpu_ms",
                torch.cuda.current_stream(hidden_states.device),
            )
        trace_offload_exec = self._pap_trace_offload_exec
        trace_pre_attn_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_pre_attn_start_ns = time.perf_counter_ns() if trace_offload_exec else 0
        try:
            with _qwen3_profile_context(
                layer_name=layer_name,
                layer_index=self.layer_index,
                stage="qkv_proj",
                hidden_states=hidden_states,
            ):
                qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            with _qwen3_profile_context(
                layer_name=layer_name,
                layer_index=self.layer_index,
                stage="qk_norm_rope",
                hidden_states=hidden_states,
            ):
                q_by_head = q.view(
                    *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
                )
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                k_by_head = k.view(
                    *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
                )
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
                q, k = self.rotary_emb(positions, q, k)
        finally:
            end_deferred_cuda_span(qkv_trace)
        direct_qkv_send_buffer: torch.Tensor | None = None
        if (
            pap_attention_enabled
            and self._pap_projection_adapter.direct_qkv_send_enabled()
            and qkv.ndim == 2
            and qkv.is_contiguous()
        ):
            repack_trace = None
            if deferred_qkv_role == "projection":
                repack_trace = begin_deferred_cuda_span(
                    "projection_qk_repack_gpu_ms",
                    torch.cuda.current_stream(hidden_states.device),
                )
            try:
                qkv[:, : self.q_size].copy_(q.reshape(qkv.shape[0], self.q_size))
                qkv[:, self.q_size : self.q_size + self.kv_size].copy_(
                    k.reshape(qkv.shape[0], self.kv_size)
                )
            finally:
                end_deferred_cuda_span(repack_trace)
            direct_qkv_send_buffer = qkv
        trace_pre_attn_done_ns = time.perf_counter_ns() if trace_offload_exec else 0
        trace_pre_attn_compute_ms = (
            (time.perf_counter() - trace_pre_attn_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        if self._pap_cudagraph_role == "projection":
            attn_output = torch.empty_like(q)
            torch.ops.vllm.pap_projection_attention_with_output(
                q,
                k,
                v,
                attn_output,
                direct_qkv_send_buffer,
                layer_name,
            )
            output, _ = self.o_proj(attn_output)
            return output
        if self._pap_cudagraph_role == "prefill":
            attn_output = self.attn(q, k, v)
            torch.ops.vllm.pap_publish_prefill_kv(attn_output, layer_name)
            output, _ = self.o_proj(attn_output)
            return output
        if pap_attention_enabled:
            projection_timeline: dict[str, Any] | None = (
                {} if trace_offload_exec else None
            )
            # Remote output readiness proves that every QKV shard has been
            # consumed, so the dead query storage can assemble routed output.
            attn_output, pap_release_messages = self._pap_projection_adapter.execute(
                q,
                k,
                v,
                pre_attn_compute_ms=trace_pre_attn_compute_ms,
                pre_attn_start_ns=trace_pre_attn_start_ns,
                pre_attn_done_ns=trace_pre_attn_done_ns,
                projection_timeline=projection_timeline,
                direct_qkv_send_buffer=direct_qkv_send_buffer,
                reuse_query_output_buffer=True,
            )
            trace_o_proj_start = time.perf_counter() if trace_offload_exec else 0.0
            try:
                with _qwen3_profile_context(
                    layer_name=layer_name,
                    layer_index=self.layer_index,
                    stage="o_proj",
                    hidden_states=hidden_states,
                ):
                    output, _ = self.o_proj(attn_output)
            finally:
                for message in pap_release_messages:
                    message.release()
            if trace_offload_exec and projection_timeline:
                trace_o_proj_done_ns = time.perf_counter_ns()
                trace_o_proj_ms = (time.perf_counter() - trace_o_proj_start) * 1000.0
                projection_timeline["o_proj_ms"] = trace_o_proj_ms
                projection_timeline["o_proj_done_ns"] = trace_o_proj_done_ns
                self._pap_projection_adapter.record_projection_timeline(
                    projection_timeline
                )
                trace_self_attn_total_ms = (
                    (trace_o_proj_done_ns - trace_pre_attn_start_ns) / 1_000_000.0
                    if trace_pre_attn_start_ns
                    else (
                        trace_pre_attn_compute_ms
                        + float(projection_timeline.get("remote_total_ms", 0.0))
                        + trace_o_proj_ms
                    )
                )
                logger.info(
                    "PAP OFFLOAD_EXEC projection timeline layer=%s batches=%d "
                    "calls=%d pre_attn_compute_ms=%.3f "
                    "send_ms=%.3f trigger_ms=%.3f yield_ms=%.3f "
                    "recv_ms=%.3f o_proj_ms=%.3f remote_total_ms=%.3f "
                    "self_attn_total_ms=%.3f batch_keys=%s "
                    "pre_attn_start_ns=%d pre_attn_done_ns=%d "
                    "send_done_ns=%d yield_start_ns=%d yield_end_ns=%d "
                    "recv_done_ns=%d o_proj_done_ns=%d "
                    "route_groups=%d contiguous_route_groups=%d "
                    "direct_qkv_groups=%d packed_qkv_groups=%d "
                    "direct_output_rows=%d scattered_output_rows=%d",
                    layer_name,
                    int(projection_timeline.get("batches", 0)),
                    int(projection_timeline.get("calls", 0)),
                    trace_pre_attn_compute_ms,
                    float(projection_timeline.get("send_ms", 0.0)),
                    float(projection_timeline.get("trigger_ms", 0.0)),
                    float(projection_timeline.get("yield_ms", 0.0)),
                    float(projection_timeline.get("recv_ms", 0.0)),
                    trace_o_proj_ms,
                    float(projection_timeline.get("remote_total_ms", 0.0)),
                    trace_self_attn_total_ms,
                    str(projection_timeline.get("batch_keys", "")),
                    trace_pre_attn_start_ns,
                    trace_pre_attn_done_ns,
                    int(projection_timeline.get("send_done_ns", 0)),
                    int(projection_timeline.get("yield_start_ns", 0)),
                    int(projection_timeline.get("yield_end_ns", 0)),
                    int(projection_timeline.get("recv_done_ns", 0)),
                    trace_o_proj_done_ns,
                    int(projection_timeline.get("route_groups", 0)),
                    int(projection_timeline.get("contiguous_route_groups", 0)),
                    int(projection_timeline.get("direct_qkv_groups", 0)),
                    int(projection_timeline.get("packed_qkv_groups", 0)),
                    int(projection_timeline.get("direct_output_rows", 0)),
                    int(projection_timeline.get("scattered_output_rows", 0)),
                )
            return output
        with _qwen3_profile_context(
            layer_name=layer_name,
            layer_index=self.layer_index,
            stage="attention",
            hidden_states=hidden_states,
        ):
            attn_output = self.attn(q, k, v)
        if self._pap_model_hooks_enabled:
            self._pap_prefill_publisher.publish(self.attn)
        with _qwen3_profile_context(
            layer_name=layer_name,
            layer_index=self.layer_index,
            stage="o_proj",
            hidden_states=hidden_states,
        ):
            output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_index = extract_layer_index(prefix)
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            is_last_layer=self.layer_index == config.num_hidden_layers - 1,
            num_hidden_layers=config.num_hidden_layers,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._pap_trace_projection_layer = pap_env_enabled(
            "PAP_OFFLOAD_EXEC_TRACE"
        ) and (
            pap_env_enabled("PAP_PROJECTION_KV_UNAWARE")
            or pap_projection_critical_trace_enabled()
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_name = self.self_attn.attn.layer_name
        trace_projection_layer = self._pap_trace_projection_layer
        trace_layer_start_ns = time.perf_counter_ns() if trace_projection_layer else 0

        # Self Attention
        trace_input_norm_start = time.perf_counter() if trace_projection_layer else 0.0
        with _qwen3_profile_context(
            layer_name=layer_name,
            layer_index=self.layer_index,
            stage="input_layernorm",
            hidden_states=hidden_states,
        ):
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        trace_input_norm_done_ns = (
            time.perf_counter_ns() if trace_projection_layer else 0
        )
        trace_input_norm_ms = (
            (time.perf_counter() - trace_input_norm_start) * 1000.0
            if trace_projection_layer
            else 0.0
        )

        trace_self_attn_start = time.perf_counter() if trace_projection_layer else 0.0
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        trace_self_attn_done_ns = (
            time.perf_counter_ns() if trace_projection_layer else 0
        )
        trace_self_attn_ms = (
            (time.perf_counter() - trace_self_attn_start) * 1000.0
            if trace_projection_layer
            else 0.0
        )

        # Fully Connected
        trace_post_norm_start = time.perf_counter() if trace_projection_layer else 0.0
        with _qwen3_profile_context(
            layer_name=layer_name,
            layer_index=self.layer_index,
            stage="post_attention_layernorm",
            hidden_states=hidden_states,
        ):
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        trace_post_norm_done_ns = (
            time.perf_counter_ns() if trace_projection_layer else 0
        )
        trace_post_norm_ms = (
            (time.perf_counter() - trace_post_norm_start) * 1000.0
            if trace_projection_layer
            else 0.0
        )
        trace_mlp_start = time.perf_counter() if trace_projection_layer else 0.0
        with _qwen3_profile_context(
            layer_name=layer_name,
            layer_index=self.layer_index,
            stage="mlp",
            hidden_states=hidden_states,
        ):
            hidden_states = self.mlp(hidden_states)
        if trace_projection_layer:
            trace_mlp_done_ns = time.perf_counter_ns()
            trace_mlp_ms = (time.perf_counter() - trace_mlp_start) * 1000.0
            trace_layer_total_ms = (
                trace_mlp_done_ns - trace_layer_start_ns
            ) / 1_000_000.0
            logger.info(
                "PAP OFFLOAD_EXEC projection layer timeline layer=%s "
                "input_norm_ms=%.3f self_attn_ms=%.3f "
                "post_attention_layernorm_ms=%.3f mlp_ms=%.3f "
                "layer_total_ms=%.3f layer_start_ns=%d "
                "input_norm_done_ns=%d self_attn_done_ns=%d "
                "post_norm_done_ns=%d mlp_done_ns=%d",
                layer_name,
                trace_input_norm_ms,
                trace_self_attn_ms,
                trace_post_norm_ms,
                trace_mlp_ms,
                trace_layer_total_ms,
                trace_layer_start_ns,
                trace_input_norm_done_ns,
                trace_self_attn_done_ns,
                trace_post_norm_done_ns,
                trace_mlp_done_ns,
            )
            projection_timeline = (
                self.self_attn._pap_projection_adapter.last_projection_timeline or {}
            )
            if pap_projection_critical_trace_enabled() and projection_timeline:
                qkv_ms = float(projection_timeline.get("pre_attn_compute_ms", 0.0))
                send_ms = float(projection_timeline.get("send_ms", 0.0))
                recv_ms = float(projection_timeline.get("recv_ms", 0.0))
                o_proj_ms = float(projection_timeline.get("o_proj_ms", 0.0))
                known_ms = (
                    trace_input_norm_ms
                    + qkv_ms
                    + send_ms
                    + recv_ms
                    + o_proj_ms
                    + trace_post_norm_ms
                    + trace_mlp_ms
                )
                logger.info(
                    "PAP OFFLOAD_EXEC projection critical path layer=%s "
                    "batch_key=%s calls=%d input_norm_ms=%.3f qkv_ms=%.3f "
                    "send_ms=%.3f recv_ms=%.3f o_proj_ms=%.3f "
                    "post_norm_ms=%.3f mlp_ms=%.3f layer_total_ms=%.3f "
                    "gaps_ms=%.3f layer_start_ns=%d input_norm_done_ns=%d "
                    "qkv_done_ns=%d send_done_ns=%d recv_done_ns=%d "
                    "o_proj_done_ns=%d post_norm_done_ns=%d mlp_done_ns=%d",
                    layer_name,
                    str(projection_timeline.get("batch_keys", "")),
                    int(projection_timeline.get("calls", 0)),
                    trace_input_norm_ms,
                    qkv_ms,
                    send_ms,
                    recv_ms,
                    o_proj_ms,
                    trace_post_norm_ms,
                    trace_mlp_ms,
                    trace_layer_total_ms,
                    max(0.0, trace_layer_total_ms - known_ms),
                    trace_layer_start_ns,
                    trace_input_norm_done_ns,
                    int(projection_timeline.get("pre_attn_done_ns", 0)),
                    int(projection_timeline.get("send_done_ns", 0)),
                    int(projection_timeline.get("recv_done_ns", 0)),
                    int(projection_timeline.get("o_proj_done_ns", 0)),
                    trace_post_norm_done_ns,
                    trace_mlp_done_ns,
                )
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


class Qwen3ForCausalLM(
    LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.vllm_config = vllm_config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        trace_projection = _pap_projection_decode_trace_enabled()
        trace_start = time.perf_counter() if trace_projection else 0.0
        trace_start_ns = time.perf_counter_ns() if trace_projection else 0
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        if trace_projection:
            trace_done_ns = time.perf_counter_ns()
            num_tokens = 0
            if isinstance(hidden_states, torch.Tensor):
                num_tokens = int(hidden_states.shape[0])
            elif input_ids is not None:
                num_tokens = int(input_ids.shape[0])
            elif inputs_embeds is not None:
                num_tokens = int(inputs_embeds.shape[0])
            logger.info(
                "PAP OFFLOAD_EXEC projection model forward num_tokens=%d "
                "model_forward_ms=%.3f model_start_ns=%d model_done_ns=%d",
                num_tokens,
                (time.perf_counter() - trace_start) * 1000.0,
                trace_start_ns,
                trace_done_ns,
            )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        trace_projection = _pap_projection_decode_trace_enabled()
        trace_start = time.perf_counter() if trace_projection else 0.0
        trace_start_ns = time.perf_counter_ns() if trace_projection else 0
        logits = self.logits_processor(self.lm_head, hidden_states)
        if trace_projection:
            trace_done_ns = time.perf_counter_ns()
            logger.info(
                "PAP OFFLOAD_EXEC projection logits num_tokens=%d logits_ms=%.3f "
                "logits_start_ns=%d logits_done_ns=%d",
                int(hidden_states.shape[0]),
                (time.perf_counter() - trace_start) * 1000.0,
                trace_start_ns,
                trace_done_ns,
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
