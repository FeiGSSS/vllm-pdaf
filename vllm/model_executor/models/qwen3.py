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

import hashlib
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
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
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _pap_env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_ENV_VALUES


def _pap_offload_exec_microbatch_count(num_reqs: int) -> int:
    raw = os.environ.get("PAP_OFFLOAD_EXEC_MICROBATCH_COUNT", "0")
    try:
        configured = int(raw)
    except ValueError:
        configured = 0
    if configured <= 1 or num_reqs <= 1:
        return 1
    return min(configured, int(num_reqs))


def _pap_contiguous_microbatches(num_reqs: int, count: int) -> list[list[int]]:
    count = max(1, min(int(count), int(num_reqs)))
    base, extra = divmod(int(num_reqs), count)
    batches: list[list[int]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        end = start + size
        if start < end:
            batches.append(list(range(start, end)))
        start = end
    return batches


def _pap_qkv_projection_split_supported(qkv_proj: Any) -> bool:
    return bool(
        qkv_proj.quant_method.__class__.__name__ == "UnquantizedLinearMethod"
        and hasattr(qkv_proj, "weight")
        and not getattr(qkv_proj, "skip_bias_add", False)
        and not getattr(qkv_proj, "gather_output", False)
    )


def _pap_qkv_projection_slice(
    qkv_proj: Any,
    hidden_states: torch.Tensor,
    start: int,
    size: int,
) -> torch.Tensor:
    weight = qkv_proj.weight.narrow(0, int(start), int(size))
    bias = getattr(qkv_proj, "bias", None)
    if bias is not None:
        bias = bias.narrow(0, int(start), int(size))
    return F.linear(hidden_states, weight, bias)


def _pap_block_ids_from_block_table(
    *,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> list[int]:
    if block_table.ndim != 2 or int(block_table.shape[0]) != 1:
        raise ValueError("PAP KV import supports one request per block table")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    num_blocks = (int(seq_len) + int(block_size) - 1) // int(block_size)
    return [
        int(block_id)
        for block_id in block_table[0, :num_blocks]
        .detach()
        .to(device="cpu", dtype=torch.long)
        .tolist()
    ]


@lru_cache(maxsize=8)
def _pap_remote_attention_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="pap-attn-rpc",
    )


def _pap_offload_exec_transport_kind() -> str:
    return os.environ.get("PAP_OFFLOAD_EXEC_TRANSPORT", "nccl").lower()


def _pap_tensor_parallel_rank() -> int:
    raw_rank = os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK")
    if raw_rank is not None:
        return int(raw_rank)
    return int(get_tensor_model_parallel_rank())


def _pap_endpoint_for_tp_rank(value: Any, *, tp_rank: int | None = None) -> Any:
    """Select this TP rank's endpoint from a ranked endpoint list.

    PAP control-plane payloads keep backward-compatible scalar endpoint values for
    TP=1. TP>1 uses comma-separated strings, or a sequence in unit tests, where
    entry N belongs to tensor-parallel rank N.
    """

    if value is None:
        return None
    rank = _pap_tensor_parallel_rank() if tp_rank is None else int(tp_rank)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) <= 1:
            return value
        if rank >= len(parts):
            raise RuntimeError(
                f"PAP endpoint list has {len(parts)} ranks, but TP rank is {rank}"
            )
        return parts[rank]
    if isinstance(value, (list, tuple)):
        if len(value) <= 1:
            return value[0] if value else None
        if rank >= len(value):
            raise RuntimeError(
                f"PAP endpoint list has {len(value)} ranks, but TP rank is {rank}"
            )
        return value[rank]
    return value


@lru_cache(maxsize=1)
def _pap_offload_exec_transport():
    from vllm.pap.data_plane import (
        build_nixl_mailbox_offload_exec_transport,
        build_p2p_nccl_offload_exec_transport,
    )

    transport = _pap_offload_exec_transport_kind()
    local_rank = _pap_tensor_parallel_rank()
    if transport in {"nixl", "nixl_mailbox"}:
        return build_nixl_mailbox_offload_exec_transport(
            actor_id=os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection"),
            local_rank=local_rank,
        )

    return build_p2p_nccl_offload_exec_transport(
        local_rank=local_rank,
        kv_port=int(os.environ.get("PAP_OFFLOAD_EXEC_ZMQ_PORT", "11300")),
        hostname=os.environ.get("PAP_OFFLOAD_EXEC_HOST", ""),
    )


@lru_cache(maxsize=32)
def _pap_nixl_mailbox_offload_exec_transport(attention_endpoint: str):
    from vllm.pap.data_plane import build_nixl_mailbox_offload_exec_transport

    local_rank = _pap_tensor_parallel_rank()
    actor_base = os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection")
    endpoint_hash = hashlib.sha1(attention_endpoint.encode("utf-8")).hexdigest()[:12]
    return build_nixl_mailbox_offload_exec_transport(
        actor_id=f"{actor_base}-{endpoint_hash}",
        local_rank=local_rank,
    )


def _pap_offload_exec_transport_for_attention_endpoint(attention_endpoint: str | None):
    transport = _pap_offload_exec_transport_kind()
    if transport in {"nixl", "nixl_mailbox"}:
        return _pap_nixl_mailbox_offload_exec_transport(str(attention_endpoint or ""))
    return _pap_offload_exec_transport()


def _pap_bind_offload_exec_mailbox_peer(
    transport: Any,
    attention_endpoint: str | None,
) -> None:
    if getattr(transport, "requires_tcp_trigger", True):
        return
    if not attention_endpoint:
        raise RuntimeError(
            "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
        )
    if getattr(transport, "_pap_mailbox_bound", False):
        return
    from vllm.pap.shadow_attention import bind_offload_exec_mailbox

    peer_metadata = bind_offload_exec_mailbox(
        attention_endpoint=attention_endpoint,
        local_agent_metadata=transport.local_agent_metadata,
    )
    transport.bind_peer(peer_metadata)
    transport._pap_mailbox_bound = True
    transport._pap_mailbox_bound_attention_endpoint = attention_endpoint


def _pap_offload_exec_local_address() -> str:
    host = os.environ.get("PAP_OFFLOAD_EXEC_HOST") or "127.0.0.1"
    port = int(
        _pap_endpoint_for_tp_rank(os.environ.get("PAP_OFFLOAD_EXEC_ZMQ_PORT", "11300"))
    )
    return os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_ADDRESS", f"{host}:{port}")


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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
        self._pap_imported_prefill_kv: set[tuple[str, str, int, str]] = set()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._should_use_pap_attention():
            microbatch_result = self._pap_attention_microbatch_pipeline(
                positions,
                hidden_states,
            )
            if microbatch_result is not None:
                return microbatch_result

            q_first_result = self._compute_pap_attention_q_first_projection(
                positions,
                hidden_states,
            )
            if q_first_result is not None:
                attn_output, pap_release_messages = q_first_result
                try:
                    output, _ = self.o_proj(attn_output)
                finally:
                    for message in pap_release_messages:
                        message.release()
                return output

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        if self._should_use_pap_attention():
            attn_output, pap_release_messages = self._compute_pap_attention(q, k, v)
            try:
                output, _ = self.o_proj(attn_output)
            finally:
                for message in pap_release_messages:
                    message.release()
            return output
        attn_output = self.attn(q, k, v)
        self._maybe_import_pap_prefill_kv_to_attention()
        output, _ = self.o_proj(attn_output)
        return output

    def _should_use_pap_attention(self) -> bool:
        def reject(reason: str) -> bool:
            if _pap_env_enabled("PAP_DEBUG_DECISION"):
                logger.info(
                    "PAP attention disabled for %s: %s",
                    getattr(self.attn, "layer_name", "<unknown>"),
                    reason,
                )
            return False

        if not is_forward_context_available():
            return reject("missing forward context")
        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        if not additional_kwargs.get("pap_enabled"):
            return reject("pap_enabled is false")

        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        if not self._select_pap_request_id(request_ids):
            return reject(f"no selected request id request_ids={request_ids[:4]}")

        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            attn_metadata = metadata.get(self.attn.layer_name)
        elif isinstance(metadata, list) and metadata:
            attn_metadata = metadata[0].get(self.attn.layer_name)
        else:
            attn_metadata = None
        if attn_metadata is None:
            metadata_keys: list[str] = []
            if isinstance(metadata, dict):
                metadata_keys = list(metadata.keys())[:4]
            elif (
                isinstance(metadata, list)
                and metadata
                and isinstance(metadata[0], dict)
            ):
                metadata_keys = list(metadata[0].keys())[:4]
            return reject(f"missing attn metadata metadata_keys={metadata_keys}")
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
            return reject(
                f"max_query_len={getattr(attn_metadata, 'max_query_len', None)}"
            )

        num_scheduled_tokens = tuple(
            int(num_tokens)
            for num_tokens in additional_kwargs.get("pap_num_scheduled_tokens") or ()
        )
        num_reqs = int(
            additional_kwargs.get("pap_num_reqs") or len(num_scheduled_tokens)
        )
        if num_reqs <= 0:
            return reject(f"num_reqs={num_reqs}")
        if len(request_ids) < num_reqs:
            return reject(
                f"request_ids too short len={len(request_ids)} num_reqs={num_reqs}"
            )
        if len(num_scheduled_tokens) < num_reqs:
            return reject(
                "num_scheduled_tokens too short "
                f"len={len(num_scheduled_tokens)} num_reqs={num_reqs}"
            )
        if any(num_tokens != 1 for num_tokens in num_scheduled_tokens[:num_reqs]):
            return reject(
                f"non-decode num_scheduled_tokens={num_scheduled_tokens[:num_reqs]}"
            )
        if not self._pap_attention_kv_ready_for_requests(request_ids[:num_reqs]):
            installed = set(
                additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
            )
            return reject(
                "attention KV not ready "
                f"request_ids={request_ids[:num_reqs]} installed={tuple(installed)[:4]}"
            )
        return True

    def _pap_attention_kv_ready_for_requests(self, request_ids: Iterable[Any]) -> bool:
        """Return True when PA-side attention KV is ready for every request."""
        if not is_forward_context_available():
            return False
        additional_kwargs = get_forward_context().additional_kwargs or {}
        installed = set(
            additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        return all(str(request_id) in installed for request_id in request_ids)

    def _pap_attention_microbatch_pipeline(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        projected_output = torch.empty_like(hidden_states)

        def consume_projected_chunk(
            req_indices: list[int],
            output: torch.Tensor,
        ) -> None:
            index_tensor = torch.tensor(
                req_indices,
                device=projected_output.device,
                dtype=torch.long,
            )
            projected_output.index_copy_(0, index_tensor, output)

        if not self._run_pap_attention_microbatch_pipeline(
            positions,
            hidden_states,
            consume_projected_chunk,
        ):
            return None
        return projected_output

    def _run_pap_attention_microbatch_pipeline(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        consume_projected_chunk: Callable[[list[int], torch.Tensor], None],
    ) -> bool:
        if not self._should_use_pap_attention():
            return False
        if not is_forward_context_available():
            return False
        additional_kwargs = get_forward_context().additional_kwargs or {}
        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        num_reqs = int(additional_kwargs.get("pap_num_reqs") or len(request_ids))
        microbatch_count = _pap_offload_exec_microbatch_count(num_reqs)
        if microbatch_count <= 1:
            return False
        if _pap_env_enabled("PAP_Q_FIRST_KV_LATER"):
            return False
        if _pap_env_enabled("PAP_SEGMENTED_QKV"):
            return False
        if int(hidden_states.shape[0]) != num_reqs:
            return False
        if int(positions.reshape(-1).shape[0]) < num_reqs:
            return False

        transport = _pap_offload_exec_transport()
        if getattr(transport, "requires_tcp_trigger", True):
            return False

        microbatches = _pap_contiguous_microbatches(num_reqs, microbatch_count)
        if len(microbatches) <= 1:
            return False

        trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_send_ms = 0.0
        trace_recv_ms = 0.0
        trace_send_done_ns = 0
        trace_recv_done_ns = 0
        trace_sent_batches: list[tuple[str, Any, list[int], Any]] = []

        full_batch_qkv_enabled = _pap_env_enabled(
            "PAP_OFFLOAD_EXEC_MICROBATCH_FULL_QKV"
        )
        positions_flat = positions.reshape(-1)
        q_all: torch.Tensor | None = None
        k_all: torch.Tensor | None = None
        v_all: torch.Tensor | None = None
        if full_batch_qkv_enabled:
            qkv, _ = self.qkv_proj(hidden_states)
            q_all, k_all, v_all = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q_by_head = q_all.view(
                *q_all.shape[:-1], q_all.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q_all = q_by_head.view(q_all.shape)
            k_by_head = k_all.view(
                *k_all.shape[:-1], k_all.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k_all = k_by_head.view(k_all.shape)
            positions_all = positions_flat[:num_reqs]
            q_all, k_all = self.rotary_emb(positions_all, q_all, k_all)

        pending_batches: list[
            tuple[
                int,
                list[int],
                torch.Tensor,
                list[tuple[str, Any, list[int], Any]],
            ]
        ] = []
        send_cursor = 0

        def send_next_microbatch() -> None:
            nonlocal send_cursor, trace_send_done_ns, trace_send_ms
            if send_cursor >= len(microbatches):
                return
            trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
            microbatch_id = send_cursor
            req_indices = microbatches[microbatch_id]
            send_cursor += 1
            start = req_indices[0]
            end = req_indices[-1] + 1
            if full_batch_qkv_enabled:
                assert q_all is not None
                assert k_all is not None
                assert v_all is not None
                q = q_all[start:end]
                k = k_all[start:end]
                v = v_all[start:end]
            else:
                hidden_chunk = hidden_states[start:end]
                positions_chunk = positions_flat[start:end]
                qkv, _ = self.qkv_proj(hidden_chunk)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
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
                q, k = self.rotary_emb(positions_chunk, q, k)
            sent = self._send_pap_attention_batch(
                q,
                k,
                v,
                request_indices=req_indices,
                transport=transport,
            )
            if trace_offload_exec:
                trace_send_ms += (time.perf_counter() - trace_send_start) * 1000.0
                trace_send_done_ns = time.perf_counter_ns()
                trace_sent_batches.extend(sent)
            pending_batches.append((microbatch_id, req_indices, q, sent))

        def consume_pending_microbatch(pending: Any) -> None:
            nonlocal trace_recv_done_ns, trace_recv_ms
            _microbatch_id, req_indices, query, sent_batches = pending
            trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
            chunk_output, chunk_release_messages = self._recv_pap_attention_batch(
                query,
                sent_batches,
                transport=transport,
            )
            if trace_offload_exec:
                trace_recv_ms += (time.perf_counter() - trace_recv_start) * 1000.0
                trace_recv_done_ns = time.perf_counter_ns()
            try:
                output, _ = self.o_proj(chunk_output)
            finally:
                for message in chunk_release_messages:
                    message.release()
            consume_projected_chunk(req_indices, output)

        if _pap_env_enabled("PAP_OFFLOAD_EXEC_MICROBATCH_STREAMING"):
            while send_cursor < min(2, len(microbatches)):
                send_next_microbatch()

            while pending_batches:
                pending = pending_batches.pop(0)
                _microbatch_id, req_indices, query, sent_batches = pending
                trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
                chunk_output, chunk_release_messages = self._recv_pap_attention_batch(
                    query,
                    sent_batches,
                    transport=transport,
                )
                if trace_offload_exec:
                    trace_recv_ms += (time.perf_counter() - trace_recv_start) * 1000.0
                    trace_recv_done_ns = time.perf_counter_ns()
                send_next_microbatch()
                try:
                    output, _ = self.o_proj(chunk_output)
                finally:
                    for message in chunk_release_messages:
                        message.release()
                consume_projected_chunk(req_indices, output)
        else:
            while send_cursor < len(microbatches):
                send_next_microbatch()
            for pending in pending_batches:
                consume_pending_microbatch(pending)
        if trace_offload_exec and trace_sent_batches:
            from vllm.pap.data_plane import pap_offload_exec_trace_id

            trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
            batch_keys = "|".join(
                pap_offload_exec_trace_id(batch[1].output_tensor_id)
                for batch in trace_sent_batches
            )
            logger.info(
                "PAP OFFLOAD_EXEC projection trace layer=%s batches=%d calls=%d "
                "send_ms=%.3f trigger_ms=%.3f yield_ms=%.3f "
                "recv_ms=%.3f total_ms=%.3f batch_keys=%s "
                "send_done_ns=%d yield_start_ns=%d yield_end_ns=%d "
                "recv_done_ns=%d",
                trace_sent_batches[0][1].layer_name,
                len(trace_sent_batches),
                sum(len(batch[1].items) for batch in trace_sent_batches),
                trace_send_ms,
                0.0,
                0.0,
                trace_recv_ms,
                trace_total_ms,
                batch_keys,
                trace_send_done_ns,
                trace_send_done_ns,
                trace_send_done_ns,
                trace_recv_done_ns,
            )
        return True

    def _start_pap_attention_wavefront_batch(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        request_indices: list[int],
        transport: Any,
    ) -> tuple[torch.Tensor, list[tuple[str, Any, list[int], Any]]] | None:
        if not self._should_use_pap_attention():
            return None
        if _pap_env_enabled("PAP_Q_FIRST_KV_LATER"):
            return None
        if _pap_env_enabled("PAP_SEGMENTED_QKV"):
            return None
        if int(hidden_states.shape[0]) != len(request_indices):
            return None
        positions_flat = positions.reshape(-1)
        if int(positions_flat.shape[0]) < len(request_indices):
            return None

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions_flat[: len(request_indices)], q, k)
        sent = self._send_pap_attention_batch(
            q,
            k,
            v,
            request_indices=request_indices,
            transport=transport,
        )
        return q, sent

    def _finish_pap_attention_wavefront_batch(
        self,
        pending: tuple[torch.Tensor, list[tuple[str, Any, list[int], Any]]],
        *,
        transport: Any,
    ) -> torch.Tensor:
        query, sent_batches = pending
        chunk_output, chunk_release_messages = self._recv_pap_attention_batch(
            query,
            sent_batches,
            transport=transport,
        )
        try:
            output, _ = self.o_proj(chunk_output)
        finally:
            for message in chunk_release_messages:
                message.release()
        return output

    def _send_pap_attention_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        request_indices: list[int],
        transport: Any,
    ) -> list[tuple[str, Any, list[int]]]:
        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            attn_metadata = metadata.get(self.attn.layer_name)
        elif isinstance(metadata, list) and metadata:
            attn_metadata = metadata[0].get(self.attn.layer_name)
        else:
            attn_metadata = None
        if attn_metadata is None:
            raise RuntimeError(
                f"PAP attention missing metadata for {self.attn.layer_name}"
            )
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
            raise RuntimeError("PAP microbatch attention supports decode-only batches")

        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        num_scheduled_tokens = tuple(
            int(num_tokens)
            for num_tokens in additional_kwargs.get("pap_num_scheduled_tokens") or ()
        )
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            raise RuntimeError("PAP attention missing scheduler seq_lens")
        positions = additional_kwargs.get("pap_positions")
        if positions is None:
            raise RuntimeError("PAP attention missing input positions")
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        positions_cpu = (
            positions.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        )

        from vllm.pap.data_plane import (
            PAPOffloadExecBatchDescriptor,
            PAPOffloadExecDescriptor,
        )
        from vllm.pap.shadow_attention import select_attention_endpoint_for_request

        offload_exec_zmq_endpoint_by_request = (
            additional_kwargs.get("pap_offload_exec_zmq_endpoint_by_request") or {}
        )
        tcp_endpoint_by_request = (
            additional_kwargs.get("pap_attention_tcp_endpoint_by_request") or {}
        )
        attention_endpoint_by_request = (
            additional_kwargs.get("pap_attention_endpoint_by_request") or {}
        )
        default_tcp_endpoint = additional_kwargs.get("pap_attention_tcp_endpoint")
        default_attention_endpoint = additional_kwargs.get("pap_attention_endpoint")
        attention_kv_installed_by_request = set(
            additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        prefix_len_by_request = (
            additional_kwargs.get("pap_prefill_prefix_len_by_request") or {}
        )
        prefill_kv_handle_by_request = (
            additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
        )

        query = q.view(-1, self.num_heads, self.head_dim)
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view(-1, self.num_kv_heads, self.head_dim)
        offload_exec_groups: dict[
            tuple[str | None, str | None, str],
            list[tuple[int, PAPOffloadExecDescriptor, tuple[torch.Tensor, ...]]],
        ] = {}
        for local_index, req_index in enumerate(request_indices):
            request_id = str(request_ids[req_index])
            if not request_id.startswith(("cmpl-", "chatcmpl-")):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            if num_scheduled_tokens and int(num_scheduled_tokens[req_index]) != 1:
                raise RuntimeError(
                    "PAP microbatch attention expects one token per request"
                )
            seq_len = int(positions_cpu[req_index].item()) + 1
            max_seq_len = int(seq_lens_cpu[req_index].item())
            if seq_len > max_seq_len:
                raise RuntimeError(
                    f"PAP attention position-derived seq_len {seq_len} exceeds "
                    f"scheduler seq_len {max_seq_len} for {request_id}"
                )
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = _pap_endpoint_for_tp_rank(tcp_endpoint)
            attention_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_attention_endpoint,
                endpoint_by_request=attention_endpoint_by_request,
            )
            attention_endpoint = _pap_endpoint_for_tp_rank(attention_endpoint)
            offload_exec_zmq_endpoint = _pap_endpoint_for_tp_rank(
                offload_exec_zmq_endpoint_by_request.get(request_id)
            )
            if offload_exec_zmq_endpoint is None:
                raise RuntimeError(
                    "PAP OFFLOAD_EXEC mailbox path missing "
                    "pap_offload_exec_zmq_endpoint"
                )
            if not attention_endpoint:
                raise RuntimeError(
                    "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
                )
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed_by_request:
                if not prefill_kv_handle:
                    raise RuntimeError("PAP missing local prefill KV handle")
                raise RuntimeError("PAP attention KV is not installed")
            descriptor = PAPOffloadExecDescriptor(
                request_id=request_id,
                layer_name=self.attn.layer_name,
                step=seq_len,
                scale=float(self.scaling),
            )
            offload_exec_groups.setdefault(
                (tcp_endpoint, attention_endpoint, offload_exec_zmq_endpoint),
                [],
            ).append(
                (
                    local_index,
                    descriptor,
                    (
                        query[local_index : local_index + 1].reshape(1, -1),
                        key[local_index : local_index + 1].reshape(1, -1),
                        value[local_index : local_index + 1].reshape(1, -1),
                    ),
                )
            )

        sent_batches: list[tuple[str, Any, list[int], Any]] = []
        for (
            _tcp_endpoint,
            attention_endpoint,
            offload_exec_zmq_endpoint,
        ), group_items in offload_exec_groups.items():
            group_transport = _pap_offload_exec_transport_for_attention_endpoint(
                attention_endpoint
            )
            _pap_bind_offload_exec_mailbox_peer(group_transport, attention_endpoint)
            batch_descriptor = PAPOffloadExecBatchDescriptor(
                layer_name=self.attn.layer_name,
                items=tuple(item[1] for item in group_items),
            )
            if len(group_items) == 1:
                qkv_batch = torch.cat(group_items[0][2], dim=-1)
            else:
                qkv_batch = torch.cat(
                    [torch.cat(item[2], dim=-1) for item in group_items],
                    dim=0,
                )
            group_transport.send_qkv_batch(
                batch_descriptor,
                qkv_batch,
                remote_address=offload_exec_zmq_endpoint,
            )
            sent_batches.append(
                (
                    offload_exec_zmq_endpoint,
                    batch_descriptor,
                    [item[0] for item in group_items],
                    group_transport,
                )
            )
        return sent_batches

    def _recv_pap_attention_batch(
        self,
        query: torch.Tensor,
        sent_batches: list[tuple[str, Any, list[int], Any]],
        *,
        transport: Any,
    ) -> tuple[torch.Tensor, list[Any]]:
        output = torch.empty_like(query)
        pap_release_messages: list[Any] = []
        for (
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            group_transport,
        ) in sent_batches:
            recv_output_batch_message = getattr(
                group_transport, "recv_output_batch_message", None
            )
            output_message = None
            if callable(recv_output_batch_message):
                output_message = recv_output_batch_message(
                    batch_descriptor,
                    remote_address=offload_exec_zmq_endpoint,
                )
                output_batch = output_message.tensor
            else:
                output_batch = group_transport.recv_output_batch(
                    batch_descriptor,
                    remote_address=offload_exec_zmq_endpoint,
                )
            try:
                if int(output_batch.shape[0]) != len(batch_descriptor.items):
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC output batch row count mismatch"
                    )
                can_use_direct_output = (
                    _pap_env_enabled("PAP_DIRECT_MAILBOX_OUTPUT")
                    and output_message is not None
                    and len(sent_batches) == 1
                    and len(req_indices) == len(batch_descriptor.items)
                    and output_batch.device == query.device
                    and output_batch.dtype == query.dtype
                    and int(output_batch.numel()) == int(query.numel())
                )
                if can_use_direct_output:
                    direct_output = output_batch.view_as(query)
                    pap_release_messages.append(output_message)
                    output_message = None
                    return direct_output, pap_release_messages
                for descriptor_index, req_index in enumerate(req_indices):
                    target = output[req_index : req_index + 1]
                    remote_output = output_batch[
                        descriptor_index : descriptor_index + 1
                    ]
                    remote_output = remote_output.to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                    if remote_output.shape != target.shape:
                        if remote_output.numel() != target.numel():
                            raise RuntimeError(
                                "PAP remote attention output shape mismatch: "
                                f"got {tuple(remote_output.shape)}, "
                                f"expected {tuple(target.shape)}"
                            )
                        remote_output = remote_output.view_as(target)
                    target.copy_(remote_output)
            finally:
                if output_message is not None:
                    output_message.release()
        return output, pap_release_messages

    def _compute_pap_attention_q_first_projection(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Any]] | None:
        if not _pap_env_enabled("PAP_Q_FIRST_PROJECTION"):
            return None
        if not _pap_qkv_projection_split_supported(self.qkv_proj):
            return None
        if not self._pap_q_first_projection_transport_supported():
            return None

        q = _pap_qkv_projection_slice(
            self.qkv_proj,
            hidden_states,
            0,
            self.q_size,
        )
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        q, _ = self.rotary_emb.forward_native(positions, q, None)
        if not self._send_pap_query_batch(q):
            return None

        kv = _pap_qkv_projection_slice(
            self.qkv_proj,
            hidden_states,
            self.q_size,
            self.kv_size * 2,
        )
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        k, _ = self.rotary_emb.forward_native(positions, k, None)
        return self._compute_pap_attention(
            q,
            k,
            v,
            query_already_sent=True,
        )

    def _pap_q_first_projection_transport_supported(self) -> bool:
        transport = _pap_offload_exec_transport()
        return bool(
            not getattr(transport, "requires_tcp_trigger", True)
            and getattr(transport, "supports_query_first_kv_later", False)
            and callable(getattr(transport, "send_query_batch", None))
            and callable(getattr(transport, "send_kv_batch", None))
        )

    def _send_pap_query_batch(self, q: torch.Tensor) -> bool:
        if not is_forward_context_available():
            return False

        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            attn_metadata = metadata.get(self.attn.layer_name)
        elif isinstance(metadata, list) and metadata:
            attn_metadata = metadata[0].get(self.attn.layer_name)
        else:
            attn_metadata = None
        if attn_metadata is None:
            return False

        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        num_reqs = int(additional_kwargs.get("pap_num_reqs") or len(request_ids))
        num_actual_tokens = int(getattr(attn_metadata, "num_actual_tokens", q.shape[0]))
        if num_reqs <= 0 or len(request_ids) < num_reqs:
            return False
        if num_actual_tokens < num_reqs:
            return False

        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None or int(seq_lens.shape[0]) < num_reqs:
            return False
        positions = additional_kwargs.get("pap_positions")
        if positions is None or int(positions.shape[-1]) < num_reqs:
            return False
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long)

        from vllm.pap.data_plane import (
            PAPOffloadExecBatchDescriptor,
            PAPOffloadExecDescriptor,
        )
        from vllm.pap.shadow_attention import select_attention_endpoint_for_request

        offload_exec_zmq_endpoint_by_request = (
            additional_kwargs.get("pap_offload_exec_zmq_endpoint_by_request") or {}
        )
        tcp_endpoint_by_request = (
            additional_kwargs.get("pap_attention_tcp_endpoint_by_request") or {}
        )
        attention_endpoint_by_request = (
            additional_kwargs.get("pap_attention_endpoint_by_request") or {}
        )
        default_tcp_endpoint = additional_kwargs.get("pap_attention_tcp_endpoint")
        default_attention_endpoint = additional_kwargs.get("pap_attention_endpoint")
        attention_kv_installed_by_request = set(
            additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        prefix_len_by_request = (
            additional_kwargs.get("pap_prefill_prefix_len_by_request") or {}
        )
        prefill_kv_handle_by_request = (
            additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
        )
        query = q.view(-1, self.num_heads, self.head_dim)
        offload_exec_groups: dict[
            tuple[str | None, str | None, str],
            list[tuple[int, PAPOffloadExecDescriptor, torch.Tensor]],
        ] = {}
        for req_index in range(num_reqs):
            request_id = str(request_ids[req_index])
            if not request_id.startswith(("cmpl-", "chatcmpl-")):
                return False
            seq_len = int(positions_cpu.reshape(-1)[req_index].item()) + 1
            max_seq_len = int(seq_lens_cpu[req_index].item())
            if seq_len > max_seq_len:
                return False
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = _pap_endpoint_for_tp_rank(tcp_endpoint)
            attention_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_attention_endpoint,
                endpoint_by_request=attention_endpoint_by_request,
            )
            attention_endpoint = _pap_endpoint_for_tp_rank(attention_endpoint)
            offload_exec_zmq_endpoint = _pap_endpoint_for_tp_rank(
                offload_exec_zmq_endpoint_by_request.get(request_id)
            )
            if offload_exec_zmq_endpoint is None:
                return False
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed_by_request:
                if not prefill_kv_handle:
                    return False
                return False
            descriptor = PAPOffloadExecDescriptor(
                request_id=request_id,
                layer_name=self.attn.layer_name,
                step=seq_len,
                scale=float(self.scaling),
            )
            offload_exec_groups.setdefault(
                (tcp_endpoint, attention_endpoint, offload_exec_zmq_endpoint),
                [],
            ).append((req_index, descriptor, query[req_index : req_index + 1]))

        probe_transport = _pap_offload_exec_transport()
        if (
            getattr(probe_transport, "requires_tcp_trigger", True)
            or not getattr(probe_transport, "supports_query_first_kv_later", False)
            or not callable(getattr(probe_transport, "send_query_batch", None))
        ):
            return False

        if any(
            not attention_endpoint
            for (
                _tcp_endpoint,
                attention_endpoint,
                _offload_exec_zmq_endpoint,
            ) in offload_exec_groups
        ):
            return False

        for (
            _tcp_endpoint,
            attention_endpoint,
            offload_exec_zmq_endpoint,
        ), group_items in offload_exec_groups.items():
            transport = _pap_offload_exec_transport_for_attention_endpoint(
                attention_endpoint
            )
            send_query_batch = getattr(transport, "send_query_batch", None)
            if (
                getattr(transport, "requires_tcp_trigger", True)
                or not getattr(transport, "supports_query_first_kv_later", False)
                or not callable(send_query_batch)
            ):
                return False
            _pap_bind_offload_exec_mailbox_peer(transport, attention_endpoint)
            batch_descriptor = PAPOffloadExecBatchDescriptor(
                layer_name=self.attn.layer_name,
                items=tuple(item[1] for item in group_items),
            )
            if len(group_items) == 1:
                query_batch = group_items[0][2].reshape(1, -1)
            else:
                query_batch = torch.cat(
                    [item[2].reshape(1, -1) for item in group_items],
                    dim=0,
                )
            send_query_batch(
                batch_descriptor,
                query_batch,
                remote_address=offload_exec_zmq_endpoint,
            )
        return True

    def _compute_pap_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_already_sent: bool = False,
    ) -> tuple[torch.Tensor, list[Any]]:
        if not is_forward_context_available():
            raise RuntimeError("PAP attention requires forward context")

        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            attn_metadata = metadata.get(self.attn.layer_name)
        elif isinstance(metadata, list):
            attn_metadata = metadata[0].get(self.attn.layer_name)
        else:
            attn_metadata = None
        if attn_metadata is None:
            raise RuntimeError(
                f"PAP attention missing metadata for {self.attn.layer_name}"
            )
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
            raise RuntimeError("PAP attention currently supports decode-only batches")

        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        num_reqs = int(additional_kwargs.get("pap_num_reqs") or len(request_ids))
        num_actual_tokens = int(
            additional_kwargs.get("pap_num_actual_tokens") or q.shape[0]
        )
        num_scheduled_tokens = tuple(
            int(num_tokens)
            for num_tokens in additional_kwargs.get("pap_num_scheduled_tokens") or ()
        )
        if num_reqs <= 0 or len(request_ids) < num_reqs:
            raise RuntimeError("PAP attention missing request ids")
        if num_actual_tokens < num_reqs:
            raise RuntimeError("PAP attention expected one actual token per request")
        if num_scheduled_tokens and any(
            num_tokens != 1 for num_tokens in num_scheduled_tokens[:num_reqs]
        ):
            raise RuntimeError("PAP attention currently supports one token per request")

        query = q.view(-1, self.num_heads, self.head_dim)
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view(-1, self.num_kv_heads, self.head_dim)

        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            raise RuntimeError("PAP attention missing scheduler seq_lens")
        if int(seq_lens.shape[0]) < num_reqs:
            raise RuntimeError("PAP attention seq_lens do not cover all requests")

        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        positions = additional_kwargs.get("pap_positions")
        if positions is None:
            raise RuntimeError("PAP attention missing input positions")
        if int(positions.shape[-1]) < num_reqs:
            raise RuntimeError("PAP attention positions do not cover all requests")
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long)

        from vllm.pap.data_plane import (
            PAPOffloadExecBatchDescriptor,
            PAPOffloadExecDescriptor,
            pap_offload_exec_trace_id,
        )
        from vllm.pap.shadow_attention import (
            select_attention_endpoint_for_request,
            trigger_offload_exec_attention_batch,
        )

        offload_exec_zmq_endpoint_by_request = (
            additional_kwargs.get("pap_offload_exec_zmq_endpoint_by_request") or {}
        )
        tcp_endpoint_by_request = (
            additional_kwargs.get("pap_attention_tcp_endpoint_by_request") or {}
        )
        attention_endpoint_by_request = (
            additional_kwargs.get("pap_attention_endpoint_by_request") or {}
        )
        default_tcp_endpoint = additional_kwargs.get("pap_attention_tcp_endpoint")
        default_attention_endpoint = additional_kwargs.get("pap_attention_endpoint")
        attention_kv_installed_by_request = set(
            additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        prefix_len_by_request = (
            additional_kwargs.get("pap_prefill_prefix_len_by_request") or {}
        )
        prefill_kv_handle_by_request = (
            additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
        )
        remote_attention_calls: list[
            tuple[int, str | None, str | None, str | None, dict[str, Any]]
        ] = []
        for req_index in range(num_reqs):
            request_id = str(request_ids[req_index])
            if not request_id.startswith(("cmpl-", "chatcmpl-")):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            seq_len = int(positions_cpu.reshape(-1)[req_index].item()) + 1
            max_seq_len = int(seq_lens_cpu[req_index].item())
            if seq_len > max_seq_len:
                raise RuntimeError(
                    f"PAP attention position-derived seq_len {seq_len} exceeds "
                    f"scheduler seq_len {max_seq_len} for {request_id}"
                )
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = _pap_endpoint_for_tp_rank(tcp_endpoint)
            attention_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_attention_endpoint,
                endpoint_by_request=attention_endpoint_by_request,
            )
            attention_endpoint = _pap_endpoint_for_tp_rank(attention_endpoint)
            offload_exec_zmq_endpoint = _pap_endpoint_for_tp_rank(
                offload_exec_zmq_endpoint_by_request.get(request_id)
            )
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed_by_request:
                if not prefill_kv_handle:
                    raise RuntimeError("PAP missing local prefill KV handle")
                continue
            remote_attention_calls.append(
                (
                    req_index,
                    tcp_endpoint,
                    attention_endpoint,
                    offload_exec_zmq_endpoint,
                    {
                        "request_id": request_id,
                        "layer_name": self.attn.layer_name,
                        "query": query[req_index : req_index + 1],
                        "key": key[req_index : req_index + 1],
                        "value": value[req_index : req_index + 1],
                        "scale": float(self.scaling),
                        "seq_len": seq_len,
                    },
                )
            )
            if offload_exec_zmq_endpoint:
                logger.debug(
                    "PAP OFFLOAD_EXEC ZMQ endpoint selected request_id=%s "
                    "layer=%s endpoint=%s",
                    request_id,
                    self.attn.layer_name,
                    offload_exec_zmq_endpoint,
                )

        all_requests_offloaded = len(remote_attention_calls) == num_reqs
        direct_mailbox_output_enabled = os.environ.get(
            "PAP_DIRECT_MAILBOX_OUTPUT", ""
        ).lower() in ("1", "true", "yes", "on")
        output: torch.Tensor | None = None
        pap_release_messages: list[Any] = []

        def get_copy_output_buffer() -> torch.Tensor:
            nonlocal output
            if output is None:
                output = (
                    torch.empty_like(query)
                    if all_requests_offloaded
                    else torch.zeros_like(query)
                )
            return output

        def apply_remote_output(req_index: int, remote_output: torch.Tensor) -> None:
            target = get_copy_output_buffer()[req_index : req_index + 1]
            remote_output = remote_output.to(device=output.device, dtype=output.dtype)
            if remote_output.shape != target.shape:
                if remote_output.numel() != target.numel():
                    raise RuntimeError(
                        "PAP remote attention output shape mismatch: "
                        f"got {tuple(remote_output.shape)}, "
                        f"expected {tuple(target.shape)}"
                    )
                remote_output = remote_output.view_as(target)
            target.copy_(remote_output)

        parallelism = int(os.environ.get("PAP_REMOTE_ATTENTION_PARALLELISM", "16"))
        q_first_kv_later_enabled = os.environ.get(
            "PAP_Q_FIRST_KV_LATER", ""
        ).lower() in ("1", "true", "yes", "on")
        segmented_qkv_enabled = os.environ.get("PAP_SEGMENTED_QKV", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        transport = _pap_offload_exec_transport()
        trace_offload_exec = os.environ.get("PAP_OFFLOAD_EXEC_TRACE", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_send_done_ns = 0
        trace_yield_start_ns = 0
        trace_yield_end_ns = 0
        trace_recv_done_ns = 0
        trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
        offload_exec_groups: dict[
            tuple[str | None, str | None, str],
            list[tuple[int, PAPOffloadExecDescriptor, tuple[torch.Tensor, ...]]],
        ] = {}
        for (
            req_index,
            tcp_endpoint,
            attention_endpoint,
            offload_exec_zmq_endpoint,
            call_kwargs,
        ) in remote_attention_calls:
            if offload_exec_zmq_endpoint is None:
                raise RuntimeError(
                    "PAP OFFLOAD_EXEC NCCL path missing pap_offload_exec_zmq_endpoint"
                )
            qkv_segments = (
                call_kwargs["query"].reshape(1, -1),
                call_kwargs["key"].reshape(1, -1),
                call_kwargs["value"].reshape(1, -1),
            )
            descriptor = PAPOffloadExecDescriptor(
                request_id=str(call_kwargs["request_id"]),
                layer_name=str(call_kwargs["layer_name"]),
                step=int(call_kwargs["seq_len"]),
                scale=float(call_kwargs["scale"]),
            )
            offload_exec_groups.setdefault(
                (tcp_endpoint, attention_endpoint, offload_exec_zmq_endpoint),
                [],
            ).append(
                (
                    req_index,
                    descriptor,
                    qkv_segments,
                )
            )

        offload_exec_batches: list[
            tuple[
                str | None,
                str | None,
                str,
                PAPOffloadExecBatchDescriptor,
                list[int],
                Any,
            ]
        ] = []
        for (
            tcp_endpoint,
            attention_endpoint,
            offload_exec_zmq_endpoint,
        ), group_items in offload_exec_groups.items():
            transport = _pap_offload_exec_transport_for_attention_endpoint(
                attention_endpoint
            )
            batch_descriptor = PAPOffloadExecBatchDescriptor(
                layer_name=self.attn.layer_name,
                items=tuple(item[1] for item in group_items),
            )
            _pap_bind_offload_exec_mailbox_peer(transport, attention_endpoint)
            send_query_batch = getattr(transport, "send_query_batch", None)
            send_kv_batch = getattr(transport, "send_kv_batch", None)
            send_qkv_batch_segments = getattr(
                transport, "send_qkv_batch_segments", None
            )
            qkv_width = sum(int(segment.shape[-1]) for segment in group_items[0][2])
            q_first_transport_ready = (
                callable(send_query_batch)
                and callable(send_kv_batch)
                and getattr(transport, "supports_query_first_kv_later", False)
                and not getattr(transport, "requires_tcp_trigger", True)
            )
            if query_already_sent and not q_first_transport_ready:
                raise RuntimeError(
                    "PAP Q-first Projection sent query before KV, but the "
                    "transport cannot accept the KV follow-up"
                )
            if (
                q_first_kv_later_enabled or query_already_sent
            ) and q_first_transport_ready:
                if len(group_items) == 1:
                    query_batch = group_items[0][2][0]
                    kv_batch = torch.cat(group_items[0][2][1:], dim=-1)
                else:
                    query_batch = torch.cat(
                        [item[2][0] for item in group_items],
                        dim=0,
                    )
                    kv_batch = torch.cat(
                        [torch.cat(item[2][1:], dim=-1) for item in group_items],
                        dim=0,
                    )
                if not query_already_sent:
                    send_query_batch(
                        batch_descriptor,
                        query_batch,
                        remote_address=offload_exec_zmq_endpoint,
                    )
                send_kv_batch(
                    batch_descriptor,
                    kv_batch,
                    remote_address=offload_exec_zmq_endpoint,
                )
            elif segmented_qkv_enabled and callable(send_qkv_batch_segments):
                if len(group_items) == 1:
                    qkv_segments = group_items[0][2]
                else:
                    qkv_segments = tuple(
                        torch.cat(
                            [item[2][segment_index] for item in group_items], dim=0
                        )
                        for segment_index in range(3)
                    )
                send_qkv_batch_segments(
                    batch_descriptor,
                    qkv_segments,
                    payload_shape=(len(group_items), qkv_width),
                    remote_address=offload_exec_zmq_endpoint,
                )
            else:
                if len(group_items) == 1:
                    qkv_batch = torch.cat(group_items[0][2], dim=-1)
                else:
                    qkv_batch = torch.cat(
                        [torch.cat(item[2], dim=-1) for item in group_items],
                        dim=0,
                    )
                transport.send_qkv_batch(
                    batch_descriptor,
                    qkv_batch,
                    remote_address=offload_exec_zmq_endpoint,
                )
            offload_exec_batches.append(
                (
                    tcp_endpoint,
                    attention_endpoint,
                    offload_exec_zmq_endpoint,
                    batch_descriptor,
                    [item[0] for item in group_items],
                    transport,
                )
            )
        trace_send_ms = (
            (time.perf_counter() - trace_send_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        if trace_offload_exec:
            trace_send_done_ns = time.perf_counter_ns()
        local_offload_exec_zmq_endpoint = _pap_offload_exec_local_address()

        def trigger_offload_exec_batch_call(
            tcp_endpoint: str | None,
            local_address: str,
            descriptor: PAPOffloadExecBatchDescriptor,
        ) -> None:
            trigger_offload_exec_attention_batch(
                tcp_endpoint=tcp_endpoint,
                layer_name=descriptor.layer_name,
                items=[
                    {
                        "request_id": item.request_id,
                        "step": item.step,
                        "scale": item.scale,
                    }
                    for item in descriptor.items
                ],
                remote_address=local_address,
            )

        trace_trigger_start = time.perf_counter() if trace_offload_exec else 0.0
        if all(
            not getattr(batch[5], "requires_tcp_trigger", True)
            for batch in offload_exec_batches
        ):
            pass
        elif len(offload_exec_batches) <= 1 or parallelism <= 1:
            for (
                tcp_endpoint,
                _attention_endpoint,
                offload_exec_zmq_endpoint,
                descriptor,
                _req_indices,
                _transport,
            ) in offload_exec_batches:
                trigger_offload_exec_batch_call(
                    tcp_endpoint,
                    local_offload_exec_zmq_endpoint,
                    descriptor,
                )
        else:
            executor = _pap_remote_attention_executor(
                min(parallelism, len(offload_exec_batches))
            )
            futures = [
                executor.submit(
                    trigger_offload_exec_batch_call,
                    tcp_endpoint,
                    local_offload_exec_zmq_endpoint,
                    descriptor,
                )
                for (
                    tcp_endpoint,
                    _attention_endpoint,
                    offload_exec_zmq_endpoint,
                    descriptor,
                    _req_indices,
                    _transport,
                ) in offload_exec_batches
            ]
            for future in futures:
                future.result()
        trace_trigger_ms = (
            (time.perf_counter() - trace_trigger_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )

        trace_yield_start = time.perf_counter() if trace_offload_exec else 0.0
        if trace_offload_exec:
            trace_yield_start_ns = time.perf_counter_ns()
        trace_yield_ms = 0.0
        if offload_exec_batches:
            from vllm.v1.worker.ubatching import dbo_enabled, dbo_yield

            if dbo_enabled():
                dbo_yield()
        if trace_offload_exec:
            trace_yield_end_ns = time.perf_counter_ns()
            trace_yield_ms = (time.perf_counter() - trace_yield_start) * 1000.0

        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
        for (
            _tcp_endpoint,
            _attention_endpoint,
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            transport,
        ) in offload_exec_batches:
            recv_output_batch_message = getattr(
                transport, "recv_output_batch_message", None
            )
            output_message = None
            if callable(recv_output_batch_message):
                output_message = recv_output_batch_message(
                    batch_descriptor,
                    remote_address=offload_exec_zmq_endpoint,
                )
                output_batch = output_message.tensor
            else:
                output_batch = transport.recv_output_batch(
                    batch_descriptor,
                    remote_address=offload_exec_zmq_endpoint,
                )
            try:
                if int(output_batch.shape[0]) != len(batch_descriptor.items):
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC output batch row count mismatch"
                    )
                can_use_direct_output = (
                    direct_mailbox_output_enabled
                    and len(offload_exec_batches) == 1
                    and len(req_indices) == num_reqs
                    and req_indices == list(range(num_reqs))
                    and output_batch.device == query.device
                    and output_batch.dtype == query.dtype
                    and int(output_batch.numel())
                    == int(q.shape[0]) * self.num_heads * self.head_dim
                )
                if can_use_direct_output:
                    direct_output = output_batch.view(
                        q.shape[0], self.num_heads * self.head_dim
                    )
                    if output_message is not None:
                        pap_release_messages.append(output_message)
                        output_message = None
                    return direct_output, pap_release_messages
                for descriptor_index, req_index in enumerate(req_indices):
                    apply_remote_output(
                        req_index,
                        output_batch[descriptor_index : descriptor_index + 1],
                    )
            finally:
                if output_message is not None:
                    output_message.release()
        if trace_offload_exec and offload_exec_batches:
            trace_recv_done_ns = time.perf_counter_ns()
            trace_recv_ms = (time.perf_counter() - trace_recv_start) * 1000.0
            trace_total_ms = (time.perf_counter() - trace_total_start) * 1000.0
            batch_keys = "|".join(
                pap_offload_exec_trace_id(batch[3].output_tensor_id)
                for batch in offload_exec_batches
            )
            logger.info(
                "PAP OFFLOAD_EXEC projection trace layer=%s batches=%d calls=%d "
                "send_ms=%.3f trigger_ms=%.3f yield_ms=%.3f "
                "recv_ms=%.3f total_ms=%.3f batch_keys=%s "
                "send_done_ns=%d yield_start_ns=%d yield_end_ns=%d "
                "recv_done_ns=%d",
                offload_exec_batches[0][3].layer_name,
                len(offload_exec_batches),
                sum(len(batch[3].items) for batch in offload_exec_batches),
                trace_send_ms,
                trace_trigger_ms,
                trace_yield_ms,
                trace_recv_ms,
                trace_total_ms,
                batch_keys,
                trace_send_done_ns,
                trace_yield_start_ns,
                trace_yield_end_ns,
                trace_recv_done_ns,
            )
        final_output = get_copy_output_buffer()
        return final_output.view(q.shape[0], self.num_heads * self.head_dim), []

    def _maybe_import_pap_prefill_kv_to_attention(self) -> None:
        if not is_forward_context_available():
            return
        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        if not additional_kwargs.get("pap_enabled"):
            return

        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        num_reqs = int(additional_kwargs.get("pap_num_reqs") or len(request_ids))
        prefill_kv_handle_by_request = (
            additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
        )
        if num_reqs <= 0 or len(request_ids) < num_reqs:
            return
        num_scheduled_tokens = tuple(
            int(num_tokens)
            for num_tokens in additional_kwargs.get("pap_num_scheduled_tokens") or ()
        )
        if len(num_scheduled_tokens) < num_reqs:
            return
        prefill_kv_handle_by_request = (
            additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
        )
        import_prefill_kv_to_attention_by_request = set(
            additional_kwargs.get("pap_import_prefill_kv_to_attention_by_request") or ()
        )
        tcp_endpoint_by_request = (
            additional_kwargs.get("pap_attention_tcp_endpoint_by_request") or {}
        )
        default_tcp_endpoint = additional_kwargs.get("pap_attention_tcp_endpoint")
        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            attn_metadata = metadata.get(self.attn.layer_name)
        elif isinstance(metadata, list) and metadata:
            attn_metadata = metadata[0].get(self.attn.layer_name)
        else:
            attn_metadata = None
        if attn_metadata is None:
            return
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None or int(seq_lens.shape[0]) < num_reqs:
            return
        block_table = getattr(attn_metadata, "block_table", None)
        if block_table is None or int(block_table.shape[0]) < num_reqs:
            return
        kv_cache = getattr(self.attn, "kv_cache", None)
        if kv_cache is None:
            return
        block_size = additional_kwargs.get("pap_block_size")
        if block_size is None:
            block_size = getattr(getattr(self.attn, "impl", None), "block_size", None)
        if block_size is None:
            block_size = getattr(self.attn, "block_size", None)
        if block_size is None:
            return

        from vllm.pap.data_plane import PAPTensorTransport
        from vllm.pap.shadow_attention import (
            import_prefill_paged_kv,
            select_attention_endpoint_for_request,
        )
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        offload_kv_transport = PAPTensorTransport(
            os.environ.get(
                "PAP_OFFLOAD_KV_TRANSPORT", PAPTensorTransport.CUDA_IPC.value
            )
        )
        if offload_kv_transport is not PAPTensorTransport.CUDA_IPC:
            raise RuntimeError("PAP paged Prefill KV export requires cuda_ipc")
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        for req_index in range(num_reqs):
            request_id = str(request_ids[req_index])
            if not request_id.startswith(("cmpl-", "chatcmpl-")):
                continue
            if request_id not in import_prefill_kv_to_attention_by_request:
                continue
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if not prefill_kv_handle:
                continue
            prefix_len = int(seq_lens_cpu[req_index].item())
            if prefix_len <= 1:
                continue
            import_key = (
                request_id,
                self.attn.layer_name,
                prefix_len,
                str(prefill_kv_handle),
            )
            if import_key in self._pap_imported_prefill_kv:
                continue
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = _pap_endpoint_for_tp_rank(tcp_endpoint)
            import_prefill_paged_kv(
                request_id=request_id,
                layer_name=self.attn.layer_name,
                kv_cache=kv_cache,
                block_ids=_pap_block_ids_from_block_table(
                    block_table=block_table[req_index : req_index + 1],
                    seq_len=prefix_len,
                    block_size=int(block_size),
                ),
                seq_len=prefix_len,
                block_size=int(block_size),
                num_kv_heads=self.num_kv_heads,
                layout=get_kv_cache_layout(),
                tcp_endpoint=tcp_endpoint,
            )
            self._pap_imported_prefill_kv.add(import_key)

    @staticmethod
    def _select_pap_request_id(request_ids: Any) -> str | None:
        if not request_ids:
            return None
        for request_id in request_ids:
            request_id_str = str(request_id)
            if request_id_str.startswith(("cmpl-", "chatcmpl-")):
                return request_id_str
        return None


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

    def _pap_microbatch_forward_after_input_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        output_hidden_states = torch.empty_like(hidden_states)
        output_residual = torch.empty_like(residual)

        def consume_projected_chunk(
            req_indices: list[int],
            projected_chunk: torch.Tensor,
        ) -> None:
            index_tensor = torch.tensor(
                req_indices,
                device=hidden_states.device,
                dtype=torch.long,
            )
            residual_chunk = residual.index_select(0, index_tensor)
            chunk_hidden_states, chunk_residual = self.post_attention_layernorm(
                projected_chunk, residual_chunk
            )
            chunk_hidden_states = self.mlp(chunk_hidden_states)
            output_hidden_states.index_copy_(0, index_tensor, chunk_hidden_states)
            output_residual.index_copy_(0, index_tensor, chunk_residual)

        if not self.self_attn._run_pap_attention_microbatch_pipeline(
            positions,
            hidden_states,
            consume_projected_chunk,
        ):
            return None
        return output_hidden_states, output_residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if _pap_env_enabled("PAP_OFFLOAD_EXEC_MICROBATCH_OVERLAP_MLP"):
            microbatch_result = self._pap_microbatch_forward_after_input_norm(
                positions,
                hidden_states,
                residual,
            )
            if microbatch_result is not None:
                return microbatch_result

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
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
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
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
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
