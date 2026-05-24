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

import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
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


@lru_cache(maxsize=8)
def _pap_remote_attention_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="pap-attn-rpc",
    )


@lru_cache(maxsize=1)
def _pap_offload_exec_transport():
    from vllm.pap.data_plane import build_p2p_nccl_offload_exec_transport

    return build_p2p_nccl_offload_exec_transport(
        local_rank=int(os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0")),
        kv_port=int(os.environ.get("PAP_OFFLOAD_EXEC_ZMQ_PORT", "11300")),
        hostname=os.environ.get("PAP_OFFLOAD_EXEC_HOST", ""),
    )


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
        self._pap_imported_prefill_kv: set[
            tuple[str, str, int, str]
        ] = set()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
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
            attn_output = self._compute_pap_attention(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output
        attn_output = self.attn(q, k, v)
        self._maybe_import_pap_prefill_kv_to_attention()
        output, _ = self.o_proj(attn_output)
        return output

    def _should_use_pap_attention(self) -> bool:
        if not is_forward_context_available():
            return False
        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        if not additional_kwargs.get("pap_enabled"):
            return False

        request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
        if not self._select_pap_request_id(request_ids):
            return False

        metadata = forward_context.attn_metadata
        if isinstance(metadata, dict):
            attn_metadata = metadata.get(self.attn.layer_name)
        elif isinstance(metadata, list) and metadata:
            attn_metadata = metadata[0].get(self.attn.layer_name)
        else:
            attn_metadata = None
        if attn_metadata is None:
            return False
        if int(getattr(attn_metadata, "max_query_len", 0)) != 1:
            return False

        num_scheduled_tokens = tuple(
            int(num_tokens)
            for num_tokens in additional_kwargs.get("pap_num_scheduled_tokens") or ()
        )
        num_reqs = int(
            additional_kwargs.get("pap_num_reqs") or len(num_scheduled_tokens)
        )
        if num_reqs <= 0:
            return False
        if len(request_ids) < num_reqs:
            return False
        if len(num_scheduled_tokens) < num_reqs:
            return False
        if any(num_tokens != 1 for num_tokens in num_scheduled_tokens[:num_reqs]):
            return False
        return self._pap_attention_kv_ready_for_requests(
            request_ids[:num_reqs]
        )

    def _pap_attention_kv_ready_for_requests(
        self, request_ids: Iterable[Any]
    ) -> bool:
        """Return True when PA-side attention KV is ready for every request."""
        if not is_forward_context_available():
            return False
        additional_kwargs = get_forward_context().additional_kwargs or {}
        installed = set(
            additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
        )
        return all(str(request_id) in installed for request_id in request_ids)

    def _compute_pap_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
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
            raise RuntimeError(
                "PAP attention currently supports one token per request"
            )

        query = q.view(-1, self.num_heads, self.head_dim)
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view(-1, self.num_kv_heads, self.head_dim)

        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is None:
            raise RuntimeError("PAP attention missing scheduler seq_lens")
        if int(seq_lens.shape[0]) < num_reqs:
            raise RuntimeError("PAP attention seq_lens do not cover all requests")

        slot_mapping_container = forward_context.slot_mapping
        if isinstance(slot_mapping_container, dict):
            slot_mapping = slot_mapping_container.get(self.attn.layer_name)
        elif isinstance(slot_mapping_container, list) and slot_mapping_container:
            slot_mapping = slot_mapping_container[0].get(self.attn.layer_name)
        else:
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            raise RuntimeError("PAP attention missing scheduler slot_mapping")
        if int(slot_mapping.shape[0]) < num_reqs:
            raise RuntimeError(
                "PAP attention slot_mapping does not cover all requests"
            )

        block_size = additional_kwargs.get("pap_block_size")
        if block_size is None:
            block_size = getattr(getattr(self.attn, "impl", None), "block_size", None)
        if block_size is None:
            block_size = getattr(self.attn, "block_size", None)
        if block_size is None or int(block_size) <= 0:
            raise RuntimeError("PAP attention missing cache block_size")
        block_size = int(block_size)

        slot_mapping_cpu = slot_mapping.detach().to(device="cpu", dtype=torch.long)
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        positions = additional_kwargs.get("pap_positions")
        if positions is None:
            raise RuntimeError("PAP attention missing input positions")
        if int(positions.shape[-1]) < num_reqs:
            raise RuntimeError("PAP attention positions do not cover all requests")
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long)

        from vllm.pap.shadow_attention import (
            select_attention_endpoint_for_request,
            trigger_offload_exec_attention,
        )
        from vllm.pap.data_plane import PAPOffloadExecDescriptor

        output = torch.zeros_like(query)
        offload_exec_zmq_endpoint_by_request = additional_kwargs.get(
            "pap_offload_exec_zmq_endpoint_by_request"
        ) or {}
        tcp_endpoint_by_request = additional_kwargs.get(
            "pap_attention_tcp_endpoint_by_request"
        ) or {}
        default_tcp_endpoint = additional_kwargs.get("pap_attention_tcp_endpoint")
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
            tuple[int, str | None, str | None, dict[str, Any]]
        ] = []
        for req_index in range(num_reqs):
            request_id = str(request_ids[req_index])
            if not request_id.startswith(("cmpl-", "chatcmpl-")):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            slot = int(slot_mapping_cpu[req_index].item())
            if slot < 0:
                raise RuntimeError(
                    f"PAP attention invalid slot_mapping for {request_id}: {slot}"
                )
            block_id = slot // block_size
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
            offload_exec_zmq_endpoint = offload_exec_zmq_endpoint_by_request.get(
                request_id
            )
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed_by_request:
                if not prefill_kv_handle:
                    raise RuntimeError(
                        "PAP missing local prefill KV handle"
                    )
                continue
            remote_attention_calls.append(
                (
                    req_index,
                    tcp_endpoint,
                    offload_exec_zmq_endpoint,
                    {
                        "request_id": request_id,
                        "layer_name": self.attn.layer_name,
                        "query": query[req_index : req_index + 1],
                        "key": key[req_index : req_index + 1],
                        "value": value[req_index : req_index + 1],
                        "scale": float(self.scaling),
                        "block_id": block_id,
                        "slot": slot,
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

        def apply_remote_output(req_index: int, remote_output: torch.Tensor) -> None:
            target = output[req_index : req_index + 1]
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
        transport = _pap_offload_exec_transport()
        trace_offload_exec = os.environ.get(
            "PAP_OFFLOAD_EXEC_TRACE", ""
        ).lower() in ("1", "true", "yes", "on")
        trace_total_start = time.perf_counter() if trace_offload_exec else 0.0
        trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
        offload_exec_calls: list[
            tuple[int, str | None, str, Any]
        ] = []
        for (
            req_index,
            tcp_endpoint,
            offload_exec_zmq_endpoint,
            call_kwargs,
        ) in remote_attention_calls:
            if offload_exec_zmq_endpoint is None:
                raise RuntimeError(
                    "PAP OFFLOAD_EXEC NCCL path missing "
                    "pap_offload_exec_zmq_endpoint"
                )
            qkv = torch.cat(
                [
                    call_kwargs["query"].reshape(1, -1),
                    call_kwargs["key"].reshape(1, -1),
                    call_kwargs["value"].reshape(1, -1),
                ],
                dim=-1,
            )
            descriptor = PAPOffloadExecDescriptor(
                request_id=str(call_kwargs["request_id"]),
                layer_name=str(call_kwargs["layer_name"]),
                step=int(call_kwargs["seq_len"]),
                scale=float(call_kwargs["scale"]),
            )
            transport.send_qkv(
                descriptor,
                qkv,
                remote_address=offload_exec_zmq_endpoint,
            )
            offload_exec_calls.append(
                (
                    req_index,
                    tcp_endpoint,
                    offload_exec_zmq_endpoint,
                    descriptor,
                )
            )
        trace_send_ms = (
            (time.perf_counter() - trace_send_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )

        def trigger_offload_exec_call(
            tcp_endpoint: str | None,
            remote_address: str,
            descriptor: Any,
        ) -> None:
            trigger_offload_exec_attention(
                tcp_endpoint=tcp_endpoint,
                request_id=descriptor.request_id,
                layer_name=descriptor.layer_name,
                step=descriptor.step,
                scale=descriptor.scale,
                remote_address=os.environ.get(
                    "PAP_OFFLOAD_EXEC_REMOTE_ADDRESS",
                    f"127.0.0.1:{os.environ.get('PAP_OFFLOAD_EXEC_ZMQ_PORT', '11300')}",
                ),
            )

        trace_trigger_start = time.perf_counter() if trace_offload_exec else 0.0
        if len(offload_exec_calls) <= 1 or parallelism <= 1:
            for (
                _req_index,
                tcp_endpoint,
                offload_exec_zmq_endpoint,
                descriptor,
            ) in offload_exec_calls:
                trigger_offload_exec_call(
                    tcp_endpoint,
                    offload_exec_zmq_endpoint,
                    descriptor,
                )
        else:
            executor = _pap_remote_attention_executor(
                min(parallelism, len(offload_exec_calls))
            )
            futures = [
                executor.submit(
                    trigger_offload_exec_call,
                    tcp_endpoint,
                    offload_exec_zmq_endpoint,
                    descriptor,
                )
                for (
                    _req_index,
                    tcp_endpoint,
                    offload_exec_zmq_endpoint,
                    descriptor,
                ) in offload_exec_calls
            ]
            for future in futures:
                future.result()
        trace_trigger_ms = (
            (time.perf_counter() - trace_trigger_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )

        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0
        for (
            req_index,
            _tcp_endpoint,
            offload_exec_zmq_endpoint,
            descriptor,
        ) in offload_exec_calls:
            apply_remote_output(
                req_index,
                transport.recv_output(
                    descriptor,
                    remote_address=offload_exec_zmq_endpoint,
                ),
            )
        if trace_offload_exec and offload_exec_calls:
            trace_recv_ms = (time.perf_counter() - trace_recv_start) * 1000.0
            trace_total_ms = (
                time.perf_counter() - trace_total_start
            ) * 1000.0
            logger.info(
                "PAP OFFLOAD_EXEC projection trace layer=%s calls=%d "
                "send_ms=%.3f trigger_ms=%.3f recv_ms=%.3f total_ms=%.3f",
                offload_exec_calls[0][3].layer_name,
                len(offload_exec_calls),
                trace_send_ms,
                trace_trigger_ms,
                trace_recv_ms,
                trace_total_ms,
            )
        return output.view(q.shape[0], self.num_heads * self.head_dim)

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
            additional_kwargs.get("pap_import_prefill_kv_to_attention_by_request")
            or ()
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

        from vllm.pap.shadow_attention import (
            import_prefill_kv_from_paged_cache,
            select_attention_endpoint_for_request,
        )
        from vllm.pap.data_plane import PAPTensorTransport
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        offload_kv_transport = PAPTensorTransport(
            os.environ.get(
                "PAP_OFFLOAD_KV_TRANSPORT", PAPTensorTransport.CUDA_IPC.value
            )
        )
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
            installed = set(
                additional_kwargs.get("pap_attention_kv_installed_by_request")
                or ()
            )
            if import_key in self._pap_imported_prefill_kv:
                installed.add(request_id)
                additional_kwargs["pap_attention_kv_installed_by_request"] = installed
                continue
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            import_prefill_kv_from_paged_cache(
                request_id=request_id,
                layer_name=self.attn.layer_name,
                kv_cache=kv_cache,
                block_table=block_table[req_index : req_index + 1],
                seq_len=prefix_len,
                block_size=int(block_size),
                num_kv_heads=self.num_kv_heads,
                layout=get_kv_cache_layout(),
                tcp_endpoint=tcp_endpoint,
                transport=offload_kv_transport,
            )
            self._pap_imported_prefill_kv.add(import_key)
            installed.add(request_id)
            additional_kwargs["pap_attention_kv_installed_by_request"] = installed

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
