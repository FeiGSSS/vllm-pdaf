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
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache, lru_cache
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
from vllm.pap.mode import is_pap_request_id, pap_request_ids_are_routable
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


def _pap_env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_ENV_VALUES


def _pap_direct_qkv_send_enabled() -> bool:
    return (
        os.environ.get("PAP_OFFLOAD_EXEC_DIRECT_QKV_SEND", "1").lower()
        in _TRUE_ENV_VALUES
    )


def _pap_batched_route_copy_enabled() -> bool:
    return os.environ.get("PAP_BATCHED_ROUTE_COPY", "1").lower() in _TRUE_ENV_VALUES


def _pap_prefill_ipc_profile_enabled() -> bool:
    return _pap_env_enabled("PAP_PREFILL_IPC_PROFILE")


def _pap_prefill_kv_async_enabled() -> bool:
    return _pap_env_enabled("PAP_PREFILL_KV_ASYNC")


def _pap_unified_kv_export_enabled() -> bool:
    return _pap_env_enabled("PAP_UNIFIED_KV")


def _pap_kv_handoff_mode() -> str:
    mode = os.environ.get("PAP_KV_HANDOFF_MODE", "layer_descriptor").lower()
    mode = mode.replace("-", "_")
    if mode not in {"layer_descriptor", "sealed_manifest"}:
        raise ValueError(f"unsupported PAP_KV_HANDOFF_MODE: {mode}")
    return mode


def _pap_unified_kv_export_decode_capacity_tokens() -> int:
    raw = os.environ.get("PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS", "")
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _pap_offload_exec_session_request_id(
    request_id: str,
    prefill_kv_handle: Any,
) -> str:
    if prefill_kv_handle:
        return str(prefill_kv_handle)
    return str(request_id)


def _pap_projection_critical_trace_enabled() -> bool:
    return _pap_env_enabled("PAP_PROJECTION_KV_UNAWARE") and _pap_env_enabled(
        "PAP_PROJECTION_CRITICAL_TRACE"
    )


def _pap_projection_decode_trace_enabled() -> bool:
    if not _pap_projection_critical_trace_enabled():
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
    if (
        trace_role == "pd_decode"
        and not pap_attention_enabled
        and max_query_len == 1
    ):
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


def _pap_prune_imported_prefill_kv(
    imported_prefill_kv: set[tuple[str, str, int, str]],
    finished_request_ids: Iterable[Any],
) -> None:
    finished = {str(request_id) for request_id in finished_request_ids}
    if not finished or not imported_prefill_kv:
        return
    imported_prefill_kv.difference_update(
        import_key
        for import_key in tuple(imported_prefill_kv)
        if import_key[0] in finished
    )


@lru_cache(maxsize=8)
def _pap_remote_attention_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="pap-attn-rpc",
    )


def _pap_pack_qkv_group_items(
    group_items: list[tuple[int, Any, tuple[torch.Tensor, ...]]],
) -> torch.Tensor:
    if len(group_items) == 1:
        return torch.cat(group_items[0][2], dim=-1)
    return torch.cat(
        [torch.cat(item[2], dim=-1) for item in group_items],
        dim=0,
    )


def _pap_direct_qkv_batch_for_indices(
    qkv_batch: torch.Tensor | None,
    req_indices: tuple[int, ...],
) -> torch.Tensor | None:
    if qkv_batch is None or not req_indices:
        return None
    if qkv_batch.ndim != 2 or not qkv_batch.is_contiguous():
        return None
    if not _pap_req_indices_are_contiguous(req_indices):
        return None
    start = int(req_indices[0])
    stop = start + len(req_indices)
    if stop > int(qkv_batch.shape[0]):
        return None
    direct = qkv_batch[start:stop]
    return direct if direct.is_contiguous() else None


def _pap_route_index_tensor(
    additional_kwargs: dict[str, Any],
    req_indices: tuple[int, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    cache = additional_kwargs.setdefault(
        "_pap_qwen3_route_index_tensors",
        {},
    )
    cache_key = (str(torch.device(device)), req_indices)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    index_tensor = torch.tensor(
        req_indices,
        dtype=torch.long,
        device=device,
    )
    cache[cache_key] = index_tensor
    return index_tensor


def _pap_qkv_batch_for_indices(
    qkv_batch: torch.Tensor | None,
    req_indices: tuple[int, ...],
    *,
    index_tensor: torch.Tensor | None,
) -> tuple[torch.Tensor | None, bool]:
    direct = _pap_direct_qkv_batch_for_indices(qkv_batch, req_indices)
    if direct is not None:
        return direct, True
    if (
        qkv_batch is None
        or qkv_batch.ndim != 2
        or not qkv_batch.is_contiguous()
        or not req_indices
        or index_tensor is None
    ):
        return None, False
    return torch.index_select(qkv_batch, 0, index_tensor), False


def _pap_scatter_attention_output_group(
    output: torch.Tensor,
    remote_output: torch.Tensor,
    *,
    req_indices: tuple[int, ...],
    index_tensor: torch.Tensor | None,
) -> None:
    if not req_indices:
        raise RuntimeError("PAP remote attention output has no route rows")
    remote_output = remote_output.to(
        device=output.device,
        dtype=output.dtype,
        non_blocking=True,
    )
    target_shape = (len(req_indices), *output.shape[1:])
    target_numel = math.prod(target_shape)
    if int(remote_output.numel()) != int(target_numel):
        raise RuntimeError(
            "PAP remote attention output shape mismatch: "
            f"got {tuple(remote_output.shape)}, expected {target_shape}"
        )
    remote_output = remote_output.reshape(target_shape)
    if _pap_req_indices_are_contiguous(req_indices):
        start = int(req_indices[0])
        output[start : start + len(req_indices)].copy_(remote_output)
        return
    if index_tensor is None:
        index_tensor = torch.tensor(
            req_indices,
            dtype=torch.long,
            device=output.device,
        )
    output.index_copy_(0, index_tensor, remote_output)


def _pap_req_indices_are_contiguous(req_indices: tuple[int, ...]) -> bool:
    if not req_indices:
        return False
    start = int(req_indices[0])
    return start >= 0 and req_indices == tuple(range(start, start + len(req_indices)))


def _pap_offload_exec_transport_kind() -> str:
    return os.environ.get("PAP_OFFLOAD_EXEC_TRANSPORT", "nixl_mailbox").lower()


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


@dataclass(frozen=True)
class _PAPOffloadExecStepGroup:
    attention_endpoint: str
    offload_exec_zmq_endpoint: str
    req_indices: tuple[int, ...]
    batch_id_suffix: str
    metadata_template: dict[str, Any]


_PAP_STEP_GROUPS_KEY = "_pap_qwen3_offload_exec_step_groups"


def _pap_offload_exec_step_groups(
    additional_kwargs: dict[str, Any],
    *,
    num_reqs: int,
    scaling: float,
) -> tuple[_PAPOffloadExecStepGroup, ...]:
    cached = additional_kwargs.get(_PAP_STEP_GROUPS_KEY)
    if cached is not None:
        return tuple(cached)

    request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
    route_groups = tuple(additional_kwargs.get("pap_offload_exec_route_groups") or ())
    if not route_groups:
        raise RuntimeError("PAP attention missing OFFLOAD_EXEC route groups")

    attention_kv_installed = set(
        additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
    )
    prefix_len_by_request = (
        additional_kwargs.get("pap_prefill_prefix_len_by_request") or {}
    )
    prefill_kv_handle_by_request = (
        additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
    )
    step_groups: list[_PAPOffloadExecStepGroup] = []
    routed_req_indices: set[int] = set()
    for route_group in route_groups:
        attention_endpoint = _pap_endpoint_for_tp_rank(
            route_group.get("attention_endpoint")
        )
        offload_exec_zmq_endpoint = _pap_endpoint_for_tp_rank(
            route_group.get("offload_exec_zmq_endpoint")
        )
        if not attention_endpoint:
            raise RuntimeError(
                "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
            )
        if not offload_exec_zmq_endpoint:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC mailbox path missing pap_offload_exec_zmq_endpoint"
            )

        req_indices = tuple(
            int(req_index) for req_index in route_group.get("req_indices", ())
        )
        group_request_ids = tuple(
            str(request_id) for request_id in route_group.get("request_ids", ())
        )
        group_steps = tuple(int(step) for step in route_group.get("steps", ()))
        if not (len(req_indices) == len(group_request_ids) == len(group_steps)):
            raise RuntimeError("PAP OFFLOAD_EXEC route group is malformed")

        session_request_ids: list[str] = []
        for group_offset, req_index in enumerate(req_indices):
            if req_index < 0 or req_index >= num_reqs:
                raise RuntimeError("PAP OFFLOAD_EXEC route index out of range")
            request_id = group_request_ids[group_offset]
            if request_id != str(request_ids[req_index]):
                raise RuntimeError("PAP OFFLOAD_EXEC route request mismatch")
            if not is_pap_request_id(request_id):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            routed_req_indices.add(req_index)
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed:
                if not prefill_kv_handle:
                    raise RuntimeError("PAP missing local prefill KV handle")
                raise RuntimeError("PAP attention KV is not installed")
            session_request_ids.append(
                _pap_offload_exec_session_request_id(
                    request_id,
                    prefill_kv_handle,
                )
            )

        batch_id_suffix = ",".join(
            f"{request_id}@{step}"
            for request_id, step in zip(session_request_ids, group_steps)
        )
        step_groups.append(
            _PAPOffloadExecStepGroup(
                attention_endpoint=str(attention_endpoint),
                offload_exec_zmq_endpoint=str(offload_exec_zmq_endpoint),
                req_indices=req_indices,
                batch_id_suffix=batch_id_suffix,
                metadata_template={
                    "r": tuple(session_request_ids),
                    "s": group_steps,
                    "a": (float(scaling),) * len(group_steps),
                },
            )
        )

    if len(routed_req_indices) != num_reqs:
        raise RuntimeError("PAP OFFLOAD_EXEC route groups do not cover batch")

    result = tuple(step_groups)
    additional_kwargs[_PAP_STEP_GROUPS_KEY] = result
    return result


def _pap_offload_exec_base_zmq_port() -> int:
    return int(
        _pap_endpoint_for_tp_rank(os.environ.get("PAP_OFFLOAD_EXEC_ZMQ_PORT", "11300"))
    )


def _pap_endpoint_port(value: str | None) -> int | None:
    if not value:
        return None
    endpoint = str(value).strip()
    if not endpoint:
        return None
    if "://" in endpoint:
        endpoint = endpoint.rsplit(":", 1)[-1]
    else:
        endpoint = endpoint.rsplit(":", 1)[-1]
    try:
        return int(endpoint)
    except ValueError:
        return None


def _pap_offload_exec_peer_port_offset(peer_endpoint: str | None) -> int:
    peer_port = _pap_endpoint_port(peer_endpoint)
    if peer_port is None:
        return 0
    rank = _pap_tensor_parallel_rank()
    for env_name in ("PAP_ATTENTION_ZMQ_PORT_BASE", "PAP_ATTENTION_PORT_BASE"):
        base_raw = os.environ.get(env_name)
        if not base_raw:
            continue
        try:
            ranked_base = int(base_raw) + rank
        except ValueError:
            continue
        offset = peer_port - ranked_base
        if offset >= 0:
            return offset
    return 0


def _pap_offload_exec_peer_zmq_port(peer_endpoint: str | None) -> int:
    return _pap_offload_exec_base_zmq_port() + _pap_offload_exec_peer_port_offset(
        peer_endpoint
    )


@lru_cache(maxsize=1)
def _pap_offload_exec_transport():
    from vllm.pap.data_plane import (
        build_local_fast_offload_exec_transport,
        build_nixl_mailbox_offload_exec_transport,
    )

    transport = _pap_offload_exec_transport_kind()
    local_rank = _pap_tensor_parallel_rank()
    if transport in {"nixl", "nixl_mailbox"}:
        return build_nixl_mailbox_offload_exec_transport(
            actor_id=os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection"),
            local_rank=local_rank,
        )
    if transport in {"local_fast", "local-fast", "cuda_ipc_fast"}:
        return build_local_fast_offload_exec_transport(
            actor_id=os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection"),
            local_rank=local_rank,
        )

    raise RuntimeError(
        f"PAP OFFLOAD_EXEC transport {transport!r} is not supported; use "
        "nixl_mailbox or local_fast"
    )


@cache
def _pap_nixl_mailbox_offload_exec_transport(attention_endpoint: str):
    from vllm.pap.data_plane import (
        build_local_fast_offload_exec_transport,
        build_nixl_mailbox_offload_exec_transport,
    )

    local_rank = _pap_tensor_parallel_rank()
    actor_base = os.environ.get("PAP_NIXL_MAILBOX_ACTOR_ID", "projection")
    endpoint_hash = hashlib.sha1(attention_endpoint.encode("utf-8")).hexdigest()[:12]
    actor_id = f"{actor_base}-r{local_rank}-{endpoint_hash}"
    transport = _pap_offload_exec_transport_kind()
    if transport in {"local_fast", "local-fast", "cuda_ipc_fast"}:
        return build_local_fast_offload_exec_transport(
            actor_id=actor_id,
            local_rank=local_rank,
        )
    return build_nixl_mailbox_offload_exec_transport(
        actor_id=actor_id,
        local_rank=local_rank,
    )


def _pap_offload_exec_transport_for_attention_endpoint(
    attention_endpoint: str | None,
    offload_exec_zmq_endpoint: str | None = None,
):
    transport = _pap_offload_exec_transport_kind()
    if transport in {"nixl", "nixl_mailbox"}:
        return _pap_nixl_mailbox_offload_exec_transport(str(attention_endpoint or ""))
    if transport in {"local_fast", "local-fast", "cuda_ipc_fast"}:
        return _pap_nixl_mailbox_offload_exec_transport(str(attention_endpoint or ""))
    raise RuntimeError(
        f"PAP OFFLOAD_EXEC transport {transport!r} is not supported; use "
        "nixl_mailbox or local_fast"
    )


def _pap_bind_offload_exec_mailbox_peer(
    transport: Any,
    attention_endpoint: str | None,
) -> None:
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
        source_id=(
            f"{os.environ.get('PAP_NIXL_MAILBOX_ACTOR_ID', 'projection')}"
            f"-r{_pap_tensor_parallel_rank()}"
        ),
    )
    transport.bind_peer(peer_metadata)
    transport._pap_mailbox_bound = True
    transport._pap_mailbox_bound_attention_endpoint = attention_endpoint


def _pap_offload_exec_local_address(peer_endpoint: str | None = None) -> str:
    host = os.environ.get("PAP_OFFLOAD_EXEC_HOST") or "127.0.0.1"
    if not peer_endpoint:
        port = _pap_offload_exec_base_zmq_port()
        return os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_ADDRESS", f"{host}:{port}")
    port = _pap_offload_exec_peer_zmq_port(peer_endpoint)
    return f"{host}:{port}"


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
        self._pap_imported_prefill_kv: set[tuple[Any, ...]] = set()
        self._pap_is_last_layer = bool(is_last_layer)
        self._pap_expected_layer_count = int(num_hidden_layers or 0)
        self._pap_prefill_kv_catalog_id = (
            os.environ.get("PAP_KV_CATALOG_ID") or f"prefill-{os.getpid()}"
        )
        self._pap_registered_kv_catalog_endpoints: set[str] = set()
        self._pap_manifest_ready_events: dict[
            tuple[str, int], torch.cuda.Event
        ] = {}
        self._pap_last_projection_timeline: dict[str, Any] | None = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self._pap_last_projection_timeline = None

        layer_name = self.attn.layer_name
        pap_attention_enabled = self._should_use_pap_attention()
        deferred_qkv_role = ""
        if deferred_cuda_trace_enabled():
            deferred_metadata = _qwen3_profile_attn_metadata(layer_name)
            deferred_qkv_role = _qwen3_deferred_qkv_trace_selected_role(
                trace_role=deferred_trace_role(),
                pap_attention_enabled=pap_attention_enabled,
                max_query_len=int(
                    getattr(deferred_metadata, "max_query_len", 0)
                ),
            )
        qkv_trace = None
        if deferred_qkv_role:
            qkv_trace = begin_deferred_cuda_span(
                "qkv_norm_rope_gpu_ms",
                torch.cuda.current_stream(hidden_states.device),
            )
        trace_offload_exec = _pap_env_enabled("PAP_OFFLOAD_EXEC_TRACE")
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
            and _pap_direct_qkv_send_enabled()
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
                qkv[:, : self.q_size].copy_(
                    q.reshape(qkv.shape[0], self.q_size)
                )
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
        if pap_attention_enabled:
            projection_timeline: dict[str, Any] | None = (
                {} if trace_offload_exec else None
            )
            attn_output, pap_release_messages = self._compute_pap_attention(
                q,
                k,
                v,
                pre_attn_compute_ms=trace_pre_attn_compute_ms,
                pre_attn_start_ns=trace_pre_attn_start_ns,
                pre_attn_done_ns=trace_pre_attn_done_ns,
                projection_timeline=projection_timeline,
                direct_qkv_send_buffer=direct_qkv_send_buffer,
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
                self._pap_last_projection_timeline = dict(projection_timeline)
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
        self._maybe_import_pap_prefill_kv_to_attention()
        with _qwen3_profile_context(
            layer_name=layer_name,
            layer_index=self.layer_index,
            stage="o_proj",
            hidden_states=hidden_states,
        ):
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
        if not pap_request_ids_are_routable(request_ids, num_reqs):
            return reject(
                "non-PAP request id in scheduled batch "
                f"request_ids={request_ids[:num_reqs][:4]}"
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
            raise RuntimeError("PAP remote attention supports decode-only batches")

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
            if not is_pap_request_id(request_id):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            if num_scheduled_tokens and int(num_scheduled_tokens[req_index]) != 1:
                raise RuntimeError("PAP remote attention expects one token per request")
            seq_len = int(positions_cpu[req_index].item()) + 1
            max_seq_len = int(seq_lens_cpu[req_index].item())
            if seq_len != max_seq_len:
                raise RuntimeError(
                    f"PAP attention position-derived seq_len {seq_len} differs from "
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
            session_request_id = _pap_offload_exec_session_request_id(
                request_id,
                prefill_kv_handle,
            )
            descriptor = PAPOffloadExecDescriptor(
                request_id=session_request_id,
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
                attention_endpoint,
                offload_exec_zmq_endpoint,
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
                if int(output_batch.shape[0]) != batch_descriptor.item_count:
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC output batch row count mismatch"
                    )
                can_use_direct_output = (
                    _pap_env_enabled("PAP_DIRECT_MAILBOX_OUTPUT")
                    and output_message is not None
                    and len(sent_batches) == 1
                    and len(req_indices) == batch_descriptor.item_count
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

    def _compute_pap_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        pre_attn_compute_ms: float = 0.0,
        pre_attn_start_ns: int = 0,
        pre_attn_done_ns: int = 0,
        projection_timeline: dict[str, Any] | None = None,
        direct_qkv_send_buffer: torch.Tensor | None = None,
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

        from vllm.pap.data_plane import (
            PAPOffloadExecBatchDescriptor,
            pap_offload_exec_trace_id,
        )

        step_groups = _pap_offload_exec_step_groups(
            additional_kwargs,
            num_reqs=num_reqs,
            scaling=float(self.scaling),
        )
        all_requests_offloaded = (
            sum(len(group.req_indices) for group in step_groups) == num_reqs
        )
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

        direct_qkv_send_enabled = _pap_direct_qkv_send_enabled()
        batched_route_copy_enabled = _pap_batched_route_copy_enabled()

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
        trace_recv_ms = 0.0
        trace_total_ms = 0.0
        trace_batch_keys = ""
        trace_contiguous_route_groups = 0
        trace_direct_qkv_groups = 0
        trace_packed_qkv_groups = 0
        trace_direct_output_rows = 0
        trace_scattered_output_rows = 0
        trace_send_start = time.perf_counter() if trace_offload_exec else 0.0
        offload_exec_batches: list[
            tuple[
                str | None,
                str,
                PAPOffloadExecBatchDescriptor,
                tuple[int, ...],
                Any,
                torch.Tensor | None,
            ]
        ] = []
        for step_group in step_groups:
            attention_endpoint = step_group.attention_endpoint
            offload_exec_zmq_endpoint = step_group.offload_exec_zmq_endpoint
            req_indices = step_group.req_indices
            route_is_contiguous = _pap_req_indices_are_contiguous(req_indices)
            if trace_offload_exec and route_is_contiguous:
                trace_contiguous_route_groups += 1
            route_index_tensor = (
                None
                if route_is_contiguous or not batched_route_copy_enabled
                else _pap_route_index_tensor(
                    additional_kwargs,
                    req_indices,
                    device=query.device,
                )
            )
            transport = _pap_offload_exec_transport_for_attention_endpoint(
                attention_endpoint,
                offload_exec_zmq_endpoint,
            )
            batch_descriptor = PAPOffloadExecBatchDescriptor(
                layer_name=self.attn.layer_name,
                items=(),
                batch_id_suffix=step_group.batch_id_suffix,
                metadata_template=step_group.metadata_template,
            )
            _pap_bind_offload_exec_mailbox_peer(transport, attention_endpoint)
            send_qkv_batch_direct = getattr(transport, "send_qkv_batch_direct", None)
            qkv_width = (
                self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
            )
            direct_qkv_batch: torch.Tensor | None = None
            direct_layout = False
            if direct_qkv_send_enabled and callable(send_qkv_batch_direct):
                if batched_route_copy_enabled:
                    direct_qkv_batch, direct_layout = _pap_qkv_batch_for_indices(
                        direct_qkv_send_buffer,
                        req_indices,
                        index_tensor=route_index_tensor,
                    )
                else:
                    direct_qkv_batch = _pap_direct_qkv_batch_for_indices(
                        direct_qkv_send_buffer,
                        req_indices,
                    )
                    direct_layout = direct_qkv_batch is not None
            if direct_qkv_batch is not None:
                if trace_offload_exec:
                    if direct_layout:
                        trace_direct_qkv_groups += 1
                    else:
                        trace_packed_qkv_groups += 1
                if int(direct_qkv_batch.shape[-1]) != qkv_width:
                    raise RuntimeError("PAP direct QKV batch width mismatch")
                send_qkv_batch_direct(
                    batch_descriptor,
                    direct_qkv_batch,
                    remote_address=offload_exec_zmq_endpoint,
                )
            else:
                if trace_offload_exec:
                    trace_packed_qkv_groups += 1
                group_items = [
                    (
                        req_index,
                        None,
                        (
                            query[req_index : req_index + 1].reshape(1, -1),
                            key[req_index : req_index + 1].reshape(1, -1),
                            value[req_index : req_index + 1].reshape(1, -1),
                        ),
                    )
                    for req_index in req_indices
                ]
                qkv_batch = _pap_pack_qkv_group_items(group_items)
                transport.send_qkv_batch(
                    batch_descriptor,
                    qkv_batch,
                    remote_address=offload_exec_zmq_endpoint,
                )
            offload_exec_batches.append(
                (
                    attention_endpoint,
                    offload_exec_zmq_endpoint,
                    batch_descriptor,
                    req_indices,
                    transport,
                    route_index_tensor,
                )
            )
        trace_send_ms = (
            (time.perf_counter() - trace_send_start) * 1000.0
            if trace_offload_exec
            else 0.0
        )
        if trace_offload_exec:
            trace_send_done_ns = time.perf_counter_ns()

        trace_trigger_ms = 0.0

        trace_yield_start = time.perf_counter() if trace_offload_exec else 0.0
        if trace_offload_exec:
            trace_yield_start_ns = time.perf_counter_ns()
        trace_yield_ms = 0.0
        if trace_offload_exec:
            trace_yield_end_ns = time.perf_counter_ns()
            trace_yield_ms = (time.perf_counter() - trace_yield_start) * 1000.0

        trace_recv_start = time.perf_counter() if trace_offload_exec else 0.0

        def record_projection_trace() -> None:
            nonlocal trace_batch_keys
            if not trace_offload_exec or not offload_exec_batches:
                return
            trace_batch_keys = "|".join(
                pap_offload_exec_trace_id(batch[2].output_tensor_id)
                for batch in offload_exec_batches
            )
            calls = sum(batch[2].item_count for batch in offload_exec_batches)
            if projection_timeline is not None:
                projection_timeline.update(
                    {
                        "layer": offload_exec_batches[0][2].layer_name,
                        "batches": len(offload_exec_batches),
                        "calls": calls,
                        "pre_attn_compute_ms": pre_attn_compute_ms,
                        "send_ms": trace_send_ms,
                        "trigger_ms": trace_trigger_ms,
                        "yield_ms": trace_yield_ms,
                        "recv_ms": trace_recv_ms,
                        "remote_total_ms": trace_total_ms,
                        "batch_keys": trace_batch_keys,
                        "pre_attn_start_ns": pre_attn_start_ns,
                        "pre_attn_done_ns": pre_attn_done_ns,
                        "send_done_ns": trace_send_done_ns,
                        "yield_start_ns": trace_yield_start_ns,
                        "yield_end_ns": trace_yield_end_ns,
                        "recv_done_ns": trace_recv_done_ns,
                        "route_groups": len(step_groups),
                        "contiguous_route_groups": (trace_contiguous_route_groups),
                        "direct_qkv_groups": trace_direct_qkv_groups,
                        "packed_qkv_groups": trace_packed_qkv_groups,
                        "direct_output_rows": trace_direct_output_rows,
                        "scattered_output_rows": trace_scattered_output_rows,
                    }
                )
            logger.info(
                "PAP OFFLOAD_EXEC projection trace layer=%s batches=%d "
                "calls=%d send_ms=%.3f trigger_ms=%.3f "
                "yield_ms=%.3f recv_ms=%.3f total_ms=%.3f batch_keys=%s "
                "send_done_ns=%d yield_start_ns=%d yield_end_ns=%d "
                "recv_done_ns=%d route_groups=%d "
                "contiguous_route_groups=%d direct_qkv_groups=%d "
                "packed_qkv_groups=%d direct_output_rows=%d "
                "scattered_output_rows=%d",
                offload_exec_batches[0][2].layer_name,
                len(offload_exec_batches),
                calls,
                trace_send_ms,
                trace_trigger_ms,
                trace_yield_ms,
                trace_recv_ms,
                trace_total_ms,
                trace_batch_keys,
                trace_send_done_ns,
                trace_yield_start_ns,
                trace_yield_end_ns,
                trace_recv_done_ns,
                len(step_groups),
                trace_contiguous_route_groups,
                trace_direct_qkv_groups,
                trace_packed_qkv_groups,
                trace_direct_output_rows,
                trace_scattered_output_rows,
            )

        for (
            _attention_endpoint,
            offload_exec_zmq_endpoint,
            batch_descriptor,
            req_indices,
            transport,
            route_index_tensor,
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
                if int(output_batch.shape[0]) != batch_descriptor.item_count:
                    raise RuntimeError(
                        "PAP OFFLOAD_EXEC output batch row count mismatch"
                    )
                can_use_direct_output = (
                    direct_mailbox_output_enabled
                    and len(offload_exec_batches) == 1
                    and len(req_indices) == num_reqs
                    and req_indices == tuple(range(num_reqs))
                    and output_batch.device == query.device
                    and output_batch.dtype == query.dtype
                    and int(output_batch.numel())
                    == int(q.shape[0]) * self.num_heads * self.head_dim
                )
                if can_use_direct_output:
                    if trace_offload_exec:
                        trace_direct_output_rows += len(req_indices)
                    direct_output = output_batch.view(
                        q.shape[0], self.num_heads * self.head_dim
                    )
                    if output_message is not None:
                        pap_release_messages.append(output_message)
                        output_message = None
                    if trace_offload_exec:
                        trace_recv_done_ns = time.perf_counter_ns()
                        trace_recv_ms = (
                            time.perf_counter() - trace_recv_start
                        ) * 1000.0
                        trace_total_ms = (
                            time.perf_counter() - trace_total_start
                        ) * 1000.0
                        record_projection_trace()
                    return direct_output, pap_release_messages
                if trace_offload_exec:
                    trace_scattered_output_rows += len(req_indices)
                if batched_route_copy_enabled:
                    _pap_scatter_attention_output_group(
                        get_copy_output_buffer(),
                        output_batch,
                        req_indices=req_indices,
                        index_tensor=route_index_tensor,
                    )
                else:
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
            record_projection_trace()
        final_output = get_copy_output_buffer()
        return final_output.view(q.shape[0], self.num_heads * self.head_dim), []

    def _maybe_import_pap_prefill_kv_to_attention(self) -> None:
        profile_ipc = _pap_prefill_ipc_profile_enabled()
        profile_total_start = time.perf_counter() if profile_ipc else 0.0
        if not is_forward_context_available():
            return
        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs or {}
        if not additional_kwargs.get("pap_enabled"):
            return
        finished_request_ids = tuple(
            str(request_id)
            for request_id in additional_kwargs.get("pap_finished_request_ids") or ()
        )
        _pap_prune_imported_prefill_kv(
            self._pap_imported_prefill_kv,
            finished_request_ids,
        )
        if finished_request_ids and self._pap_manifest_ready_events:
            finished = set(finished_request_ids)
            self._pap_manifest_ready_events = {
                key: event
                for key, event in self._pap_manifest_ready_events.items()
                if key[0] not in finished
            }

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
        if _pap_kv_handoff_mode() == "sealed_manifest":
            self._publish_pap_prefill_kv_manifests(
                request_ids=request_ids,
                num_reqs=num_reqs,
                num_scheduled_tokens=num_scheduled_tokens,
                prefill_kv_handle_by_request=prefill_kv_handle_by_request,
                import_request_ids=import_prefill_kv_to_attention_by_request,
                tcp_endpoint_by_request=tcp_endpoint_by_request,
                default_tcp_endpoint=default_tcp_endpoint,
                seq_lens=seq_lens,
                block_table=block_table,
                kv_cache=kv_cache,
                block_size=int(block_size),
                layout=get_kv_cache_layout(),
            )
            return
        seq_lens_start = time.perf_counter() if profile_ipc else 0.0
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        seq_lens_cpu_ms = (
            (time.perf_counter() - seq_lens_start) * 1000.0 if profile_ipc else 0.0
        )
        import_count = 0
        block_ids_total_ms = 0.0
        import_total_ms = 0.0
        async_import = _pap_prefill_kv_async_enabled()
        for req_index in range(num_reqs):
            req_profile_start = time.perf_counter() if profile_ipc else 0.0
            request_id = str(request_ids[req_index])
            if not is_pap_request_id(request_id):
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
                async_import,
            )
            if import_key in self._pap_imported_prefill_kv:
                continue
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = _pap_endpoint_for_tp_rank(tcp_endpoint)
            block_ids_start = time.perf_counter() if profile_ipc else 0.0
            unified_export = _pap_unified_kv_export_enabled()
            unified_capacity_tokens = 0
            if unified_export:
                unified_capacity_tokens = (
                    _pap_unified_kv_export_decode_capacity_tokens()
                )
            block_seq_len = prefix_len
            if unified_export and unified_capacity_tokens > 0:
                block_seq_len = prefix_len + unified_capacity_tokens
            block_ids = _pap_block_ids_from_block_table(
                block_table=block_table[req_index : req_index + 1],
                seq_len=block_seq_len,
                block_size=int(block_size),
            )

            block_ids_ms = (
                (time.perf_counter() - block_ids_start) * 1000.0 if profile_ipc else 0.0
            )
            publish_start = time.perf_counter() if profile_ipc else 0.0
            import_prefill_paged_kv(
                request_id=request_id,
                layer_name=self.attn.layer_name,
                kv_cache=kv_cache,
                block_ids=block_ids,
                seq_len=prefix_len,
                block_size=int(block_size),
                num_kv_heads=self.num_kv_heads,
                layout=get_kv_cache_layout(),
                tcp_endpoint=tcp_endpoint,
            )
            publish_ms = (
                (time.perf_counter() - publish_start) * 1000.0 if profile_ipc else 0.0
            )
            self._pap_imported_prefill_kv.add(import_key)
            if profile_ipc:
                import_count += 1
                block_ids_total_ms += block_ids_ms
                import_total_ms += publish_ms
                logger.info(
                    "PAP prefill IPC model profile request_id=%s layer=%s "
                    "prefix_len=%d blocks=%d async=%s seq_lens_cpu_ms=%.3f "
                    "block_ids_ms=%.3f publish_response_wait_ms=%.3f "
                    "total_ms=%.3f",
                    request_id,
                    self.attn.layer_name,
                    prefix_len,
                    len(block_ids),
                    async_import,
                    seq_lens_cpu_ms,
                    block_ids_ms,
                    publish_ms,
                    (time.perf_counter() - req_profile_start) * 1000.0,
                )
        if profile_ipc and import_count:
            logger.info(
                "PAP prefill IPC model aggregate layer=%s imports=%d "
                "seq_lens_cpu_ms=%.3f block_ids_total_ms=%.3f "
                "publish_response_wait_total_ms=%.3f total_ms=%.3f async=%s",
                self.attn.layer_name,
                import_count,
                seq_lens_cpu_ms,
                block_ids_total_ms,
                import_total_ms,
                (time.perf_counter() - profile_total_start) * 1000.0,
                async_import,
            )

    def _publish_pap_prefill_kv_manifests(
        self,
        *,
        request_ids: tuple[Any, ...],
        num_reqs: int,
        num_scheduled_tokens: tuple[int, ...],
        prefill_kv_handle_by_request: dict[Any, Any],
        import_request_ids: set[Any],
        tcp_endpoint_by_request: dict[Any, Any],
        default_tcp_endpoint: Any,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        kv_cache: torch.Tensor,
        block_size: int,
        layout: str,
    ) -> None:
        """Register static KV backing and publish request-level layouts."""

        from vllm.pap.shadow_attention import (
            publish_prefill_kv_session_manifest,
            register_prefill_kv_catalog,
            select_attention_endpoint_for_request,
        )

        eligible: list[tuple[int, str, str, str]] = []
        for req_index in range(num_reqs):
            if num_scheduled_tokens[req_index] <= 0:
                continue
            request_id = str(request_ids[req_index])
            if not is_pap_request_id(request_id):
                continue
            if request_id not in import_request_ids:
                continue
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if not prefill_kv_handle:
                continue
            tcp_endpoint = select_attention_endpoint_for_request(
                request_id,
                default_endpoint=default_tcp_endpoint,
                endpoint_by_request=tcp_endpoint_by_request,
            )
            tcp_endpoint = str(_pap_endpoint_for_tp_rank(tcp_endpoint))
            eligible.append(
                (req_index, request_id, str(prefill_kv_handle), tcp_endpoint)
            )
        if not eligible:
            return

        for tcp_endpoint in dict.fromkeys(item[3] for item in eligible):
            if tcp_endpoint in self._pap_registered_kv_catalog_endpoints:
                continue
            status = register_prefill_kv_catalog(
                catalog_id=self._pap_prefill_kv_catalog_id,
                layer_name=self.attn.layer_name,
                kv_cache=kv_cache,
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                layout=layout,
                tcp_endpoint=tcp_endpoint,
            )
            self._pap_registered_kv_catalog_endpoints.add(tcp_endpoint)
            logger.info(
                "PAP Prefill KV catalog %s catalog_id=%s layer=%s endpoint=%s",
                status,
                self._pap_prefill_kv_catalog_id,
                self.attn.layer_name,
                tcp_endpoint,
            )
        if not self._pap_is_last_layer:
            return
        if self._pap_expected_layer_count <= 0:
            raise RuntimeError("sealed Prefill KV handoff requires model layer count")
        if kv_cache.device.type != "cuda":
            raise RuntimeError("sealed Prefill KV handoff requires CUDA KV cache")

        ready_event = torch.cuda.Event(interprocess=True)
        ready_event.record(torch.cuda.current_stream(kv_cache.device))
        ready_event_handle = ready_event.ipc_handle()
        seq_lens_cpu = seq_lens.detach().to(device="cpu", dtype=torch.long)
        published = 0
        for req_index, request_id, prefill_kv_handle, tcp_endpoint in eligible:
            prefix_len = int(seq_lens_cpu[req_index].item())
            if prefix_len <= 1:
                continue
            import_key = (
                request_id,
                "sealed_manifest",
                prefix_len,
                prefill_kv_handle,
                tcp_endpoint,
            )
            if import_key in self._pap_imported_prefill_kv:
                continue
            block_seq_len = prefix_len
            unified_capacity_tokens = (
                _pap_unified_kv_export_decode_capacity_tokens()
            )
            if unified_capacity_tokens > 0:
                block_seq_len += unified_capacity_tokens
            block_ids = _pap_block_ids_from_block_table(
                block_table=block_table[req_index : req_index + 1],
                seq_len=block_seq_len,
                block_size=block_size,
            )
            publish_prefill_kv_session_manifest(
                request_id=request_id,
                catalog_id=self._pap_prefill_kv_catalog_id,
                block_ids=block_ids,
                prefix_len=prefix_len,
                block_size=block_size,
                expected_layer_count=self._pap_expected_layer_count,
                ready_event_handle=ready_event_handle,
                tcp_endpoint=tcp_endpoint,
            )
            self._pap_imported_prefill_kv.add(import_key)
            self._pap_manifest_ready_events[(request_id, prefix_len)] = ready_event
            published += 1
        if published:
            logger.info(
                "PAP Prefill KV manifests published catalog_id=%s requests=%d "
                "prefix_min=%d prefix_max=%d",
                self._pap_prefill_kv_catalog_id,
                published,
                min(
                    key[1]
                    for key in self._pap_manifest_ready_events
                    if key[0] in {item[1] for item in eligible}
                ),
                max(
                    key[1]
                    for key in self._pap_manifest_ready_events
                    if key[0] in {item[1] for item in eligible}
                ),
            )


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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_name = self.self_attn.attn.layer_name
        trace_projection_layer = _pap_env_enabled("PAP_OFFLOAD_EXEC_TRACE") and (
            _pap_env_enabled("PAP_PROJECTION_KV_UNAWARE")
            or _pap_projection_critical_trace_enabled()
        )
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
            projection_timeline = self.self_attn._pap_last_projection_timeline or {}
            if _pap_projection_critical_trace_enabled() and projection_timeline:
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
