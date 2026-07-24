# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed access to PAP fields in the vLLM model forward context."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context, is_forward_context_available

_PAP_FORWARD_BATCH_BASE_KEY = "_pap_model_forward_batch_base"


def _attention_metadata_for_layer(metadata: Any, layer_name: str) -> Any | None:
    if isinstance(metadata, dict):
        return metadata.get(layer_name)
    if isinstance(metadata, list) and metadata:
        return metadata[0].get(layer_name)
    return None


@dataclass(frozen=True, slots=True)
class _PAPModelForwardBatchBase:
    request_ids: tuple[str, ...]
    num_scheduled_tokens: tuple[int, ...]
    num_reqs: int
    num_actual_tokens: int


@dataclass(frozen=True, slots=True)
class PAPModelForwardBatch:
    """PAP request and Attention metadata for one model forward."""

    additional_kwargs: dict[str, Any]
    attention_metadata: Any | None
    request_ids: tuple[str, ...]
    num_scheduled_tokens: tuple[int, ...]
    num_reqs: int
    num_actual_tokens: int

    @property
    def enabled(self) -> bool:
        """Whether PAP is enabled for this forward."""
        return bool(self.additional_kwargs.get("pap_enabled"))

    @classmethod
    def current(cls, layer_name: str) -> PAPModelForwardBatch | None:
        """Read the current vLLM forward context for one Attention layer."""
        if not is_forward_context_available():
            return None
        forward_context = get_forward_context()
        additional_kwargs = forward_context.additional_kwargs
        if additional_kwargs is None:
            additional_kwargs = {}
        base = additional_kwargs.get(_PAP_FORWARD_BATCH_BASE_KEY)
        if base is None:
            base = _PAPModelForwardBatchBase(
                request_ids=tuple(
                    str(request_id)
                    for request_id in additional_kwargs.get("pap_request_ids") or ()
                ),
                num_scheduled_tokens=tuple(
                    int(num_tokens)
                    for num_tokens in (
                        additional_kwargs.get("pap_num_scheduled_tokens") or ()
                    )
                ),
                num_reqs=int(
                    additional_kwargs.get("pap_num_reqs")
                    or len(
                        additional_kwargs.get("pap_num_scheduled_tokens") or ()
                    )
                ),
                num_actual_tokens=int(
                    additional_kwargs.get("pap_num_actual_tokens")
                    or additional_kwargs.get("pap_num_reqs")
                    or len(
                        additional_kwargs.get("pap_num_scheduled_tokens") or ()
                    )
                ),
            )
            additional_kwargs[_PAP_FORWARD_BATCH_BASE_KEY] = base
        return cls(
            additional_kwargs=additional_kwargs,
            attention_metadata=_attention_metadata_for_layer(
                forward_context.attn_metadata,
                layer_name,
            ),
            request_ids=base.request_ids,
            num_scheduled_tokens=base.num_scheduled_tokens,
            num_reqs=base.num_reqs,
            num_actual_tokens=base.num_actual_tokens,
        )


def pap_tensor_parallel_rank() -> int:
    """Return the PAP-local tensor-parallel rank."""
    raw_rank = os.environ.get("PAP_OFFLOAD_EXEC_LOCAL_RANK")
    if raw_rank is not None:
        return int(raw_rank)
    return int(get_tensor_model_parallel_rank())


def pap_endpoint_for_tp_rank(value: Any, *, tp_rank: int | None = None) -> Any:
    """Select the endpoint assigned to one tensor-parallel rank."""
    if value is None:
        return None
    rank = pap_tensor_parallel_rank() if tp_rank is None else int(tp_rank)
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
