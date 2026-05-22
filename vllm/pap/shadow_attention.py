# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side PAP shadow attention hook.

This module reports metadata at the true Qwen3 q/k/v -> attention boundary
and provides the first remote-output path for the PAP prototype. The current
path is deliberately conservative: projection can still run local attention for
KV update/fallback, then ask the internal attention executor for an output to
feed into ``o_proj``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch

logger = logging.getLogger(__name__)

_OPENAI_REQUEST_ID_PREFIXES = ("cmpl-", "chatcmpl-")


def _pap_config() -> dict[str, Any]:
    try:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    except Exception:
        return {}
    if vllm_config is None or vllm_config.kv_transfer_config is None:
        return {}
    return vllm_config.kv_transfer_config.kv_connector_extra_config


def _enabled(enabled: bool | str | None = None) -> bool:
    if enabled is not None:
        return str(enabled).lower() in {"1", "true", "yes"}
    config = _pap_config()
    configured = config.get(
        "pap_shadow_attention", os.environ.get("PAP_SHADOW_ATTENTION", "")
    )
    return str(configured).lower() in {"1", "true", "yes"}


def _attention_endpoint(endpoint: str | None = None) -> str:
    if endpoint is not None:
        return str(endpoint).rstrip("/")
    config = _pap_config()
    configured = config.get(
        "pap_attention_endpoint",
        os.environ.get("PAP_ATTENTION_ENDPOINT", "http://127.0.0.1:8300"),
    )
    return str(configured).rstrip("/")


def _select_request_id(request_ids: Sequence[str] | None) -> str | None:
    if not request_ids:
        return None
    for request_id in request_ids:
        request_id_str = str(request_id)
        if request_id_str.startswith(_OPENAI_REQUEST_ID_PREFIXES):
            return request_id_str
    return None


def _is_decode(num_scheduled_tokens: Sequence[int] | None, query: torch.Tensor) -> bool:
    if num_scheduled_tokens is not None and len(num_scheduled_tokens) > 0:
        return all(int(num_tokens) == 1 for num_tokens in num_scheduled_tokens)
    return query.shape[0] == 1


def build_layer_event_payload(
    *,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    request_ids: Sequence[str] | None,
    num_scheduled_tokens: Sequence[int] | None,
    num_reqs: int | None,
    num_actual_tokens: int | None,
    max_seq_len: int | None,
) -> dict[str, Any] | None:
    request_id = _select_request_id(request_ids)
    if request_id is None:
        return None

    return {
        "request_id": request_id,
        "layer_name": layer_name,
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "dtype": str(query.dtype),
        "device": str(query.device),
        "is_decode": _is_decode(num_scheduled_tokens, query),
        "num_reqs": None if num_reqs is None else int(num_reqs),
        "num_actual_tokens": None
        if num_actual_tokens is None
        else int(num_actual_tokens),
        "max_seq_len": None if max_seq_len is None else int(max_seq_len),
    }


def maybe_report_qkv_boundary(
    *,
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    request_ids: Sequence[str] | None,
    num_scheduled_tokens: Sequence[int] | None,
    num_reqs: int | None,
    num_actual_tokens: int | None,
    max_seq_len: int | None,
    enabled: bool | str | None = None,
    endpoint: str | None = None,
) -> None:
    if not _enabled(enabled):
        return

    payload = build_layer_event_payload(
        layer_name=layer_name,
        query=query,
        key=key,
        value=value,
        request_ids=request_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_seq_len=max_seq_len,
    )
    if payload is None:
        return

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{_attention_endpoint(endpoint)}/v1/pap/attention/layer-event",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=0.5):
            return
    except (TimeoutError, URLError, OSError):
        logger.debug(
            "failed to report PAP shadow attention event request_id=%s endpoint=%s",
            payload.get("request_id"),
            _attention_endpoint(endpoint),
            exc_info=True,
        )


def build_remote_attention_request(
    *,
    request_id: str,
    layer_name: str,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    scale: float,
    layout: str,
) -> dict[str, Any]:
    from vllm.pap.remote_attention import gather_paged_kv, serialize_tensor

    if layout not in {"NHD", "HND"}:
        raise ValueError(f"unsupported KV cache layout: {layout}")
    key, value = gather_paged_kv(
        kv_cache=kv_cache,
        block_table=block_table,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        layout=layout,
    )
    return {
        "request_id": request_id,
        "layer_name": layer_name,
        "query": serialize_tensor(query),
        "key": serialize_tensor(key),
        "value": serialize_tensor(value),
        "scale": float(scale),
    }


def compute_remote_attention_output(
    *,
    request_id: str,
    layer_name: str,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    scale: float,
    layout: str,
    endpoint: str | None = None,
    timeout: float | None = None,
) -> torch.Tensor:
    from vllm.pap.remote_attention import deserialize_attention_result

    payload = build_remote_attention_request(
        request_id=request_id,
        layer_name=layer_name,
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        scale=scale,
        layout=layout,
    )
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{_attention_endpoint(endpoint)}/v1/pap/attention/compute",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request_timeout = (
        float(timeout)
        if timeout is not None
        else float(os.environ.get("PAP_REMOTE_ATTENTION_TIMEOUT", "5.0"))
    )
    with urlopen(request, timeout=request_timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return deserialize_attention_result(result["output"])
