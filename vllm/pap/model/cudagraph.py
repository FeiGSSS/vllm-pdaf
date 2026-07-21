# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph boundaries for PAP model-side effects."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch

from vllm.forward_context import get_forward_context
from vllm.pap.model.context import PAPModelForwardBatch
from vllm.utils.torch_utils import direct_register_custom_op

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_PAP_CUDAGRAPH_ROLES = {"prefill", "projection"}


def pap_model_hooks_enabled(values: Mapping[str, str] | None = None) -> bool:
    """Return whether this process owns a PAP model-side role."""
    source = os.environ if values is None else values
    explicit = source.get("PAP_MODEL_HOOKS")
    if explicit is not None:
        return explicit.lower() in _TRUE_ENV_VALUES
    projection = source.get("PAP_PROJECTION_KV_UNAWARE", "").lower()
    return projection in _TRUE_ENV_VALUES or bool(source.get("PAP_TOPOLOGY"))


def pap_cudagraph_role(values: Mapping[str, str] | None = None) -> str | None:
    """Return the statically configured PAP role for piecewise CUDA Graphs."""
    source = os.environ if values is None else values
    enabled = source.get("PAP_CUDAGRAPH_COMPATIBLE", "").lower()
    if enabled not in _TRUE_ENV_VALUES:
        return None
    role = source.get("PAP_CUDAGRAPH_ROLE", "").lower()
    if role not in _PAP_CUDAGRAPH_ROLES:
        choices = ", ".join(sorted(_PAP_CUDAGRAPH_ROLES))
        raise RuntimeError(
            "PAP_CUDAGRAPH_ROLE must be one of "
            f"{choices} when PAP_CUDAGRAPH_COMPATIBLE is enabled"
        )
    return role


def bind_pap_cudagraph_adapters(
    attention: Any,
    *,
    projection_adapter: Any,
    prefill_publisher: Any,
) -> None:
    """Expose graph-excluded PAP adapters through the Attention context."""
    attention._pap_cudagraph_projection_adapter = projection_adapter
    attention._pap_cudagraph_prefill_publisher = prefill_publisher


def _pap_attention_layer(layer_name: str) -> Any:
    forward_context = get_forward_context()
    return forward_context.no_compile_layers[layer_name]


def _pap_projection_attention_with_output_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    direct_qkv_send_buffer: torch.Tensor | None,
    layer_name: str,
) -> None:
    attention = _pap_attention_layer(layer_name)
    adapter = attention._pap_cudagraph_projection_adapter
    batch = PAPModelForwardBatch.current(layer_name)

    # vLLM profiles and captures with synthetic forwards that do not carry PAP
    # request metadata. They only need shape-correct data to size later graphs.
    if batch is None or not batch.enabled:
        output.zero_()
        return
    if not adapter.should_execute():
        raise RuntimeError("PAP Projection CUDA Graph received an invalid batch")

    remote_output, release_messages = adapter.execute(
        query,
        key,
        value,
        direct_qkv_send_buffer=direct_qkv_send_buffer,
    )
    try:
        output.copy_(remote_output.reshape_as(output))
    finally:
        for message in release_messages:
            message.release()


def _pap_projection_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    direct_qkv_send_buffer: torch.Tensor | None,
    layer_name: str,
) -> None:
    return


def _pap_publish_prefill_kv_impl(
    attention_output: torch.Tensor,
    layer_name: str,
) -> None:
    attention = _pap_attention_layer(layer_name)
    publisher = attention._pap_cudagraph_prefill_publisher
    publisher.publish(attention)


def _pap_publish_prefill_kv_fake(
    attention_output: torch.Tensor,
    layer_name: str,
) -> None:
    return


_CUDAGRAPH_UNSAFE_TAGS = (torch.Tag.cudagraph_unsafe,)

direct_register_custom_op(
    op_name="pap_projection_attention_with_output",
    op_func=_pap_projection_attention_with_output_impl,
    mutates_args=["output"],
    fake_impl=_pap_projection_attention_with_output_fake,
    tags=_CUDAGRAPH_UNSAFE_TAGS,
)
direct_register_custom_op(
    op_name="pap_publish_prefill_kv",
    op_func=_pap_publish_prefill_kv_impl,
    mutates_args=["attention_output"],
    fake_impl=_pap_publish_prefill_kv_fake,
    tags=_CUDAGRAPH_UNSAFE_TAGS,
)
