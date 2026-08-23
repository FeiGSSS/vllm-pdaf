# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-unsafe boundaries for PAP model-side effects."""

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


def pap_cudagraph_role(values: Mapping[str, str] | None = None) -> str | None:
    """Return the statically configured PAP role for piecewise Graphs."""
    source = os.environ if values is None else values
    if source.get("PAP_CUDAGRAPH_COMPATIBLE", "").lower() not in _TRUE_ENV_VALUES:
        return None
    role = source.get("PAP_CUDAGRAPH_ROLE", "").lower()
    if role not in _PAP_CUDAGRAPH_ROLES:
        choices = ", ".join(sorted(_PAP_CUDAGRAPH_ROLES))
        raise RuntimeError(f"PAP_CUDAGRAPH_ROLE must be one of {choices}")
    return role


def bind_pap_cudagraph_adapters(
    attention: Any,
    *,
    projection_adapter: Any,
    prefill_publisher: Any,
) -> None:
    """Expose graph-excluded PAP adapters through Attention context."""
    attention._pap_cudagraph_projection_adapter = projection_adapter
    attention._pap_cudagraph_prefill_publisher = prefill_publisher


def _pap_attention_layer(layer_name: str) -> Any:
    return get_forward_context().no_compile_layers[layer_name]


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
    attention._pap_cudagraph_prefill_publisher.publish(attention)


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
