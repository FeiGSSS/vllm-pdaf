# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for PAP Projection-side control helpers."""

from vllm.pap.attention.client import (
    bind_offload_exec_mailbox,
    select_attention_endpoint_for_request,
    update_offload_exec_mailbox_activity,
)
from vllm.pap.kv.handoff import (
    publish_prefill_kv_session_manifest,
    register_prefill_kv_catalog,
)

__all__ = [
    "bind_offload_exec_mailbox",
    "publish_prefill_kv_session_manifest",
    "register_prefill_kv_catalog",
    "select_attention_endpoint_for_request",
    "update_offload_exec_mailbox_activity",
]
