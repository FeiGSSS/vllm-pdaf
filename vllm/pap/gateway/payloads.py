# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-payload construction for the PAP gateway."""

from __future__ import annotations

from typing import Any

_PREFILL_KV_TRANSPORT_KEYS = {
    "remote_block_ids",
    "remote_engine_id",
    "remote_request_id",
    "remote_host",
    "remote_port",
}


def requested_decode_capacity(req_data: dict[str, Any]) -> int | None:
    value = req_data.get("max_completion_tokens")
    if value is None:
        value = req_data.get("max_tokens")
    if value is None:
        return None
    try:
        capacity = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("decode token limit must be an integer") from exc
    if capacity <= 0:
        raise ValueError("decode token limit must be positive")
    return capacity


def build_prefill_payload(
    req_data: dict[str, Any],
) -> dict[str, Any]:
    decode_capacity = requested_decode_capacity(req_data)
    payload = req_data.copy()
    payload["stream"] = False
    payload["max_tokens"] = 1
    # Projection must consume the exact prompt produced by this render pass.
    # Returning the IDs avoids rendering and tokenizing the same long Chat
    # request again on the Projection API process.
    payload["return_token_ids"] = True
    kv_params: dict[str, Any] = {}
    payload["kv_transfer_params"] = kv_params
    if decode_capacity is not None:
        payload["kv_transfer_params"]["pap_decode_capacity_tokens"] = decode_capacity
    payload.pop("max_completion_tokens", None)
    payload.pop("min_tokens", None)
    payload.pop("stream_options", None)
    return payload


def attach_pap_prefill_attention_params(
    payload: dict[str, Any],
    *,
    pap_attention_endpoint: str,
    pap_prefill_kv_handle: str,
    pap_mode: str,
    pap_attention_tcp_endpoint: str | None = None,
) -> dict[str, Any]:
    """Attach PAP Prefill->Attention control-plane hints."""
    updated = payload.copy()
    kv_params = dict(updated.get("kv_transfer_params") or {})
    kv_params["pap_attention_endpoint"] = str(pap_attention_endpoint)
    if pap_attention_tcp_endpoint:
        kv_params["pap_attention_tcp_endpoint"] = str(pap_attention_tcp_endpoint)
    kv_params["pap_prefill_kv_handle"] = str(pap_prefill_kv_handle)
    kv_params["pap_import_prefill_kv_to_attention"] = True
    kv_params["pap_mode"] = str(pap_mode)
    updated["kv_transfer_params"] = kv_params
    return updated


def build_projection_kv_unaware_payload(
    req_data: dict[str, Any],
    kv_transfer_params: dict[str, Any],
    *,
    prompt_token_ids: list[int],
    prompt_text: str | None = None,
    pap_attention_endpoint: str | None = None,
    pap_attention_tcp_endpoint: str | None = None,
    pap_prefill_kv_handle: str | None = None,
    pap_attention_kv_installed: bool = False,
) -> dict[str, Any]:
    """Build a PAP Projection request that carries metadata, not Prefill KV."""
    if not prompt_token_ids:
        raise ValueError("PAP Projection requires Prefill prompt token IDs")
    payload = req_data.copy()
    kv_params: dict[str, Any] = {
        "pap_projection_kv_unaware": True,
        "pap_prompt_token_ids": prompt_token_ids,
    }
    if prompt_text is not None:
        kv_params["pap_prompt_text"] = prompt_text

    remote_num_tokens = kv_transfer_params.get("remote_num_tokens")
    if remote_num_tokens is not None:
        kv_params["pap_remote_prefix_len"] = int(remote_num_tokens)

    for key, value in kv_transfer_params.items():
        if key in _PREFILL_KV_TRANSPORT_KEYS:
            continue
        if key in ("remote_num_tokens", "tp_size"):
            continue
        if key.startswith("pap_"):
            kv_params[key] = value

    if pap_attention_endpoint:
        kv_params["pap_attention_endpoint"] = str(pap_attention_endpoint)
    if pap_attention_tcp_endpoint:
        kv_params["pap_attention_tcp_endpoint"] = str(pap_attention_tcp_endpoint)
    if pap_prefill_kv_handle:
        kv_params["pap_prefill_kv_handle"] = str(pap_prefill_kv_handle)
    if pap_attention_kv_installed:
        kv_params["pap_attention_kv_installed"] = True

    payload["kv_transfer_params"] = kv_params
    return payload
