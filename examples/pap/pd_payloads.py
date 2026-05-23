# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared request payload helpers for PAP and native PD proxy experiments."""

from __future__ import annotations

from typing import Any


def build_prefill_payload(req_data: dict[str, Any]) -> dict[str, Any]:
    payload = req_data.copy()
    payload["stream"] = False
    payload["max_tokens"] = 1
    payload["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    payload.pop("max_completion_tokens", None)
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
    kv_params["pap_mode"] = str(pap_mode)
    updated["kv_transfer_params"] = kv_params
    return updated


def build_decode_payload(
    req_data: dict[str, Any],
    kv_transfer_params: dict[str, Any],
    *,
    pap_prefill_kv_handle: str | None = None,
    pap_attention_kv_installed: bool = False,
) -> dict[str, Any]:
    payload = req_data.copy()
    payload["kv_transfer_params"] = dict(kv_transfer_params)
    if pap_prefill_kv_handle:
        payload["kv_transfer_params"]["pap_prefill_kv_handle"] = str(
            pap_prefill_kv_handle
        )
    if pap_attention_kv_installed:
        payload["kv_transfer_params"]["pap_attention_kv_installed"] = True
    return payload


def enrich_prefill_kv_params(
    kv_transfer_params: dict[str, Any],
    *,
    prefill_host: str,
    prefill_nixl_port: int | None,
) -> dict[str, Any]:
    kv_params = dict(kv_transfer_params)
    kv_params.setdefault("remote_host", prefill_host)
    if prefill_nixl_port is not None:
        kv_params.setdefault("remote_port", prefill_nixl_port)
    return kv_params
