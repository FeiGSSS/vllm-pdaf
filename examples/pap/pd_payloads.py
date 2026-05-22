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


def build_decode_payload(
    req_data: dict[str, Any], kv_transfer_params: dict[str, Any]
) -> dict[str, Any]:
    payload = req_data.copy()
    payload["kv_transfer_params"] = dict(kv_transfer_params)
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
