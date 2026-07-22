# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Topology-owned PAP request routing plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_offload_exec_route_groups(
    request_ids: Sequence[str],
    *,
    attention_endpoint_by_request: Mapping[str, str],
    offload_exec_zmq_endpoint_by_request: Mapping[str, str],
    steps_by_request: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for req_index, req_id in enumerate(request_ids):
        req_id = str(req_id)
        attention_endpoint = attention_endpoint_by_request.get(req_id)
        offload_exec_zmq_endpoint = offload_exec_zmq_endpoint_by_request.get(req_id)
        step = steps_by_request.get(req_id)
        if not attention_endpoint or not offload_exec_zmq_endpoint or step is None:
            continue
        key = (attention_endpoint, offload_exec_zmq_endpoint)
        group = groups.setdefault(
            key,
            {
                "attention_endpoint": attention_endpoint,
                "offload_exec_zmq_endpoint": offload_exec_zmq_endpoint,
                "req_indices": [],
                "request_ids": [],
                "steps": [],
            },
        )
        group["req_indices"].append(req_index)
        group["request_ids"].append(req_id)
        group["steps"].append(int(step))
    return tuple(
        {
            "attention_endpoint": group["attention_endpoint"],
            "offload_exec_zmq_endpoint": group["offload_exec_zmq_endpoint"],
            "req_indices": tuple(group["req_indices"]),
            "request_ids": tuple(group["request_ids"]),
            "steps": tuple(group["steps"]),
            "batch_id_suffix": ",".join(
                f"{request_id}@{step}"
                for request_id, step in zip(group["request_ids"], group["steps"])
            ),
        }
        for group in groups.values()
    )
