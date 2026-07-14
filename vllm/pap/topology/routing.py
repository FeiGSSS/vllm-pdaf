# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Topology-owned PAP request routing plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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

def filter_offload_exec_route_groups_for_request_slice(
    route_groups: Iterable[Mapping[str, Any]],
    request_slice: slice,
) -> tuple[dict[str, Any], ...]:
    start = int(request_slice.start or 0)
    stop = int(request_slice.stop or start)
    filtered_groups: list[dict[str, Any]] = []
    for group in route_groups:
        req_indices = tuple(int(index) for index in group.get("req_indices", ()))
        request_ids = tuple(str(req_id) for req_id in group.get("request_ids", ()))
        steps = tuple(int(step) for step in group.get("steps", ()))
        local_indices: list[int] = []
        local_request_ids: list[str] = []
        local_steps: list[int] = []
        for offset, req_index in enumerate(req_indices):
            if start <= req_index < stop:
                local_indices.append(req_index - start)
                if offset < len(request_ids):
                    local_request_ids.append(request_ids[offset])
                if offset < len(steps):
                    local_steps.append(steps[offset])
        if not local_indices:
            continue
        filtered_groups.append(
            {
                "attention_endpoint": group.get("attention_endpoint"),
                "offload_exec_zmq_endpoint": group.get("offload_exec_zmq_endpoint"),
                "req_indices": tuple(local_indices),
                "request_ids": tuple(local_request_ids),
                "steps": tuple(local_steps),
                "batch_id_suffix": ",".join(
                    f"{request_id}@{step}"
                    for request_id, step in zip(local_request_ids, local_steps)
                ),
            }
        )
    return tuple(filtered_groups)
