# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection-side decode-step route validation and grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.pap.mode import is_pap_request_id
from vllm.pap.model.context import pap_endpoint_for_tp_rank

_PAP_STEP_GROUPS_KEY = "_pap_qwen3_offload_exec_step_groups"


def _pap_offload_exec_session_request_id(
    request_id: str,
    prefill_kv_handle: Any,
) -> str:
    return str(prefill_kv_handle or request_id)


@dataclass(frozen=True)
class _PAPOffloadExecStepGroup:
    attention_endpoint: str
    offload_exec_zmq_endpoint: str
    req_indices: tuple[int, ...]
    batch_id_suffix: str
    metadata_template: dict[str, Any]


def _pap_offload_exec_step_groups(
    additional_kwargs: dict[str, Any],
    *,
    num_reqs: int,
    scaling: float,
) -> tuple[_PAPOffloadExecStepGroup, ...]:
    cached = additional_kwargs.get(_PAP_STEP_GROUPS_KEY)
    if cached is not None:
        return tuple(cached)

    request_ids = tuple(additional_kwargs.get("pap_request_ids") or ())
    route_groups = tuple(additional_kwargs.get("pap_offload_exec_route_groups") or ())
    if not route_groups:
        raise RuntimeError("PAP attention missing OFFLOAD_EXEC route groups")

    attention_kv_installed = set(
        additional_kwargs.get("pap_attention_kv_installed_by_request") or ()
    )
    prefix_len_by_request = (
        additional_kwargs.get("pap_prefill_prefix_len_by_request") or {}
    )
    prefill_kv_handle_by_request = (
        additional_kwargs.get("pap_prefill_kv_handle_by_request") or {}
    )
    step_groups: list[_PAPOffloadExecStepGroup] = []
    routed_req_indices: set[int] = set()
    for route_group in route_groups:
        attention_endpoint = pap_endpoint_for_tp_rank(
            route_group.get("attention_endpoint")
        )
        offload_exec_zmq_endpoint = pap_endpoint_for_tp_rank(
            route_group.get("offload_exec_zmq_endpoint")
        )
        if not attention_endpoint:
            raise RuntimeError(
                "PAP NIXL mailbox OFFLOAD_EXEC requires pap_attention_endpoint"
            )
        if not offload_exec_zmq_endpoint:
            raise RuntimeError(
                "PAP OFFLOAD_EXEC mailbox path missing pap_offload_exec_zmq_endpoint"
            )

        req_indices = tuple(
            int(req_index) for req_index in route_group.get("req_indices", ())
        )
        group_request_ids = tuple(
            str(request_id) for request_id in route_group.get("request_ids", ())
        )
        group_steps = tuple(int(step) for step in route_group.get("steps", ()))
        if not (len(req_indices) == len(group_request_ids) == len(group_steps)):
            raise RuntimeError("PAP OFFLOAD_EXEC route group is malformed")

        prepared_session_request_ids = tuple(
            str(request_id) for request_id in route_group.get("session_request_ids", ())
        )
        if prepared_session_request_ids and len(prepared_session_request_ids) != len(
            group_request_ids
        ):
            raise RuntimeError("PAP OFFLOAD_EXEC session route is malformed")
        session_request_ids: list[str] = []
        for group_offset, req_index in enumerate(req_indices):
            if req_index < 0 or req_index >= num_reqs:
                raise RuntimeError("PAP OFFLOAD_EXEC route index out of range")
            request_id = group_request_ids[group_offset]
            if request_id != str(request_ids[req_index]):
                raise RuntimeError("PAP OFFLOAD_EXEC route request mismatch")
            if not is_pap_request_id(request_id):
                raise RuntimeError(
                    f"PAP attention cannot route non-OpenAI request id {request_id}"
                )
            routed_req_indices.add(req_index)
            prefix_len = int(prefix_len_by_request.get(request_id) or 0)
            prefill_kv_handle = prefill_kv_handle_by_request.get(request_id)
            if prefix_len > 0 and request_id not in attention_kv_installed:
                if not prefill_kv_handle:
                    raise RuntimeError("PAP missing local prefill KV handle")
                raise RuntimeError("PAP attention KV is not installed")
            session_request_id = _pap_offload_exec_session_request_id(
                request_id,
                prefill_kv_handle,
            )
            if (
                prepared_session_request_ids
                and prepared_session_request_ids[group_offset] != session_request_id
            ):
                raise RuntimeError("PAP OFFLOAD_EXEC session route is stale")
            session_request_ids.append(session_request_id)

        computed_batch_id_suffix = ",".join(
            f"{request_id}@{step}"
            for request_id, step in zip(session_request_ids, group_steps)
        )
        batch_id_suffix = str(
            route_group.get("batch_id_suffix") or computed_batch_id_suffix
        )
        if prepared_session_request_ids and (
            batch_id_suffix != computed_batch_id_suffix
        ):
            raise RuntimeError("PAP OFFLOAD_EXEC batch route is stale")
        prepared_metadata_template = route_group.get("metadata_template")
        if prepared_metadata_template is not None and (
            tuple(prepared_metadata_template.get("r", ())) != tuple(session_request_ids)
            or tuple(prepared_metadata_template.get("s", ())) != group_steps
        ):
            raise RuntimeError("PAP OFFLOAD_EXEC metadata route is stale")
        step_groups.append(
            _PAPOffloadExecStepGroup(
                attention_endpoint=str(attention_endpoint),
                offload_exec_zmq_endpoint=str(offload_exec_zmq_endpoint),
                req_indices=req_indices,
                batch_id_suffix=batch_id_suffix,
                metadata_template={
                    "r": tuple(session_request_ids),
                    "s": group_steps,
                    "a": (float(scaling),) * len(group_steps),
                },
            )
        )

    if len(routed_req_indices) != num_reqs:
        raise RuntimeError("PAP OFFLOAD_EXEC route groups do not cover batch")

    result = tuple(step_groups)
    additional_kwargs[_PAP_STEP_GROUPS_KEY] = result
    return result
