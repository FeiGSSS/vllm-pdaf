# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection model-runner adapters for PAP request batches."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vllm.logger import init_logger
from vllm.pap.integration.request import PAPProjectionRequestStore


def build_offload_exec_route_groups(
    request_ids: Sequence[str],
    *,
    attention_endpoint_by_request: Mapping[str, str],
    steps_by_request: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    groups: dict[str, dict[str, Any]] = {}
    for req_index, req_id in enumerate(request_ids):
        req_id = str(req_id)
        attention_endpoint = attention_endpoint_by_request.get(req_id)
        step = steps_by_request.get(req_id)
        if not attention_endpoint or step is None:
            continue
        key = attention_endpoint
        group = groups.setdefault(
            key,
            {
                "attention_endpoint": attention_endpoint,
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


logger = init_logger(__name__)


def select_projection_request_ids(
    store: PAPProjectionRequestStore,
    request_ids: Sequence[str],
    *,
    globally_enabled: bool,
) -> frozenset[str]:
    """Return requests that should use PAP in one Projection batch."""
    normalized_ids = tuple(str(request_id) for request_id in request_ids)
    if globally_enabled:
        return frozenset(normalized_ids)

    selected_ids: list[str] = []
    for request_id in normalized_ids:
        if request_id in store.attention_endpoint_by_request:
            logger.debug(
                "PAP enabled via per-request mailbox endpoint req_id=%s",
                request_id,
            )
            selected_ids.append(request_id)
    return frozenset(selected_ids)


def _filter_mapping(
    mapping: Mapping[str, Any],
    request_ids: Sequence[str],
) -> dict[str, Any]:
    request_id_set = set(request_ids)
    return {key: value for key, value in mapping.items() if key in request_id_set}


def _prepare_offload_exec_route_groups(
    store: PAPProjectionRequestStore,
    route_groups: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Attach session-owned wire metadata during runner input preparation."""
    prepared_groups: list[dict[str, Any]] = []
    for route_group in route_groups:
        request_ids = tuple(
            str(request_id) for request_id in route_group["request_ids"]
        )
        steps = tuple(int(step) for step in route_group["steps"])
        session_request_ids = tuple(
            str(store.prefill_kv_handle_by_request.get(request_id) or request_id)
            for request_id in request_ids
        )
        prepared_groups.append(
            {
                **route_group,
                "session_request_ids": session_request_ids,
                "batch_id_suffix": ",".join(
                    f"{request_id}@{step}"
                    for request_id, step in zip(session_request_ids, steps)
                ),
                "metadata_template": {
                    "r": session_request_ids,
                    "s": steps,
                },
            }
        )
    return tuple(prepared_groups)


def build_projection_forward_context(
    store: PAPProjectionRequestStore,
    *,
    request_ids: Sequence[str],
    num_scheduled_tokens: Iterable[int],
    num_actual_tokens: int,
    positions: Any,
    seq_lens_cpu_upper_bound: Iterable[int],
    pap_enabled: bool,
    attention_tcp_endpoint: Any,
    block_size: int,
    finished_request_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the PAP portion of vLLM's model forward context."""
    normalized_ids = tuple(str(request_id) for request_id in request_ids)
    steps_by_request = {
        request_id: int(step)
        for request_id, step in zip(
            normalized_ids,
            seq_lens_cpu_upper_bound,
        )
    }

    route_groups = build_offload_exec_route_groups(
        normalized_ids,
        attention_endpoint_by_request=store.attention_endpoint_by_request,
        steps_by_request=steps_by_request,
    )
    route_groups = _prepare_offload_exec_route_groups(store, route_groups)
    return {
        "pap_request_ids": normalized_ids,
        "pap_num_scheduled_tokens": tuple(
            int(num_tokens) for num_tokens in num_scheduled_tokens
        ),
        "pap_num_reqs": len(normalized_ids),
        "pap_num_actual_tokens": num_actual_tokens,
        "pap_positions": positions,
        "pap_enabled": pap_enabled,
        "pap_attention_tcp_endpoint": attention_tcp_endpoint,
        "pap_block_size": block_size,
        "pap_attention_tcp_endpoint_by_request": _filter_mapping(
            store.attention_tcp_endpoint_by_request, normalized_ids
        ),
        "pap_attention_endpoint_by_request": _filter_mapping(
            store.attention_endpoint_by_request, normalized_ids
        ),
        "pap_offload_exec_route_groups": route_groups,
        "pap_prefill_prefix_len_by_request": _filter_mapping(
            store.prefill_prefix_len_by_request, normalized_ids
        ),
        "pap_decode_capacity_tokens_by_request": _filter_mapping(
            store.decode_capacity_tokens_by_request, normalized_ids
        ),
        "pap_prefill_kv_handle_by_request": _filter_mapping(
            store.prefill_kv_handle_by_request, normalized_ids
        ),
        "pap_attention_kv_installed_by_request": {
            request_id
            for request_id in normalized_ids
            if request_id in store.attention_kv_installed_requests
        },
        "pap_import_prefill_kv_to_attention_by_request": {
            request_id
            for request_id in normalized_ids
            if request_id in store.import_prefill_kv_to_attention_requests
        },
        "pap_finished_request_ids": tuple(
            str(request_id) for request_id in finished_request_ids
        ),
    }
