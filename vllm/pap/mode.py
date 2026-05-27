# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable

PAP_REQUEST_ID_PREFIXES = ("cmpl-", "chatcmpl-")


def is_pap_enabled(additional_kwargs: dict | None) -> bool:
    """Return True when the PAP attention path should be used for this forward."""
    if not additional_kwargs:
        return False
    return bool(additional_kwargs.get("pap_enabled"))


def is_pap_request_id(request_id: object) -> bool:
    return str(request_id).startswith(PAP_REQUEST_ID_PREFIXES)


def pap_request_ids_are_routable(
    request_ids: Iterable[object] | None, num_reqs: int
) -> bool:
    if request_ids is None or num_reqs <= 0:
        return False
    selected_request_ids = tuple(request_ids)[:num_reqs]
    return len(selected_request_ids) == num_reqs and all(
        is_pap_request_id(request_id) for request_id in selected_request_ids
    )
