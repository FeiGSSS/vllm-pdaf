# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway request profiling helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _PrefillCacheUsage:
    prompt_tokens: int
    cached_tokens: int


def _pap_prefill_ipc_profile_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_IPC_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _extract_prefill_cache_usage(
    prefill_response: dict[str, Any],
) -> _PrefillCacheUsage | None:
    usage = prefill_response.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        return None
    if not isinstance(details, dict):
        return None
    cached_tokens = details.get("cached_tokens")
    if (
        not isinstance(cached_tokens, int)
        or cached_tokens < 0
        or cached_tokens > prompt_tokens
    ):
        return None
    return _PrefillCacheUsage(prompt_tokens, cached_tokens)


def _merge_prefill_cache_usage(
    response: dict[str, Any],
    prefill_usage: _PrefillCacheUsage | None,
) -> dict[str, Any]:
    if prefill_usage is None:
        return response
    response_usage = response.get("usage")
    if not isinstance(response_usage, dict):
        return response

    merged_response = response.copy()
    merged_usage = response_usage.copy()
    details = merged_usage.get("prompt_tokens_details")
    merged_details = details.copy() if isinstance(details, dict) else {}
    merged_details["cached_tokens"] = prefill_usage.cached_tokens
    merged_usage["prompt_tokens"] = prefill_usage.prompt_tokens
    merged_usage["prompt_tokens_details"] = merged_details
    completion_tokens = merged_usage.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens >= 0:
        merged_usage["total_tokens"] = prefill_usage.prompt_tokens + completion_tokens
    merged_response["usage"] = merged_usage
    return merged_response


def _merge_prefill_cache_usage_sse_event(
    event: bytes,
    prefill_usage: _PrefillCacheUsage | None,
) -> bytes:
    if prefill_usage is None or b'"usage"' not in event:
        return event
    prefix = b"data:"
    if not event.startswith(prefix):
        return event
    try:
        response = json.loads(event[len(prefix) :].strip())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return event
    if not isinstance(response, dict):
        return event
    merged = _merge_prefill_cache_usage(response, prefill_usage)
    if merged is response:
        return event
    payload = json.dumps(
        merged,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return b"data: " + payload


def _prefill_usage_headers(prefill_response: dict[str, Any]) -> dict[str, str]:
    usage = prefill_response.get("usage")
    if not isinstance(usage, dict):
        return {}

    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        return {}

    headers = {
        "X-PAP-Prefill-Prompt-Tokens": str(prompt_tokens),
    }
    cache_usage = _extract_prefill_cache_usage(prefill_response)
    if cache_usage is None:
        return headers

    headers["X-PAP-Prefill-Cached-Tokens"] = str(cache_usage.cached_tokens)
    headers["X-PAP-Prefill-Computed-Tokens"] = str(
        prompt_tokens - cache_usage.cached_tokens
    )
    return headers
