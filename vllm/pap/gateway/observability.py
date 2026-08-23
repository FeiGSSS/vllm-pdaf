# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP gateway request profiling helpers."""

from __future__ import annotations

import os
from typing import Any


def _pap_prefill_ipc_profile_enabled() -> bool:
    return os.environ.get("PAP_PREFILL_IPC_PROFILE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return headers

    cached_tokens = details.get("cached_tokens")
    if not isinstance(cached_tokens, int) or cached_tokens < 0:
        return headers

    headers["X-PAP-Prefill-Cached-Tokens"] = str(cached_tokens)
    if cached_tokens <= prompt_tokens:
        headers["X-PAP-Prefill-Computed-Tokens"] = str(prompt_tokens - cached_tokens)
    return headers
