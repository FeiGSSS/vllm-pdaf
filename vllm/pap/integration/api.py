# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP ownership boundary for routes installed into the vLLM API server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.pap.integration.settings import PAPRuntimeSettings


def install_pap_control_routes(
    app: Any,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Install Prefill control routes when unified KV is enabled."""
    settings = PAPRuntimeSettings.from_environ(environ)
    if settings.unified_kv_decode_capacity_tokens <= 0:
        return False

    from vllm.pap.prefill_control_router import build_prefill_control_router

    app.include_router(build_prefill_control_router())
    return True
