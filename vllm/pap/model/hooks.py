# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP model-hook activation."""

from __future__ import annotations

import os
from collections.abc import Mapping

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def pap_model_hooks_enabled(values: Mapping[str, str] | None = None) -> bool:
    """Return whether this process owns a PAP model-side role."""
    source = os.environ if values is None else values
    explicit = source.get("PAP_MODEL_HOOKS")
    if explicit is not None:
        return explicit.lower() in _TRUE_ENV_VALUES
    projection = source.get("PAP_PROJECTION_KV_UNAWARE", "").lower()
    return projection in _TRUE_ENV_VALUES or bool(source.get("PAP_TOPOLOGY"))


__all__ = ["pap_model_hooks_enabled"]
