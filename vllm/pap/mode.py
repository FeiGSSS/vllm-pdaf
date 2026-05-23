# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations


def is_pap_enabled(additional_kwargs: dict | None) -> bool:
    """Return True when the PAP attention path should be used for this forward."""
    if not additional_kwargs:
        return False
    return bool(additional_kwargs.get("pap_enabled"))
