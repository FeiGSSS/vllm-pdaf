# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from enum import StrEnum


class PAPMode(StrEnum):
    DEBUG_REMOTE_ATTENTION = "debug_remote_attention"
    TRUE_SPLIT = "true_split"
    TRUE_SPLIT_PERFORMANCE = "true_split_performance"


def parse_pap_mode(value: str | None) -> PAPMode:
    if value is None or value == "":
        return PAPMode.DEBUG_REMOTE_ATTENTION
    try:
        return PAPMode(value)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in PAPMode)
        msg = f"unsupported PAP mode {value!r}; supported: {supported}"
        raise ValueError(msg) from exc


def is_debug_remote_attention(value: str | None) -> bool:
    return parse_pap_mode(value) is PAPMode.DEBUG_REMOTE_ATTENTION
