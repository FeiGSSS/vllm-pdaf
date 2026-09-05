# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime settings shared by PAP's vLLM integration adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from vllm.pap.config import read_env_bool, read_env_int


def pap_env_enabled(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether one PAP boolean environment setting is enabled."""
    values = os.environ if environ is None else environ
    return read_env_bool(values, name)


@dataclass(frozen=True, slots=True)
class PAPRuntimeSettings:
    """PAP environment values consumed from vLLM-owned execution paths."""

    projection_kv_unaware: bool
    critical_trace: bool
    debug_decision: bool
    unified_kv_decode_capacity_tokens: int

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> PAPRuntimeSettings:
        """Parse PAP integration settings once at an owning boundary."""
        values = os.environ if environ is None else environ
        projection_kv_unaware = pap_env_enabled(
            "PAP_PROJECTION_KV_UNAWARE",
            values,
        )
        return cls(
            projection_kv_unaware=projection_kv_unaware,
            critical_trace=(
                projection_kv_unaware
                and pap_env_enabled("PAP_PROJECTION_CRITICAL_TRACE", values)
            ),
            debug_decision=pap_env_enabled("PAP_DEBUG_DECISION", values),
            unified_kv_decode_capacity_tokens=read_env_int(
                values, "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS", minimum=0
            ),
        )
