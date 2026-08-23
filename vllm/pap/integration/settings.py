# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime settings shared by PAP's vLLM integration adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import regex as re

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_LAYER_INDEX_RE = re.compile(r"(?:^|\.layers\.)(\d+)(?:\.|$)")


def pap_env_enabled(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether one PAP boolean environment setting is enabled."""
    values = os.environ if environ is None else environ
    return values.get(name, "").lower() in _TRUE_VALUES


def pap_offload_exec_trace_enabled(
    layer: str | int,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether OFFLOAD_EXEC timing is selected for one model layer."""
    values = os.environ if environ is None else environ
    if not pap_env_enabled("PAP_OFFLOAD_EXEC_TRACE", values):
        return False
    selected = values.get("PAP_OFFLOAD_EXEC_TRACE_LAYER", "").strip()
    if not selected:
        return True
    try:
        selected_index = int(selected)
    except ValueError as exc:
        raise ValueError(
            "PAP_OFFLOAD_EXEC_TRACE_LAYER must be a non-negative layer index"
        ) from exc
    if selected_index < 0:
        raise ValueError(
            "PAP_OFFLOAD_EXEC_TRACE_LAYER must be a non-negative layer index"
        )
    if isinstance(layer, int):
        layer_index = layer
    else:
        match = _LAYER_INDEX_RE.search(str(layer))
        if match is None:
            raise ValueError(f"cannot extract PAP trace layer index from {layer!r}")
        layer_index = int(match.group(1))
    return layer_index == selected_index


def pap_projection_critical_trace_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether Projection critical-path tracing is enabled."""
    values = os.environ if environ is None else environ
    return pap_env_enabled(
        "PAP_PROJECTION_KV_UNAWARE",
        values,
    ) and pap_env_enabled("PAP_PROJECTION_CRITICAL_TRACE", values)


def _integer(environ: Mapping[str, str], name: str, default: int = 0) -> int:
    try:
        return int(environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class PAPRuntimeSettings:
    """PAP environment values consumed from vLLM-owned execution paths."""

    projection_kv_unaware: bool
    critical_trace: bool
    debug_decision: bool
    unified_kv_decode_capacity_tokens: int
    cuda_context_role: str

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
            unified_kv_decode_capacity_tokens=max(
                0,
                _integer(values, "PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS"),
            ),
            cuda_context_role=(
                values.get("PAP_RUNTIME_CUDA_CONTEXT_ROLE") or "vllm_worker"
            ),
        )
