# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional execution overrides for opaque Attention operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

AttentionExecution = Callable[..., None]
AttentionExecutionFactory = Callable[[Any, Any], AttentionExecution | None]

_FACTORIES: dict[str, AttentionExecutionFactory] = {}


def register_attention_execution_factory(
    name: str,
    factory: AttentionExecutionFactory,
) -> None:
    """Register one process-wide Attention execution factory."""
    existing = _FACTORIES.get(name)
    if existing is not None and existing is not factory:
        raise RuntimeError(f"Attention execution factory already registered: {name}")
    _FACTORIES[name] = factory


def resolve_attention_execution(
    attention: Any,
    vllm_config: Any,
) -> AttentionExecution | None:
    """Resolve at most one external execution implementation for a layer."""
    selected = [
        (name, execution)
        for name, factory in _FACTORIES.items()
        if (execution := factory(attention, vllm_config)) is not None
    ]
    if len(selected) > 1:
        names = ", ".join(name for name, _execution in selected)
        raise RuntimeError(f"Multiple Attention execution overrides selected: {names}")
    return selected[0][1] if selected else None


__all__ = [
    "AttentionExecution",
    "AttentionExecutionFactory",
    "register_attention_execution_factory",
    "resolve_attention_execution",
]
