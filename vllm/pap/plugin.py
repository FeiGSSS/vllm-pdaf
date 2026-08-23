# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP registration through vLLM's general plugin interface."""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.attention.execution import (
    register_attention_execution_factory,
)
from vllm.pap.model.attention_execution import (
    create_projection_attention_execution,
)
from vllm.pap.model.hooks import pap_model_hooks_enabled
from vllm.pap.model.step_graph import pap_projection_step_graph_enabled


def _install_engine_control() -> None:
    from vllm.v1.engine.core import EngineCore

    if hasattr(EngineCore, "pap_control"):
        return

    def pap_control(
        engine_core: Any, operation: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        from vllm.pap.integration.control import PAPEngineControl

        control = getattr(engine_core, "_pap_engine_control", None)
        if control is None:
            control = PAPEngineControl(engine_core.scheduler)
            engine_core._pap_engine_control = control
        return control.apply(operation, payload)

    type.__setattr__(EngineCore, "pap_control", pap_control)


def _projection_execution_factory(attention: Any, vllm_config: Any):
    if not pap_projection_step_graph_enabled():
        return None
    layer_count = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    return create_projection_attention_execution(attention, layer_count)


def register_pap_plugin() -> None:
    """Register PAP extensions only in explicitly enabled PAP processes."""
    if not pap_model_hooks_enabled():
        return
    _install_engine_control()
    register_attention_execution_factory("pap", _projection_execution_factory)


__all__ = ["register_pap_plugin"]
