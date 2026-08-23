# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP routes registered through vLLM's endpoint plugin interface."""

from __future__ import annotations

from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from starlette.datastructures import State

from vllm.pap.integration.settings import PAPRuntimeSettings
from vllm.pap.prefill_control_router import (
    PAPControlDispatcher,
    build_prefill_control_router,
)


class PAPEndpointPlugin:
    """Expose Prefill control only in a PAP unified-KV process."""

    name = "pap"
    required_tasks = ("generate",)

    def __init__(self) -> None:
        settings = PAPRuntimeSettings.from_environ()
        self._enabled = settings.unified_kv_decode_capacity_tokens > 0

    def attach_router(self, app: FastAPI) -> None:
        if not self._enabled:
            return
        app.include_router(build_prefill_control_router())
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def pap_lifespan(lifespan_app: FastAPI):
            async with parent_lifespan(lifespan_app):
                try:
                    yield
                finally:
                    dispatcher = getattr(
                        lifespan_app.state,
                        "pap_control_dispatcher",
                        None,
                    )
                    if dispatcher is not None:
                        await dispatcher.close()

        app.router.lifespan_context = pap_lifespan

    async def init_state(
        self,
        engine_client: Any | None,
        state: State,
        args: Namespace,
    ) -> None:
        del args
        if not self._enabled:
            return
        if engine_client is None:
            raise RuntimeError("PAP control endpoints require an EngineClient")
        dispatcher = PAPControlDispatcher(engine_client)
        dispatcher.start()
        state.pap_control_dispatcher = dispatcher


__all__ = ["PAPEndpointPlugin"]
