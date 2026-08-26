# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cancellation-safe lifecycle ownership for one PAP request."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import httpx

from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.clients import PAPServiceClient, request_headers
from vllm.pap.gateway.topology import PAPGroup, ProjectionInstance

logger = logging.getLogger("pap_gateway")

_ATTENTION_RELEASE_MAX_ATTEMPTS = 5
_ATTENTION_RELEASE_RETRY_INITIAL_S = 0.02
_ATTENTION_RELEASE_RETRY_MAX_S = 0.2
_RECENT_FAILURE_LIMIT = 100


class PAPLifecycleCleanupError(RuntimeError):
    """One or more remote PAP resources could not be released."""


async def release_attention_sessions(
    attention_clients: list[PAPServiceClient],
    request_id: str,
) -> None:
    """Idempotently release every Attention session with bounded retries."""

    async def release_one(attention: PAPServiceClient) -> None:
        delay_s = _ATTENTION_RELEASE_RETRY_INITIAL_S
        last_error: Exception | None = None
        for attempt in range(1, _ATTENTION_RELEASE_MAX_ATTEMPTS + 1):
            try:
                response = await attention.client.delete(
                    f"/v1/pap/attention/sessions/{request_id}",
                    headers=request_headers(request_id),
                )
                response.raise_for_status()
                return
            except Exception as exc:
                last_error = exc
                if attempt == _ATTENTION_RELEASE_MAX_ATTEMPTS:
                    break
                await asyncio.sleep(delay_s)
                delay_s = min(
                    _ATTENTION_RELEASE_RETRY_MAX_S,
                    delay_s * 2,
                )
        raise PAPLifecycleCleanupError(
            "failed to release PAP Attention session "
            f"request_id={request_id} endpoint={attention.base_url} "
            f"attempts={_ATTENTION_RELEASE_MAX_ATTEMPTS} error={last_error}"
        )

    results = await asyncio.gather(
        *(release_one(attention) for attention in attention_clients),
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise PAPLifecycleCleanupError("; ".join(str(error) for error in errors))


async def quiesce_engine_request(
    client: PAPServiceClient,
    request_id: str,
    *,
    role: str,
) -> None:
    """Abort a vLLM request and wait for its EngineCore abort barrier."""
    delay_s = _ATTENTION_RELEASE_RETRY_INITIAL_S
    last_error: Exception | None = None
    for attempt in range(1, _ATTENTION_RELEASE_MAX_ATTEMPTS + 1):
        try:
            response = await client.client.post(
                f"/v1/pap/{role}/quiesce",
                json={
                    "request_ids": [
                        request_id,
                        f"chatcmpl-{request_id}",
                        f"cmpl-{request_id}",
                    ]
                },
                headers=request_headers(request_id),
            )
            response.raise_for_status()
            result = response.json()
            if result.get("quiesced", False):
                return
            raise PAPLifecycleCleanupError(
                f"{role} did not acknowledge request quiescence "
                f"request_id={request_id} result={result}"
            )
        except Exception as exc:
            last_error = exc
            if attempt == _ATTENTION_RELEASE_MAX_ATTEMPTS:
                break
            await asyncio.sleep(delay_s)
            delay_s = min(_ATTENTION_RELEASE_RETRY_MAX_S, delay_s * 2)
    raise PAPLifecycleCleanupError(
        f"failed to quiesce PAP {role} request "
        f"request_id={request_id} attempts={_ATTENTION_RELEASE_MAX_ATTEMPTS} "
        f"error={last_error}"
    )


async def quiesce_projection_request(
    projection_client: PAPServiceClient,
    request_id: str,
) -> None:
    await quiesce_engine_request(
        projection_client,
        request_id,
        role="projection",
    )


async def quiesce_prefill_request(
    prefill_client: PAPServiceClient,
    request_id: str,
) -> None:
    await quiesce_engine_request(
        prefill_client,
        request_id,
        role="prefill",
    )


class PAPRequestLifecycle:
    """Own all resources acquired for one PAP request."""

    def __init__(
        self,
        *,
        manager: PAPLifecycleManager,
        request_id: str,
        attention_clients: list[PAPServiceClient],
        prefill_client: PAPServiceClient | None,
        projection_client: PAPServiceClient | None,
        admission: PAPProjectionAdmission,
        group: PAPGroup,
        projection: ProjectionInstance,
        on_finished: Callable[[], None],
    ) -> None:
        self.manager = manager
        self.request_id = request_id
        self.attention_clients = attention_clients
        self.prefill_client = prefill_client
        self.projection_client = projection_client
        self.admission = admission
        self.group = group
        self.projection = projection
        self.on_finished = on_finished
        self.attention_registered = False
        self.prefill_started = False
        self.prefill_completed = False
        self.projection_admitted = False
        self.projection_response: httpx.Response | None = None
        self.projection_started = False
        self.projection_completed = False
        self.state = "active"
        self.termination_reason: str | None = None
        self._termination_task: asyncio.Task[None] | None = None

    def mark_attention_registered(self) -> None:
        self.attention_registered = True

    def mark_prefill_started(self) -> None:
        self.prefill_started = True

    def mark_prefill_completed(self) -> None:
        self.prefill_completed = True

    def mark_projection_admitted(self) -> None:
        self.projection_admitted = True

    def mark_projection_started(self) -> None:
        self.projection_started = True

    def mark_projection_completed(self) -> None:
        self.projection_completed = True

    def attach_projection_response(self, response: httpx.Response) -> None:
        self.projection_response = response

    async def terminate(self, reason: str) -> None:
        """Start termination once and shield it from caller cancellation."""
        if self._termination_task is None:
            self.termination_reason = reason
            self.state = "terminating"
            self._termination_task = asyncio.create_task(
                self._terminate(),
                name=f"pap-lifecycle-{self.request_id}",
            )
        await asyncio.shield(self._termination_task)

    async def _terminate(self) -> None:
        errors: list[BaseException] = []
        projection_quiesced = not self.projection_started or self.projection_completed
        prefill_quiesced = not self.prefill_started or self.prefill_completed
        if self.projection_response is not None:
            try:
                await self.projection_response.aclose()
            except Exception as exc:
                errors.append(exc)
            self.projection_response = None

        if self.projection_started and not self.projection_completed:
            if self.projection_client is None:
                errors.append(RuntimeError("Projection client is missing"))
            else:
                try:
                    await quiesce_projection_request(
                        self.projection_client,
                        self.request_id,
                    )
                    projection_quiesced = True
                except Exception as exc:
                    errors.append(exc)

        if self.prefill_started and not self.prefill_completed:
            if self.prefill_client is None:
                errors.append(RuntimeError("Prefill client is missing"))
            else:
                try:
                    await quiesce_prefill_request(
                        self.prefill_client,
                        self.request_id,
                    )
                    prefill_quiesced = True
                except Exception as exc:
                    errors.append(exc)

        if self.attention_registered and projection_quiesced and prefill_quiesced:
            try:
                await release_attention_sessions(
                    self.attention_clients,
                    self.request_id,
                )
            except Exception as exc:
                errors.append(exc)
            self.attention_registered = False

        if self.projection_admitted and projection_quiesced:
            try:
                await self.admission.release(self.group, self.projection)
            except Exception as exc:
                errors.append(exc)
            self.projection_admitted = False

        error: BaseException | None = None
        if errors:
            error = PAPLifecycleCleanupError(
                "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
            )
            self.state = "failed"
        else:
            self.state = "terminated"
        try:
            self.on_finished()
        finally:
            self.manager._finish(self, error)
        if error is not None:
            raise error


class PAPLifecycleManager:
    """Track live coordinators and expose cleanup failures to health checks."""

    def __init__(self) -> None:
        self._active: dict[str, PAPRequestLifecycle] = {}
        self._recent_failures: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._completed = 0

    def create(
        self,
        *,
        request_id: str,
        attention_clients: list[PAPServiceClient],
        prefill_client: PAPServiceClient | None,
        projection_client: PAPServiceClient | None,
        admission: PAPProjectionAdmission,
        group: PAPGroup,
        projection: ProjectionInstance,
        on_finished: Callable[[], None],
    ) -> PAPRequestLifecycle:
        if request_id in self._active:
            raise ValueError(f"duplicate active PAP request: {request_id}")
        lifecycle = PAPRequestLifecycle(
            manager=self,
            request_id=request_id,
            attention_clients=attention_clients,
            prefill_client=prefill_client,
            projection_client=projection_client,
            admission=admission,
            group=group,
            projection=projection,
            on_finished=on_finished,
        )
        self._active[request_id] = lifecycle
        return lifecycle

    def _finish(
        self,
        lifecycle: PAPRequestLifecycle,
        error: BaseException | None,
    ) -> None:
        self._active.pop(lifecycle.request_id, None)
        self._completed += 1
        if error is None:
            return
        self._recent_failures[lifecycle.request_id] = {
            "reason": lifecycle.termination_reason or "unknown",
            "error": str(error),
        }
        self._recent_failures.move_to_end(lifecycle.request_id)
        while len(self._recent_failures) > _RECENT_FAILURE_LIMIT:
            self._recent_failures.popitem(last=False)
        logger.error(
            "PAP lifecycle cleanup failed request_id=%s reason=%s error=%s",
            lifecycle.request_id,
            lifecycle.termination_reason,
            error,
        )

    async def shutdown(self) -> None:
        lifecycles = tuple(self._active.values())
        if not lifecycles:
            return
        await asyncio.gather(
            *(lifecycle.terminate("gateway_shutdown") for lifecycle in lifecycles),
            return_exceptions=True,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "active": len(self._active),
            "completed": self._completed,
            "failed": len(self._recent_failures),
            "active_requests": {
                request_id: lifecycle.state
                for request_id, lifecycle in self._active.items()
            },
            "recent_failures": dict(self._recent_failures),
        }


__all__ = [
    "PAPLifecycleCleanupError",
    "PAPLifecycleManager",
    "PAPRequestLifecycle",
    "quiesce_engine_request",
    "quiesce_prefill_request",
    "quiesce_projection_request",
    "release_attention_sessions",
]
