# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP surface and ordered dispatch for Prefill-owned PAP control."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DecodeCommitRequest(BaseModel):
    request_id: str
    session_request_id: str | None = None
    commit_seq: int = Field(gt=0)
    new_seq_len: int = Field(gt=0)
    new_token_ids: list[int]
    layer_complete: bool = True
    submit_only: bool = False


class LeaseReleaseRequest(BaseModel):
    request_id: str
    lease_id: str
    final_commit_seq: int | None = Field(default=None, ge=0)
    submit_only: bool = False


class RequestQuiesceRequest(BaseModel):
    request_ids: list[str] = Field(min_length=1)


class DecodeAllocationRequest(BaseModel):
    session_handle: str
    lease_id: str
    generation: int = Field(ge=0)
    required_tokens: int = Field(gt=0)
    reserve_tokens: int = Field(default=256, ge=0, le=4096)


@dataclass(slots=True)
class _ControlItem:
    operation: str
    payload: dict[str, Any]
    result: asyncio.Future[dict[str, Any]]


class PAPControlDispatcher:
    """Serialize control calls before they enter EngineCore."""

    def __init__(self, engine_client: Any, queue_size: int = 4096) -> None:
        self._engine_client = engine_client
        self._queue: asyncio.Queue[_ControlItem | None] = asyncio.Queue(queue_size)
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(
                self._run(), name="pap-engine-control-dispatcher"
            )

    async def submit(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        wait: bool,
    ) -> dict[str, Any]:
        self.start()
        result = asyncio.get_running_loop().create_future()
        await self._queue.put(_ControlItem(operation, payload, result))
        if wait:
            return await result

        result.add_done_callback(self._report_deferred_result)
        return {"accepted": True}

    async def close(self) -> None:
        worker = self._worker
        if worker is None:
            return
        await self._queue.put(None)
        await worker
        self._worker = None

    async def abort_and_quiesce(self, request_ids: list[str]) -> dict[str, Any]:
        await self._engine_client.abort(request_ids)
        utility_client = getattr(
            self._engine_client,
            "engine_core",
            self._engine_client,
        )
        return await utility_client.call_utility_async(
            "pap_control",
            "request_quiesce",
            {"request_ids": request_ids},
        )

    async def _run(self) -> None:
        while (item := await self._queue.get()) is not None:
            try:
                utility_client = getattr(
                    self._engine_client,
                    "engine_core",
                    self._engine_client,
                )
                value = await utility_client.call_utility_async(
                    "pap_control", item.operation, item.payload
                )
            except Exception as exc:
                if not item.result.done():
                    item.result.set_exception(exc)
            else:
                if not item.result.done():
                    item.result.set_result(value)
            finally:
                self._queue.task_done()
        self._queue.task_done()

    @staticmethod
    def _report_deferred_result(result: asyncio.Future[dict[str, Any]]) -> None:
        try:
            value = result.result()
        except Exception:
            logger.exception("deferred PAP EngineCore control failed")
            return
        if not (
            value.get("applied") or value.get("released") or value.get("idempotent")
        ):
            logger.error("deferred PAP EngineCore control was not applied: %s", value)


def _dispatcher(raw_request: Request) -> PAPControlDispatcher:
    dispatcher = getattr(raw_request.app.state, "pap_control_dispatcher", None)
    if dispatcher is None:
        raise HTTPException(status_code=503, detail="PAP control is not initialized")
    return dispatcher


def build_prefill_control_router() -> APIRouter:
    """Build PAP control routes without patching vLLM's API server."""
    router = APIRouter()

    @router.get("/v1/pap/prefill/kv-load")
    async def kv_load(raw_request: Request) -> dict[str, Any]:
        return await _dispatcher(raw_request).submit("kv_load_snapshot", {}, wait=True)

    @router.post("/v1/pap/prefill/quiesce")
    async def quiesce(
        req: RequestQuiesceRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        result = await _dispatcher(raw_request).abort_and_quiesce(req.request_ids)
        if not result.get("quiesced", False):
            raise HTTPException(status_code=409, detail=result)
        return result

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest, raw_request: Request) -> dict[str, Any]:
        payload = {
            "request_id": req.request_id,
            "session_request_id": req.session_request_id or req.request_id,
            "commit_seq": req.commit_seq,
            "new_seq_len": req.new_seq_len,
            "new_token_ids": req.new_token_ids,
            "layer_complete": req.layer_complete,
        }
        result = await _dispatcher(raw_request).submit(
            "decode_commit", payload, wait=not req.submit_only
        )
        if req.submit_only:
            return {
                **result,
                "request_id": req.request_id,
                "commit_seq": req.commit_seq,
                "accepted_commit_seq": req.commit_seq,
                "new_seq_len": req.new_seq_len,
            }
        if not (result.get("applied") or result.get("idempotent")):
            raise HTTPException(status_code=409, detail=result)
        return result

    @router.post("/v1/pap/prefill/decode-allocate")
    async def allocate(
        req: DecodeAllocationRequest, raw_request: Request
    ) -> dict[str, Any]:
        return await _dispatcher(raw_request).submit(
            "decode_allocate", req.model_dump(), wait=True
        )

    @router.post("/v1/pap/prefill/lease-release")
    async def release_lease(
        req: LeaseReleaseRequest, raw_request: Request
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": req.request_id,
            "lease_id": req.lease_id,
        }
        if req.final_commit_seq is not None:
            payload["final_commit_seq"] = req.final_commit_seq
        result = await _dispatcher(raw_request).submit(
            "lease_release", payload, wait=not req.submit_only
        )
        if req.submit_only:
            return {
                **result,
                "request_id": req.request_id,
                "lease_id": req.lease_id,
            }
        return result

    return router


def build_projection_control_router() -> APIRouter:
    """Build the Projection abort-and-quiesce acknowledgment route."""
    router = APIRouter()

    @router.post("/v1/pap/projection/quiesce")
    async def quiesce(
        req: RequestQuiesceRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        result = await _dispatcher(raw_request).abort_and_quiesce(req.request_ids)
        if not result.get("quiesced", False):
            raise HTTPException(status_code=409, detail=result)
        return result

    return router


__all__ = [
    "DecodeAllocationRequest",
    "DecodeCommitRequest",
    "LeaseReleaseRequest",
    "PAPControlDispatcher",
    "RequestQuiesceRequest",
    "build_prefill_control_router",
    "build_projection_control_router",
]
