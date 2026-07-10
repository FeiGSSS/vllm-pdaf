from __future__ import annotations

import asyncio
import inspect
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


class DecodeCommitRequest(BaseModel):
    request_id: str
    commit_seq: int = Field(gt=0)
    new_seq_len: int
    new_token_ids: list[int]
    layer_complete: bool = True


class LeaseReleaseRequest(BaseModel):
    request_id: str
    lease_id: str


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_engine_control_method(
    engine_client: Any,
    async_name: str,
    sync_name: str,
    *args: Any,
) -> dict[str, Any]:
    async_method = getattr(engine_client, async_name, None)
    if async_method is not None:
        try:
            return await _maybe_await(async_method(*args))
        except NotImplementedError:
            pass
    return await _maybe_await(getattr(engine_client, sync_name)(*args))


def build_prefill_control_router() -> APIRouter:
    router = APIRouter()
    commit_lock = asyncio.Lock()
    acked_commit_state: dict[str, tuple[int, int]] = {}

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest, raw_request: Request) -> dict[str, Any]:
        request_id = str(req.request_id)
        async with commit_lock:
            acked_seq, acked_seq_len = acked_commit_state.get(request_id, (0, 0))
            if req.commit_seq <= acked_seq:
                return {
                    "request_id": request_id,
                    "commit_seq": req.commit_seq,
                    "acked_commit_seq": acked_seq,
                    "new_seq_len": acked_seq_len,
                    "applied": False,
                    "idempotent": True,
                }

            engine_client = raw_request.app.state.engine_client
            result = await _call_engine_control_method(
                engine_client,
                "pap_apply_decode_commit_async",
                "pap_apply_decode_commit",
                request_id,
                req.new_seq_len,
                tuple(req.new_token_ids),
            )
            if not result.get("applied", False):
                raise HTTPException(status_code=409, detail=result)

            acked_commit_state[request_id] = (req.commit_seq, req.new_seq_len)
            return {
                **result,
                "commit_seq": req.commit_seq,
                "acked_commit_seq": req.commit_seq,
                "idempotent": False,
            }

    @router.post("/v1/pap/prefill/lease-release")
    async def release_lease(
        req: LeaseReleaseRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        request_id = str(req.request_id)
        async with commit_lock:
            engine_client = raw_request.app.state.engine_client
            result = await _call_engine_control_method(
                engine_client,
                "pap_release_kv_lease_async",
                "pap_release_kv_lease",
                request_id,
                req.lease_id,
            )
            if result.get("released", False) or result.get("reason") == (
                "unknown_or_released_lease"
            ):
                acked_commit_state.pop(request_id, None)
        return result

    return router
