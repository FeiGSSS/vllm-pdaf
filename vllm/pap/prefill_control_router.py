from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


class DecodeCommitRequest(BaseModel):
    request_id: str
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

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest, raw_request: Request) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        result = await _call_engine_control_method(
            engine_client,
            "pap_apply_decode_commit_async",
            "pap_apply_decode_commit",
            req.request_id,
            req.new_seq_len,
            tuple(req.new_token_ids),
        )
        if not result.get("applied", False):
            raise HTTPException(status_code=404, detail=result)
        return result

    @router.post("/v1/pap/prefill/lease-release")
    async def release_lease(
        req: LeaseReleaseRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        return await _call_engine_control_method(
            engine_client,
            "pap_release_kv_lease_async",
            "pap_release_kv_lease",
            req.request_id,
            req.lease_id,
        )

    return router
