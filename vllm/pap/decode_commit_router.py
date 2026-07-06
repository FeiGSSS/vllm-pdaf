from __future__ import annotations
from typing import Any, Callable
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class DecodeCommitRequest(BaseModel):
    request_id: str
    new_seq_len: int
    new_token_ids: list[int]
    layer_complete: bool = True


def build_commit_router(
    *,
    manager: Any,
    requests: dict[str, Any],
    lookup_request: Callable[[str], Any] | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest) -> dict:
        request = (
            lookup_request(req.request_id)
            if lookup_request is not None
            else requests.get(req.request_id)
        )
        if request is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown PAP request {req.request_id}",
            )
        manager.apply_decode_commit(
            request=request,
            new_seq_len=req.new_seq_len,
            new_token_ids=tuple(req.new_token_ids),
        )
        return {"request_id": req.request_id, "applied": True}

    return router
