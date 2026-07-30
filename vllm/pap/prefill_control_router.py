from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DecodeCommitRequest(BaseModel):
    request_id: str
    session_request_id: str | None = None
    commit_seq: int = Field(gt=0)
    new_seq_len: int
    new_token_ids: list[int]
    layer_complete: bool = True
    submit_only: bool = False


class LeaseReleaseRequest(BaseModel):
    request_id: str
    lease_id: str
    retain: bool = False
    submit_only: bool = False


class KVExportRequest(BaseModel):
    request_id: str


class KVMigrationRequest(BaseModel):
    request_id: str
    source_kv_params: dict[str, Any]
    prefix_len: int = Field(gt=0)
    prefix_token_ids: list[int]
    prefix_block_hashes: list[str]
    decode_capacity_tokens: int = Field(ge=0)
    session_handle: str
    attention_tcp_endpoint: str


class KVMigrationStatusRequest(BaseModel):
    job_id: str


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


async def _submit_engine_control_method(
    engine_client: Any,
    async_name: str,
    *args: Any,
) -> Any:
    """Submit one EngineCore call and return its execution future."""
    async_method = getattr(engine_client, async_name, None)
    if async_method is None:
        raise NotImplementedError
    submission = async_method(*args)
    if inspect.iscoroutine(submission):
        return await submission
    return submission


def _track_submitted_control(
    pending: Any,
    *,
    operation: str,
    request_id: str,
    success_key: str,
) -> None:
    """Consume a submit-only result and report deferred failures."""

    def report_result(result: Any) -> None:
        if not isinstance(result, dict) or not result.get(success_key, False):
            logger.error(
                "PAP submitted control was not applied operation=%s "
                "request_id=%s result=%s",
                operation,
                request_id,
                result,
            )

    def completed(future: Any) -> None:
        try:
            report_result(future.result())
        except Exception:
            logger.exception(
                "PAP submitted control failed operation=%s request_id=%s",
                operation,
                request_id,
            )

    add_done_callback = getattr(pending, "add_done_callback", None)
    if add_done_callback is None:
        report_result(pending)
        return
    add_done_callback(completed)


def build_prefill_control_router() -> APIRouter:
    router = APIRouter()
    session_locks: dict[str, asyncio.Lock] = {}
    acked_commit_state: dict[str, tuple[int, int]] = {}
    submitted_commit_state: dict[str, tuple[int, int]] = {}
    targets_by_session: dict[str, set[str]] = {}

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest, raw_request: Request) -> dict[str, Any]:
        request_id = str(req.request_id)
        session_request_id = str(req.session_request_id or request_id)
        session_lock = session_locks.setdefault(session_request_id, asyncio.Lock())
        async with session_lock:
            targets_by_session.setdefault(session_request_id, set()).add(request_id)
            acked_seq, acked_seq_len = acked_commit_state.get(request_id, (0, 0))
            submitted_seq, submitted_seq_len = submitted_commit_state.get(
                request_id, (0, 0)
            )
            known_seq = max(acked_seq, submitted_seq)
            known_seq_len = (
                submitted_seq_len if submitted_seq >= acked_seq else acked_seq_len
            )
            if req.commit_seq <= known_seq:
                if req.submit_only:
                    return {
                        "request_id": request_id,
                        "commit_seq": req.commit_seq,
                        "accepted_commit_seq": known_seq,
                        "new_seq_len": known_seq_len,
                        "accepted": True,
                        "idempotent": True,
                    }
                return {
                    "request_id": request_id,
                    "commit_seq": req.commit_seq,
                    "acked_commit_seq": known_seq,
                    "new_seq_len": known_seq_len,
                    "applied": False,
                    "idempotent": True,
                }

            engine_client = raw_request.app.state.engine_client
            if req.submit_only:
                try:
                    pending = await _submit_engine_control_method(
                        engine_client,
                        "pap_submit_decode_commit_async",
                        request_id,
                        req.new_seq_len,
                        tuple(req.new_token_ids),
                    )
                except NotImplementedError:
                    pass
                else:
                    submitted_commit_state[request_id] = (
                        req.commit_seq,
                        req.new_seq_len,
                    )
                    _track_submitted_control(
                        pending,
                        operation="decode_commit",
                        request_id=request_id,
                        success_key="applied",
                    )
                    return {
                        "request_id": request_id,
                        "commit_seq": req.commit_seq,
                        "accepted_commit_seq": req.commit_seq,
                        "new_seq_len": req.new_seq_len,
                        "accepted": True,
                        "idempotent": False,
                    }

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
        session_lock = session_locks.setdefault(request_id, asyncio.Lock())
        async with session_lock:
            engine_client = raw_request.app.state.engine_client
            control_args = (
                (request_id, req.lease_id, True)
                if req.retain
                else (request_id, req.lease_id)
            )
            if req.submit_only:
                try:
                    pending = await _submit_engine_control_method(
                        engine_client,
                        "pap_submit_release_kv_lease_async",
                        *control_args,
                    )
                except NotImplementedError:
                    pass
                else:
                    _track_submitted_control(
                        pending,
                        operation="lease_retain" if req.retain else "lease_release",
                        request_id=request_id,
                        success_key="retained" if req.retain else "released",
                    )
                    targets = targets_by_session.pop(request_id, {request_id})
                    for target in targets:
                        acked_commit_state.pop(target, None)
                        submitted_commit_state.pop(target, None)
                    session_locks.pop(request_id, None)
                    return {
                        "request_id": request_id,
                        "lease_id": req.lease_id,
                        "accepted": True,
                        "retain": req.retain,
                    }

            result = await _call_engine_control_method(
                engine_client,
                "pap_release_kv_lease_async",
                "pap_release_kv_lease",
                *control_args,
            )
            completed = result.get("released", False) or result.get("retained", False)
            if completed or result.get("reason") in {
                "unknown_or_released_lease",
                "unknown_expired_or_released_lease",
            }:
                targets = targets_by_session.pop(request_id, {request_id})
                for target in targets:
                    acked_commit_state.pop(target, None)
                    submitted_commit_state.pop(target, None)
        session_locks.pop(request_id, None)
        return result

    @router.post("/v1/pap/prefill/kv-export")
    async def export_kv(
        req: KVExportRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        return await _call_engine_control_method(
            engine_client,
            "pap_export_kv_lease_async",
            "pap_export_kv_lease",
            str(req.request_id),
        )

    @router.post("/v1/pap/prefill/kv-import")
    async def import_kv(
        req: KVMigrationRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        return await _call_engine_control_method(
            engine_client,
            "pap_submit_kv_migration_async",
            "pap_submit_kv_migration",
            req.model_dump(),
        )

    @router.post("/v1/pap/prefill/kv-import/status")
    async def import_kv_status(
        req: KVMigrationStatusRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        return await _call_engine_control_method(
            engine_client,
            "pap_kv_migration_status_async",
            "pap_kv_migration_status",
            str(req.job_id),
        )

    return router
