# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention internal executor.

This first PAP slice keeps the data-plane in vLLM's NIXL connector and exposes
the Attention role as an internal state owner. The process is intentionally not
an OpenAI-compatible vLLM server: it records which prefill KV handle belongs to
which PAP request so the proxy and future remote-attention backend have a stable
control-plane contract.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_attention")


class PAPAttentionRegistration(BaseModel):
    """KV ownership metadata registered after Prefill completes."""

    request_id: str
    conversation_id: str = ""
    prefill_endpoint: str
    kv_transfer_params: dict[str, Any] = Field(default_factory=dict)
    prefix_len: int | None = None


class PAPAttentionComputeRequest(BaseModel):
    """Tensor payload for the first PAP remote-attention compute path."""

    request_id: str
    layer_name: str
    query: dict[str, Any]
    key: dict[str, Any]
    value: dict[str, Any]
    scale: float


class PAPAttentionLayerEventRequest(BaseModel):
    """Shape-only event emitted at Projection's q/k/v -> attention boundary."""

    request_id: str
    layer_name: str
    query_shape: list[int]
    key_shape: list[int]
    value_shape: list[int]
    dtype: str
    device: str
    is_decode: bool
    num_reqs: int | None = None
    num_actual_tokens: int | None = None
    max_seq_len: int | None = None


@dataclass
class PAPAttentionLayerEvent:
    """Trace event for one Projection-side layer attention invocation."""

    request_id: str
    session_request_id: str
    layer_name: str
    query_shape: list[int]
    key_shape: list[int]
    value_shape: list[int]
    dtype: str
    device: str
    is_decode: bool
    num_reqs: int | None
    num_actual_tokens: int | None
    max_seq_len: int | None
    created_at: float = field(default_factory=time.time)

    def copy(self) -> PAPAttentionLayerEvent:
        return PAPAttentionLayerEvent(
            request_id=self.request_id,
            session_request_id=self.session_request_id,
            layer_name=self.layer_name,
            query_shape=list(self.query_shape),
            key_shape=list(self.key_shape),
            value_shape=list(self.value_shape),
            dtype=self.dtype,
            device=self.device,
            is_decode=self.is_decode,
            num_reqs=self.num_reqs,
            num_actual_tokens=self.num_actual_tokens,
            max_seq_len=self.max_seq_len,
            created_at=self.created_at,
        )


@dataclass
class PAPAttentionSession:
    """Snapshot of one PAP request known by the Attention executor."""

    request_id: str
    conversation_id: str
    prefill_endpoint: str
    kv_transfer_params: dict[str, Any]
    prefix_len: int | None
    created_at: float = field(default_factory=time.time)
    role: str = "attention"

    def copy(self) -> PAPAttentionSession:
        return PAPAttentionSession(
            request_id=self.request_id,
            conversation_id=self.conversation_id,
            prefill_endpoint=self.prefill_endpoint,
            kv_transfer_params=dict(self.kv_transfer_params),
            prefix_len=self.prefix_len,
            created_at=self.created_at,
            role=self.role,
        )


class PAPAttentionRegistry:
    """Thread-safe in-memory registry for PAP Attention control-plane state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, PAPAttentionSession] = {}
        self._layer_events: dict[str, list[PAPAttentionLayerEvent]] = {}

    def register_prefill_kv(
        self, registration: PAPAttentionRegistration
    ) -> PAPAttentionSession:
        session = PAPAttentionSession(
            request_id=registration.request_id,
            conversation_id=registration.conversation_id,
            prefill_endpoint=registration.prefill_endpoint,
            kv_transfer_params=dict(registration.kv_transfer_params),
            prefix_len=registration.prefix_len,
        )
        with self._lock:
            self._sessions[registration.request_id] = session
            self._layer_events.setdefault(registration.request_id, [])
        logger.info(
            "registered PAP attention session request_id=%s "
            "conversation_id=%s kv_keys=%s",
            registration.request_id,
            registration.conversation_id,
            sorted(registration.kv_transfer_params.keys()),
        )
        return session.copy()

    def get_session(self, request_id: str) -> PAPAttentionSession | None:
        with self._lock:
            session = self._sessions.get(request_id)
            return None if session is None else session.copy()

    def release_session(self, request_id: str) -> bool:
        with self._lock:
            existed = self._sessions.pop(request_id, None) is not None
            self._layer_events.pop(request_id, None)
            return existed

    def resolve_session_request_id(self, request_id: str) -> str | None:
        """Map vLLM-wrapped request ids back to the proxy-level PAP id."""
        with self._lock:
            return self._resolve_session_request_id_locked(request_id)

    def _resolve_session_request_id_locked(self, request_id: str) -> str | None:
        if request_id in self._sessions:
            return request_id

        candidates = [request_id]
        for prefix in ("cmpl-", "chatcmpl-"):
            if request_id.startswith(prefix):
                candidates.append(request_id[len(prefix) :])

        for candidate in candidates:
            if candidate in self._sessions:
                return candidate
            for session_request_id in self._sessions:
                if candidate.startswith(
                    f"{session_request_id}-"
                ) or candidate.startswith(f"{session_request_id}_"):
                    return session_request_id
        return None

    def record_layer_event(
        self,
        *,
        request_id: str,
        layer_name: str,
        query_shape: list[int],
        key_shape: list[int],
        value_shape: list[int],
        dtype: str,
        device: str,
        is_decode: bool,
        num_reqs: int | None = None,
        num_actual_tokens: int | None = None,
        max_seq_len: int | None = None,
    ) -> PAPAttentionLayerEvent:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                raise KeyError(request_id)
            event = PAPAttentionLayerEvent(
                request_id=request_id,
                session_request_id=session_request_id,
                layer_name=layer_name,
                query_shape=list(query_shape),
                key_shape=list(key_shape),
                value_shape=list(value_shape),
                dtype=dtype,
                device=device,
                is_decode=is_decode,
                num_reqs=num_reqs,
                num_actual_tokens=num_actual_tokens,
                max_seq_len=max_seq_len,
            )
            self._layer_events.setdefault(session_request_id, []).append(event)
        logger.info(
            "recorded PAP attention layer event request_id=%s "
            "session=%s layer=%s decode=%s",
            request_id,
            session_request_id,
            layer_name,
            is_decode,
        )
        return event.copy()

    def get_layer_events(self, request_id: str) -> list[PAPAttentionLayerEvent]:
        with self._lock:
            session_request_id = self._resolve_session_request_id_locked(request_id)
            if session_request_id is None:
                return []
            return [
                event.copy() for event in self._layer_events.get(session_request_id, [])
            ]

    def size(self) -> int:
        with self._lock:
            return len(self._sessions)


def create_app(registry: PAPAttentionRegistry | None = None) -> FastAPI:
    registry = registry or PAPAttentionRegistry()
    app = FastAPI(title="PAP Attention Executor")
    app.state.registry = registry

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "role": "attention",
            "sessions": registry.size(),
        }

    @app.post("/v1/pap/attention/register")
    async def register(
        registration: PAPAttentionRegistration,
    ) -> dict[str, Any]:
        return registry.register_prefill_kv(registration).__dict__

    @app.get("/v1/pap/attention/sessions/{request_id}")
    async def get_session(request_id: str) -> dict[str, Any]:
        session = registry.get_session(request_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown PAP request")
        return session.__dict__

    @app.post("/v1/pap/attention/compute")
    async def compute_attention(
        request: PAPAttentionComputeRequest,
    ) -> dict[str, Any]:
        from vllm.pap.remote_attention import (
            compute_attention_output,
            deserialize_tensor,
            serialize_attention_result,
        )

        query = deserialize_tensor(request.query)
        key = deserialize_tensor(request.key)
        value = deserialize_tensor(request.value)
        if torch.cuda.is_available():
            query = query.cuda(non_blocking=True)
            key = key.cuda(non_blocking=True)
            value = value.cuda(non_blocking=True)
        output = compute_attention_output(
            query=query,
            key=key,
            value=value,
            scale=request.scale,
        )
        logger.info(
            "computed PAP attention output request_id=%s layer=%s query_shape=%s",
            request.request_id,
            request.layer_name,
            list(query.shape),
        )
        return {
            "request_id": request.request_id,
            "layer_name": request.layer_name,
            "output": serialize_attention_result(output),
        }

    @app.post("/v1/pap/attention/layer-event")
    async def record_layer_event(
        event: PAPAttentionLayerEventRequest,
    ) -> dict[str, Any]:
        try:
            recorded = registry.record_layer_event(**event.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown PAP request") from exc
        return recorded.__dict__

    @app.get("/v1/pap/attention/sessions/{request_id}/layer-events")
    async def get_layer_events(request_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "events": [
                event.__dict__ for event in registry.get_layer_events(request_id)
            ],
        }

    @app.delete("/v1/pap/attention/sessions/{request_id}")
    async def release_session(request_id: str) -> dict[str, Any]:
        return {"released": registry.release_session(request_id)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAP Attention executor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    return parser.parse_args()


app = create_app()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
