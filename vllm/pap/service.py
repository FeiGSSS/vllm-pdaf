# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention service composition root.

This first PAP slice keeps Attention as an internal compute endpoint. The
process is intentionally not an OpenAI-compatible vLLM server: it records which
Prefill KV handle belongs to which PAP request so the proxy and remote-attention
path have a stable control-plane contract.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import socketserver
import time
from contextlib import asynccontextmanager
from threading import Thread
from typing import Any

import pybase64 as base64
from fastapi import FastAPI, HTTPException

from vllm.pap.attention import PAPAttentionRuntime
from vllm.pap.attention.peers import (
    PAPAttentionPeerConflict,
    PAPAttentionPeerManager,
)
from vllm.pap.config import PAPRuntimeConfig
from vllm.pap.kv.handoff import accept_prefill_kv_handoff
from vllm.pap.kv.registry import PAPAttentionRegistry
from vllm.pap.protocol.models import (
    PAPAttentionRegistration,
    PAPDecodeTokenBatchRequest,
    PAPDecodeTokenRequest,
    PAPOffloadExecNVSHMEMBindRequest,
)
from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pap_attention")


def compute_binary_attention_response(
    registry: PAPAttentionRegistry,
    payload: bytes,
) -> bytes:
    """Handle the sealed KV handoff wire protocol."""
    return accept_prefill_kv_handoff(registry, payload)


def _recv_exact(sock: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("PAP attention TCP peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class _AttentionTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_attention_tcp_server(
    registry: PAPAttentionRegistry,
    *,
    host: str,
    port: int,
) -> socketserver.TCPServer:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    header = _recv_exact(self.request, 8)
                    request_len = int.from_bytes(header, byteorder="little")
                    if request_len <= 0:
                        raise ValueError("PAP attention TCP request length <= 0")
                    payload = _recv_exact(self.request, request_len)
                    response = compute_binary_attention_response(
                        registry,
                        payload,
                    )
                    self.request.sendall(
                        len(response).to_bytes(8, byteorder="little") + response
                    )
                except EOFError:
                    return
                except Exception:
                    logger.exception("PAP attention TCP request failed")
                    return

    server = _AttentionTCPServer((host, port), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("PAP Attention TCP data plane listening on %s:%d", host, port)
    return server


def create_app(
    registry: PAPAttentionRegistry | None = None,
    *,
    config: PAPRuntimeConfig | None = None,
) -> FastAPI:
    runtime_config = config or PAPRuntimeConfig.from_env()
    runtime = PAPAttentionRuntime(config=runtime_config, registry=registry)
    registry = runtime.registry
    peer_manager = PAPAttentionPeerManager(
        runtime=runtime,
        config=runtime_config,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            peer_manager.stop()

    app = FastAPI(title="PAP Attention Service", lifespan=lifespan)
    app.state.pap_config = runtime_config
    app.state.pap_runtime = runtime
    app.state.pap_peer_manager = peer_manager
    app.state.registry = registry

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/v1/pap/attention/stats")
    async def attention_stats() -> dict[str, Any]:
        return runtime.stats(peer_manager.membership_stats())

    @app.post("/v1/pap/attention/register")
    async def register(
        registration: PAPAttentionRegistration,
    ) -> dict[str, Any]:
        return runtime.register_prefill_kv(registration)

    def record_one_decode_token(
        request: PAPDecodeTokenRequest,
    ) -> dict[str, Any]:
        try:
            status = runtime.record_decode_token(
                request_id=request.request_id,
                new_seq_len=request.new_seq_len,
                token_id=request.token_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="unknown PAP request",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "status": status,
            "request_id": request.request_id,
            "new_seq_len": request.new_seq_len,
        }

    @app.post("/v1/pap/attention/decode-token")
    async def record_decode_token(request: PAPDecodeTokenRequest) -> dict[str, Any]:
        return record_one_decode_token(request)

    @app.post("/v1/pap/attention/decode-tokens")
    async def record_decode_tokens(
        request: PAPDecodeTokenBatchRequest,
    ) -> dict[str, Any]:
        return {
            "status": "accepted",
            "results": [record_one_decode_token(token) for token in request.tokens],
        }

    @app.get("/v1/pap/attention/sessions/{request_id}")
    async def get_session(request_id: str) -> dict[str, Any]:
        session = runtime.get_session(request_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown PAP request")
        return session.__dict__

    @app.post("/v1/pap/attention/offload-exec-nvshmem/bind")
    async def bind_offload_exec_nvshmem(
        request: PAPOffloadExecNVSHMEMBindRequest,
    ) -> dict[str, Any]:
        peer_metadata = base64.b64decode(request.agent_metadata_b64.encode("ascii"))
        try:
            local_metadata = peer_manager.bind(
                peer_metadata=peer_metadata,
                source_id=request.source_id,
            )
        except PAPAttentionPeerConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"agent_metadata_b64": base64.b64encode(local_metadata).decode("ascii")}

    @app.get("/v1/pap/attention/sessions")
    async def get_active_session_count() -> dict[str, int]:
        return {"active_sessions": runtime.active_session_count()}

    @app.get("/v1/pap/attention/sessions/{request_id}/prefill-readiness")
    async def get_prefill_readiness(
        request_id: str,
        expected_prefix_len: int | None = None,
        expected_session_handle: str | None = None,
        timeout_s: float = 0.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            snapshot = runtime.get_prefill_readiness_snapshot(
                request_id,
                include_layers=False,
            )
            session_handle = snapshot["session_handle"]
            session_ready = expected_session_handle is None or (
                session_handle is not None
                and str(session_handle) == expected_session_handle
            )
            ready_prefix_len = snapshot["ready_prefix_len"]
            prefix_ready = expected_prefix_len is None or (
                ready_prefix_len is not None
                and int(ready_prefix_len) >= expected_prefix_len
            )
            ready = bool(snapshot["ready"] and session_ready and prefix_ready)
            timed_out = time.monotonic() >= deadline
            if ready or snapshot["failed"] or timed_out:
                break
            await asyncio.sleep(0.005)
        return {
            "request_id": request_id,
            "session_handle": session_handle,
            "manifest_prefix_len": snapshot["manifest_prefix_len"],
            "ready_prefix_len": ready_prefix_len,
            "ready": ready,
            "failed": snapshot["failed"],
            "error": snapshot["error"],
            "timed_out": timed_out and not ready,
        }

    @app.delete("/v1/pap/attention/sessions/{request_id}")
    def release_session(request_id: str) -> dict[str, Any]:
        return {
            "released": runtime.release_session(request_id),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PAP Attention service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--tcp-port", type=int, default=None)
    return parser.parse_args()


def maybe_start_offload_exec_transport(
    *,
    app: FastAPI,
    config: PAPRuntimeConfig | None = None,
) -> None:
    """Initialize the NVSHMEM whole-step Graph data plane."""
    runtime_config = config or app.state.pap_config
    local_rank = runtime_config.attention.local_rank
    peer_manager: PAPAttentionPeerManager = app.state.pap_peer_manager
    if runtime_config != peer_manager.config:
        raise ValueError("PAP Attention transport config must match app config")
    peer_manager.initialize(enabled=True)
    logger.info(
        "PAP Attention OFFLOAD_EXEC NVSHMEM whole-step Graph initialized local_rank=%d",
        local_rank,
    )


app = create_app()


def main() -> None:
    """Run the PAP Attention service."""
    import uvicorn

    args = parse_args()
    if args.tcp_port is not None:
        start_attention_tcp_server(
            app.state.registry,
            host=args.host,
            port=args.tcp_port,
        )
    maybe_start_offload_exec_transport(app=app)
    write_runtime_cuda_context_audit(role="attention")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
