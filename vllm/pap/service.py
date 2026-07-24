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
import base64
import logging
import socket
import socketserver
from threading import Thread
from typing import Any

from fastapi import FastAPI, HTTPException

from vllm.pap.attention import PAPAttentionRuntime
from vllm.pap.attention.peers import (
    PAPAttentionPeerConflict,
    PAPAttentionPeerManager,
)
from vllm.pap.config import (
    PAPOffloadExecTransport,
    PAPRuntimeConfig,
)
from vllm.pap.kv.handoff import accept_prefill_kv_handoff
from vllm.pap.kv.registry import PAPAttentionRegistry
from vllm.pap.protocol.models import (
    PAPAttentionRegistration,
    PAPDecodeTokenBatchRequest,
    PAPDecodeTokenRequest,
    PAPOffloadExecMailboxActivityRequest,
    PAPOffloadExecMailboxBindRequest,
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
    app = FastAPI(title="PAP Attention Service")
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
            "results": [
                record_one_decode_token(token) for token in request.tokens
            ],
        }

    @app.get("/v1/pap/attention/sessions/{request_id}")
    async def get_session(request_id: str) -> dict[str, Any]:
        session = runtime.get_session(request_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown PAP request")
        return session.__dict__

    @app.post("/v1/pap/attention/offload-exec-mailbox/activity")
    async def update_offload_exec_mailbox_activity(
        request: PAPOffloadExecMailboxActivityRequest,
    ) -> dict[str, Any]:
        try:
            return peer_manager.update_activity(
                source_id=str(request.source_id),
                generation=int(request.membership_generation),
                active=bool(request.active),
            )
        except PAPAttentionPeerConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/pap/attention/offload-exec-mailbox/bind")
    async def bind_offload_exec_mailbox(
        request: PAPOffloadExecMailboxBindRequest,
    ) -> dict[str, Any]:
        peer_metadata = base64.b64decode(request.agent_metadata_b64.encode("ascii"))
        try:
            local_metadata = peer_manager.bind(
                peer_metadata=peer_metadata,
                source_id=request.source_id,
            )
        except PAPAttentionPeerConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "agent_metadata_b64": base64.b64encode(local_metadata).decode("ascii")
        }

    async def stop_peer_manager() -> None:
        peer_manager.stop()

    app.add_event_handler("shutdown", stop_peer_manager)

    @app.get("/v1/pap/attention/sessions")
    async def get_active_session_count() -> dict[str, int]:
        return {"active_sessions": runtime.active_session_count()}

    @app.get("/v1/pap/attention/sessions/{request_id}/prefill-readiness")
    async def get_prefill_readiness(request_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "layers": [
                readiness.__dict__
                for readiness in runtime.get_prefill_readiness(request_id)
            ],
        }

    @app.delete("/v1/pap/attention/sessions/{request_id}")
    def release_session(request_id: str) -> dict[str, Any]:
        return {"released": runtime.release_session(request_id)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PAP Attention service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument(
        "--offload-exec-zmq-port",
        type=int,
        default=None,
        help=(
            "Reserved PAP OFFLOAD_EXEC ZMQ control port for the "
            "Projection<->Attention data plane."
        ),
    )
    return parser.parse_args()


def maybe_start_offload_exec_transport(
    *,
    app: FastAPI,
    host: str,
    zmq_port: int | None,
    config: PAPRuntimeConfig | None = None,
) -> None:
    """Initialize the optional OFFLOAD_EXEC data plane."""
    del host
    if zmq_port is None:
        return
    runtime_config = config or app.state.pap_config
    local_rank = runtime_config.attention.local_rank
    peer_manager: PAPAttentionPeerManager = app.state.pap_peer_manager
    if runtime_config != peer_manager.config:
        raise ValueError("PAP Attention transport config must match app config")
    peer_manager.initialize(enabled=True)
    transport = runtime_config.offload_exec_transport
    if transport is PAPOffloadExecTransport.NIXL_MAILBOX:
        logger.info(
            "PAP Attention OFFLOAD_EXEC NIXL mailbox initialized local_rank=%d",
            local_rank,
        )
        return
    if transport is PAPOffloadExecTransport.LOCAL_FAST:
        logger.info(
            "PAP Attention OFFLOAD_EXEC local_fast serial CUDA IPC buffer "
            "initialized local_rank=%d",
            local_rank,
        )
        return
    raise AssertionError(f"unsupported PAP OFFLOAD_EXEC transport: {transport}")


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
    if args.offload_exec_zmq_port is not None:
        logger.info(
            "PAP Attention OFFLOAD_EXEC ZMQ endpoint reserved at %s:%d",
            args.host,
            args.offload_exec_zmq_port,
        )
    maybe_start_offload_exec_transport(
        app=app,
        host=args.host,
        zmq_port=args.offload_exec_zmq_port,
    )
    write_runtime_cuda_context_audit(role="attention")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
