# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP Attention service composition and compatibility runtime.

This first PAP slice keeps Attention as an internal compute endpoint. The
process is intentionally not an OpenAI-compatible vLLM server: it records which
Prefill KV handle belongs to which PAP request so the proxy and remote-attention
path have a stable control-plane contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import logging
import socket
import socketserver
from threading import Lock, Thread
from typing import Any

from fastapi import FastAPI, HTTPException

from vllm.pap.attention import (
    PAPAttentionDispatcher,
    PAPAttentionRuntime,
    PAPAttentionWorkItem,
)
from vllm.pap.config import (
    PAPOffloadExecTransport,
    PAPRuntimeConfig,
)
from vllm.pap.transport.factory import build_offload_exec_transport
from vllm.pap.protocol import (
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)
from vllm.pap.deferred_cuda_trace import deferred_cuda_trace_snapshot
from vllm.pap.runtime_cuda_context_audit import write_runtime_cuda_context_audit

# Retain the historical ``vllm.pap.attention_executor`` symbol surface while
# service composition itself depends on ``PAPAttentionRuntime``.
from vllm.pap.attention.compute import (
    _combine_offload_exec_outputs,
    _compute_unified_paged_flash_batch,
    _finalize_offload_exec_compute_trace,
    _offload_exec_attention_shapes,
    _offload_exec_batch_rows,
    _offload_exec_session,
    _run_paged_flash_varlen,
    compute_offload_exec_batch_output,
    run_offload_exec_batch_once,
)
from vllm.pap.attention.runtime import (
    _combine_offload_exec_work_items,
    _execute_offload_exec_work_item,
    _execute_offload_exec_work_items,
    _new_offload_exec_compute_trace_stats,
    _offload_exec_work_item_compatibility_key,
    _qkv_message_recv_trace,
    _QKVBatchMessagePrefetcher,
    _record_offload_exec_ready_event,
    _recv_next_qkv_batch_message_or_tensor,
    _wait_offload_exec_ready_event,
    run_offload_exec_mailbox_loop,
    run_offload_exec_mailbox_receiver_loop,
)
from vllm.pap.kv.metadata import (
    PAPPagedFlashMetadata,
    _coerce_block_id,
    _UNIFIED_MD_CACHE,
    build_unified_paged_flash_metadata,
    reset_unified_paged_flash_metadata_cache,
    unified_paged_flash_metadata_cache_stats,
)
from vllm.pap.kv.handoff import accept_prefill_kv_handoff
from vllm.pap.kv.state import (
    PAPAttentionRegistry,
    PAPAttentionSession,
    PAPOffloadExecSessionEntry,
    PAPPrefillKVCacheCatalogEntry,
    PAPPrefillLayerReadiness,
    PAPUnifiedPagedKVState,
    PAPUnifiedSlotActivation,
    PAPUnifiedSlotTopology,
    _block_locality_stats,
    _DECODE_COMMIT_PATH,
    _DEFERRED_CUDA_TRACE_ENABLED,
    _get_commit_client,
    _get_lease_release_client,
    _KV_LOCALITY_PROFILE_SEEN,
    _LEASE_RELEASE_PATH,
    _log_kv_locality_profile,
    _pap_attention_pool_profile_enabled,
    _pap_env_flag,
    _pap_kv_lease_profile_enabled,
    _pap_kv_locality_profile_enabled,
    _prefill_control_endpoint,
    _trace_add_elapsed_ms,
    open_ipc_tensor_handle,
    open_prefill_manifest_event,
)
from vllm.pap.protocol.models import (
    PAPAttentionRegistration,
    PAPDecodeTokenBatchRequest,
    PAPDecodeTokenRequest,
    PAPOffloadExecMailboxActivityRequest,
    PAPOffloadExecMailboxBindRequest,
)

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
    dispatch_mode = runtime.dispatch_mode
    active_peer_tracking = runtime.active_peer_tracking
    app = FastAPI(title="PAP Attention Executor")
    app.state.pap_config = runtime_config
    app.state.pap_runtime = runtime
    app.state.registry = registry
    app.state.offload_exec_transport = None
    app.state.offload_exec_transports = {}
    app.state.offload_exec_source_ids = {}
    app.state.offload_exec_active_source_ids = set()
    app.state.offload_exec_membership_generations = {}
    app.state.offload_exec_membership_updates = 0
    app.state.offload_exec_membership_stale_updates = 0
    app.state.offload_exec_lock = Lock()
    app.state.offload_exec_mailbox_loop_started = False
    app.state.offload_exec_mailbox_loop_peers = set()
    app.state.offload_exec_local_rank = 0
    app.state.offload_exec_actor_base = "attention"
    app.state.offload_exec_dispatch_mode = dispatch_mode
    app.state.offload_exec_dispatcher = runtime.dispatcher

    def sync_dispatcher_membership() -> None:
        if dispatch_mode != "central_combine":
            return
        if active_peer_tracking:
            source_ids = set(app.state.offload_exec_active_source_ids)
        else:
            source_ids = set(app.state.offload_exec_source_ids.values())
        runtime.sync_dispatcher_membership(source_ids)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/v1/pap/attention/stats")
    async def attention_stats() -> dict[str, Any]:
        with app.state.offload_exec_lock:
            active_source_ids = sorted(app.state.offload_exec_active_source_ids)
            membership_generations = dict(
                sorted(app.state.offload_exec_membership_generations.items())
            )
            membership_updates = app.state.offload_exec_membership_updates
            membership_stale_updates = app.state.offload_exec_membership_stale_updates
        membership_stats = {
            "attention_active_source_ids": active_source_ids,
            "attention_membership_generations": membership_generations,
            "attention_membership_updates": membership_updates,
            "attention_membership_stale_updates": membership_stale_updates,
        }
        return runtime.stats(membership_stats)

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
        source_id = str(request.source_id)
        generation = int(request.membership_generation)
        active = bool(request.active)
        with app.state.offload_exec_lock:
            previous_generation = app.state.offload_exec_membership_generations.get(
                source_id
            )
            previous_active = source_id in app.state.offload_exec_active_source_ids
            if previous_generation is not None and generation < previous_generation:
                app.state.offload_exec_membership_stale_updates += 1
                return {
                    "source_id": source_id,
                    "active": previous_active,
                    "membership_generation": previous_generation,
                    "applied": False,
                    "stale": True,
                }
            if previous_generation == generation:
                if previous_active != active:
                    raise HTTPException(
                        status_code=409,
                        detail=("PAP mailbox membership generation changed activity"),
                    )
                return {
                    "source_id": source_id,
                    "active": active,
                    "membership_generation": generation,
                    "applied": False,
                    "stale": False,
                }
            app.state.offload_exec_membership_generations[source_id] = generation
            if active:
                app.state.offload_exec_active_source_ids.add(source_id)
            else:
                app.state.offload_exec_active_source_ids.discard(source_id)
            app.state.offload_exec_membership_updates += 1
            sync_dispatcher_membership()
        return {
            "source_id": source_id,
            "active": active,
            "membership_generation": generation,
            "applied": True,
            "stale": False,
        }

    @app.post("/v1/pap/attention/offload-exec-mailbox/bind")
    async def bind_offload_exec_mailbox(
        request: PAPOffloadExecMailboxBindRequest,
    ) -> dict[str, Any]:
        peer_metadata = base64.b64decode(request.agent_metadata_b64.encode("ascii"))
        peer_key = hashlib.sha1(peer_metadata).hexdigest()[:16]
        source_id = str(request.source_id or peer_key)
        with app.state.offload_exec_lock:
            existing_source_id = app.state.offload_exec_source_ids.get(peer_key)
            if existing_source_id is not None and existing_source_id != source_id:
                raise HTTPException(
                    status_code=409,
                    detail="PAP mailbox peer changed its stable source id",
                )
            if any(
                existing_peer_key != peer_key and existing_source_id == source_id
                for existing_peer_key, existing_source_id in (
                    app.state.offload_exec_source_ids.items()
                )
            ):
                raise HTTPException(
                    status_code=409,
                    detail="PAP mailbox source id is already bound",
                )
            transport = app.state.offload_exec_transports.get(peer_key)
            if transport is None:
                initial_transport = app.state.offload_exec_transport
                if (
                    not app.state.offload_exec_transports
                    and initial_transport is not None
                ):
                    transport = initial_transport
                else:
                    transport = _build_attention_offload_exec_transport(
                        actor_id=(f"{app.state.offload_exec_actor_base}-{peer_key}"),
                        local_rank=app.state.offload_exec_local_rank,
                        transport=runtime_config.offload_exec_transport,
                    )
                if not hasattr(transport, "local_agent_metadata"):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "PAP OFFLOAD_EXEC mailbox transport is not initialized"
                        ),
                    )
                transport.bind_peer(peer_metadata)
                transport._pap_mailbox_bound = True
                app.state.offload_exec_transports[peer_key] = transport
            app.state.offload_exec_source_ids[peer_key] = source_id
            if dispatch_mode == "central_combine":
                sync_dispatcher_membership()
            if peer_key not in app.state.offload_exec_mailbox_loop_peers:
                dispatcher = app.state.offload_exec_dispatcher
                if dispatch_mode == "central_combine":
                    assert dispatcher is not None
                    dispatcher.start()
                    target = run_offload_exec_mailbox_receiver_loop
                    kwargs = {
                        "registry": registry,
                        "transport": transport,
                        "dispatcher": dispatcher,
                        "peer_id": source_id,
                    }
                    thread_kind = "receiver"
                else:
                    target = run_offload_exec_mailbox_loop
                    kwargs = {
                        "registry": registry,
                        "transport": transport,
                        "peer_id": peer_key,
                    }
                    thread_kind = "loop"
                Thread(
                    target=target,
                    kwargs=kwargs,
                    daemon=True,
                    name=(f"pap-offload-exec-mailbox-{thread_kind}-{peer_key}"),
                ).start()
                app.state.offload_exec_mailbox_loop_peers.add(peer_key)
                app.state.offload_exec_mailbox_loop_started = True
        return {
            "agent_metadata_b64": base64.b64encode(
                transport.local_agent_metadata
            ).decode("ascii")
        }

    async def stop_offload_exec_dispatcher() -> None:
        runtime.stop()

    app.add_event_handler("shutdown", stop_offload_exec_dispatcher)


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
    parser = argparse.ArgumentParser(description="Run PAP Attention executor")
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

    if zmq_port is None:
        return
    runtime_config = config or app.state.pap_config
    local_rank = runtime_config.attention.local_rank
    actor_base = runtime_config.attention.actor_id
    app.state.offload_exec_local_rank = local_rank
    app.state.offload_exec_actor_base = actor_base
    app.state.offload_exec_transport = _build_attention_offload_exec_transport(
        actor_id=actor_base,
        local_rank=local_rank,
        transport=runtime_config.offload_exec_transport,
    )
    transport = runtime_config.offload_exec_transport
    if transport is PAPOffloadExecTransport.NIXL_MAILBOX:
        logger.info(
            "PAP Attention OFFLOAD_EXEC NIXL mailbox initialized local_rank=%d",
            local_rank,
        )
        return
    if transport is PAPOffloadExecTransport.LOCAL_FAST:
        logger.info(
            "PAP Attention OFFLOAD_EXEC local_fast CUDA IPC ring "
            "initialized local_rank=%d",
            local_rank,
        )
        return
    raise AssertionError(f"unsupported PAP OFFLOAD_EXEC transport: {transport}")


def _build_attention_offload_exec_transport(
    *,
    actor_id: str,
    local_rank: int,
    transport: PAPOffloadExecTransport,
) -> Any:
    return build_offload_exec_transport(
        transport=transport,
        actor_id=actor_id,
        local_rank=local_rank,
    )


app = create_app()


def main() -> None:
    """Run the PAP Attention executor service."""
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
