# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible gateway for arbitrary PAP topologies."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from contextlib import asynccontextmanager
from itertools import count
from typing import Any

from fastapi import FastAPI, Request

from vllm.pap.config import reject_removed_pap_flags
from vllm.pap.gateway.admission import PAPProjectionAdmission
from vllm.pap.gateway.dynamo_routing import (
    PAPDynamoRouter,
    PAPDynamoRouterDisabled,
)
from vllm.pap.gateway.lifecycle import PAPLifecycleManager
from vllm.pap.gateway.load_tracker import PAPLoadTracker
from vllm.pap.gateway.prefix_cache import PAPPrefixCacheTracker
from vllm.pap.gateway.request_pipeline import _handle_openai_request
from vllm.pap.gateway.routing import PAPConversationRouter
from vllm.pap.gateway.topology import (
    _make_client,
    parse_pap_groups,
    parse_projection_instances,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    reject_removed_pap_flags(os.environ)
    app.state.groups = parse_pap_groups(args.pap_groups)
    app.state.projections = parse_projection_instances(args.projections)
    app.state.request_counter = count()
    app.state.pap_active_request_ids = set()
    app.state.pap_pending_route_request_ids = set()
    app.state.pair_counts = Counter()
    app.state.prefill_clients = {
        group: _make_client(group.prefill_host, group.prefill_port, "prefill")
        for group in app.state.groups
    }
    app.state.pap_load_tracker = PAPLoadTracker(app.state.prefill_clients)
    await app.state.pap_load_tracker.start()
    event_endpoints = tuple(
        endpoint.strip()
        for endpoint in args.kv_event_endpoints.split(",")
        if endpoint.strip()
    )
    app.state.pap_prefix_cache_tracker = PAPPrefixCacheTracker(
        app.state.groups,
        model=args.model,
        event_endpoints=event_endpoints,
        block_size=args.prefix_block_size,
        max_model_len=args.max_model_len,
        hf_overrides=json.loads(args.hf_overrides),
        generation_config=args.generation_config,
    )
    if args.routing_policy != "dynamo":
        app.state.pap_prefix_cache_tracker.start()
    if args.routing_policy == "dynamo":
        total_kv_blocks = args.dynamo_total_kv_blocks
        if total_kv_blocks is None:
            total_kv_blocks = [
                int(snapshot["total_kv_blocks"])
                for snapshot in app.state.pap_load_tracker.snapshot().values()
            ]
        app.state.pap_dynamo_router = await PAPDynamoRouter.create(
            app.state.groups,
            event_endpoints=event_endpoints,
            site_packages=args.dynamo_site_packages,
            model_name=args.dynamo_model_name,
            block_size=args.prefix_block_size,
            max_num_batched_tokens=args.dynamo_max_num_batched_tokens,
            total_kv_blocks=total_kv_blocks,
            prefill_load_scale=args.dynamo_prefill_load_scale,
        )
    else:
        app.state.pap_dynamo_router = PAPDynamoRouterDisabled()
    app.state.attention_clients = {}
    for group in app.state.groups:
        if isinstance(group.attention_port, int):
            ports = (group.attention_port,)
        else:
            ports = group.attention_port
        app.state.attention_clients[group] = [
            _make_client(group.attention_host, port, "attention") for port in ports
        ]
    app.state.projection_clients = {
        projection: _make_client(projection.host, projection.port, "projection")
        for projection in app.state.projections
    }
    app.state.conversation_router = PAPConversationRouter(app.state.groups)
    app.state.projection_admission = PAPProjectionAdmission(app.state.groups)
    app.state.pap_lifecycle_manager = PAPLifecycleManager()
    try:
        yield
    finally:
        await app.state.pap_dynamo_router.shutdown()
        app.state.pap_prefix_cache_tracker.stop()
        await app.state.pap_lifecycle_manager.shutdown()
        attention_clients = [
            client
            for clients in app.state.attention_clients.values()
            for client in clients
        ]
        for client in [
            *app.state.prefill_clients.values(),
            *attention_clients,
            *app.state.projection_clients.values(),
        ]:
            await client.client.aclose()


app = FastAPI(title="PAP Gateway", lifespan=lifespan)


@app.post("/v1/completions")
async def completions(request: Request):
    return await _handle_openai_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle_openai_request("/v1/chat/completions", request)


@app.get("/health")
async def health() -> dict[str, Any]:
    pending_route_requests = len(app.state.pap_pending_route_request_ids)
    active_requests = len(app.state.pap_active_request_ids)
    return {
        "status": "ok",
        "role": "multi-pap-proxy",
        "groups": len(app.state.groups),
        "projections": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "inflight_requests": pending_route_requests + active_requests,
        "pending_route_requests": pending_route_requests,
        "active_requests": active_requests,
        "pair_counts": dict(sorted(app.state.pair_counts.items())),
        "conversation_routing": app.state.conversation_router.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
        "load_tracker": app.state.pap_load_tracker.stats(),
        "prefix_cache": app.state.pap_prefix_cache_tracker.stats(),
        "dynamo_router": app.state.pap_dynamo_router.stats(),
        "lifecycle": app.state.pap_lifecycle_manager.stats(),
    }


@app.get("/v1/pap/topology/stats")
async def topology_stats() -> dict[str, Any]:
    pair_counts = dict(sorted(app.state.pair_counts.items()))
    pending_route_requests = len(app.state.pap_pending_route_request_ids)
    return {
        "pa_count": len(app.state.groups),
        "projection_count": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "total_requests": sum(pair_counts.values()),
        "pending_route_requests": pending_route_requests,
        "pair_counts": pair_counts,
        "conversation_routing": app.state.conversation_router.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
        "load_tracker": app.state.pap_load_tracker.stats(),
        "prefix_cache": app.state.pap_prefix_cache_tracker.stats(),
        "dynamo_router": app.state.pap_dynamo_router.stats(),
        "lifecycle": app.state.pap_lifecycle_manager.stats(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PAP request gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--pap-groups",
        required=True,
        help=(
            "Comma-separated prefill_host:prefill_port:attention_host:"
            "attention_port[:attention_tcp_port] entries"
        ),
    )
    parser.add_argument(
        "--projections",
        required=True,
        help="Comma-separated projection_host:projection_port entries",
    )
    parser.add_argument("--pap-mode", default=os.environ.get("PAP_MODE", "pap"))
    parser.add_argument("--model", default=os.environ.get("PAP_MODEL_PATH"))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument(
        "--hf-overrides",
        default=os.environ.get("PAP_HF_OVERRIDES", "{}"),
    )
    parser.add_argument("--generation-config", default="vllm")
    parser.add_argument(
        "--kv-event-endpoints",
        default=os.environ.get("PAP_KV_EVENT_ENDPOINTS", ""),
    )
    parser.add_argument(
        "--prefix-block-size",
        type=int,
        default=int(os.environ.get("PAP_BLOCK_SIZE", "16")),
    )
    parser.add_argument(
        "--dynamo-site-packages",
        default=os.environ.get("PAP_DYNAMO_SITE_PACKAGES", ""),
    )
    parser.add_argument(
        "--dynamo-model-name",
        default=os.environ.get("PAP_DYNAMO_MODEL_NAME", "pap"),
    )
    parser.add_argument(
        "--dynamo-max-num-batched-tokens",
        type=int,
        default=int(os.environ.get("PAP_PREFILL_MAX_NUM_BATCHED_TOKENS", "32768")),
    )
    parser.add_argument(
        "--dynamo-total-kv-blocks",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dynamo-prefill-load-scale",
        type=float,
        default=float(os.environ.get("PAP_DYNAMO_PREFILL_LOAD_SCALE", "2.0")),
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=300,
        help="Keep pooled benchmark connections alive between conversation turns",
    )
    parser.add_argument(
        "--routing-policy",
        default=os.environ.get("PAP_ROUTING_POLICY", "conversation_affinity"),
        choices=(
            "round_robin",
            "crossbar_round_robin",
            "projection_affinity",
            "projection_sticky",
            "conversation_affinity",
            "dynamo",
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the PAP gateway."""
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(
        app,
        host=parsed.host,
        port=parsed.port,
        timeout_keep_alive=parsed.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()
