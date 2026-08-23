# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible gateway for arbitrary PAP topologies."""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from contextlib import asynccontextmanager
from itertools import count
from typing import Any

from fastapi import FastAPI, Request

from vllm.pap.config import reject_removed_pap_flags
from vllm.pap.gateway.admission import PAPProjectionAdmission
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
    app.state.pair_counts = Counter()
    app.state.prefill_clients = {
        group: _make_client(group.prefill_host, group.prefill_port, "prefill")
        for group in app.state.groups
    }
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
    yield
    attention_clients = [
        client for clients in app.state.attention_clients.values() for client in clients
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
    return {
        "status": "ok",
        "role": "multi-pap-proxy",
        "groups": len(app.state.groups),
        "projections": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "inflight_requests": len(app.state.pap_active_request_ids),
        "pair_counts": dict(sorted(app.state.pair_counts.items())),
        "conversation_routing": app.state.conversation_router.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
    }


@app.get("/v1/pap/topology/stats")
async def topology_stats() -> dict[str, Any]:
    pair_counts = dict(sorted(app.state.pair_counts.items()))
    return {
        "pa_count": len(app.state.groups),
        "projection_count": len(app.state.projections),
        "routing_policy": app.state.args.routing_policy,
        "total_requests": sum(pair_counts.values()),
        "pair_counts": pair_counts,
        "conversation_routing": app.state.conversation_router.snapshot(),
        "projection_admission": await app.state.projection_admission.snapshot(),
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
    parser.add_argument(
        "--routing-policy",
        default=os.environ.get("PAP_ROUTING_POLICY", "conversation_affinity"),
        choices=(
            "round_robin",
            "crossbar_round_robin",
            "projection_affinity",
            "projection_sticky",
            "conversation_affinity",
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the PAP gateway."""
    import uvicorn

    parsed = parse_args()
    app.state.args = parsed
    uvicorn.run(app, host=parsed.host, port=parsed.port)


if __name__ == "__main__":
    main()
