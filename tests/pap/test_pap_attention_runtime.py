# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace

from vllm.pap.service import create_app, maybe_start_offload_exec_transport


def test_attention_health_reports_nvshmem_graph() -> None:
    app = create_app()
    health = app.state.pap_runtime.health()
    assert health["offload_exec_transport"] == "nvshmem_graph"


def test_attention_service_exposes_only_nvshmem_bind_route() -> None:
    paths = {route.path for route in create_app().routes}
    assert "/v1/pap/attention/offload-exec-nvshmem/bind" in paths
    assert not any("mailbox" in path for path in paths)


def test_attention_transport_start_is_unconditional(monkeypatch) -> None:
    app = create_app()
    calls = []
    app.state.pap_peer_manager = SimpleNamespace(
        config=app.state.pap_config,
        initialize=lambda **kwargs: calls.append(kwargs),
    )
    maybe_start_offload_exec_transport(app=app)
    assert calls == [{"enabled": True}]


def test_attention_health_returns_503_for_dead_receiver(monkeypatch) -> None:
    app = create_app()
    monkeypatch.setattr(
        app.state.pap_peer_manager,
        "health",
        lambda: {"status": "error", "receiver_state": "dead"},
    )

    health_route = next(route for route in app.routes if route.path == "/health")
    response = asyncio.run(health_route.endpoint())

    assert response.status_code == 503
