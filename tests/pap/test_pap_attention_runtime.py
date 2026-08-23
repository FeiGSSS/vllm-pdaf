# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
