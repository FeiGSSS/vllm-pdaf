# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace

import pytest

from vllm.pap.kv.models import PAPAttentionStepContext
from vllm.pap.kv.step_context_registry import _PAPAttentionStepContextMixin
from vllm.pap.service import create_app, maybe_start_offload_exec_transport


def _successor_owner(*, seq_len: int = 16):
    states = {
        "layer0": (SimpleNamespace(seq_len=seq_len),),
        "layer1": (SimpleNamespace(seq_len=seq_len),),
    }
    entry = SimpleNamespace(session_request_id="req", session_epoch=3)
    previous = PAPAttentionStepContext(
        cache_key=("previous",),
        request_ids=("req",),
        decode_seq_lens=(seq_len,),
        session_entries=(entry,),
        session_request_ids=("req",),
        prior_seq_lens=(seq_len - 1,),
        result_seq_lens=(seq_len,),
        commit_new_seq_lens=(seq_len,),
        active_indices=(0,),
        active_prior_seq_lens=(seq_len - 1,),
        expected_layers=frozenset(states),
        layer_states=states,
        topology_ids=(11,),
        q_size=4096,
        kv_size=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        scale=0.125,
        completed_layers=set(states),
        kv_ready_published=True,
    )
    activation = SimpleNamespace(
        complete=True,
        conflict_latched=False,
        canonical_topology_id=11,
    )
    return SimpleNamespace(
        _last_attention_step_context=previous,
        _session_epochs={"req": 3},
        _unified_slot_activations={"req": activation},
    )


def _build_successor(owner, *, request_ids=("req",), steps=(17,)):
    return _PAPAttentionStepContextMixin._successor_attention_step_context(
        owner,
        cache_key=(request_ids, steps),
        request_ids=request_ids,
        decode_seq_lens=steps,
        scales=(0.125,) * len(request_ids),
        default_q_size=4096,
        default_kv_size=1024,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


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


def test_attention_step_successor_reuses_structure_across_page_boundary() -> None:
    owner = _successor_owner(seq_len=16)

    successor = _build_successor(owner, steps=(17,))

    assert successor is not None
    assert successor.prior_seq_lens == (16,)
    assert successor.result_seq_lens == (17,)
    assert successor.layer_states is owner._last_attention_step_context.layer_states
    assert successor.graph_slot_tensor is None
    assert successor.metadata is None
    assert successor.attention_kernel_plan is None


@pytest.mark.parametrize("request_ids", [(), ("req", "new")])
def test_attention_step_successor_rejects_request_membership_change(
    request_ids,
) -> None:
    owner = _successor_owner()

    assert (
        _build_successor(
            owner,
            request_ids=request_ids,
            steps=(17,) * len(request_ids),
        )
        is None
    )


@pytest.mark.parametrize("change", ["epoch", "topology"])
def test_attention_step_successor_rejects_replaced_session(change) -> None:
    owner = _successor_owner()
    if change == "epoch":
        owner._session_epochs["req"] = 4
    else:
        owner._unified_slot_activations["req"].canonical_topology_id = 12

    assert _build_successor(owner) is None
