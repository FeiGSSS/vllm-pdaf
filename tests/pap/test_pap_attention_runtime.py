# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from vllm.pap.config import PAPConfigError
from vllm.pap.kv.models import PAPAttentionStepContext
from vllm.pap.kv.registry import PAPAttentionRegistry
from vllm.pap.kv.step_context_registry import _PAPAttentionStepContextMixin
from vllm.pap.protocol import (
    PAPAttentionRegistration,
    PAPCudaIPCTensorHandle,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)
from vllm.pap.service import create_app, maybe_start_offload_exec_transport


def test_step_preparation_builds_only_row_stable_graph_slots():
    from threading import RLock

    import torch

    from vllm.pap.attention.compute import prepare_offload_exec_step

    state = SimpleNamespace(block_ids=(5, 7, 9), block_size=16)
    context = SimpleNamespace(
        lock=RLock(),
        prepare_event=None,
        graph_slot_tensor=None,
        request_ids=("a", "b", "c"),
        active_indices=(0, 2),
        prior_seq_lens=(17, 0, 32),
        layer_states={"layer0": (state, state, state)},
        metadata=object(),
        paged_decode_workspace=object(),
        attention_kernel_plan_prepared=True,
    )
    copied = []

    class TensorCache:
        def copy(self, *, kind, values, dtype, device):
            copied.append(kind)
            return torch.tensor(values, dtype=dtype, device=device)

    registry = SimpleNamespace(
        offload_exec_shape_defaults=(1, 1, 1, 1, 1),
        storage_device=torch.device("cpu"),
        get_or_create_attention_step_context=lambda **kwargs: context,
        record_attention_step_slot_plan_build=lambda: None,
        ensure_decode_capacity=lambda request_ids, steps: None,
    )
    descriptor = SimpleNamespace(
        layer_name="layer0",
        metadata_template={"r": ("a", "b", "c"), "s": (18, 1, 33), "a": (1, 1, 1)},
    )
    prepared = prepare_offload_exec_step(
        registry=registry,
        descriptor=descriptor,
        dtype=torch.float16,
        step_tensor_cache=TensorCache(),
    )
    assert prepared.graph_slot_tensor.tolist() == [113, -1, 144]
    assert copied == ["graph_slots"]


def test_step_preflight_allocates_before_metadata_preparation() -> None:
    from vllm.pap.attention.compute import ensure_offload_exec_capacity

    calls = []
    registry = SimpleNamespace(
        ensure_decode_capacity=lambda request_ids, steps: calls.append(
            (request_ids, steps)
        )
    )
    descriptor = SimpleNamespace(
        metadata_template={"r": ("a", "b"), "s": (17, 33), "a": (1, 1)}
    )

    ensure_offload_exec_capacity(registry=registry, descriptor=descriptor)

    assert calls == [(("a", "b"), (17, 33))]


@pytest.mark.parametrize(
    "name,value",
    [
        ("PAP_DECODE_SLOT_PLAN_CACHE_LIMIT", "-1"),
        ("PAP_DECODE_TOKEN_FLUSH_TIMEOUT", "nan"),
        ("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "inf"),
    ],
)
def test_registry_direct_constructor_uses_strict_config_validation(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    with pytest.raises(PAPConfigError, match=name):
        PAPAttentionRegistry()


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


def _registry_with_prefill_layout() -> tuple[PAPAttentionRegistry, str]:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    session = registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req",
            prefill_endpoint="http://127.0.0.1:8100",
            block_size=16,
            max_seq_len=128,
            q_size=4096,
            kv_size=1024,
        )
    )
    registry.register_prefill_kv_catalog(
        descriptor=PAPPrefillKVCacheCatalogDescriptor(
            catalog_id="catalog",
            layer_name="layer0",
            block_size=16,
            num_kv_heads=8,
            layout="NHD",
            kv_cache=PAPCudaIPCTensorHandle(
                dtype="float16",
                shape=(8, 2, 16, 8, 128),
                ipc_handle={"GPU-test": (b"unused",)},
            ),
        ),
        kv_cache=torch.empty((8, 2, 16, 8, 128), dtype=torch.float16),
    )
    registry.install_prefill_kv_session_manifest(
        manifest=PAPPrefillKVSessionManifest(
            request_id="req",
            session_handle=session.prefill_kv_handle,
            catalog_id="catalog",
            prefix_len=17,
            block_ids=(1, 2),
            block_size=16,
            expected_layer_count=1,
            lease_id="lease-0",
            leased_block_ids=(1, 2),
            lease_capacity_tokens=32,
            writable_start_token=17,
            writable_end_token=32,
            generation=0,
        ),
        ready_event=None,
    )
    return registry, session.prefill_kv_handle


def test_ready_endpoint_starts_decode_capacity_prefetch(monkeypatch) -> None:
    registry, session_handle = _registry_with_prefill_layout()
    app = create_app(registry)
    calls = []
    monkeypatch.setattr(
        app.state.pap_runtime,
        "prefetch_decode_capacity",
        lambda request_id, required: calls.append((request_id, required)),
    )
    route = next(
        route
        for route in app.routes
        if route.path == "/v1/pap/attention/sessions/{request_id}/prefill-readiness"
    )

    result = asyncio.run(
        route.endpoint(
            request_id="req",
            expected_prefix_len=17,
            expected_session_handle=session_handle,
            timeout_s=0.0,
        )
    )

    assert result["ready"] is True
    assert calls == [("req", 17)]


def test_attention_prefetches_capacity_and_waits_only_at_boundary(monkeypatch) -> None:
    registry, _session_handle = _registry_with_prefill_layout()
    request_started = Event()
    allow_response = Event()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "allocated": True,
                "block_ids": list(range(1, 9)),
                "writable_end_token": 128,
                "allocation_limit_token": 128,
            }

    calls: list[tuple[str, dict]] = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        request_started.set()
        assert allow_response.wait(timeout=1.0)
        return Response()

    monkeypatch.setattr(
        "vllm.pap.kv.session_registry.httpx.post",
        post,
    )

    registry.ensure_decode_capacity(("req",), (32,))
    assert request_started.wait(timeout=1.0)
    assert len(calls) == 1
    old_topology = registry._unified_paged_kv["req"]["layer0"].slot_topology_id
    assert registry._unified_paged_kv["req"]["layer0"].block_ids == (1, 2)

    errors: list[Exception] = []

    def cross_boundary() -> None:
        try:
            registry.ensure_decode_capacity(("req",), (33,))
        except Exception as exc:
            errors.append(exc)

    waiter = Thread(target=cross_boundary)
    waiter.start()
    assert waiter.is_alive()
    allow_response.set()
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert errors == []

    state = registry._unified_paged_kv["req"]["layer0"]
    assert state.block_ids == tuple(range(1, 9))
    assert state.writable_end_token == 128
    assert state.slot_topology_id != old_topology
    assert len(calls) == 1
    assert calls[0][1]["json"]["required_tokens"] == 32
    assert calls[0][1]["json"]["reserve_tokens"] == 256
    stats = registry.decode_capacity_stats()
    assert stats.pop("decode_capacity_wait_ns") > 0
    assert stats == {
        "decode_capacity_requests": 1,
        "decode_capacity_prefetches": 1,
        "decode_capacity_installs": 1,
        "decode_capacity_blocks_added": 6,
        "decode_capacity_waits": 1,
        "decode_capacity_failures": 0,
        "decode_capacity_pending": 0,
    }


def test_prefill_revocation_fences_late_manifest_and_allows_new_generation() -> None:
    registry, session_handle = _registry_with_prefill_layout()

    result = registry.revoke_prefill_kv(session_handle=session_handle, generation=0)
    assert result == {"revoked": True, "generation": 1}
    assert registry._unified_paged_kv.get("req") is None

    stale = PAPPrefillKVSessionManifest(
        request_id="req",
        session_handle=session_handle,
        catalog_id="catalog",
        prefix_len=17,
        block_ids=(1, 2),
        block_size=16,
        expected_layer_count=1,
        lease_id="lease-0",
        leased_block_ids=(1, 2),
        lease_capacity_tokens=32,
        writable_start_token=17,
        writable_end_token=32,
        generation=0,
    )
    assert (
        registry.install_prefill_kv_session_manifest(manifest=stale, ready_event=None)
        == 0
    )

    current = replace(stale, lease_id="lease-1", generation=1)
    assert (
        registry.install_prefill_kv_session_manifest(manifest=current, ready_event=None)
        == 17
    )


def test_prefill_revocation_rejects_layout_already_claimed_by_decode() -> None:
    registry, session_handle = _registry_with_prefill_layout()
    registry.get_unified_paged_states(session_request_ids=("req",), layer_name="layer0")

    with pytest.raises(RuntimeError, match="claimed"):
        registry.revoke_prefill_kv(session_handle=session_handle, generation=0)


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
