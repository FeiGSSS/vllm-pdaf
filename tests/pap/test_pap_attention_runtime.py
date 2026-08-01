# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention runtime, service composition, and unified-KV integration tests."""

import base64
import hashlib
import inspect
import logging
import sys
import time
from contextlib import suppress
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from httpx import ASGITransport, AsyncClient, Response

from vllm.pap.attention import PAPAttentionDispatcher, PAPAttentionWorkItem
from vllm.pap.attention import compute as attention_compute_module
from vllm.pap.attention import execution as attention_runtime_module
from vllm.pap.attention import peers as attention_peers_module
from vllm.pap.attention.compute import (
    _offload_exec_batch_rows,
    compute_offload_exec_batch_output,
    prepare_offload_exec_step,
)
from vllm.pap.attention.execution import (
    _execute_offload_exec_work_item,
    _execute_offload_exec_work_items,
    _offload_exec_work_item_compatibility_key,
    run_offload_exec_mailbox_loop,
    run_offload_exec_mailbox_receiver_loop,
)
from vllm.pap.config import PAPOffloadExecTransport
from vllm.pap.kv import (
    PAPAttentionRegistry,
    PAPUnifiedPagedKVState,
    build_unified_paged_flash_step_metadata,
)
from vllm.pap.kv import metadata as kv_metadata_module
from vllm.pap.kv import registry as kv_registry_module
from vllm.pap.protocol import (
    PAPAttentionRegistration,
    PAPCudaIPCTensorHandle,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPPrefillKVCacheCatalogDescriptor,
    PAPPrefillKVSessionManifest,
)
from vllm.pap.service import (
    compute_binary_attention_response,
    create_app,
    maybe_start_offload_exec_transport,
    parse_args,
)


class _ASGITestClient:
    def __init__(self, app: Any) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs: Any) -> Response:
        async def run_request() -> Response:
            transport = ASGITransport(app=self.app)
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return anyio.run(run_request)

    def get(self, url: str, **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> Response:
        return self.request("DELETE", url, **kwargs)


def _install_unified_activation(
    registry: PAPAttentionRegistry,
    *,
    request_id: str,
    layer_names: tuple[str, ...],
    kv_cache: Any,
    block_ids: tuple[int, ...],
    seq_len: int,
    block_size: int = 4,
) -> None:
    session_request_id = registry.resolve_session_request_id(request_id)
    assert session_request_id is not None
    capacity_tokens = len(block_ids) * block_size
    with registry._lock:
        existing_activation = registry._unified_slot_activations.get(session_request_id)
        if (
            existing_activation is not None
            and existing_activation.prefix_len == seq_len
        ):
            existing_activation.expected_layers = None
        states = registry._unified_paged_kv.setdefault(session_request_id, {})
        for layer_name in layer_names:
            state = PAPUnifiedPagedKVState(
                kv_cache=kv_cache,
                block_ids=block_ids,
                prefix_len=seq_len,
                seq_len=seq_len,
                capacity_tokens=capacity_tokens,
                writable_start_token=seq_len,
                writable_end_token=capacity_tokens,
                lease_id=f"lease-{request_id}",
                block_size=block_size,
                num_kv_heads=1,
                layout="NHD",
            )
            registry._record_unified_slot_topology_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
                state=state,
            )
            states[layer_name] = state
        activation = registry._unified_slot_activations[session_request_id]
        activation.expected_layers = frozenset(states)
        activation.complete = activation.expected_layers.issubset(
            activation.layer_observations
        )
        registry._session_manifest_prefix_lens[session_request_id] = seq_len
        registry._session_lease_ids[session_request_id] = f"lease-{request_id}"
        registry._session_leased_block_ids[session_request_id] = block_ids
        registry._session_lease_capacity_tokens[session_request_id] = capacity_tokens
        session = registry._sessions[session_request_id]
        session.prefix_len = seq_len
        session.seq_len = max(session.seq_len, seq_len)
        session.block_ids = block_ids
        for layer_name in layer_names:
            registry._mark_prefill_ready_locked(
                session_request_id=session_request_id,
                layer_name=layer_name,
            )


def _catalog_descriptor(
    *,
    layer_name: str,
    shape: tuple[int, ...],
) -> PAPPrefillKVCacheCatalogDescriptor:
    return PAPPrefillKVCacheCatalogDescriptor(
        catalog_id="prefill-test",
        layer_name=layer_name,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
        kv_cache=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=shape,
            ipc_handle={"gpu-test": ("storage",)},
        ),
    )


def _step_metadata_state(
    torch_module: Any,
    *,
    block_ids: tuple[int, ...],
    topology_id: int,
) -> PAPUnifiedPagedKVState:
    return PAPUnifiedPagedKVState(
        kv_cache=torch_module.zeros((2, 2, 4, 1, 2)),
        block_ids=block_ids,
        prefix_len=8,
        seq_len=8,
        capacity_tokens=32,
        writable_start_token=8,
        writable_end_token=32,
        lease_id=f"lease-{topology_id}",
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
        slot_generation=1 if topology_id else 0,
        slot_topology_id=topology_id,
    )


def test_step_metadata_reuses_static_table_with_dynamic_sequence_lengths() -> None:
    import torch

    kv_metadata_module.reset_unified_paged_flash_metadata_cache()
    states = [
        _step_metadata_state(torch, block_ids=(3, 5), topology_id=101),
        _step_metadata_state(torch, block_ids=(7,), topology_id=102),
    ]

    first = build_unified_paged_flash_step_metadata(
        states=states,
        seq_lens=(8, 12),
        device=torch.device("cpu"),
    )
    second = build_unified_paged_flash_step_metadata(
        states=states,
        seq_lens=(9, 13),
        device=torch.device("cpu"),
    )

    assert first.block_table.tolist() == [[3, 5], [7, 7]]
    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    assert first.seq_lens.tolist() == [8, 12]
    assert second.seq_lens.tolist() == [9, 13]
    assert second.cu_seqlens_q.data_ptr() == first.cu_seqlens_q.data_ptr()
    assert kv_metadata_module.unified_paged_flash_metadata_cache_stats() == {
        "hits": 1,
        "misses": 1,
        "entries": 1,
        "fast_key_lookups": 2,
        "fast_key_hits": 1,
        "full_key_scans": 1,
        "block_ids_scanned": 3,
    }


def test_step_metadata_falls_back_for_unknown_ragged_topology() -> None:
    import torch

    kv_metadata_module.reset_unified_paged_flash_metadata_cache()
    states = [
        _step_metadata_state(torch, block_ids=(3,), topology_id=0),
        _step_metadata_state(torch, block_ids=(7, 9), topology_id=0),
    ]

    first = build_unified_paged_flash_step_metadata(
        states=states,
        seq_lens=(8, 12),
        device=torch.device("cpu"),
    )
    second = build_unified_paged_flash_step_metadata(
        states=states,
        seq_lens=(9, 13),
        device=torch.device("cpu"),
    )

    assert first.block_table.tolist() == [[3, 3], [7, 9]]
    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    stats = kv_metadata_module.unified_paged_flash_metadata_cache_stats()
    assert stats["fast_key_lookups"] == 0
    assert stats["full_key_scans"] == 2
    assert stats["block_ids_scanned"] == 6


def test_step_metadata_updates_reusable_block_table_buffer() -> None:
    import torch

    buffer = kv_metadata_module.PAPPagedBlockTableBuffer(row_capacity=4)
    first_states = [
        _step_metadata_state(torch, block_ids=(3, 5), topology_id=101),
        _step_metadata_state(torch, block_ids=(7,), topology_id=102),
    ]
    second_states = [
        _step_metadata_state(torch, block_ids=(11,), topology_id=103),
        _step_metadata_state(torch, block_ids=(13, 17), topology_id=104),
    ]

    first = build_unified_paged_flash_step_metadata(
        states=first_states,
        seq_lens=(8, 12),
        device=torch.device("cpu"),
        block_table_buffer=buffer,
    )
    second = build_unified_paged_flash_step_metadata(
        states=second_states,
        seq_lens=(9, 13),
        device=torch.device("cpu"),
        block_table_buffer=buffer,
    )

    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    assert second.block_table.tolist() == [[11, 11], [13, 17]]


def test_sealed_prefill_manifest_installs_all_layers_atomically() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    session = registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-sealed",
            conversation_id="conv-sealed",
            prefill_endpoint="http://localhost:8100",
            block_size=4,
            max_seq_len=32,
        )
    )
    shape = (8, 2, 4, 1, 2)
    for layer_name in ("layer0", "layer1"):
        assert registry.register_prefill_kv_catalog(
            descriptor=_catalog_descriptor(
                layer_name=layer_name,
                shape=shape,
            ),
            kv_cache=torch.zeros(shape),
        )

    manifest = PAPPrefillKVSessionManifest(
        request_id="req-sealed",
        session_handle=session.prefill_kv_handle,
        catalog_id="prefill-test",
        prefix_len=5,
        block_ids=(1, 2, 3),
        block_size=4,
        expected_layer_count=2,
        lease_id="lease-sealed",
        leased_block_ids=(1, 2, 3),
        lease_capacity_tokens=9,
        writable_start_token=5,
        writable_end_token=9,
    )
    assert (
        registry.install_prefill_kv_session_manifest(
            manifest=manifest,
            ready_event=None,
        )
        == 5
    )

    states = registry.get_unified_paged_states(
        session_request_ids=("req-sealed",),
        layer_name="layer0",
    )
    assert states is not None
    assert states[0].block_ids == (1, 2, 3)
    assert states[0].seq_len == 5
    assert states[0].writable_start_token == 5
    layer1 = registry.get_unified_paged_states(
        session_request_ids=("req-sealed",),
        layer_name="layer1",
    )
    assert layer1 is not None
    assert layer1[0].block_ids == states[0].block_ids


def test_sealed_prefill_manifest_updates_before_claim_and_freezes_after() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    session = registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-chunked",
            conversation_id="conv-chunked",
            prefill_endpoint="http://localhost:8100",
            block_size=4,
            max_seq_len=32,
        )
    )
    shape = (8, 2, 4, 1, 2)
    registry.register_prefill_kv_catalog(
        descriptor=_catalog_descriptor(layer_name="layer0", shape=shape),
        kv_cache=torch.zeros(shape),
    )

    def manifest(prefix_len: int) -> PAPPrefillKVSessionManifest:
        return PAPPrefillKVSessionManifest(
            request_id="req-chunked",
            session_handle=session.prefill_kv_handle,
            catalog_id="prefill-test",
            prefix_len=prefix_len,
            block_ids=(1, 2, 3),
            block_size=4,
            expected_layer_count=1,
            lease_id="lease-chunked",
            leased_block_ids=(1, 2, 3),
            lease_capacity_tokens=10,
            writable_start_token=prefix_len,
            writable_end_token=10,
        )

    registry.install_prefill_kv_session_manifest(
        manifest=manifest(5),
        ready_event=None,
    )
    registry.install_prefill_kv_session_manifest(
        manifest=manifest(7),
        ready_event=None,
    )
    states = registry.get_unified_paged_states(
        session_request_ids=("req-chunked",),
        layer_name="layer0",
    )
    assert states is not None
    assert states[0].prefix_len == 7

    with pytest.raises(RuntimeError, match="after Decode claimed"):
        registry.install_prefill_kv_session_manifest(
            manifest=manifest(8),
            ready_event=None,
        )


def test_sealed_prefill_manifest_rejects_stale_session_generation() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registration = PAPAttentionRegistration(
        request_id="req-aba",
        conversation_id="conv-aba",
        prefill_endpoint="http://localhost:8100",
        prefix_len=5,
        block_size=4,
    )
    old_session = registry.register_prefill_kv(registration)
    shape = (8, 2, 4, 1, 2)
    registry.register_prefill_kv_catalog(
        descriptor=_catalog_descriptor(layer_name="layer0", shape=shape),
        kv_cache=torch.zeros(shape),
    )
    new_session = registry.register_prefill_kv(registration)

    assert old_session.prefill_kv_handle != new_session.prefill_kv_handle

    def manifest(
        session_handle: str,
        lease_id: str,
    ) -> PAPPrefillKVSessionManifest:
        return PAPPrefillKVSessionManifest(
            request_id=registration.request_id,
            session_handle=session_handle,
            catalog_id="prefill-test",
            prefix_len=5,
            block_ids=(0, 1),
            block_size=4,
            expected_layer_count=1,
            lease_id=lease_id,
            leased_block_ids=(0, 1),
            lease_capacity_tokens=8,
            writable_start_token=5,
            writable_end_token=8,
        )

    with pytest.raises(KeyError, match="pap-session-1"):
        registry.install_prefill_kv_session_manifest(
            manifest=manifest(old_session.prefill_kv_handle, "lease-stale"),
            ready_event=None,
        )

    assert (
        registry.install_prefill_kv_session_manifest(
            manifest=manifest(new_session.prefill_kv_handle, "lease-current"),
            ready_event=None,
        )
        == 5
    )
    state = registry._unified_paged_kv[registration.request_id]["layer0"]
    assert state.lease_id == "lease-current"


def test_sealed_prefill_manifest_requires_complete_catalog() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    session = registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-incomplete",
            conversation_id="conv-incomplete",
            prefill_endpoint="http://localhost:8100",
            block_size=4,
        )
    )
    shape = (8, 2, 4, 1, 2)
    registry.register_prefill_kv_catalog(
        descriptor=_catalog_descriptor(layer_name="layer0", shape=shape),
        kv_cache=torch.zeros(shape),
    )
    manifest = PAPPrefillKVSessionManifest(
        request_id="req-incomplete",
        session_handle=session.prefill_kv_handle,
        catalog_id="prefill-test",
        prefix_len=5,
        block_ids=(1, 2),
        block_size=4,
        expected_layer_count=2,
        lease_id="lease-incomplete",
        leased_block_ids=(1, 2),
        lease_capacity_tokens=8,
        writable_start_token=5,
        writable_end_token=8,
    )

    with pytest.raises(RuntimeError, match="layer count mismatch"):
        registry.install_prefill_kv_session_manifest(
            manifest=manifest,
            ready_event=None,
        )


def test_sealed_manifests_preserve_shared_prefix_and_private_tails() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    sessions = {}
    for request_id in ("req-shared-a", "req-shared-b"):
        sessions[request_id] = registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id=request_id,
                conversation_id=request_id,
                prefill_endpoint="http://localhost:8100",
                block_size=4,
                max_seq_len=32,
            )
        )
    shape = (8, 2, 4, 1, 2)
    registry.register_prefill_kv_catalog(
        descriptor=_catalog_descriptor(layer_name="layer0", shape=shape),
        kv_cache=torch.zeros(shape),
    )

    for request_id, private_tail in (
        ("req-shared-a", 3),
        ("req-shared-b", 4),
    ):
        block_ids = (1, 2, private_tail)
        registry.install_prefill_kv_session_manifest(
            manifest=PAPPrefillKVSessionManifest(
                request_id=request_id,
                session_handle=sessions[request_id].prefill_kv_handle,
                catalog_id="prefill-test",
                prefix_len=8,
                block_ids=block_ids,
                block_size=4,
                expected_layer_count=1,
                lease_id=f"lease-{request_id}",
                leased_block_ids=block_ids,
                lease_capacity_tokens=12,
                writable_start_token=8,
                writable_end_token=12,
            ),
            ready_event=None,
        )

    state_a = registry.get_unified_paged_states(
        session_request_ids=("req-shared-a",),
        layer_name="layer0",
    )
    state_b = registry.get_unified_paged_states(
        session_request_ids=("req-shared-b",),
        layer_name="layer0",
    )
    assert state_a is not None and state_b is not None
    assert state_a[0].block_ids[:2] == state_b[0].block_ids[:2] == (1, 2)
    assert state_a[0].block_ids[2] != state_b[0].block_ids[2]
    assert state_a[0].writable_start_token == 8
    assert state_b[0].writable_start_token == 8


def test_attention_service_rejects_legacy_prefill_wire_command() -> None:
    from vllm.pap.protocol.wire import serialize_tensor_bundle

    payload = serialize_tensor_bundle(
        {"command": "import_prefill_paged_kv_ipc", "descriptor": {}},
        {},
    )

    with pytest.raises(ValueError, match="sealed KV handoff"):
        compute_binary_attention_response(PAPAttentionRegistry(), payload)


def test_offload_exec_batch_rows_uses_template_without_items() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template={
            "r": ("req-a", "req-b"),
            "s": (7, 8),
            "a": (0.125, 0.125),
        },
    )

    assert _offload_exec_batch_rows(descriptor) == (
        ("req-a", "req-b"),
        (7, 8),
        (0.125, 0.125),
    )


def test_attention_service_parses_offload_exec_zmq_port(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["pap-attention", "--offload-exec-zmq-port", "8401"],
    )

    assert parse_args().offload_exec_zmq_port == 8401


def test_attention_executor_skips_offload_exec_when_zmq_port_is_none(
    monkeypatch,
) -> None:
    app = create_app()
    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=None)
    assert app.state.pap_peer_manager.initial_transport is None


def test_attention_executor_starts_offload_exec_transport(monkeypatch) -> None:
    fake_transport = object()

    def fake_build_transport(**kwargs):
        fake_build_transport.kwargs = kwargs
        return fake_transport

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "2")
    monkeypatch.setattr(
        "vllm.pap.attention.peers.build_offload_exec_transport",
        fake_build_transport,
    )
    app = create_app()

    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=10300)

    assert app.state.pap_peer_manager.initial_transport is fake_transport
    assert fake_build_transport.kwargs == {
        "transport": PAPOffloadExecTransport.LOCAL_FAST,
        "actor_id": "attention",
        "local_rank": 2,
    }


def test_attention_executor_binds_each_projection_to_distinct_transport(
    monkeypatch,
) -> None:
    class FakeTransport:
        def __init__(self, actor_id: str, local_rank: int) -> None:
            self.actor_id = actor_id
            self.local_rank = local_rank
            self.local_agent_metadata = f"local:{actor_id}".encode()
            self.bound_peers: list[bytes] = []

        def bind_peer(self, peer_metadata: bytes) -> None:
            self.bound_peers.append(peer_metadata)

    transports: list[FakeTransport] = []
    loops_started: list[FakeTransport] = []
    loop_peer_ids: list[str] = []
    loops_ready = Event()

    def fake_build_transport(
        *,
        actor_id: str,
        local_rank: int,
        transport,
    ) -> FakeTransport:
        assert transport.value == "local_fast"
        fake_transport = FakeTransport(actor_id, local_rank)
        transports.append(fake_transport)
        return fake_transport

    def fake_mailbox_loop(*, registry, transport, peer_id) -> None:
        loops_started.append(transport)
        loop_peer_ids.append(peer_id)
        if len(loops_started) == 2:
            loops_ready.set()

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "2")
    monkeypatch.setenv("PAP_NIXL_MAILBOX_ACTOR_ID", "attention-4")
    monkeypatch.setattr(
        attention_peers_module,
        "build_offload_exec_transport",
        fake_build_transport,
    )
    monkeypatch.setattr(
        attention_peers_module,
        "run_offload_exec_mailbox_loop",
        fake_mailbox_loop,
    )

    app = create_app()
    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=10300)
    client = _ASGITestClient(app)
    peers = (b"projection-a", b"projection-b")
    responses = []
    for peer_metadata in peers:
        responses.append(
            client.post(
                "/v1/pap/attention/offload-exec-mailbox/bind",
                json={"agent_metadata_b64": base64.b64encode(peer_metadata).decode()},
            )
        )

    assert all(response.status_code == 200 for response in responses)
    assert len(transports) == 2
    assert transports[0].actor_id == "attention-4"
    peer_b_hash = hashlib.sha1(peers[1]).hexdigest()[:16]
    assert transports[1].actor_id == f"attention-4-{peer_b_hash}"
    assert transports[0].local_rank == transports[1].local_rank == 2
    assert transports[0].bound_peers == [peers[0]]
    assert transports[1].bound_peers == [peers[1]]
    assert loops_ready.wait(timeout=1.0)
    assert set(loops_started) == set(transports)
    assert set(loop_peer_ids) == {hashlib.sha1(peer).hexdigest()[:16] for peer in peers}

    rebound = client.post(
        "/v1/pap/attention/offload-exec-mailbox/bind",
        json={"agent_metadata_b64": base64.b64encode(peers[0]).decode()},
    )
    assert rebound.status_code == 200
    assert len(transports) == 2
    assert transports[0].bound_peers == [peers[0]]


def test_attention_executor_central_mode_shares_one_dispatcher(
    monkeypatch,
) -> None:
    class FakeTransport:
        def __init__(self, actor_id: str, local_rank: int) -> None:
            self.actor_id = actor_id
            self.local_rank = local_rank
            self.local_agent_metadata = f"local:{actor_id}".encode()

        def bind_peer(self, peer_metadata: bytes) -> None:
            pass

    transports = []
    receiver_contexts = []
    receivers_ready = Event()

    def fake_build_transport(*, actor_id, local_rank, transport):
        assert transport.value == "local_fast"
        fake_transport = FakeTransport(actor_id, local_rank)
        transports.append(fake_transport)
        return fake_transport

    def fake_receiver_loop(*, registry, transport, dispatcher, peer_id):
        receiver_contexts.append((transport, dispatcher, peer_id))
        if len(receiver_contexts) == 2:
            receivers_ready.set()

    monkeypatch.setenv("PAP_TOPOLOGY", "1pa2p")
    monkeypatch.setenv("PAP_PA_COUNT", "1")
    monkeypatch.setenv("PAP_PROJECTION_COUNT", "2")
    monkeypatch.setattr(
        attention_peers_module,
        "build_offload_exec_transport",
        fake_build_transport,
    )
    monkeypatch.setattr(
        attention_peers_module,
        "run_offload_exec_mailbox_receiver_loop",
        fake_receiver_loop,
    )
    app = create_app()
    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=10300)
    client = _ASGITestClient(app)
    peers = (b"projection-a", b"projection-b")

    for peer_index, peer_metadata in enumerate(peers):
        source_id = f"projection-{peer_index}"
        response = client.post(
            "/v1/pap/attention/offload-exec-mailbox/bind",
            json={
                "agent_metadata_b64": base64.b64encode(peer_metadata).decode(),
                "source_id": source_id,
            },
        )
        assert response.status_code == 200
        response = client.post(
            "/v1/pap/attention/offload-exec-mailbox/activity",
            json={
                "source_id": source_id,
                "membership_generation": 1,
                "active": True,
            },
        )
        assert response.status_code == 200

    assert receivers_ready.wait(timeout=1.0)
    dispatcher = app.state.pap_peer_manager.dispatcher
    assert dispatcher is not None
    assert {id(context[1]) for context in receiver_contexts} == {id(dispatcher)}
    assert {context[2] for context in receiver_contexts} == {
        "projection-0",
        "projection-1",
    }
    stats = client.get("/v1/pap/attention/stats").json()
    assert stats["attention_dispatch_mode"] == "central_combine"
    assert stats["dispatcher_running"] is True
    assert stats["dispatcher_expected_group_size"] == 2
    assert stats["dispatcher_preferred_peer_id"] == "projection-0"
    dispatcher.stop(drain=True, timeout=1.0)


def test_attention_executor_rejects_unknown_dispatch_mode(monkeypatch) -> None:
    monkeypatch.setenv("PAP_ATTENTION_DISPATCH_MODE", "separate_cohorts")
    with pytest.raises(ValueError, match="was removed"):
        create_app()


def test_attention_executor_builds_central_combine_dispatcher(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAP_TOPOLOGY", "1pa2p")
    monkeypatch.setenv("PAP_PA_COUNT", "1")
    monkeypatch.setenv("PAP_PROJECTION_COUNT", "2")

    app = create_app()

    assert app.state.pap_peer_manager.dispatch_mode == "central_combine"
    dispatcher = app.state.pap_peer_manager.dispatcher
    assert dispatcher is not None
    assert dispatcher._batch_handler is not None
    assert dispatcher._compatibility_key is not None
    stats = dispatcher.stats()
    assert stats["dispatcher_coalesce_timeout_us"] == 200
    assert stats["dispatcher_expected_group_size"] == 1


def test_attention_executor_tracks_active_projection_membership(monkeypatch) -> None:
    monkeypatch.setenv("PAP_TOPOLOGY", "1pa2p")
    monkeypatch.setenv("PAP_PA_COUNT", "1")
    monkeypatch.setenv("PAP_PROJECTION_COUNT", "2")
    app = create_app()
    client = _ASGITestClient(app)

    response = client.post(
        "/v1/pap/attention/offload-exec-mailbox/activity",
        json={
            "source_id": "projection-1-r0",
            "active": True,
            "membership_generation": 1,
        },
    )
    assert response.status_code == 200
    response = client.post(
        "/v1/pap/attention/offload-exec-mailbox/activity",
        json={
            "source_id": "projection-0-r0",
            "active": True,
            "membership_generation": 1,
        },
    )
    assert response.status_code == 200

    stats = client.get("/v1/pap/attention/stats").json()
    assert stats["attention_active_peer_tracking"] is True
    assert stats["attention_active_source_ids"] == [
        "projection-0-r0",
        "projection-1-r0",
    ]
    assert stats["attention_membership_generations"] == {
        "projection-0-r0": 1,
        "projection-1-r0": 1,
    }
    assert stats["dispatcher_expected_group_size"] == 2
    assert stats["dispatcher_preferred_peer_id"] == "projection-0-r0"

    response = client.post(
        "/v1/pap/attention/offload-exec-mailbox/activity",
        json={
            "source_id": "projection-0-r0",
            "active": False,
            "membership_generation": 2,
        },
    )
    assert response.status_code == 200
    assert response.json()["applied"] is True

    stale = client.post(
        "/v1/pap/attention/offload-exec-mailbox/activity",
        json={
            "source_id": "projection-0-r0",
            "active": True,
            "membership_generation": 1,
        },
    )
    assert stale.status_code == 200
    assert stale.json() == {
        "source_id": "projection-0-r0",
        "active": False,
        "membership_generation": 2,
        "applied": False,
        "stale": True,
    }

    stats = client.get("/v1/pap/attention/stats").json()
    assert stats["attention_active_source_ids"] == ["projection-1-r0"]
    assert stats["dispatcher_expected_group_size"] == 1
    assert stats["dispatcher_preferred_peer_id"] == "projection-1-r0"


def test_mailbox_bind_sends_stable_projection_source_id(monkeypatch) -> None:
    import json

    from vllm.pap.attention import client as attention_client

    response_metadata = b"attention-metadata"
    response_body = json.dumps(
        {"agent_metadata_b64": base64.b64encode(response_metadata).decode("ascii")}
    ).encode("utf-8")

    class FakeSocket:
        def __init__(self) -> None:
            self.sent = b""
            self.responses = [
                b"HTTP/1.1 200 OK\r\nContent-Length: "
                + str(len(response_body)).encode("ascii")
                + b"\r\n\r\n"
                + response_body,
                b"",
            ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _timeout):
            pass

        def sendall(self, payload):
            self.sent += payload

        def recv(self, _size):
            return self.responses.pop(0)

    fake_socket = FakeSocket()
    monkeypatch.setattr(
        attention_client.socket,
        "create_connection",
        lambda *_args, **_kwargs: fake_socket,
    )

    result = attention_client.bind_offload_exec_mailbox(
        attention_endpoint="http://127.0.0.1:8300",
        local_agent_metadata=b"projection-metadata",
        source_id="projection-0-r0",
    )

    request_body = fake_socket.sent.partition(b"\r\n\r\n")[2]
    assert json.loads(request_body)["source_id"] == "projection-0-r0"
    assert result == response_metadata


def test_mailbox_activity_sends_membership_generation(monkeypatch) -> None:
    import json

    from vllm.pap.attention import client as attention_client

    response_body = json.dumps({"applied": True}).encode("utf-8")

    class FakeSocket:
        def __init__(self) -> None:
            self.sent = b""
            self.responses = [
                b"HTTP/1.1 200 OK\r\nContent-Length: "
                + str(len(response_body)).encode("ascii")
                + b"\r\n\r\n"
                + response_body,
                b"",
            ]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _timeout):
            pass

        def sendall(self, payload):
            self.sent += payload

        def recv(self, _size):
            return self.responses.pop(0)

    fake_socket = FakeSocket()
    monkeypatch.setattr(
        attention_client.socket,
        "create_connection",
        lambda *_args, **_kwargs: fake_socket,
    )

    result = attention_client.update_offload_exec_mailbox_activity(
        attention_endpoint="http://127.0.0.1:8300",
        source_id="projection-0-r0",
        active=False,
        membership_generation=7,
    )

    header, _, request_body = fake_socket.sent.partition(b"\r\n\r\n")
    assert b"POST /v1/pap/attention/offload-exec-mailbox/activity " in header
    assert json.loads(request_body) == {
        "source_id": "projection-0-r0",
        "active": False,
        "membership_generation": 7,
    }
    assert result == {"applied": True}


def test_attention_executor_rejects_nccl_offload_exec_transport(monkeypatch) -> None:
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_TRANSPORT", "nccl")

    with pytest.raises(ValueError, match="PAP_OFFLOAD_EXEC_TRANSPORT"):
        create_app()


def test_unified_offload_exec_commit_waits_for_async_decode_token(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")

    commits = []

    class FakeCommitClient:
        def commit(
            self,
            *,
            request_id,
            session_request_id,
            new_seq_len,
            new_token_ids,
            endpoint,
        ):
            commits.append(
                (
                    request_id,
                    session_request_id,
                    new_seq_len,
                    tuple(new_token_ids),
                    endpoint,
                )
            )

    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
        )
    )
    registry._unified_paged_kv["req-a"] = {
        "layer0": PAPUnifiedPagedKVState(
            kv_cache=torch.zeros((2, 2, 4, 1, 2)),
            block_ids=(0,),
            prefix_len=1,
            seq_len=1,
            capacity_tokens=4,
            writable_start_token=1,
            writable_end_token=4,
            lease_id="lease-a",
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
        )
    }
    monkeypatch.setattr(
        registry,
        "append_decode_kv_to_unified_prefill_cache",
        lambda **kwargs: 1,
    )
    monkeypatch.setattr(
        attention_compute_module,
        "_compute_unified_paged_attention_batch",
        lambda **kwargs: torch.tensor([[2.0, 0.0]]),
    )
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="cmpl-req-a-0",
                layer_name="layer0",
                step=2,
                scale=1.0,
            ),
        ),
    )

    compute_offload_exec_batch_output(
        registry=registry,
        descriptor=descriptor,
        qkv_batch=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
    )

    assert commits == []
    assert registry.decode_token_stats()["decode_token_pending_kv"] == 1
    assert (
        registry.record_decode_token(
            request_id="cmpl-req-a-0",
            new_seq_len=2,
            token_id=42,
        )
        == "matched"
    )
    assert commits == []
    assert registry._decode_token_committer.flush_request("req-a")
    assert commits == [
        (
            "cmpl-req-a-0",
            "req-a",
            2,
            (42,),
            "http://localhost:8100/v1/pap/prefill/decode-commit",
        )
    ]


def test_attention_step_context_reuses_plan_and_publishes_once(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    kv_metadata_module.reset_unified_paged_flash_metadata_cache()

    commits = []

    class FakeCommitClient:
        def commit(
            self,
            *,
            request_id,
            session_request_id,
            new_seq_len,
            new_token_ids,
            endpoint,
        ):
            commits.append((request_id, new_seq_len, tuple(new_token_ids), endpoint))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-step",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
        )
    )
    _install_unified_activation(
        registry,
        request_id="req-step",
        layer_names=("layer0", "layer1"),
        kv_cache=torch.zeros((2, 2, 4, 1, 2)),
        block_ids=(0,),
        seq_len=1,
    )
    reshape_calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: reshape_calls.append(args),
        raising=False,
    )
    workspace_builds = []
    workspaces = []

    original_workspace_builder = attention_compute_module.build_paged_decode_workspace

    def build_workspace(query):
        workspace = original_workspace_builder(query)
        workspace_builds.append(workspace)
        return workspace

    def compute_attention(**kwargs):
        workspaces.append(kwargs["workspace"])
        return torch.tensor([[2.0, 0.0]])

    monkeypatch.setattr(
        attention_compute_module,
        "build_paged_decode_workspace",
        build_workspace,
    )
    monkeypatch.setattr(
        attention_compute_module,
        "_compute_unified_paged_attention_batch",
        compute_attention,
    )
    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )

    assert (
        registry.record_decode_token(
            request_id="req-step",
            new_seq_len=2,
            token_id=42,
        )
        == "pending"
    )
    qkv = torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])
    prepare_offload_exec_step(
        registry=registry,
        descriptor=PAPOffloadExecBatchDescriptor(
            layer_name="layer0",
            items=(
                PAPOffloadExecDescriptor(
                    request_id="req-step",
                    layer_name="layer0",
                    step=2,
                    scale=1.0,
                ),
            ),
        ),
        dtype=qkv.dtype,
    )
    for layer_name in ("layer0", "layer1"):
        descriptor = PAPOffloadExecBatchDescriptor(
            layer_name=layer_name,
            items=(
                PAPOffloadExecDescriptor(
                    request_id="req-step",
                    layer_name=layer_name,
                    step=2,
                    scale=1.0,
                ),
            ),
        )
        compute_offload_exec_batch_output(
            registry=registry,
            descriptor=descriptor,
            qkv_batch=qkv,
        )
        if layer_name == "layer0":
            assert commits == []

    assert len(reshape_calls) == 2
    assert reshape_calls[1][4].data_ptr() == reshape_calls[0][4].data_ptr()
    assert len(workspace_builds) == 1
    assert len(workspaces) == 2
    assert all(workspace is workspace_builds[0] for workspace in workspaces)
    assert commits == []
    assert registry._decode_token_committer.flush_request("req-step")
    assert commits == [
        (
            "req-step",
            2,
            (42,),
            "http://localhost:8100/v1/pap/prefill/decode-commit",
        )
    ]
    assert registry.attention_step_context_stats() == {
        "step_context_hits": 2,
        "step_context_misses": 1,
        "step_context_entries": 1,
        "step_slot_plan_builds": 1,
        "step_metadata_builds": 1,
        "step_kv_ready_publishes": 1,
    }
    token_stats = registry.decode_token_stats()
    assert token_stats["decode_kv_ready"] == 1
    assert token_stats["decode_token_duplicates"] == 0


def test_decode_token_http_endpoint_is_idempotent_and_rejects_mismatch() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    client = _ASGITestClient(create_app(registry))
    payload = {"request_id": "req-a", "new_seq_len": 2, "token_id": 42}

    first = client.post("/v1/pap/attention/decode-token", json=payload)
    duplicate = client.post("/v1/pap/attention/decode-token", json=payload)
    mismatch = client.post(
        "/v1/pap/attention/decode-token",
        json={**payload, "token_id": 43},
    )

    assert first.status_code == 200
    assert first.json()["status"] == "pending"
    assert duplicate.status_code == 200
    assert duplicate.json()["status"] == "duplicate"
    assert mismatch.status_code == 409


def test_decode_token_batch_http_endpoint_records_one_forward() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    for request_id in ("req-a", "req-b"):
        registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id=request_id,
                conversation_id="conv",
                prefill_endpoint="http://localhost:8100",
            )
        )
    client = _ASGITestClient(create_app(registry))

    response = client.post(
        "/v1/pap/attention/decode-tokens",
        json={
            "tokens": [
                {"request_id": "req-a", "new_seq_len": 2, "token_id": 42},
                {"request_id": "req-b", "new_seq_len": 3, "token_id": 43},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "results": [
            {"request_id": "req-a", "new_seq_len": 2, "status": "pending"},
            {"request_id": "req-b", "new_seq_len": 3, "status": "pending"},
        ],
    }


def test_decode_token_endpoint_accepts_released_handle_only() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    session = registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    client = _ASGITestClient(create_app(registry))

    assert registry.release_session("req-a")
    released = client.post(
        "/v1/pap/attention/decode-tokens",
        json={
            "tokens": [
                {
                    "request_id": session.prefill_kv_handle,
                    "new_seq_len": 2,
                    "token_id": 42,
                }
            ]
        },
    )
    unknown = client.post(
        "/v1/pap/attention/decode-token",
        json={"request_id": "never-seen", "new_seq_len": 2, "token_id": 42},
    )

    assert released.status_code == 200
    assert released.json()["results"][0]["status"] == "released"
    assert unknown.status_code == 404


def test_attention_release_waits_for_kv_ready_token_but_not_final_token(
    monkeypatch,
) -> None:
    commits = []

    class FakeCommitClient:
        def commit(self, **kwargs):
            commits.append(kwargs)

        def flush_request(self, request_id):
            return True

        def forget_request(self, request_id):
            return None

    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    registry._decode_token_committer.record_kv_ready(
        request_id="req-a",
        new_seq_len=2,
        endpoint="http://localhost:8100/v1/pap/prefill/decode-commit",
        commit_request_id="req-a",
    )

    released: list[bool] = []
    release_thread = Thread(
        target=lambda: released.append(registry.release_session("req-a"))
    )
    release_thread.start()
    time.sleep(0.02)
    assert release_thread.is_alive()

    registry.record_decode_token(
        request_id="req-a",
        new_seq_len=2,
        token_id=42,
    )
    release_thread.join(timeout=1.0)
    assert released == [True]
    assert len(commits) == 1

    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-b",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    registry.record_decode_token(
        request_id="req-b",
        new_seq_len=2,
        token_id=99,
    )
    assert registry.release_session("req-b")
    stats = registry.decode_token_stats()
    assert stats["decode_token_pending_tokens"] == 0
    assert stats["decode_token_pending_kv"] == 0
    assert stats["decode_token_only_dropped"] == 1


def test_attention_release_endpoint_runs_in_fastapi_threadpool() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    app = create_app(registry)
    release_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", "") == "/v1/pap/attention/sessions/{request_id}"
        and "DELETE" in getattr(route, "methods", set())
    )

    assert not inspect.iscoroutinefunction(release_endpoint)


def test_unified_offload_exec_overlap_step_does_not_commit(
    monkeypatch,
) -> None:
    import torch

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
            num_heads=1,
            num_kv_heads=1,
            head_dim=2,
        )
    )
    registry._unified_paged_kv["req-a"] = {
        "layer0": PAPUnifiedPagedKVState(
            kv_cache=torch.zeros((2, 2, 4, 1, 2)),
            block_ids=(0,),
            prefix_len=1,
            seq_len=1,
            capacity_tokens=4,
            writable_start_token=1,
            writable_end_token=4,
            lease_id="lease-a",
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
        )
    }

    commits = []

    class FakeCommitClient:
        enabled = True

        def commit(
            self,
            *,
            request_id,
            session_request_id,
            new_seq_len,
            new_token_ids,
            endpoint,
        ):
            commits.append((request_id, new_seq_len, tuple(new_token_ids), endpoint))

    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        attention_compute_module,
        "_compute_unified_paged_attention_batch",
        lambda **kwargs: torch.tensor([[2.0, 0.0]]),
    )
    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )

    output = compute_offload_exec_batch_output(
        registry=registry,
        descriptor=descriptor,
        qkv_batch=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
    )

    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))
    assert commits == []
    assert registry._unified_paged_kv["req-a"]["layer0"].seq_len == 1


def test_unified_decode_append_all_active_reuses_inputs_and_scales(
    monkeypatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    registry._unified_paged_kv = {
        request_id: {
            "layer0": PAPUnifiedPagedKVState(
                kv_cache=kv_cache,
                block_ids=(block_id,),
                prefix_len=1,
                seq_len=1,
                capacity_tokens=4,
                writable_start_token=1,
                writable_end_token=4,
                lease_id=f"lease-{request_id}",
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
            )
        }
        for request_id, block_id in (("req-a", 0), ("req-b", 1))
    }
    calls = []

    def fake_reshape_and_cache_flash(*args):
        calls.append(args)

    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        fake_reshape_and_cache_flash,
        raising=False,
    )
    key = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    value = key + 10

    written = registry.append_decode_kv_to_unified_prefill_cache(
        session_request_ids=("req-a", "req-b"),
        layer_name="layer0",
        key_batch=key,
        value_batch=value,
        decode_seq_lens=(2, 2),
    )
    first_scales = (calls[0][6], calls[0][7])
    next_key = key + 20
    next_value = value + 20
    registry.append_decode_kv_to_unified_prefill_cache(
        session_request_ids=("req-a", "req-b"),
        layer_name="layer0",
        key_batch=next_key,
        value_batch=next_value,
        decode_seq_lens=(3, 3),
    )

    assert written == 2
    assert calls[0][0].data_ptr() == key.data_ptr()
    assert calls[0][1].data_ptr() == value.data_ptr()
    assert calls[1][0].data_ptr() == next_key.data_ptr()
    assert calls[1][1].data_ptr() == next_value.data_ptr()
    assert calls[1][6].data_ptr() == first_scales[0].data_ptr()
    assert calls[1][7].data_ptr() == first_scales[1].data_ptr()
    assert registry.decode_append_fast_path_stats() == {
        "fast_path_hits": 2,
        "fallbacks": 0,
        "scale_cache_entries": 1,
        "slot_plan_hits": 0,
        "slot_plan_misses": 0,
        "slot_plan_entries": 0,
        "slot_topology_mismatches": 0,
    }


def test_unified_decode_append_does_not_hold_registry_lock_during_gpu_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-lock-scope",
            conversation_id="conv-lock-scope",
            prefill_endpoint="http://localhost:8100",
        )
    )
    _install_unified_activation(
        registry,
        request_id="req-lock-scope",
        layer_names=("layer0",),
        kv_cache=torch.zeros((1, 2, 4, 1, 2)),
        block_ids=(0,),
        seq_len=1,
    )
    gpu_work_started = Event()
    allow_gpu_work = Event()
    append_errors: list[BaseException] = []

    def blocking_reshape_and_cache_flash(*_args: Any) -> None:
        gpu_work_started.set()
        assert allow_gpu_work.wait(timeout=5)

    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        blocking_reshape_and_cache_flash,
        raising=False,
    )

    def append() -> None:
        try:
            registry.append_decode_kv_to_unified_prefill_cache(
                session_request_ids=("req-lock-scope",),
                layer_name="layer0",
                key_batch=torch.ones((1, 1, 2)),
                value_batch=torch.ones((1, 1, 2)),
                decode_seq_lens=(2,),
            )
        except BaseException as exc:
            append_errors.append(exc)

    append_thread = Thread(target=append)
    append_thread.start()
    assert gpu_work_started.wait(timeout=5)

    stats_finished = Event()

    def read_stats() -> None:
        registry.decode_append_fast_path_stats()
        stats_finished.set()

    stats_thread = Thread(target=read_stats)
    stats_thread.start()
    assert stats_finished.wait(timeout=1)

    allow_gpu_work.set()
    append_thread.join(timeout=5)
    stats_thread.join(timeout=5)
    assert not append_thread.is_alive()
    assert not stats_thread.is_alive()
    assert append_errors == []
    assert registry._unified_paged_kv["req-lock-scope"]["layer0"].seq_len == 2


def test_unified_decode_append_reuses_slot_plan_across_layers(
    monkeypatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    for request_id in ("req-a", "req-b"):
        registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id=request_id,
                conversation_id="conv",
                prefill_endpoint="http://localhost:8100",
            )
        )
    layer_caches = {
        layer_name: torch.zeros((4, 2, 4, 1, 2)) for layer_name in ("layer0", "layer1")
    }
    for layer_name, kv_cache in layer_caches.items():
        for request_id, block_id in (("req-a", 0), ("req-b", 1)):
            _install_unified_activation(
                registry,
                request_id=request_id,
                layer_names=(layer_name,),
                kv_cache=kv_cache,
                block_ids=(block_id,),
                seq_len=1,
            )
    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)

    for layer_name in ("layer0", "layer1"):
        written = registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=("req-a", "req-b"),
            layer_name=layer_name,
            key_batch=key,
            value_batch=key + 10,
            decode_seq_lens=(2, 2),
        )
        assert written == 2

    assert calls[0][4].tolist() == [1, 5]
    assert calls[1][4].data_ptr() == calls[0][4].data_ptr()
    assert registry.decode_append_fast_path_stats() == {
        "fast_path_hits": 2,
        "fallbacks": 0,
        "scale_cache_entries": 1,
        "slot_plan_hits": 1,
        "slot_plan_misses": 1,
        "slot_plan_entries": 1,
        "slot_topology_mismatches": 0,
    }


def test_unified_decode_append_recovers_slot_plan_after_chunk_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    layer_names = tuple(f"layer{index}" for index in range(36))
    kv_cache = torch.zeros((1002, 2, 16, 1, 2))
    for block_count, seq_len in (
        (256, 4096),
        (512, 8192),
        (768, 12288),
        (1002, 16018),
    ):
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=layer_names,
            kv_cache=kv_cache,
            block_ids=tuple(range(block_count)),
            seq_len=seq_len,
            block_size=16,
        )

    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((1, 1, 2))
    for layer_name in layer_names:
        registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=("req-a",),
            layer_name=layer_name,
            key_batch=key,
            value_batch=key,
            decode_seq_lens=(16019,),
        )

    assert calls[0][4].tolist() == [16018]
    assert all(call[4].data_ptr() == calls[0][4].data_ptr() for call in calls[1:])
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 35
    assert stats["slot_plan_misses"] == 1
    assert stats["slot_topology_mismatches"] == 0


def test_unified_decode_append_latches_same_generation_topology_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    for block_ids in ((0,), (1,), (0,)):
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=("layer0",),
            kv_cache=kv_cache,
            block_ids=block_ids,
            seq_len=1,
        )
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer1",),
        kv_cache=kv_cache,
        block_ids=(0,),
        seq_len=1,
    )

    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((1, 1, 2))
    for layer_name in ("layer0", "layer1"):
        registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=("req-a",),
            layer_name=layer_name,
            key_batch=key,
            value_batch=key,
            decode_seq_lens=(2,),
        )

    assert calls[0][4].data_ptr() != calls[1][4].data_ptr()
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 0
    assert stats["slot_plan_misses"] == 0
    assert stats["slot_topology_mismatches"] == 1


def test_unified_decode_append_waits_for_complete_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0", "layer1"),
        kv_cache=kv_cache,
        block_ids=(0,),
        seq_len=1,
    )
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0",),
        kv_cache=kv_cache,
        block_ids=(0, 1),
        seq_len=5,
    )

    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((1, 1, 2))
    registry.append_decode_kv_to_unified_prefill_cache(
        session_request_ids=("req-a",),
        layer_name="layer0",
        key_batch=key,
        value_batch=key,
        decode_seq_lens=(6,),
    )

    assert calls[0][4].tolist() == [5]
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 0
    assert stats["slot_plan_misses"] == 0
    assert stats["slot_topology_mismatches"] == 0


def test_unified_decode_append_batch_falls_back_for_incomplete_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    kv_cache = torch.zeros((8, 2, 4, 1, 2))
    for request_id, initial_blocks, next_blocks in (
        ("req-a", (0,), (0, 1)),
        ("req-b", (2,), (2, 3)),
    ):
        registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id=request_id,
                conversation_id="conv",
                prefill_endpoint="http://localhost:8100",
            )
        )
        _install_unified_activation(
            registry,
            request_id=request_id,
            layer_names=("layer0", "layer1"),
            kv_cache=kv_cache,
            block_ids=initial_blocks,
            seq_len=1,
        )
        _install_unified_activation(
            registry,
            request_id=request_id,
            layer_names=("layer0", "layer1") if request_id == "req-a" else ("layer0",),
            kv_cache=kv_cache,
            block_ids=next_blocks,
            seq_len=5,
        )

    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((2, 1, 2))
    registry.append_decode_kv_to_unified_prefill_cache(
        session_request_ids=("req-a", "req-b"),
        layer_name="layer0",
        key_batch=key,
        value_batch=key,
        decode_seq_lens=(6, 6),
    )

    assert calls[0][4].tolist() == [5, 13]
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 0
    assert stats["slot_plan_misses"] == 0


def test_unified_slot_conflict_clears_only_on_new_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0",),
        kv_cache=kv_cache,
        block_ids=(0,),
        seq_len=1,
    )
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer1",),
        kv_cache=kv_cache,
        block_ids=(1,),
        seq_len=1,
    )
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0", "layer1"),
        kv_cache=kv_cache,
        block_ids=(0, 1),
        seq_len=5,
    )

    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((1, 1, 2))
    for layer_name in ("layer0", "layer1"):
        registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=("req-a",),
            layer_name=layer_name,
            key_batch=key,
            value_batch=key,
            decode_seq_lens=(6,),
        )

    assert calls[1][4].data_ptr() == calls[0][4].data_ptr()
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 1
    assert stats["slot_plan_misses"] == 1
    assert stats["slot_topology_mismatches"] == 1


def test_unified_slot_generation_rejects_stale_activation() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0", "layer1"),
        kv_cache=kv_cache,
        block_ids=(0,),
        seq_len=1,
    )
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0", "layer1"),
        kv_cache=kv_cache,
        block_ids=(0, 1),
        seq_len=5,
    )

    with pytest.raises(RuntimeError, match="stale.*activation"):
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=("layer0",),
            kv_cache=kv_cache,
            block_ids=(0,),
            seq_len=1,
        )

    assert registry._unified_paged_kv["req-a"]["layer0"].prefix_len == 5


def test_unified_slot_generation_rejects_advance_before_complete() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0", "layer1"),
        kv_cache=kv_cache,
        block_ids=(0,),
        seq_len=1,
    )
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0",),
        kv_cache=kv_cache,
        block_ids=(0, 1),
        seq_len=5,
    )

    with pytest.raises(RuntimeError, match="advanced before all layers"):
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=("layer0",),
            kv_cache=kv_cache,
            block_ids=(0, 1, 2),
            seq_len=9,
        )

    activation = registry._unified_slot_activations["req-a"]
    assert activation.prefix_len == 5
    assert not activation.complete
    assert registry._unified_paged_kv["req-a"]["layer0"].prefix_len == 5


def test_unified_slot_generation_rejects_new_layer_after_freeze() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    _install_unified_activation(
        registry,
        request_id="req-a",
        layer_names=("layer0", "layer1"),
        kv_cache=kv_cache,
        block_ids=(0,),
        seq_len=1,
    )

    with pytest.raises(RuntimeError, match="unexpected layer"):
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=("layer2",),
            kv_cache=kv_cache,
            block_ids=(0, 1),
            seq_len=5,
        )

    activation = registry._unified_slot_activations["req-a"]
    assert activation.prefix_len == 1
    assert "layer2" not in registry._unified_paged_kv["req-a"]


def test_unified_slot_topology_ids_are_unique_across_registries() -> None:
    import torch

    topology_ids = []
    for index in range(2):
        registry = PAPAttentionRegistry(storage_device="cpu")
        request_id = f"req-{index}"
        registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id=request_id,
                conversation_id="conv",
                prefill_endpoint="http://localhost:8100",
            )
        )
        _install_unified_activation(
            registry,
            request_id=request_id,
            layer_names=("layer0",),
            kv_cache=torch.zeros((1, 2, 4, 1, 2)),
            block_ids=(0,),
            seq_len=1,
        )
        topology_ids.append(
            registry._unified_paged_kv[request_id]["layer0"].slot_topology_id
        )

    assert topology_ids[0] > 0
    assert topology_ids[1] > topology_ids[0]


def test_unified_decode_append_disables_slot_plan_for_mixed_topology(
    monkeypatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    for layer_name, block_id in (("layer0", 0), ("layer1", 1)):
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=(layer_name,),
            kv_cache=torch.zeros((4, 2, 4, 1, 2)),
            block_ids=(block_id,),
            seq_len=1,
        )
    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((1, 1, 2))

    for layer_name in ("layer0", "layer1"):
        registry.append_decode_kv_to_unified_prefill_cache(
            session_request_ids=("req-a",),
            layer_name=layer_name,
            key_batch=key,
            value_batch=key,
            decode_seq_lens=(2,),
        )

    assert calls[0][4].tolist() == [1]
    assert calls[1][4].tolist() == [5]
    assert calls[1][4].data_ptr() != calls[0][4].data_ptr()
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 0
    assert stats["slot_plan_misses"] == 0
    assert stats["slot_plan_entries"] == 0
    assert stats["slot_topology_mismatches"] == 1


def test_unified_decode_append_slot_plan_uses_session_generation(
    monkeypatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")

    def install_session(block_id: int) -> None:
        registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id="req-a",
                conversation_id="conv",
                prefill_endpoint="http://localhost:8100",
            )
        )
        _install_unified_activation(
            registry,
            request_id="req-a",
            layer_names=("layer0", "layer1"),
            kv_cache=torch.zeros((4, 2, 4, 1, 2)),
            block_ids=(block_id,),
            seq_len=1,
        )

    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.ones((1, 1, 2))

    def append_both_layers() -> None:
        for layer_name in ("layer0", "layer1"):
            registry.append_decode_kv_to_unified_prefill_cache(
                session_request_ids=("req-a",),
                layer_name=layer_name,
                key_batch=key,
                value_batch=key,
                decode_seq_lens=(2,),
            )

    install_session(0)
    append_both_layers()
    with registry._lock:
        registry._release_session_locked("req-a")
    install_session(1)
    append_both_layers()

    assert calls[0][4].tolist() == [1]
    assert calls[1][4].data_ptr() == calls[0][4].data_ptr()
    assert calls[2][4].tolist() == [5]
    assert calls[2][4].data_ptr() != calls[0][4].data_ptr()
    assert calls[3][4].data_ptr() == calls[2][4].data_ptr()
    stats = registry.decode_append_fast_path_stats()
    assert stats["slot_plan_hits"] == 2
    assert stats["slot_plan_misses"] == 2
    assert stats["slot_plan_entries"] == 2


def test_unified_decode_append_partial_batch_uses_fallback_gather(
    monkeypatch,
) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    kv_cache = torch.zeros((4, 2, 4, 1, 2))
    registry._unified_paged_kv = {
        "req-a": {
            "layer0": PAPUnifiedPagedKVState(
                kv_cache=kv_cache,
                block_ids=(0,),
                prefix_len=1,
                seq_len=2,
                capacity_tokens=4,
                writable_start_token=1,
                writable_end_token=4,
                lease_id="lease-a",
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
            )
        },
        "req-b": {
            "layer0": PAPUnifiedPagedKVState(
                kv_cache=kv_cache,
                block_ids=(1,),
                prefix_len=1,
                seq_len=2,
                capacity_tokens=4,
                writable_start_token=1,
                writable_end_token=4,
                lease_id="lease-b",
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
            )
        },
    }
    calls = []
    monkeypatch.setattr(
        kv_registry_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: calls.append(args),
        raising=False,
    )
    key = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)

    written = registry.append_decode_kv_to_unified_prefill_cache(
        session_request_ids=("req-a", "req-b"),
        layer_name="layer0",
        key_batch=key,
        value_batch=key + 10,
        decode_seq_lens=(2, 3),
    )

    assert written == 1
    assert calls[0][0].shape == (1, 1, 2)
    assert calls[0][0].data_ptr() != key.data_ptr()
    assert registry.decode_append_fast_path_stats()["fallbacks"] == 1


def test_attention_registry_caches_resolved_request_ids() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-cache",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )

    assert registry.resolve_session_request_id("req-cache-0") == "req-cache"
    assert registry._request_id_resolution_cache["req-cache-0"] == "req-cache"

    assert registry.release_session("req-cache")
    assert "req-cache-0" not in registry._request_id_resolution_cache


def test_attention_registry_reports_active_session_count() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    assert registry.active_session_count() == 0

    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-count",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )

    assert registry.active_session_count() == 1
    assert registry.release_session("req-count")
    assert registry.active_session_count() == 0


def test_attention_fast_path_stats_endpoint() -> None:
    kv_metadata_module.reset_unified_paged_flash_metadata_cache()
    registry = PAPAttentionRegistry(storage_device="cpu")
    client = _ASGITestClient(create_app(registry=registry))

    response = client.get("/v1/pap/attention/stats")

    assert response.status_code == 200
    assert response.json() == {
        "attention_dispatch_mode": "direct",
        "attention_active_peer_tracking": False,
        "paged_decode_visible_sms": 0,
        "paged_decode_kernel_config": {
            "num_splits": 4,
            "block_h": 16,
            "num_warps": 4,
            "num_stages": 2,
        },
        "attention_active_source_ids": [],
        "attention_membership_generations": {},
        "attention_membership_updates": 0,
        "attention_membership_stale_updates": 0,
        "fast_path_hits": 0,
        "fallbacks": 0,
        "scale_cache_entries": 0,
        "slot_plan_hits": 0,
        "slot_plan_misses": 0,
        "slot_plan_entries": 0,
        "slot_topology_mismatches": 0,
        "step_context_hits": 0,
        "step_context_misses": 0,
        "step_context_entries": 0,
        "step_slot_plan_builds": 0,
        "step_metadata_builds": 0,
        "step_kv_ready_publishes": 0,
        "unified_md_hits": 0,
        "unified_md_misses": 0,
        "unified_md_entries": 0,
        "unified_md_fast_key_lookups": 0,
        "unified_md_fast_key_hits": 0,
        "unified_md_full_key_scans": 0,
        "unified_md_block_ids_scanned": 0,
        "offload_exec_peer_batches": 0,
        "offload_exec_peer_rows": 0,
        "offload_exec_compute_calls": 0,
        "offload_exec_compute_rows": 0,
        "offload_exec_source_batches_per_compute_sum": 0,
        "offload_exec_max_source_batches_per_compute": 0,
        "offload_exec_peer_batches_by_source": {},
        "offload_exec_peer_rows_by_source": {},
        "offload_exec_compute_calls_by_layer": {},
        "paged_decode_warmup_started": False,
        "paged_decode_warmup_done": False,
        "paged_decode_warmup_failed": False,
        "decode_token_received": 0,
        "decode_kv_ready": 0,
        "decode_token_matched": 0,
        "decode_token_duplicates": 0,
        "decode_token_mismatches": 0,
        "decode_token_dispatch_failures": 0,
        "decode_token_pending_tokens": 0,
        "decode_token_pending_kv": 0,
        "decode_token_dispatching": 0,
        "decode_token_only_dropped": 0,
    }


def test_attention_registry_release_session_notifies_prefill_lease(
    monkeypatch,
) -> None:
    released = []
    events = []

    class FakeCommitClient:
        enabled = True

        def flush_request(self, request_id):
            events.append(("flush", request_id))
            return True

        def forget_request(self, request_id):
            events.append(("forget", request_id))

    class FakeLeaseReleaseClient:
        enabled = True

        def release(self, *, request_id, lease_id, endpoint):
            events.append(("release", request_id, lease_id, endpoint))
            released.append((request_id, lease_id))

    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        kv_registry_module,
        "_get_lease_release_client",
        lambda: FakeLeaseReleaseClient(),
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-lease",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    registry._session_lease_ids["req-lease"] = "lease-1"
    registry._session_leased_block_ids["req-lease"] = (3, 4)

    assert registry.release_session("req-lease")

    assert released == [("req-lease", "lease-1")]
    assert events == [
        ("flush", "req-lease"),
        (
            "release",
            "req-lease",
            "lease-1",
            "http://localhost:8100/v1/pap/prefill/lease-release",
        ),
        ("forget", "req-lease"),
    ]
    assert registry.get_session("req-lease") is None


def test_attention_registry_can_detach_session_and_retain_prefill_lease(
    monkeypatch,
) -> None:
    events = []

    class FakeCommitClient:
        enabled = True

        def flush_request(self, request_id):
            events.append(("flush", request_id))
            return True

        def forget_request(self, request_id):
            events.append(("forget", request_id))

    class FakeLeaseReleaseClient:
        def release(self, **kwargs):
            events.append(("release", kwargs))

    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        kv_registry_module,
        "_get_lease_release_client",
        lambda: FakeLeaseReleaseClient(),
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-retain",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    registry._session_lease_ids["req-retain"] = "lease-1"

    assert registry.release_session("req-retain", retain_lease=True)
    assert events == [
        ("flush", "req-retain"),
        (
            "release",
            {
                "request_id": "req-retain",
                "lease_id": "lease-1",
                "endpoint": ("http://localhost:8100/v1/pap/prefill/lease-release"),
                "retain": True,
            },
        ),
        ("forget", "req-retain"),
    ]
    assert registry.get_session("req-retain") is None


def test_attention_registry_does_not_release_lease_before_commit_ack(
    monkeypatch,
) -> None:
    events = []

    class FakeCommitClient:
        enabled = True

        def flush_request(self, request_id):
            events.append(("flush", request_id))
            return False

        def forget_request(self, request_id):
            events.append(("forget", request_id))

    class FakeLeaseReleaseClient:
        def release(self, *, request_id, lease_id, endpoint):
            events.append(("release", request_id, lease_id, endpoint))

    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        kv_registry_module,
        "_get_lease_release_client",
        lambda: FakeLeaseReleaseClient(),
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-unacked",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    registry._session_lease_ids["req-unacked"] = "lease-1"

    assert registry.release_session("req-unacked")
    assert events == [("flush", "req-unacked")]


def test_attention_registry_reregister_releases_replaced_prefill_lease(
    monkeypatch,
) -> None:
    released = []
    events = []

    class FakeCommitClient:
        enabled = True

        def flush_request(self, request_id):
            events.append(("flush", request_id))
            return True

        def forget_request(self, request_id):
            events.append(("forget", request_id))

    class FakeLeaseReleaseClient:
        enabled = True

        def release(self, *, request_id, lease_id, endpoint):
            events.append(("release", request_id, lease_id, endpoint))
            released.append((request_id, lease_id))

    monkeypatch.setattr(
        kv_registry_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        kv_registry_module,
        "_get_lease_release_client",
        lambda: FakeLeaseReleaseClient(),
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-reuse",
            conversation_id="conv-a",
            prefill_endpoint="http://localhost:8100",
        )
    )
    registry._session_lease_ids["req-reuse"] = "lease-old"
    registry._session_leased_block_ids["req-reuse"] = (3, 4)

    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-reuse",
            conversation_id="conv-b",
            prefill_endpoint="http://localhost:8100",
        )
    )

    assert released == [("req-reuse", "lease-old")]
    assert events == [
        ("flush", "req-reuse"),
        ("forget", "req-reuse"),
        (
            "release",
            "req-reuse",
            "lease-old",
            "http://localhost:8100/v1/pap/prefill/lease-release",
        ),
    ]
    session = registry.get_session("req-reuse")
    assert session is not None
    assert session.conversation_id == "conv-b"


def test_attention_registry_lists_prefill_readiness() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-ready",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )

    assert registry.get_prefill_readiness("req-ready") == []

    _install_unified_activation(
        registry,
        request_id="req-ready",
        layer_names=("layer0",),
        kv_cache=torch.zeros(2, 2, 4, 1, 2),
        block_ids=(0,),
        seq_len=2,
    )

    readiness = registry.get_prefill_readiness("req-ready")

    assert len(readiness) == 1
    assert readiness[0].request_id == "req-ready"
    assert readiness[0].layer_name == "layer0"
    assert readiness[0].descriptor_received is True
    assert readiness[0].descriptor_opened is True
    assert readiness[0].ready is True


def test_attention_prefill_readiness_endpoint() -> None:
    import anyio
    import torch
    from httpx import ASGITransport, AsyncClient

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-ready",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    _install_unified_activation(
        registry,
        request_id="req-ready",
        layer_names=("layer0",),
        kv_cache=torch.zeros(2, 2, 4, 1, 2),
        block_ids=(0,),
        seq_len=2,
    )
    app = create_app(registry=registry)

    async def run_request():
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(
                "/v1/pap/attention/sessions/req-ready/prefill-readiness"
            )

    response = anyio.run(run_request)

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-ready"
    assert body["layers"][0]["layer_name"] == "layer0"
    assert body["layers"][0]["ready"] is True


def test_offload_exec_batch_session_entries_reuses_and_invalidates_cache() -> None:
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-entry-cache",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )

    first = registry.offload_exec_batch_session_entries(
        ("req-entry-cache",),
        default_q_size=2,
        default_kv_size=2,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
    )
    second = registry.offload_exec_batch_session_entries(
        ("req-entry-cache",),
        default_q_size=2,
        default_kv_size=2,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
    )

    assert second[0] is first[0]

    changed_defaults = registry.offload_exec_batch_session_entries(
        ("req-entry-cache",),
        default_q_size=4,
        default_kv_size=4,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
    )
    assert changed_defaults[0] is not first[0]
    assert changed_defaults[0].q_size == 4
    assert changed_defaults[0].kv_size == 4

    assert registry.release_session("req-entry-cache")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-entry-cache",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    after_release = registry.offload_exec_batch_session_entries(
        ("req-entry-cache",),
        default_q_size=2,
        default_kv_size=2,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
    )

    assert after_release[0] is not first[0]


def test_run_offload_exec_mailbox_loop_releases_qkv_message(monkeypatch) -> None:
    import torch

    events = []

    class FakeMessage:
        def __init__(self):
            self.tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def release(self):
            events.append("release")

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.message = FakeMessage()
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, self.message

        def send_output_batch(self, descriptor, output, *, remote_address):
            events.append("send")

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    transport = FakeTransport(descriptor)

    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        lambda **kwargs: torch.tensor([[2.0, 0.0]]),
    )

    with suppress(KeyboardInterrupt):
        run_offload_exec_mailbox_loop(
            registry=registry,
            transport=transport,
            peer_id="projection-a",
        )

    assert events == ["release", "send"]
    assert registry.offload_exec_dispatch_stats() == {
        "offload_exec_peer_batches": 1,
        "offload_exec_peer_rows": 1,
        "offload_exec_compute_calls": 1,
        "offload_exec_compute_rows": 1,
        "offload_exec_source_batches_per_compute_sum": 1,
        "offload_exec_max_source_batches_per_compute": 1,
        "offload_exec_peer_batches_by_source": {"projection-a": 1},
        "offload_exec_peer_rows_by_source": {"projection-a": 1},
        "offload_exec_compute_calls_by_layer": {"layer0": 1},
        "paged_decode_warmup_started": False,
        "paged_decode_warmup_done": False,
        "paged_decode_warmup_failed": False,
    }


def test_mailbox_receiver_enqueues_without_computing(monkeypatch) -> None:
    import torch

    released = []

    class FakeMessage:
        tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def release(self):
            released.append("release")

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, FakeMessage()

    class FakeDispatcher:
        def __init__(self):
            self.items = []

        def enqueue(self, item):
            self.items.append(item)
            item.mark_completed()

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    transport = FakeTransport(descriptor)
    dispatcher = FakeDispatcher()
    registry = PAPAttentionRegistry(storage_device="cpu")
    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        lambda **_kwargs: pytest.fail("receiver must not compute"),
    )

    with suppress(KeyboardInterrupt):
        run_offload_exec_mailbox_receiver_loop(
            registry=registry,
            transport=transport,
            dispatcher=dispatcher,
            peer_id="projection-a",
        )

    assert len(dispatcher.items) == 1
    assert released == []
    assert registry.offload_exec_dispatch_stats()["offload_exec_peer_batches"] == 1
    assert registry.offload_exec_dispatch_stats()["offload_exec_compute_calls"] == 0
    dispatcher.items[0].release_input()
    assert released == ["release"]


def test_mailbox_receiver_waits_for_dispatch_before_busy_spinning_again() -> None:
    import torch

    enqueued = Event()
    receiver_done = Event()

    class FakeMessage:
        tensor = torch.ones((1, 6))

        def release(self):
            pass

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, FakeMessage()

    class FakeDispatcher:
        def __init__(self):
            self.items = []

        def enqueue(self, item):
            self.items.append(item)
            enqueued.set()

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    transport = FakeTransport(descriptor)
    dispatcher = FakeDispatcher()

    def receive() -> None:
        with suppress(KeyboardInterrupt):
            run_offload_exec_mailbox_receiver_loop(
                registry=PAPAttentionRegistry(storage_device="cpu"),
                transport=transport,
                dispatcher=dispatcher,
                peer_id="projection-a",
            )
        receiver_done.set()

    thread = Thread(target=receive, daemon=True)
    thread.start()

    assert enqueued.wait(timeout=1.0)
    assert transport.recv_calls == 1
    assert not receiver_done.wait(timeout=0.05)
    dispatcher.items[0].release_input()
    dispatcher.items[0].mark_completed()

    assert receiver_done.wait(timeout=1.0)
    assert transport.recv_calls == 2


def test_central_dispatcher_computes_and_sends_to_each_source(monkeypatch) -> None:
    import torch

    events = []

    class FakeMessage:
        def __init__(self, peer_id):
            self.peer_id = peer_id
            self.tensor = torch.ones((1, 6))

        def release(self):
            events.append(f"release:{self.peer_id}")

    class FakeTransport:
        def __init__(self, peer_id, descriptor):
            self.peer_id = peer_id
            self.descriptor = descriptor
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, FakeMessage(self.peer_id)

        def send_output_batch(self, descriptor, output, *, remote_address):
            events.append(f"send:{self.peer_id}")

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    registry = PAPAttentionRegistry(storage_device="cpu")
    compute_peers = []

    def fake_compute(**kwargs):
        compute_peers.append(float(kwargs["qkv_batch"][0, 0]))
        return torch.ones((1, 2))

    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        fake_compute,
    )
    dispatcher = PAPAttentionDispatcher(
        handler=lambda item: _execute_offload_exec_work_item(
            registry=registry,
            item=item,
        )
    )
    transports = [
        FakeTransport("projection-a", descriptor),
        FakeTransport("projection-b", descriptor),
    ]

    def receive(source) -> None:
        with suppress(KeyboardInterrupt):
            run_offload_exec_mailbox_receiver_loop(
                registry=registry,
                transport=source,
                dispatcher=dispatcher,
                peer_id=source.peer_id,
            )

    dispatcher.start()
    receiver_threads = []
    for transport in transports:
        thread = Thread(
            target=receive,
            args=(transport,),
            daemon=True,
        )
        receiver_threads.append(thread)
        thread.start()
    for thread in receiver_threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    dispatcher.stop(drain=True, timeout=1.0)

    assert compute_peers == [1.0, 1.0]
    for peer_id in ("projection-a", "projection-b"):
        assert events.index(f"send:{peer_id}") < events.index(f"release:{peer_id}")
    stats = registry.offload_exec_dispatch_stats()
    assert stats["offload_exec_peer_batches"] == 2
    assert stats["offload_exec_compute_calls"] == 2
    assert stats["offload_exec_max_source_batches_per_compute"] == 1


def test_central_combine_executes_once_and_scatters_to_sources(
    monkeypatch,
) -> None:
    import torch

    class FakeTransport:
        def __init__(self) -> None:
            self.sent = []

        def send_output_batch(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output.clone(), remote_address))

    descriptors = [
        PAPOffloadExecBatchDescriptor(
            layer_name="layer0",
            items=(
                PAPOffloadExecDescriptor(
                    request_id="req-a",
                    layer_name="layer0",
                    step=3,
                    scale=0.5,
                ),
            ),
        ),
        PAPOffloadExecBatchDescriptor(
            layer_name="layer0",
            items=(),
            batch_id_suffix="req-b@7",
            metadata_template={
                "r": ("req-b",),
                "s": (7,),
                "a": (0.5,),
            },
        ),
    ]
    transports = [FakeTransport(), FakeTransport()]
    items = tuple(
        PAPAttentionWorkItem(
            descriptor=descriptor,
            qkv_batch=torch.full((1, 6), float(index + 1)),
            transport=transport,
            peer_id=f"projection-{index}",
            arrival_ns=1,
        )
        for index, (descriptor, transport) in enumerate(zip(descriptors, transports))
    )
    compute_calls = []

    def fake_compute(**kwargs):
        rows = _offload_exec_batch_rows(kwargs["descriptor"])
        compute_calls.append((rows, kwargs["qkv_batch"].clone()))
        return torch.tensor([[10.0, 11.0], [20.0, 21.0]])

    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        fake_compute,
    )
    registry = PAPAttentionRegistry(storage_device="cpu")

    _execute_offload_exec_work_items(registry=registry, items=items)

    assert len(compute_calls) == 1
    request_ids, steps, scales = compute_calls[0][0]
    assert request_ids == ("req-a", "req-b")
    assert steps == (3, 7)
    assert scales == (0.5, 0.5)
    torch.testing.assert_close(
        compute_calls[0][1],
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ]
        ),
    )
    assert transports[0].sent[0][0] is descriptors[0]
    assert transports[1].sent[0][0] is descriptors[1]
    torch.testing.assert_close(
        transports[0].sent[0][1],
        torch.tensor([[10.0, 11.0]]),
    )
    torch.testing.assert_close(
        transports[1].sent[0][1],
        torch.tensor([[20.0, 21.0]]),
    )
    stats = registry.offload_exec_dispatch_stats()
    assert stats["offload_exec_compute_calls"] == 1
    assert stats["offload_exec_compute_rows"] == 2
    assert stats["offload_exec_source_batches_per_compute_sum"] == 2
    assert stats["offload_exec_max_source_batches_per_compute"] == 2


def test_central_combine_single_item_reuses_fifo_executor(monkeypatch) -> None:
    import torch

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=0.5,
            ),
        ),
    )
    item = PAPAttentionWorkItem(
        descriptor=descriptor,
        qkv_batch=torch.ones((1, 6)),
        transport=object(),
        peer_id="projection-0",
        arrival_ns=1,
    )
    calls = []
    monkeypatch.setattr(
        attention_runtime_module,
        "_execute_offload_exec_work_item",
        lambda **kwargs: calls.append(kwargs),
    )
    registry = PAPAttentionRegistry(storage_device="cpu")

    _execute_offload_exec_work_items(registry=registry, items=(item,))

    assert calls == [{"registry": registry, "item": item}]


def test_central_combine_compatibility_key_rejects_layer_or_scale() -> None:
    import torch

    def make_item(layer_name: str, scale: float) -> PAPAttentionWorkItem:
        descriptor = PAPOffloadExecBatchDescriptor(
            layer_name=layer_name,
            items=(
                PAPOffloadExecDescriptor(
                    request_id=f"req-{layer_name}-{scale}",
                    layer_name=layer_name,
                    step=1,
                    scale=scale,
                ),
            ),
        )
        return PAPAttentionWorkItem(
            descriptor=descriptor,
            qkv_batch=torch.ones((1, 6)),
            transport=object(),
            peer_id="projection",
            arrival_ns=1,
        )

    baseline = _offload_exec_work_item_compatibility_key(make_item("layer0", 0.5))

    assert baseline == _offload_exec_work_item_compatibility_key(
        make_item("layer0", 0.5)
    )
    assert baseline != _offload_exec_work_item_compatibility_key(
        make_item("layer1", 0.5)
    )
    assert baseline != _offload_exec_work_item_compatibility_key(
        make_item("layer0", 1.0)
    )


def test_central_dispatcher_preserves_cuda_ready_dependency(monkeypatch) -> None:
    import torch

    events = []
    ready_event = object()

    class FakeMessage:
        tensor = torch.ones((1, 6))

        def release(self):
            events.append("release")

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, FakeMessage()

        def send_output_batch(self, descriptor, output, *, remote_address):
            events.append("send")

    class FakeDispatcher:
        def __init__(self):
            self.items = []

        def enqueue(self, item):
            self.items.append(item)
            item.mark_completed()

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(
        attention_runtime_module,
        "_record_offload_exec_ready_event",
        lambda _tensor: events.append("record") or ready_event,
        raising=False,
    )
    with suppress(KeyboardInterrupt):
        run_offload_exec_mailbox_receiver_loop(
            registry=PAPAttentionRegistry(storage_device="cpu"),
            transport=FakeTransport(descriptor),
            dispatcher=dispatcher,
            peer_id="projection-a",
        )

    assert events == ["record"]
    assert dispatcher.items[0].ready_event is ready_event

    def wait_ready(item):
        assert item.ready_event is ready_event
        events.append("wait")

    def fake_compute(**_kwargs):
        events.append("compute")
        return torch.ones((1, 2))

    monkeypatch.setattr(
        attention_runtime_module,
        "_wait_offload_exec_ready_event",
        wait_ready,
        raising=False,
    )
    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        fake_compute,
    )
    _execute_offload_exec_work_item(
        registry=PAPAttentionRegistry(storage_device="cpu"),
        item=dispatcher.items[0],
    )
    dispatcher.items[0].release_input()

    assert events == ["record", "wait", "compute", "send", "release"]


def test_central_dispatcher_preserves_mailbox_trace_contract(
    monkeypatch,
    caplog,
) -> None:
    import time

    import torch

    from vllm.pap.attention import PAPAttentionWorkItem

    class FakeTransport:
        def send_output_batch(self, descriptor, output, *, remote_address):
            pass

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    now_ns = time.perf_counter_ns()
    item = PAPAttentionWorkItem(
        descriptor=descriptor,
        qkv_batch=torch.ones((1, 6)),
        transport=FakeTransport(),
        peer_id="projection-a",
        arrival_ns=now_ns,
        trace_context={
            "enabled": True,
            "total_start": time.perf_counter(),
            "recv_start_ns": now_ns - 100_000,
            "recv_done_ns": now_ns,
            "recv_ms": 0.1,
            "recv_stats": {
                "wait_ms": 0.05,
                "read_ms": 0.02,
                "materialize_ms": 0.01,
                "transfer_ms": 0.01,
                "wait_other_ms": 0.03,
                "unaccounted_ms": 0.05,
            },
        },
    )

    def fake_compute(**kwargs):
        assert kwargs["trace_stats"] is not None
        return torch.ones((1, 2))

    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        fake_compute,
    )
    dispatcher = PAPAttentionDispatcher(
        handler=lambda work_item: _execute_offload_exec_work_item(
            registry=PAPAttentionRegistry(storage_device="cpu"),
            item=work_item,
        )
    )
    dispatcher.enqueue(item)

    with caplog.at_level(logging.INFO, logger="pap_attention"):
        assert dispatcher.dispatch_next(timeout=0.1)

    assert (
        "PAP OFFLOAD_EXEC attention mailbox batch trace layer=layer0 calls=1"
        in caplog.text
    )
    assert "queue_wait_ms=" in caplog.text
    assert "peer=projection-a" in caplog.text


def test_run_offload_exec_mailbox_loop_emits_trace(monkeypatch, caplog) -> None:
    import torch

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.sent = []
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            message = SimpleNamespace(
                tensor=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
                release=lambda: None,
                recv_trace={},
            )
            return self.descriptor, message

        def send_output_batch(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_TRACE", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    transport = FakeTransport(descriptor)

    def fake_compute_offload_exec_batch_output(**kwargs):
        trace_stats = kwargs["trace_stats"]
        trace_stats["paged_metadata_ms"] = 0.125
        trace_stats["paged_flash_ms"] = 0.250
        trace_stats["metadata_build_ms"] = 0.125
        trace_stats["paged_flash_kernel_ms"] = 0.250
        trace_stats["attention_output_reshape_ms"] = 0.031
        trace_stats["compute_unaccounted_ms"] = 0.004
        trace_stats["pre_compute_done_ns"] = 200
        trace_stats["paged_flash_done_ns"] = 300
        trace_stats["post_compute_done_ns"] = 400
        return torch.tensor([[2.0, 0.0]])

    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        fake_compute_offload_exec_batch_output,
    )

    with (
        caplog.at_level(logging.INFO, logger="pap_attention"),
        suppress(KeyboardInterrupt),
    ):
        run_offload_exec_mailbox_loop(registry=registry, transport=transport)

    assert len(transport.sent) == 1
    assert (
        "PAP OFFLOAD_EXEC attention mailbox batch trace layer=layer0 calls=1"
        in caplog.text
    )
    assert "recv_qkv_ms=" in caplog.text
    assert "compute_ms=" in caplog.text
    assert "send_output_ms=" in caplog.text
    assert "total_ms=" in caplog.text
    assert "metadata_build_ms=0.125" in caplog.text
    assert "paged_flash_kernel_ms=0.250" in caplog.text
    assert "attention_output_reshape_ms=0.031" in caplog.text
    assert "compute_unaccounted_ms=0.004" in caplog.text


def test_run_offload_exec_mailbox_loop_emits_recv_breakdown(
    monkeypatch,
    caplog,
) -> None:
    import torch

    class FakeMessage:
        tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])
        recv_trace = {
            "wait_ms": 0.600,
            "read_total_ms": 0.098,
            "materialize_ms": 0.009,
            "transfer_ms": 0.080,
        }

        def release(self):
            pass

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.sent = []
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, FakeMessage()

        def send_output_batch(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_TRACE", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    transport = FakeTransport(descriptor)

    monkeypatch.setattr(
        attention_runtime_module,
        "compute_offload_exec_batch_output",
        lambda **kwargs: torch.tensor([[2.0, 0.0]]),
    )

    with (
        caplog.at_level(logging.INFO, logger="pap_attention"),
        suppress(KeyboardInterrupt),
    ):
        run_offload_exec_mailbox_loop(registry=registry, transport=transport)

    assert "recv_wait_ms=0.600" in caplog.text
    assert "recv_read_ms=0.098" in caplog.text
    assert "recv_materialize_ms=0.009" in caplog.text
    assert "recv_transfer_ms=0.080" in caplog.text
    assert "recv_wait_other_ms=0.502" in caplog.text
    assert "recv_unaccounted_ms=0.000" in caplog.text


def test_attention_registry_records_prefill_kv_handle() -> None:
    registry = PAPAttentionRegistry()
    registration = PAPAttentionRegistration(
        request_id="req-1",
        conversation_id="conv-1",
        prefill_endpoint="http://localhost:8100",
        kv_transfer_params={
            "remote_engine_id": "prefill-0",
            "remote_block_ids": [[1, 2, 3]],
            "remote_host": "127.0.0.1",
            "remote_port": 5559,
        },
        prefix_len=17,
    )

    snapshot = registry.register_prefill_kv(registration)

    assert snapshot.request_id == "req-1"
    assert snapshot.conversation_id == "conv-1"
    assert snapshot.prefill_endpoint == "http://localhost:8100"
    assert snapshot.kv_transfer_params["remote_block_ids"] == [[1, 2, 3]]
    assert snapshot.prefix_len == 17
    assert snapshot.prefill_kv_handle == "req-1@pap-session-1"
    assert snapshot.role == "attention"


def test_attention_registry_returns_registered_snapshot_copy() -> None:
    registry = PAPAttentionRegistry()
    registration = PAPAttentionRegistration(
        request_id="req-2",
        conversation_id="conv-2",
        prefill_endpoint="http://localhost:8100",
        kv_transfer_params={"remote_engine_id": "prefill-0"},
        prefix_len=None,
    )

    registry.register_prefill_kv(registration)
    snapshot = registry.get_session("req-2")

    assert snapshot is not None
    assert snapshot.request_id == "req-2"
    assert snapshot.kv_transfer_params == {"remote_engine_id": "prefill-0"}

    snapshot.kv_transfer_params["remote_engine_id"] = "mutated"
    stored = registry.get_session("req-2")
    assert stored is not None
    assert stored.kv_transfer_params == {"remote_engine_id": "prefill-0"}


def test_attention_registry_matches_wrapped_vllm_request_ids() -> None:
    registry = PAPAttentionRegistry()
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="abcd",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
        )
    )

    assert registry.resolve_session_request_id("cmpl-abcd-0-deadbeef") == "abcd"
    assert registry.resolve_session_request_id("chatcmpl-abcd-1-deadbeef") == "abcd"
    assert registry.resolve_session_request_id("abcd") == "abcd"
    assert registry.resolve_session_request_id("unknown") is None
