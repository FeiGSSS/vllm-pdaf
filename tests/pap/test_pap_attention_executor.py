import base64
import hashlib
import inspect
import logging
import time
from contextlib import suppress
from pathlib import Path
from threading import Event, Thread
from typing import Any

import anyio
import pytest
from httpx import ASGITransport, AsyncClient, Response

from examples.pap.pap_attention_executor import (
    PAPAttentionRegistration,
    PAPAttentionRegistry,
    PAPUnifiedPagedKVState,
    _execute_offload_exec_work_item,
    _execute_offload_exec_work_items,
    _offload_exec_batch_rows,
    _offload_exec_work_item_compatibility_key,
    build_unified_paged_flash_metadata,
    compute_binary_attention_response,
    compute_offload_exec_batch_output,
    compute_offload_exec_output,
    create_app,
    maybe_start_offload_exec_transport,
    run_offload_exec_batch_once,
    run_offload_exec_mailbox_loop,
    run_offload_exec_mailbox_receiver_loop,
    run_offload_exec_once,
)
from vllm.pap.attention_scheduler import (
    PAPAttentionDispatcher,
    PAPAttentionWorkItem,
)
from vllm.pap.data_plane import (
    PAPCudaIPCTensorHandle,
    PAPOffloadExecBatchDescriptor,
    PAPOffloadExecDescriptor,
    PAPOffloadKVIPCDescriptor,
    PAPOffloadKVPagedIPCDescriptor,
)

ROOT = Path(__file__).resolve().parents[2]


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


def _make_unified_state(
    torch_module: Any,
    *,
    block_ids: tuple[int, ...],
    seq_len: int,
    lease_id: str,
) -> PAPUnifiedPagedKVState:
    return PAPUnifiedPagedKVState(
        kv_cache=torch_module.zeros((1, 2, 1, 1, 1)),
        block_ids=block_ids,
        prefix_len=seq_len,
        seq_len=seq_len,
        capacity_tokens=seq_len + 1,
        writable_start_token=seq_len,
        writable_end_token=seq_len + 1,
        lease_id=lease_id,
        block_size=16,
        num_kv_heads=1,
        layout="NHD",
    )


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
    for layer_name in layer_names:
        registry.import_prefill_paged_kv(
            request_id=request_id,
            layer_name=layer_name,
            kv_cache=kv_cache,
            block_ids=list(block_ids),
            seq_len=seq_len,
            block_size=block_size,
            num_kv_heads=1,
            layout="NHD",
            lease_id=f"lease-{request_id}",
            lease_capacity_tokens=len(block_ids) * block_size,
            unified_kv_mode=True,
            prefix_len=seq_len,
            writable_start_token=seq_len,
            writable_end_token=len(block_ids) * block_size,
        )


def test_offload_exec_batch_rows_uses_template_without_items() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(),
        batch_id_suffix="req-a@7,req-b@8",
        metadata_template={
            "r": ("req-a", "req-b"),
            "s": (7, 8),
            "a": (0.125, 0.125),
            "t": ((42,), (99,)),
        },
    )

    assert _offload_exec_batch_rows(descriptor) == (
        ("req-a", "req-b"),
        (7, 8),
        (0.125, 0.125),
        ((42,), (99,)),
    )


def test_unified_paged_flash_metadata_reuses_identical_decode_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    reset_cache = getattr(
        executor_module,
        "reset_unified_paged_flash_metadata_cache",
        None,
    )
    cache_stats = getattr(
        executor_module,
        "unified_paged_flash_metadata_cache_stats",
        None,
    )
    assert callable(reset_cache)
    assert callable(cache_stats)
    reset_cache()

    arange_calls = 0
    real_arange = torch.arange

    def counted_arange(*args, **kwargs):
        nonlocal arange_calls
        arange_calls += 1
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(executor_module.torch, "arange", counted_arange)
    kv_cache = torch.zeros((4, 2, 4, 1, 2), dtype=torch.float32)
    states = [
        PAPUnifiedPagedKVState(
            kv_cache=kv_cache,
            block_ids=(0, 1),
            prefix_len=5,
            seq_len=6,
            capacity_tokens=8,
            writable_start_token=5,
            writable_end_token=8,
            lease_id="lease-a",
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
        ),
        PAPUnifiedPagedKVState(
            kv_cache=kv_cache,
            block_ids=(2, 3),
            prefix_len=5,
            seq_len=6,
            capacity_tokens=8,
            writable_start_token=5,
            writable_end_token=8,
            lease_id="lease-b",
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
        ),
    ]

    first = build_unified_paged_flash_metadata(
        states=states,
        device=torch.device("cpu"),
    )
    second = build_unified_paged_flash_metadata(
        states=states,
        device=torch.device("cpu"),
    )

    assert cache_stats() == {
        "hits": 1,
        "misses": 1,
        "entries": 1,
        "fast_key_lookups": 0,
        "fast_key_hits": 0,
        "full_key_scans": 2,
        "block_ids_scanned": 8,
    }
    assert arange_calls == 1
    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    assert second.seq_lens.data_ptr() == first.seq_lens.data_ptr()
    assert second.cu_seqlens_q.data_ptr() == first.cu_seqlens_q.data_ptr()


def test_unified_paged_flash_metadata_fast_key_avoids_hit_block_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    state = _make_unified_state(
        torch,
        block_ids=tuple(range(1024)),
        seq_len=16384,
        lease_id="lease-long",
    )
    state.slot_generation = 3
    state.slot_topology_id = 41
    coerce_calls = 0
    real_coerce = executor_module._coerce_block_id

    def counted_coerce(value: Any) -> int:
        nonlocal coerce_calls
        coerce_calls += 1
        return real_coerce(value)

    monkeypatch.setattr(executor_module, "_coerce_block_id", counted_coerce)
    first = build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )
    second = build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )

    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    assert coerce_calls == 1024
    assert executor_module.unified_paged_flash_metadata_cache_stats() == {
        "hits": 1,
        "misses": 1,
        "entries": 1,
        "fast_key_lookups": 2,
        "fast_key_hits": 1,
        "full_key_scans": 1,
        "block_ids_scanned": 1024,
    }


def test_unified_paged_flash_metadata_fast_key_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    monkeypatch.setenv("PAP_UNIFIED_MD_FAST_KEY", "0")
    executor_module.reset_unified_paged_flash_metadata_cache()
    state = _make_unified_state(
        torch,
        block_ids=tuple(range(32)),
        seq_len=512,
        lease_id="lease-toggle",
    )
    state.slot_generation = 1
    state.slot_topology_id = 7
    coerce_calls = 0
    real_coerce = executor_module._coerce_block_id

    def counted_coerce(value: Any) -> int:
        nonlocal coerce_calls
        coerce_calls += 1
        return real_coerce(value)

    monkeypatch.setattr(executor_module, "_coerce_block_id", counted_coerce)
    build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )
    build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )

    assert coerce_calls == 64
    stats = executor_module.unified_paged_flash_metadata_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["fast_key_lookups"] == 0
    assert stats["full_key_scans"] == 2
    assert stats["block_ids_scanned"] == 64


def test_unified_paged_flash_metadata_fast_key_snapshots_sequence_length() -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    state = _make_unified_state(
        torch,
        block_ids=(3, 5),
        seq_len=17,
        lease_id="lease-seq",
    )
    state.slot_generation = 2
    state.slot_topology_id = 99

    first = build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )
    state.seq_len = 18
    second = build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )
    state.seq_len = 17
    first_again = build_unified_paged_flash_metadata(
        states=[state],
        device=torch.device("cpu"),
    )

    assert first.seq_lens.tolist() == [17]
    assert second.seq_lens.tolist() == [18]
    assert second.seq_lens.data_ptr() != first.seq_lens.data_ptr()
    assert first_again.seq_lens.data_ptr() == first.seq_lens.data_ptr()
    stats = executor_module.unified_paged_flash_metadata_cache_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["fast_key_lookups"] == 3
    assert stats["full_key_scans"] == 2
    assert stats["block_ids_scanned"] == 4


def test_unified_paged_flash_metadata_fast_key_preserves_row_order() -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    first_state = _make_unified_state(
        torch,
        block_ids=(3,),
        seq_len=8,
        lease_id="lease-a",
    )
    first_state.slot_generation = 1
    first_state.slot_topology_id = 101
    second_state = _make_unified_state(
        torch,
        block_ids=(7,),
        seq_len=8,
        lease_id="lease-b",
    )
    second_state.slot_generation = 1
    second_state.slot_topology_id = 102

    forward = build_unified_paged_flash_metadata(
        states=[first_state, second_state],
        device=torch.device("cpu"),
    )
    reverse = build_unified_paged_flash_metadata(
        states=[second_state, first_state],
        device=torch.device("cpu"),
    )

    assert forward.block_table.tolist() == [[3], [7]]
    assert reverse.block_table.tolist() == [[7], [3]]
    assert forward.block_table.data_ptr() != reverse.block_table.data_ptr()


def test_unified_paged_flash_metadata_lru_hit_is_atomic_with_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from collections import OrderedDict

    from examples.pap import pap_attention_executor as executor_module

    monkeypatch.setenv("PAP_UNIFIED_MD_CACHE_LIMIT", "1")
    executor_module.reset_unified_paged_flash_metadata_cache()
    first_state = _make_unified_state(
        torch,
        block_ids=(3,),
        seq_len=8,
        lease_id="lease-a",
    )
    first_state.slot_generation = 1
    first_state.slot_topology_id = 201
    second_state = _make_unified_state(
        torch,
        block_ids=(7,),
        seq_len=8,
        lease_id="lease-b",
    )
    second_state.slot_generation = 1
    second_state.slot_topology_id = 202
    build_unified_paged_flash_metadata(
        states=[first_state],
        device=torch.device("cpu"),
    )
    first_key = next(iter(executor_module._UNIFIED_MD_CACHE))
    hit_read = Event()
    first_evicted = Event()

    class CoordinatedCache(OrderedDict):
        def __init__(self, items):
            super().__init__(items)
            self.wait_once = True

        def get(self, key, default=None):
            value = super().get(key, default)
            if key == first_key and value is not None and self.wait_once:
                self.wait_once = False
                hit_read.set()
                first_evicted.wait(timeout=0.2)
            return value

        def popitem(self, last=True):
            item = super().popitem(last=last)
            if item[0] == first_key:
                first_evicted.set()
            return item

    coordinated_cache = CoordinatedCache(executor_module._UNIFIED_MD_CACHE.items())
    monkeypatch.setattr(executor_module, "_UNIFIED_MD_CACHE", coordinated_cache)
    errors: list[BaseException] = []

    def hit_first() -> None:
        try:
            build_unified_paged_flash_metadata(
                states=[first_state],
                device=torch.device("cpu"),
            )
        except BaseException as error:
            errors.append(error)

    def evict_first() -> None:
        assert hit_read.wait(timeout=1.0)
        build_unified_paged_flash_metadata(
            states=[second_state],
            device=torch.device("cpu"),
        )

    hit_thread = Thread(target=hit_first)
    evict_thread = Thread(target=evict_first)
    hit_thread.start()
    evict_thread.start()
    hit_thread.join(timeout=2.0)
    evict_thread.join(timeout=2.0)

    assert not hit_thread.is_alive()
    assert not evict_thread.is_alive()
    assert errors == []


def test_unified_paged_flash_metadata_avoids_scalar_tensor_writes_on_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    from examples.pap import pap_attention_executor as executor_module

    class ScalarCopyCounter(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.scalar_tensor_writes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            if func is torch.ops.aten.copy_.default and args[0].ndim == 0:
                self.scalar_tensor_writes += 1
            return func(*args, **kwargs)

    monkeypatch.setenv("PAP_UNIFIED_MD_CACHE_LIMIT", "0")
    executor_module.reset_unified_paged_flash_metadata_cache()
    block_ids = tuple(range(1024))
    state = _make_unified_state(
        torch,
        block_ids=block_ids,
        seq_len=16384,
        lease_id="lease-long",
    )
    counter = ScalarCopyCounter()

    with counter:
        metadata = build_unified_paged_flash_metadata(
            states=[state],
            device=torch.device("cpu"),
        )

    assert counter.scalar_tensor_writes == 0
    assert metadata.block_table.shape == (1, 1024)
    assert metadata.block_table.dtype == torch.int32
    assert metadata.seq_lens.dtype == torch.int32
    assert metadata.cu_seqlens_q.dtype == torch.int32
    assert metadata.block_table.device == torch.device("cpu")
    assert metadata.block_table.is_contiguous()
    assert metadata.block_table.tolist() == [list(block_ids)]
    assert metadata.seq_lens.tolist() == [16384]
    assert metadata.max_seq_len == 16384


def test_unified_paged_flash_metadata_pads_ragged_rows() -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    states = [
        _make_unified_state(
            torch,
            block_ids=block_ids,
            seq_len=seq_len,
            lease_id=f"lease-{index}",
        )
        for index, (block_ids, seq_len) in enumerate(
            (
                ((3, 5, 7), 40),
                ((11,), 11),
                ((13, 17), 25),
            )
        )
    ]

    metadata = build_unified_paged_flash_metadata(
        states=states,
        device=torch.device("cpu"),
    )

    assert metadata.block_table.tolist() == [
        [3, 5, 7],
        [11, 11, 11],
        [13, 17, 17],
    ]
    assert metadata.seq_lens.tolist() == [40, 11, 25]
    assert metadata.cu_seqlens_q.tolist() == [0, 1, 2, 3]
    assert metadata.max_seq_len == 40


def test_unified_paged_flash_metadata_key_uses_unpadded_rows() -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()

    def build(rows: tuple[tuple[int, ...], ...]):
        states = [
            _make_unified_state(
                torch,
                block_ids=row,
                seq_len=8,
                lease_id=f"lease-{index}",
            )
            for index, row in enumerate(rows)
        ]
        return build_unified_paged_flash_metadata(
            states=states,
            device=torch.device("cpu"),
        )

    first = build(((7,), (10, 11)))
    second = build(((7, 7), (10, 11)))

    assert torch.equal(first.block_table, second.block_table)
    assert executor_module.unified_paged_flash_metadata_cache_stats() == {
        "hits": 0,
        "misses": 2,
        "entries": 2,
        "fast_key_lookups": 0,
        "fast_key_hits": 0,
        "full_key_scans": 2,
        "block_ids_scanned": 7,
    }
    assert first.block_table.data_ptr() != second.block_table.data_ptr()


def test_unified_paged_flash_metadata_preserves_lru_recency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    monkeypatch.setenv("PAP_UNIFIED_MD_CACHE_LIMIT", "2")
    executor_module.reset_unified_paged_flash_metadata_cache()

    def build(block_id: int):
        state = _make_unified_state(
            torch,
            block_ids=(block_id,),
            seq_len=8,
            lease_id=f"lease-{block_id}",
        )
        return build_unified_paged_flash_metadata(
            states=[state],
            device=torch.device("cpu"),
        )

    build(1)
    first_b = build(2)
    build(1)
    build(3)
    second_b = build(2)

    assert executor_module.unified_paged_flash_metadata_cache_stats() == {
        "hits": 1,
        "misses": 4,
        "entries": 2,
        "fast_key_lookups": 0,
        "fast_key_hits": 0,
        "full_key_scans": 5,
        "block_ids_scanned": 5,
    }
    assert first_b.block_table.data_ptr() != second_b.block_table.data_ptr()


def test_attention_executor_declares_offload_exec_zmq_port() -> None:
    text = (ROOT / "examples" / "pap" / "pap_attention_executor.py").read_text()

    assert "--offload-exec-zmq-port" in text
    assert "OFFLOAD_EXEC NIXL mailbox initialized" in text


def test_attention_executor_logs_ipc_prefill_import() -> None:
    text = (ROOT / "examples" / "pap" / "pap_attention_executor.py").read_text()

    assert "PAP prefill KV imported via IPC descriptor" in text


def test_attention_executor_skips_offload_exec_when_zmq_port_is_none(
    monkeypatch,
) -> None:
    app = create_app()
    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=None)
    assert app.state.offload_exec_transport is None


def test_attention_executor_starts_offload_exec_transport(monkeypatch) -> None:
    app = create_app()
    fake_transport = object()

    def fake_build_transport(**kwargs):
        fake_build_transport.kwargs = kwargs
        return fake_transport

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "2")
    monkeypatch.setattr(
        "examples.pap.pap_attention_executor.build_nixl_mailbox_offload_exec_transport",
        fake_build_transport,
    )

    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=10300)

    assert app.state.offload_exec_transport is fake_transport
    assert fake_build_transport.kwargs == {
        "actor_id": "attention",
        "local_rank": 2,
    }


def test_attention_executor_binds_each_projection_to_distinct_transport(
    monkeypatch,
) -> None:
    from examples.pap import pap_attention_executor as executor_module

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

    def fake_build_transport(*, actor_id: str, local_rank: int) -> FakeTransport:
        transport = FakeTransport(actor_id, local_rank)
        transports.append(transport)
        return transport

    def fake_mailbox_loop(*, registry, transport, peer_id) -> None:
        loops_started.append(transport)
        loop_peer_ids.append(peer_id)
        if len(loops_started) == 2:
            loops_ready.set()

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "2")
    monkeypatch.setenv("PAP_NIXL_MAILBOX_ACTOR_ID", "attention-4")
    monkeypatch.setattr(
        executor_module,
        "_build_attention_offload_exec_transport",
        fake_build_transport,
    )
    monkeypatch.setattr(
        executor_module,
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


@pytest.mark.parametrize("dispatch_mode", ["central_fifo", "central_combine"])
def test_attention_executor_central_mode_shares_one_dispatcher(
    monkeypatch,
    dispatch_mode,
) -> None:
    from examples.pap import pap_attention_executor as executor_module

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

    def fake_build_transport(*, actor_id, local_rank):
        transport = FakeTransport(actor_id, local_rank)
        transports.append(transport)
        return transport

    def fake_receiver_loop(*, registry, transport, dispatcher, peer_id):
        receiver_contexts.append((transport, dispatcher, peer_id))
        if len(receiver_contexts) == 2:
            receivers_ready.set()

    monkeypatch.setenv("PAP_ATTENTION_DISPATCH_MODE", dispatch_mode)
    monkeypatch.setattr(
        executor_module,
        "_build_attention_offload_exec_transport",
        fake_build_transport,
    )
    monkeypatch.setattr(
        executor_module,
        "run_offload_exec_mailbox_receiver_loop",
        fake_receiver_loop,
    )
    app = create_app()
    maybe_start_offload_exec_transport(app=app, host="127.0.0.1", zmq_port=10300)
    client = _ASGITestClient(app)
    peers = (b"projection-a", b"projection-b")

    for peer_index, peer_metadata in enumerate(peers):
        response = client.post(
            "/v1/pap/attention/offload-exec-mailbox/bind",
            json={
                "agent_metadata_b64": base64.b64encode(peer_metadata).decode(),
                "source_id": f"projection-{peer_index}",
            },
        )
        assert response.status_code == 200

    assert receivers_ready.wait(timeout=1.0)
    dispatcher = app.state.offload_exec_dispatcher
    assert dispatcher is not None
    assert {id(context[1]) for context in receiver_contexts} == {id(dispatcher)}
    assert {context[2] for context in receiver_contexts} == {
        "projection-0",
        "projection-1",
    }
    stats = client.get("/v1/pap/attention/stats").json()
    assert stats["attention_dispatch_mode"] == dispatch_mode
    assert stats["dispatcher_running"] is True
    if dispatch_mode == "central_combine":
        assert stats["dispatcher_expected_group_size"] == 2
        assert stats["dispatcher_preferred_peer_id"] == "projection-0"
    dispatcher.stop(drain=True, timeout=1.0)


def test_attention_executor_rejects_unknown_dispatch_mode(monkeypatch) -> None:
    monkeypatch.setenv("PAP_ATTENTION_DISPATCH_MODE", "separate_cohorts")

    with pytest.raises(ValueError, match="PAP_ATTENTION_DISPATCH_MODE"):
        create_app()


def test_attention_executor_builds_central_combine_dispatcher(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PAP_ATTENTION_DISPATCH_MODE", "central_combine")
    monkeypatch.setenv("PAP_ATTENTION_COMBINE_WAIT_US", "250")

    app = create_app()

    assert app.state.offload_exec_dispatch_mode == "central_combine"
    dispatcher = app.state.offload_exec_dispatcher
    assert dispatcher is not None
    assert dispatcher._batch_handler is not None
    assert dispatcher._compatibility_key is not None
    stats = dispatcher.stats()
    assert stats["dispatcher_coalesce_timeout_us"] == 250
    assert stats["dispatcher_expected_group_size"] == 1


def test_attention_executor_tracks_active_projection_membership(monkeypatch) -> None:
    monkeypatch.setenv("PAP_ATTENTION_DISPATCH_MODE", "central_combine")
    monkeypatch.setenv("PAP_ATTENTION_ACTIVE_PEER_TRACKING", "1")
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

    from vllm.pap import shadow_attention

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
        shadow_attention.socket,
        "create_connection",
        lambda *_args, **_kwargs: fake_socket,
    )

    result = shadow_attention.bind_offload_exec_mailbox(
        attention_endpoint="http://127.0.0.1:8300",
        local_agent_metadata=b"projection-metadata",
        source_id="projection-0-r0",
    )

    request_body = fake_socket.sent.partition(b"\r\n\r\n")[2]
    assert json.loads(request_body)["source_id"] == "projection-0-r0"
    assert result == response_metadata


def test_mailbox_activity_sends_membership_generation(monkeypatch) -> None:
    import json

    from vllm.pap import shadow_attention

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
        shadow_attention.socket,
        "create_connection",
        lambda *_args, **_kwargs: fake_socket,
    )

    result = shadow_attention.update_offload_exec_mailbox_activity(
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
    app = create_app()

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_TRANSPORT", "nccl")

    with pytest.raises(RuntimeError, match="use nixl_mailbox"):
        maybe_start_offload_exec_transport(
            app=app,
            host="127.0.0.1",
            zmq_port=10300,
        )


def test_compute_offload_exec_output_from_packed_qkv(monkeypatch) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_Q_SIZE", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_KV_SIZE", "2")

    output = compute_offload_exec_output(
        registry=registry,
        request_id="req-offload",
        layer_name="layer0",
        qkv=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
        scale=1.0,
        step=1,
    )

    assert output.shape == (1, 2)
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_compute_offload_exec_output_uses_step_block_descriptor(monkeypatch) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
            block_size=16,
            prefix_len=497,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    registry.import_prefill_kv(
        request_id="req-offload",
        layer_name="layer0",
        key=torch.zeros(497, 1, 2),
        value=torch.zeros(497, 1, 2),
        seq_len=497,
        block_ids=list(range(32)),
    )

    output = compute_offload_exec_output(
        registry=registry,
        request_id="req-offload",
        layer_name="layer0",
        qkv=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
        scale=1.0,
        step=498,
    )

    session = registry.get_session("req-offload")
    assert output.shape == (1, 2)
    assert session is not None
    assert session.block_ids[-1] == 31
    assert session.decode_seq_lens["layer0"] == 498


def test_compute_offload_exec_output_keeps_decode_kv_attention_local(
    monkeypatch,
) -> None:
    import torch

    layer_name = "model.layers.0.self_attn.attn"
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-local-decode",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
            prefix_len=4,
            block_size=4,
        )
    )
    kv_cache = torch.zeros((2, 2, 4, 1, 2))
    registry.import_prefill_paged_kv(
        request_id="req-local-decode",
        layer_name=layer_name,
        kv_cache=kv_cache,
        block_ids=[0],
        seq_len=4,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")

    output = compute_offload_exec_output(
        registry=registry,
        request_id="req-local-decode",
        layer_name=layer_name,
        qkv=torch.tensor([[1.0, 0.0, 5.0, 6.0, 7.0, 8.0]]),
        scale=1.0,
        step=5,
    )

    session_id = registry.resolve_session_request_id("req-local-decode")
    assert output.shape == (1, 2)
    assert torch.equal(kv_cache[0, 1, 0, :, :], torch.zeros((1, 2)))
    assert torch.equal(kv_cache[1, 1, 0, :, :], torch.zeros((1, 2)))
    decode_key, decode_value = registry._decode_kv[session_id][layer_name].view()
    assert torch.equal(decode_key, torch.tensor([[[5.0, 6.0]]]))
    assert torch.equal(decode_value, torch.tensor([[[7.0, 8.0]]]))


def test_run_offload_exec_once_receives_qkv_and_sends_output(monkeypatch) -> None:
    import torch

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert remote_address == "127.0.0.1:11300"
            assert descriptor.request_id == "req-offload"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    descriptor = PAPOffloadExecDescriptor(
        request_id="req-offload",
        layer_name="layer0",
        step=1,
        scale=1.0,
    )

    run_offload_exec_once(
        registry=registry,
        transport=transport,
        remote_address="127.0.0.1:11300",
        descriptor=descriptor,
    )

    assert len(transport.sent) == 1
    _, output, remote_address = transport.sent[0]
    assert remote_address == "127.0.0.1:11300"
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_run_offload_exec_batch_once_uses_batched_compute(monkeypatch) -> None:
    import torch

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv_batch(self, descriptor, *, remote_address):
            return torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 3.0],
                ]
            )

        def send_output_batch(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    from examples.pap import pap_attention_executor as executor_module

    calls = []

    def fake_batch_compute(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    def fail_per_request_compute(**kwargs):
        raise AssertionError("batch OFFLOAD_EXEC should use batched compute")

    monkeypatch.setattr(
        executor_module,
        "compute_offload_exec_batch_output",
        fake_batch_compute,
    )
    monkeypatch.setattr(
        executor_module,
        "compute_offload_exec_output",
        fail_per_request_compute,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
            PAPOffloadExecDescriptor(
                request_id="req-b",
                layer_name="layer0",
                step=1,
                scale=1.0,
            ),
        ),
    )
    transport = FakeTransport()

    run_offload_exec_batch_once(
        registry=registry,
        transport=transport,
        remote_address="127.0.0.1:11300",
        descriptor=descriptor,
    )

    assert len(calls) == 1
    assert calls[0]["registry"] is registry
    assert calls[0]["descriptor"] is descriptor
    assert calls[0]["qkv_batch"].shape == (2, 6)
    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))


def test_unified_offload_exec_commit_uses_descriptor_decode_token_ids(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

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

    def fake_append_decode_kv_to_unified_prefill_cache(**kwargs):
        return 1

    commits = []

    class FakeCommitClient:
        enabled = True

        def commit(self, *, request_id, new_seq_len, new_token_ids, endpoint):
            commits.append((request_id, new_seq_len, tuple(new_token_ids), endpoint))

    monkeypatch.setattr(
        registry,
        "append_decode_kv_to_unified_prefill_cache",
        fake_append_decode_kv_to_unified_prefill_cache,
    )
    monkeypatch.setattr(
        executor_module,
        "_compute_unified_paged_flash_batch",
        lambda **kwargs: torch.tensor([[2.0, 0.0]]),
    )
    monkeypatch.setattr(
        executor_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )

    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                request_id="req-a",
                layer_name="layer0",
                step=2,
                scale=1.0,
                decode_token_ids=(42,),
            ),
        ),
    )

    output = compute_offload_exec_batch_output(
        registry=registry,
        descriptor=descriptor,
        qkv_batch=torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]]),
    )

    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))
    assert commits == [
        (
            "req-a",
            2,
            (42,),
            "http://localhost:8100/v1/pap/prefill/decode-commit",
        )
    ]


def test_unified_offload_exec_commit_waits_for_async_decode_token(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")

    commits = []

    class FakeCommitClient:
        def commit(self, *, request_id, new_seq_len, new_token_ids, endpoint):
            commits.append((request_id, new_seq_len, tuple(new_token_ids), endpoint))

    monkeypatch.setattr(
        executor_module,
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
        executor_module,
        "_compute_unified_paged_flash_batch",
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
                decode_token_ids=(),
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
    assert commits == [
        (
            "cmpl-req-a-0",
            2,
            (42,),
            "http://localhost:8100/v1/pap/prefill/decode-commit",
        )
    ]


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


def test_attention_release_waits_for_kv_ready_token_but_not_final_token(
    monkeypatch,
) -> None:
    from examples.pap import pap_attention_executor as executor_module

    commits = []

    class FakeCommitClient:
        def commit(self, **kwargs):
            commits.append(kwargs)

        def flush_request(self, request_id):
            return True

        def forget_request(self, request_id):
            return None

    monkeypatch.setattr(
        executor_module,
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
    registry.record_decode_kv_ready(
        request_id="req-a",
        new_seq_len=2,
        endpoint="http://localhost:8100/v1/pap/prefill/decode-commit",
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
        if getattr(route, "path", "")
        == "/v1/pap/attention/sessions/{request_id}"
        and "DELETE" in getattr(route, "methods", set())
    )

    assert not inspect.iscoroutinefunction(release_endpoint)


def test_unified_offload_exec_overlap_step_does_not_commit(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

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

        def commit(self, *, request_id, new_seq_len, new_token_ids, endpoint):
            commits.append((request_id, new_seq_len, tuple(new_token_ids), endpoint))

    monkeypatch.setattr(
        executor_module.torch.ops._C_cache_ops,
        "reshape_and_cache_flash",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(
        executor_module,
        "_compute_unified_paged_flash_batch",
        lambda **kwargs: torch.tensor([[2.0, 0.0]]),
    )
    monkeypatch.setattr(
        executor_module,
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
                decode_token_ids=(42,),
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module.torch.ops._C_cache_ops,
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


def test_unified_decode_append_reuses_slot_plan_across_layers(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

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
            registry.import_prefill_paged_kv(
                request_id=request_id,
                layer_name=layer_name,
                kv_cache=kv_cache,
                block_ids=[block_id],
                seq_len=1,
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
                lease_id=f"lease-{request_id}",
                lease_capacity_tokens=4,
                unified_kv_mode=True,
                prefix_len=1,
                writable_start_token=1,
                writable_end_token=4,
            )
    calls = []
    monkeypatch.setattr(
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

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
            layer_names=("layer0", "layer1")
            if request_id == "req-a"
            else ("layer0",),
            kv_cache=kv_cache,
            block_ids=next_blocks,
            seq_len=5,
        )

    calls = []
    monkeypatch.setattr(
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module.torch.ops._C_cache_ops,
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


def test_unified_metadata_fast_key_rejects_reused_request_aba() -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    registry = PAPAttentionRegistry(storage_device="cpu")

    def install(block_id: int) -> PAPUnifiedPagedKVState:
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
            layer_names=("layer0",),
            kv_cache=torch.zeros((2, 2, 4, 1, 2)),
            block_ids=(block_id,),
            seq_len=1,
        )
        return registry._unified_paged_kv["req-a"]["layer0"]

    first_state = install(0)
    first = build_unified_paged_flash_metadata(
        states=[first_state],
        device=torch.device("cpu"),
    )
    with registry._lock:
        registry._release_session_locked("req-a")
    second_state = install(1)
    second = build_unified_paged_flash_metadata(
        states=[second_state],
        device=torch.device("cpu"),
    )

    assert second_state.slot_topology_id > first_state.slot_topology_id
    assert first.block_table.tolist() == [[0]]
    assert second.block_table.tolist() == [[1]]
    assert second.block_table.data_ptr() != first.block_table.data_ptr()


def test_unified_metadata_fast_key_mixed_unknown_batch_falls_back() -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    states = [
        _make_unified_state(
            torch,
            block_ids=(3,),
            seq_len=8,
            lease_id="lease-a",
        ),
        _make_unified_state(
            torch,
            block_ids=(7,),
            seq_len=8,
            lease_id="lease-b",
        ),
    ]
    states[0].slot_generation = 1
    states[0].slot_topology_id = 301

    first = build_unified_paged_flash_metadata(
        states=states,
        device=torch.device("cpu"),
    )
    second = build_unified_paged_flash_metadata(
        states=states,
        device=torch.device("cpu"),
    )

    assert second.block_table.data_ptr() == first.block_table.data_ptr()
    stats = executor_module.unified_paged_flash_metadata_cache_stats()
    assert stats["fast_key_lookups"] == 0
    assert stats["full_key_scans"] == 2
    assert stats["block_ids_scanned"] == 4


def test_unified_decode_append_disables_slot_plan_for_mixed_topology(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-a",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
        )
    )
    for layer_name, block_id in (("layer0", 0), ("layer1", 1)):
        registry.import_prefill_paged_kv(
            request_id="req-a",
            layer_name=layer_name,
            kv_cache=torch.zeros((4, 2, 4, 1, 2)),
            block_ids=[block_id],
            seq_len=1,
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            lease_id="lease-a",
            lease_capacity_tokens=4,
            unified_kv_mode=True,
            prefix_len=1,
            writable_start_token=1,
            writable_end_token=4,
        )
    calls = []
    monkeypatch.setattr(
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

    registry = PAPAttentionRegistry(storage_device="cpu")

    def install_session(block_id: int) -> None:
        registry.register_prefill_kv(
            PAPAttentionRegistration(
                request_id="req-a",
                conversation_id="conv",
                prefill_endpoint="http://localhost:8100",
            )
        )
        for layer_name in ("layer0", "layer1"):
            registry.import_prefill_paged_kv(
                request_id="req-a",
                layer_name=layer_name,
                kv_cache=torch.zeros((4, 2, 4, 1, 2)),
                block_ids=[block_id],
                seq_len=1,
                block_size=4,
                num_kv_heads=1,
                layout="NHD",
                lease_id="lease-a",
                lease_capacity_tokens=4,
                unified_kv_mode=True,
                prefix_len=1,
                writable_start_token=1,
                writable_end_token=4,
            )

    calls = []
    monkeypatch.setattr(
        executor_module.torch.ops._C_cache_ops,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module.torch.ops._C_cache_ops,
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
    from examples.pap import pap_attention_executor as executor_module

    executor_module.reset_unified_paged_flash_metadata_cache()
    registry = PAPAttentionRegistry(storage_device="cpu")
    client = _ASGITestClient(create_app(registry=registry))

    response = client.get("/v1/pap/attention/stats")

    assert response.status_code == 200
    assert response.json() == {
        "attention_dispatch_mode": "legacy",
        "attention_active_peer_tracking": False,
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
    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        executor_module,
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


def test_attention_registry_does_not_release_lease_before_commit_ack(
    monkeypatch,
) -> None:
    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        executor_module,
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
    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
        "_get_commit_client",
        lambda: FakeCommitClient(),
    )
    monkeypatch.setattr(
        executor_module,
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

    registry.import_prefill_kv(
        request_id="req-ready",
        layer_name="layer0",
        key=torch.zeros(2, 1, 2),
        value=torch.zeros(2, 1, 2),
        seq_len=2,
        block_ids=[0],
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
    registry.import_prefill_kv(
        request_id="req-ready",
        layer_name="layer0",
        key=torch.zeros(2, 1, 2),
        value=torch.zeros(2, 1, 2),
        seq_len=2,
        block_ids=[0],
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


def test_run_offload_exec_batch_once_single_item_avoids_output_cat(
    monkeypatch,
) -> None:
    import torch

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv_batch(self, descriptor, *, remote_address):
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

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
    transport = FakeTransport()

    from examples.pap import pap_attention_executor as executor_module

    def fake_compute_offload_exec_batch_output(**kwargs):
        return torch.tensor([[2.0, 0.0]])

    def fail_cat(*args, **kwargs):
        raise AssertionError("single-item OFFLOAD_EXEC batch should not cat output")

    monkeypatch.setattr(
        executor_module,
        "compute_offload_exec_batch_output",
        fake_compute_offload_exec_batch_output,
    )
    monkeypatch.setattr(torch, "cat", fail_cat)

    run_offload_exec_batch_once(
        registry=registry,
        transport=transport,
        remote_address="127.0.0.1:11300",
        descriptor=descriptor,
    )

    assert len(transport.sent) == 1
    _, output, remote_address = transport.sent[0]
    assert remote_address == "127.0.0.1:11300"
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


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

        def recv_next_qkv_batch(self):
            raise AssertionError("mailbox loop should preserve message lifetime")

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
    from examples.pap import pap_attention_executor as executor_module

    monkeypatch.setattr(
        executor_module,
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
    }


def test_mailbox_receiver_enqueues_without_computing(monkeypatch) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
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

    from examples.pap import pap_attention_executor as executor_module

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
                    decode_token_ids=(41,),
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
                "t": ((42,),),
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
        executor_module,
        "compute_offload_exec_batch_output",
        fake_compute,
    )
    registry = PAPAttentionRegistry(storage_device="cpu")

    _execute_offload_exec_work_items(registry=registry, items=items)

    assert len(compute_calls) == 1
    request_ids, steps, scales, token_rows = compute_calls[0][0]
    assert request_ids == ("req-a", "req-b")
    assert steps == (3, 7)
    assert scales == (0.5, 0.5)
    assert token_rows == ((41,), (42,))
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
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

    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
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
        executor_module,
        "_wait_offload_exec_ready_event",
        wait_ready,
        raising=False,
    )
    monkeypatch.setattr(
        executor_module,
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

    from examples.pap import pap_attention_executor as executor_module
    from vllm.pap.attention_scheduler import PAPAttentionWorkItem

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
        executor_module,
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


def test_run_offload_exec_mailbox_loop_prefetches_next_qkv_message(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor as executor_module

    events = []
    second_recv_started = Event()

    class FakeMessage:
        def __init__(self, index):
            self.index = index
            self.tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def release(self):
            events.append(f"release{self.index}")

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.recv_calls = 0

        def recv_next_qkv_batch_message(self):
            self.recv_calls += 1
            events.append(f"recv{self.recv_calls}")
            if self.recv_calls == 2:
                second_recv_started.set()
            if self.recv_calls > 2:
                raise KeyboardInterrupt
            return self.descriptor, FakeMessage(self.recv_calls)

        def recv_next_qkv_batch(self):
            raise AssertionError("prefetch should use mailbox message receive")

        def send_output_batch(self, descriptor, output, *, remote_address):
            events.append("send")

    compute_calls = 0

    def fake_compute_batch_output(**kwargs):
        nonlocal compute_calls
        compute_calls += 1
        events.append(f"compute{compute_calls}")
        if compute_calls == 1:
            assert second_recv_started.wait(timeout=1.0)
        return torch.tensor([[2.0, 0.0]])

    monkeypatch.setenv("PAP_ATTENTION_MAILBOX_PREFETCH", "1")
    monkeypatch.setattr(
        executor_module,
        "compute_offload_exec_batch_output",
        fake_compute_batch_output,
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
    transport = FakeTransport(descriptor)

    with suppress(KeyboardInterrupt):
        run_offload_exec_mailbox_loop(
            registry=PAPAttentionRegistry(storage_device="cpu"),
            transport=transport,
        )

    assert events.index("recv2") < events.index("release1")
    assert events.index("recv2") < events.index("send")


def test_run_offload_exec_mailbox_loop_emits_trace(monkeypatch, caplog) -> None:
    import torch

    class FakeTransport:
        def __init__(self, descriptor):
            self.descriptor = descriptor
            self.sent = []
            self.recv_calls = 0

        def recv_next_qkv_batch(self):
            self.recv_calls += 1
            if self.recv_calls > 1:
                raise KeyboardInterrupt
            return self.descriptor, torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

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
    from examples.pap import pap_attention_executor as executor_module

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
        executor_module,
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
    from examples.pap import pap_attention_executor as executor_module

    monkeypatch.setattr(
        executor_module,
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


def test_run_offload_exec_once_resolves_wrapped_request_id(monkeypatch) -> None:
    import torch

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "cmpl-req-offload-0-deadbeef"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    descriptor = PAPOffloadExecDescriptor(
        request_id="cmpl-req-offload-0-deadbeef",
        layer_name="layer0",
        step=1,
        scale=1.0,
    )

    run_offload_exec_once(
        registry=registry,
        transport=transport,
        remote_address="127.0.0.1:11300",
        descriptor=descriptor,
    )

    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_offload_exec_endpoint_triggers_transport(monkeypatch) -> None:
    import anyio
    import torch
    from httpx import ASGITransport, AsyncClient

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "req-offload"
            assert descriptor.step == 1
            assert remote_address == "127.0.0.1:11300"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    app = create_app(registry)
    transport = FakeTransport()
    app.state.offload_exec_transport = transport

    async def run_request():
        asgi_transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=asgi_transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/pap/attention/offload-exec",
                json={
                    "request_id": "req-offload",
                    "layer_name": "layer0",
                    "step": 1,
                    "scale": 1.0,
                    "remote_address": "127.0.0.1:11300",
                },
            )

    response = anyio.run(run_request)

    assert response.status_code == 200
    assert response.json()["remote_address"] == "127.0.0.1:11300"
    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


def test_offload_exec_binary_command_triggers_transport(monkeypatch) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "req-offload"
            assert descriptor.step == 1
            assert remote_address == "127.0.0.1:11300"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    class TrackingLock:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, exc_type, exc, traceback):
            return False

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    lock = TrackingLock()

    response = compute_binary_attention_response(
        registry,
        serialize_tensor_bundle(
            {
                "command": "offload_exec",
                "request_id": "req-offload",
                "layer_name": "layer0",
                "step": 1,
                "scale": 1.0,
                "remote_address": "127.0.0.1:11300",
            },
            {},
        ),
        offload_exec_transport=transport,
        offload_exec_lock=lock,
    )

    metadata, tensors = deserialize_tensor_bundle(response)
    assert metadata["request_id"] == "req-offload"
    assert tensors == {}
    assert lock.entered
    assert len(transport.sent) == 1


def test_offload_exec_compact_command_triggers_transport(monkeypatch) -> None:
    import torch

    from vllm.pap.remote_attention import (
        deserialize_compact_offload_exec_ack,
        serialize_compact_offload_exec_command,
    )

    class FakeTransport:
        def __init__(self):
            self.sent = []

        def recv_qkv(self, descriptor, *, remote_address):
            assert descriptor.request_id == "req-offload"
            assert descriptor.step == 1
            assert remote_address == "127.0.0.1:11300"
            return torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])

        def send_output(self, descriptor, output, *, remote_address):
            self.sent.append((descriptor, output, remote_address))

    class TrackingLock:
        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, exc_type, exc, traceback):
            return False

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-offload",
            conversation_id="conv",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    transport = FakeTransport()
    lock = TrackingLock()

    response = compute_binary_attention_response(
        registry,
        serialize_compact_offload_exec_command(
            request_id="req-offload",
            layer_name="layer0",
            step=1,
            scale=1.0,
            remote_address="127.0.0.1:11300",
        ),
        offload_exec_transport=transport,
        offload_exec_lock=lock,
    )

    deserialize_compact_offload_exec_ack(response)
    assert lock.entered
    assert len(transport.sent) == 1
    _, output, _ = transport.sent[0]
    torch.testing.assert_close(output, torch.tensor([[2.0, 0.0]]))


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
    assert snapshot.prefill_kv_handle == "req-1"
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


def test_attention_registry_can_pin_state_to_configured_device() -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-device",
            conversation_id="conv-device",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
            prefix_len=1,
        )
    )

    registry.import_prefill_kv(
        request_id="req-device",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[1.0, 0.0]]]),
        value=torch.tensor([[[2.0, 0.0]]]),
        seq_len=1,
    )
    segments, seq_len = registry.append_decode_kv(
        request_id="req-device",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[0.0, 1.0]]]),
        value=torch.tensor([[[0.0, 4.0]]]),
    )

    assert seq_len == 2
    assert registry.storage_device.type == "cpu"
    assert all(
        tensor.device.type == "cpu" for segment in segments for tensor in segment
    )


def test_attention_registry_waits_for_layer_prefill_before_decode(monkeypatch) -> None:
    import torch

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-wait",
            conversation_id="conv-wait",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={},
        )
    )
    monkeypatch.setenv("PAP_ATTENTION_PREFILL_WAIT_TIMEOUT", "2.0")

    result = {}

    def append_decode() -> None:
        segments, seq_len = registry.append_decode_kv(
            request_id="req-wait",
            layer_name="model.layers.0.self_attn.attn",
            key=torch.tensor([[[0.0, 1.0]]]),
            value=torch.tensor([[[0.0, 4.0]]]),
            block_id=0,
            slot=1,
            seq_len=2,
        )
        result["seq_len"] = seq_len
        result["segments"] = segments

    thread = Thread(target=append_decode)
    thread.start()
    registry.import_prefill_kv(
        request_id="req-wait",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[1.0, 0.0]]]),
        value=torch.tensor([[[2.0, 0.0]]]),
        seq_len=1,
    )
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert result["seq_len"] == 2
    assert len(result["segments"]) == 2


def test_attention_registry_records_layer_event_without_payload_tensors() -> None:
    registry = PAPAttentionRegistry()
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-4",
            conversation_id="conv-4",
            prefill_endpoint="http://localhost:8100",
            kv_transfer_params={"remote_engine_id": "prefill-0"},
        )
    )

    event = registry.record_layer_event(
        request_id="cmpl-req-4-0-deadbeef",
        layer_name="model.layers.0.self_attn.attn",
        query_shape=[1, 32, 128],
        key_shape=[1, 8, 128],
        value_shape=[1, 8, 128],
        dtype="torch.bfloat16",
        device="cuda:0",
        is_decode=True,
        num_reqs=1,
        num_actual_tokens=1,
        max_seq_len=9,
    )

    assert event.request_id == "cmpl-req-4-0-deadbeef"
    assert event.session_request_id == "req-4"
    assert event.layer_name == "model.layers.0.self_attn.attn"
    assert event.query_shape == [1, 32, 128]
    assert event.is_decode is True

    stored = registry.get_layer_events("req-4")
    assert len(stored) == 1
    assert stored[0].max_seq_len == 9


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


def test_attention_executor_compute_route_returns_attention_output() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    response = client.post(
        "/v1/pap/attention/compute",
        json={
            "request_id": "cmpl-unit-0-deadbeef",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(query),
            "key": serialize_tensor(key),
            "value": serialize_tensor(value),
            "scale": 1.0,
        },
    )

    assert response.status_code == 200
    output = deserialize_attention_result(response.json()["output"])
    assert output.shape == (1, 2, 2)
    assert output.dtype == torch.float32


def test_attention_executor_append_and_compute_keeps_stateful_kv() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-stateful",
            "conversation_id": "conv-stateful",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 0,
        },
    )

    first = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-stateful",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
        },
    )
    assert first.status_code == 200
    first_output = deserialize_attention_result(first.json()["output"])
    assert torch.allclose(first_output, torch.tensor([[[2.0, 0.0]]]))

    second = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-stateful",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "key": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[0.0, 4.0]]])),
            "scale": 1.0,
        },
    )
    assert second.status_code == 200
    second_output = deserialize_attention_result(second.json()["output"])

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(second_output, expected)

    session = client.get("/v1/pap/attention/sessions/req-stateful").json()
    assert session["decode_seq_lens"] == {"model.layers.0.self_attn.attn": 2}


def test_attention_executor_binary_append_and_compute_keeps_stateful_kv() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-binary-stateful",
            "conversation_id": "conv-binary-stateful",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute-binary",
        content=serialize_tensor_bundle(
            {
                "request_id": "req-binary-stateful",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
            },
            {
                "query": torch.tensor([[[1.0, 0.0]]]),
                "key": torch.tensor([[[1.0, 0.0]]]),
                "value": torch.tensor([[[2.0, 0.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(response.content)
    assert metadata["request_id"] == "req-binary-stateful"
    assert metadata["seq_len"] == 1
    assert torch.allclose(tensors["output"], torch.tensor([[[2.0, 0.0]]]))


def test_attention_executor_batch_binary_append_and_compute_keeps_stateful_kv() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    app = create_app()
    client = _ASGITestClient(app)
    for request_id in ("req-batch-0", "req-batch-1"):
        client.post(
            "/v1/pap/attention/register",
            json={
                "request_id": request_id,
                "conversation_id": "conv-batch",
                "prefill_endpoint": "http://localhost:8100",
                "kv_transfer_params": {},
            },
        )

    response = client.post(
        "/v1/pap/attention/append-and-compute-batch-binary",
        content=serialize_tensor_bundle(
            {
                "items": [
                    {
                        "request_id": "req-batch-0",
                        "layer_name": "model.layers.0.self_attn.attn",
                        "scale": 1.0,
                    },
                    {
                        "request_id": "req-batch-1",
                        "layer_name": "model.layers.0.self_attn.attn",
                        "scale": 1.0,
                    },
                ]
            },
            {
                "query_0": torch.tensor([[[1.0, 0.0]]]),
                "key_0": torch.tensor([[[1.0, 0.0]]]),
                "value_0": torch.tensor([[[2.0, 0.0]]]),
                "query_1": torch.tensor([[[0.0, 1.0]]]),
                "key_1": torch.tensor([[[0.0, 1.0]]]),
                "value_1": torch.tensor([[[0.0, 4.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(response.content)
    assert [item["request_id"] for item in metadata["items"]] == [
        "req-batch-0",
        "req-batch-1",
    ]
    assert torch.allclose(tensors["output_0"], torch.tensor([[[2.0, 0.0]]]))
    assert torch.allclose(tensors["output_1"], torch.tensor([[[0.0, 4.0]]]))


def test_attention_executor_batch_binary_accepts_packed_qkv(monkeypatch) -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_NUM_KV_HEADS", "1")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_HEAD_DIM", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_Q_SIZE", "2")
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_KV_SIZE", "2")
    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-packed",
            "conversation_id": "conv-packed",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute-batch-binary",
        content=serialize_tensor_bundle(
            {
                "items": [
                    {
                        "request_id": "req-packed",
                        "layer_name": "model.layers.0.self_attn.attn",
                        "scale": 1.0,
                    },
                ]
            },
            {"qkv_0": torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    _metadata, tensors = deserialize_tensor_bundle(response.content)
    assert torch.allclose(tensors["output_0"], torch.tensor([[[2.0, 0.0]]]))


def test_attention_executor_compact_tcp_payload_keeps_stateful_kv() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
        compute_binary_attention_response,
    )
    from vllm.pap.remote_attention import (
        deserialize_compact_attention_response,
        serialize_compact_attention_batch,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-compact",
            conversation_id="conv-compact",
            prefill_endpoint="http://localhost:8100",
            q_size=2,
            kv_size=2,
        )
    )
    payload = serialize_compact_attention_batch(
        [
            {
                "request_id": "req-compact",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
                "block_id": 0,
                "slot": 0,
                "seq_len": 1,
                "q_size": 2,
                "kv_size": 2,
                "num_heads": 1,
                "num_kv_heads": 1,
                "head_dim": 2,
            },
        ],
        [torch.tensor([[1.0, 0.0, 1.0, 0.0, 2.0, 0.0]])],
    )

    response = compute_binary_attention_response(registry, payload)
    outputs = deserialize_compact_attention_response(response)

    assert torch.allclose(outputs[0], torch.tensor([[[2.0, 0.0]]]))


def test_attention_executor_rejects_decode_when_prompt_kv_is_missing() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-missing",
            "conversation_id": "conv-prefix-missing",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-prefix-missing",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
        },
    )

    assert response.status_code == 409
    assert "prefill KV" in response.json()["detail"]


def test_attention_executor_imports_prefill_kv_before_stateful_decode() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-imported",
            "conversation_id": "conv-prefix-imported",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv",
        json={
            "request_id": "req-prefix-imported",
            "layer_name": "model.layers.0.self_attn.attn",
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])),
            "seq_len": 2,
        },
    )
    assert imported.status_code == 200
    assert imported.json()["seq_len"] == 2

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-prefix-imported",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "key": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[0.0, 8.0]]])),
            "scale": 1.0,
        },
    )

    assert response.status_code == 200
    output = deserialize_attention_result(response.json()["output"])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]], [[0.0, 8.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(output, expected)

    session = client.get("/v1/pap/attention/sessions/req-prefix-imported").json()
    assert session["prefill_seq_lens"] == {"model.layers.0.self_attn.attn": 2}
    assert session["decode_seq_lens"] == {"model.layers.0.self_attn.attn": 3}


def test_attention_executor_binary_imports_prefill_kv_before_decode() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-binary",
            "conversation_id": "conv-prefix-binary",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_kv",
                "request_id": "req-prefix-binary",
                "layer_name": "model.layers.0.self_attn.attn",
                "seq_len": 2,
                "block_ids": [4],
            },
            {
                "key": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                "value": torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 2
    assert tensors == {}

    response = client.post(
        "/v1/pap/attention/append-and-compute-binary",
        content=serialize_tensor_bundle(
            {
                "request_id": "req-prefix-binary",
                "layer_name": "model.layers.0.self_attn.attn",
                "scale": 1.0,
                "block_id": 0,
                "slot": 2,
                "seq_len": 3,
            },
            {
                "query": torch.tensor([[[0.0, 1.0]]]),
                "key": torch.tensor([[[0.0, 1.0]]]),
                "value": torch.tensor([[[0.0, 8.0]]]),
            },
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(response.content)
    assert metadata["seq_len"] == 3
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]], [[0.0, 8.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(tensors["output"], expected)


def test_attention_executor_binary_imports_prefill_kv_ipc_descriptor(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor
    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    def fake_open_ipc_prefill_kv(descriptor):
        assert descriptor.request_id == "req-prefix-ipc"
        assert descriptor.seq_len == 2
        return key, value

    monkeypatch.setattr(
        pap_attention_executor,
        "open_ipc_prefill_kv",
        fake_open_ipc_prefill_kv,
    )

    descriptor = PAPOffloadKVIPCDescriptor(
        request_id="req-prefix-ipc",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=2,
        block_ids=(4,),
        key=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(key.shape),
            ipc_handle={"GPU-test": ("key", 1, 2, 3, 4, 5, 0)},
        ),
        value=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(value.shape),
            ipc_handle={"GPU-test": ("value", 1, 2, 3, 4, 5, 0)},
        ),
    )
    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-ipc",
            "conversation_id": "conv-prefix-ipc",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_kv_ipc",
                "descriptor": descriptor.to_dict(),
            },
            {},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 2
    assert tensors == {}
    session = client.get("/v1/pap/attention/sessions/req-prefix-ipc").json()
    assert session["prefill_seq_lens"] == {"model.layers.0.self_attn.attn": 2}


def test_attention_executor_ipc_import_keeps_opened_tensor_views(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor
    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    def fake_open_ipc_prefill_kv(descriptor):
        return key, value

    monkeypatch.setattr(
        pap_attention_executor,
        "open_ipc_prefill_kv",
        fake_open_ipc_prefill_kv,
    )

    descriptor = PAPOffloadKVIPCDescriptor(
        request_id="req-prefill-ipc",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=2,
        block_ids=(4,),
        key=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(key.shape),
            ipc_handle={"GPU-test": ("key", 1, 2, 3, 4, 5, 0)},
        ),
        value=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(value.shape),
            ipc_handle={"GPU-test": ("value", 1, 2, 3, 4, 5, 0)},
        ),
    )
    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefill-ipc",
            "conversation_id": "conv-prefill-ipc",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_kv_ipc",
                "descriptor": descriptor.to_dict(),
            },
            {},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 2
    assert tensors == {}
    registry = app.state.registry
    session_id = registry.resolve_session_request_id("req-prefill-ipc")
    [(stored_key, stored_value)] = registry._prefill_kv[session_id][
        "model.layers.0.self_attn.attn"
    ]
    assert stored_key.data_ptr() == key.data_ptr()
    assert stored_value.data_ptr() == value.data_ptr()


def test_attention_executor_paged_ipc_import_keeps_prefill_block_views(
    monkeypatch,
) -> None:
    import torch

    from examples.pap import pap_attention_executor
    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    kv_cache = torch.zeros((2, 2, 4, 1, 2))
    for block in range(2):
        for offset in range(4):
            kv_cache[0, block, offset] = block * 100 + offset * 10 + 1
            kv_cache[1, block, offset] = block * 100 + offset * 10 + 2

    def fake_open_ipc_paged_kv_cache(descriptor):
        return kv_cache

    monkeypatch.setattr(
        pap_attention_executor,
        "open_ipc_paged_kv_cache",
        fake_open_ipc_paged_kv_cache,
    )

    descriptor = PAPOffloadKVPagedIPCDescriptor(
        request_id="req-paged-ipc",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=5,
        block_ids=(0, 1),
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
        kv_cache=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(kv_cache.shape),
            ipc_handle={"GPU-test": ("kv", 1, 2, 3, 4, 5, 0)},
        ),
    )
    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-paged-ipc",
            "conversation_id": "conv-paged-ipc",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 5,
            "block_size": 4,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_paged_kv_ipc",
                "descriptor": descriptor.to_dict(),
            },
            {},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 5
    assert tensors == {}
    registry = app.state.registry
    session_id = registry.resolve_session_request_id("req-paged-ipc")
    segments, seq_len = registry.append_decode_kv(
        request_id="req-paged-ipc",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[9.0, 9.0]]]),
        value=torch.tensor([[[10.0, 10.0]]]),
        block_id=1,
        slot=5,
        seq_len=6,
    )

    assert seq_len == 6
    assert len(segments) == 3
    assert (
        segments[0][0].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )
    assert torch.equal(
        torch.cat([key.detach().cpu() for key, _ in segments], dim=0),
        torch.cat(
            [
                kv_cache[0, 0, :4, :1, :],
                kv_cache[0, 1, :1, :1, :],
                torch.tensor([[[9.0, 9.0]]]),
            ],
            dim=0,
        ),
    )
    assert "model.layers.0.self_attn.attn" in registry._decode_kv[session_id]


def test_attention_registry_keeps_decode_kv_out_of_prefill_paged_block() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-paged-local-decode",
            conversation_id="conv-paged-local-decode",
            prefill_endpoint="http://localhost:8100",
            prefix_len=5,
            block_size=4,
        )
    )

    kv_cache = torch.zeros((2, 2, 4, 1, 2))
    kv_cache[0, 0, :, :, :] = 1
    kv_cache[1, 0, :, :, :] = 2
    kv_cache[0, 1, 0, :, :] = 3
    kv_cache[1, 1, 0, :, :] = 4

    registry.import_prefill_paged_kv(
        request_id="req-paged-local-decode",
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        block_ids=[0, 1],
        seq_len=5,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
    )

    decode_key = torch.tensor([[[9.0, 10.0]]])
    decode_value = torch.tensor([[[11.0, 12.0]]])
    segments, seq_len = registry.append_decode_kv(
        request_id="req-paged-local-decode",
        layer_name="model.layers.0.self_attn.attn",
        key=decode_key,
        value=decode_value,
        block_id=1,
        slot=5,
        seq_len=6,
    )

    assert seq_len == 6
    assert len(segments) == 3
    assert torch.equal(kv_cache[0, 1, 1, :, :], torch.zeros((1, 2)))
    assert torch.equal(kv_cache[1, 1, 1, :, :], torch.zeros((1, 2)))
    assert torch.equal(
        torch.cat([key for key, _ in segments], dim=0),
        torch.cat(
            [kv_cache[0, 0, :4, :1, :], kv_cache[0, 1, :1, :1, :], decode_key],
            dim=0,
        ),
    )
    session_id = registry.resolve_session_request_id("req-paged-local-decode")
    decode_key_buffer, decode_value_buffer = registry._decode_kv[session_id][
        "model.layers.0.self_attn.attn"
    ].view()
    assert torch.equal(decode_key_buffer, decode_key)
    assert torch.equal(decode_value_buffer, decode_value)


def test_attention_registry_keeps_in_block_decode_tokens_attention_local() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-paged-capacity",
            conversation_id="conv-paged-capacity",
            prefill_endpoint="http://localhost:8100",
            prefix_len=12,
            block_size=16,
        )
    )

    kv_cache = torch.zeros((2, 1, 16, 1, 2))
    registry.import_prefill_paged_kv(
        request_id="req-paged-capacity",
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        block_ids=[0],
        seq_len=12,
        block_size=16,
        num_kv_heads=1,
        layout="NHD",
    )

    for seq_len in range(13, 17):
        segments, written_seq_len = registry.append_decode_kv(
            request_id="req-paged-capacity",
            layer_name="model.layers.0.self_attn.attn",
            key=torch.full((1, 1, 2), float(seq_len)),
            value=torch.full((1, 1, 2), float(seq_len + 100)),
            block_id=0,
            slot=seq_len - 1,
            seq_len=seq_len,
        )

        assert written_seq_len == seq_len
        assert len(segments) == 2
        assert torch.equal(kv_cache[0, 0, seq_len - 1], torch.zeros((1, 2)))
        assert torch.equal(kv_cache[1, 0, seq_len - 1], torch.zeros((1, 2)))

    session_id = registry.resolve_session_request_id("req-paged-capacity")
    decode_key_buffer, decode_value_buffer = registry._decode_kv[session_id][
        "model.layers.0.self_attn.attn"
    ].view()
    assert decode_key_buffer.shape[0] == 4
    assert torch.equal(decode_key_buffer[:, 0, 0], torch.arange(13, 17).float())
    assert torch.equal(decode_value_buffer[:, 0, 0], torch.arange(113, 117).float())


def test_attention_registry_uses_local_decode_when_prefill_block_full() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-paged-full",
            conversation_id="conv-paged-full",
            prefill_endpoint="http://localhost:8100",
            prefix_len=16,
            block_size=32,
        )
    )

    kv_cache = torch.zeros((2, 1, 16, 1, 2))
    registry.import_prefill_paged_kv(
        request_id="req-paged-full",
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        block_ids=[0],
        seq_len=16,
        block_size=32,
        num_kv_heads=1,
        layout="NHD",
    )

    segments, seq_len = registry.append_decode_kv(
        request_id="req-paged-full",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[17.0, 17.0]]]),
        value=torch.tensor([[[117.0, 117.0]]]),
        block_id=0,
        slot=16,
        seq_len=17,
    )

    assert seq_len == 17
    assert len(segments) == 2
    assert segments[0][0].shape[0] == 16
    assert torch.equal(segments[1][0], torch.tensor([[[17.0, 17.0]]]))
    session_id = registry.resolve_session_request_id("req-paged-full")
    assert "model.layers.0.self_attn.attn" in registry._decode_kv[session_id]


def test_attention_registry_keeps_new_decode_block_attention_local() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
    )

    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-paged-new-block",
            conversation_id="conv-paged-new-block",
            prefill_endpoint="http://localhost:8100",
            prefix_len=4,
            block_size=4,
        )
    )

    kv_cache = torch.zeros((2, 2, 4, 1, 2))
    kv_cache[0, 0, :, :, :] = 1
    kv_cache[1, 0, :, :, :] = 2
    registry.import_prefill_paged_kv(
        request_id="req-paged-new-block",
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=kv_cache,
        block_ids=[0],
        seq_len=4,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
    )

    segments, seq_len = registry.append_decode_kv(
        request_id="req-paged-new-block",
        layer_name="model.layers.0.self_attn.attn",
        key=torch.tensor([[[5.0, 6.0]]]),
        value=torch.tensor([[[7.0, 8.0]]]),
        block_id=1,
        slot=4,
        seq_len=5,
    )

    assert seq_len == 5
    assert len(segments) == 2
    assert torch.equal(kv_cache[0, 1, 0, :, :], torch.zeros((1, 2)))
    assert torch.equal(kv_cache[1, 1, 0, :, :], torch.zeros((1, 2)))
    assert torch.equal(
        torch.cat([key for key, _ in segments], dim=0),
        torch.cat(
            [kv_cache[0, 0, :4, :1, :], torch.tensor([[[5.0, 6.0]]])],
            dim=0,
        ),
    )
    session_id = registry.resolve_session_request_id("req-paged-new-block")
    assert registry._prefill_paged_kv[session_id][
        "model.layers.0.self_attn.attn"
    ].block_ids == [0]
    decode_key, decode_value = registry._decode_kv[session_id][
        "model.layers.0.self_attn.attn"
    ].view()
    assert torch.equal(decode_key, torch.tensor([[[5.0, 6.0]]]))
    assert torch.equal(decode_value, torch.tensor([[[7.0, 8.0]]]))


def test_attention_registry_reimport_preserves_local_decode_kv() -> None:
    import torch

    from examples.pap.pap_attention_executor import (
        PAPAttentionRegistration,
        PAPAttentionRegistry,
    )

    layer_name = "model.layers.0.self_attn.attn"
    registry = PAPAttentionRegistry(storage_device="cpu")
    registry.register_prefill_kv(
        PAPAttentionRegistration(
            request_id="req-prefill-reimport",
            conversation_id="conv-prefill-reimport",
            prefill_endpoint="http://localhost:8100",
            prefix_len=None,
            block_size=4,
        )
    )
    kv_cache = torch.zeros((2, 2, 4, 1, 2))
    registry.import_prefill_paged_kv(
        request_id="req-prefill-reimport",
        layer_name=layer_name,
        kv_cache=kv_cache,
        block_ids=[0],
        seq_len=4,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
    )
    registry.append_decode_kv(
        request_id="req-prefill-reimport",
        layer_name=layer_name,
        key=torch.tensor([[[5.0, 6.0]]]),
        value=torch.tensor([[[7.0, 8.0]]]),
        block_id=1,
        slot=4,
        seq_len=5,
    )

    registry.import_prefill_paged_kv(
        request_id="req-prefill-reimport",
        layer_name=layer_name,
        kv_cache=kv_cache,
        block_ids=[0],
        seq_len=4,
        block_size=4,
        num_kv_heads=1,
        layout="NHD",
    )

    session_id = registry.resolve_session_request_id("req-prefill-reimport")

    assert registry._prefill_paged_kv[session_id][layer_name].block_ids == [0]
    assert (
        sum(
            int(segment_key.shape[0])
            for segment_key, _ in registry._prefill_kv[session_id][layer_name]
        )
        == 4
    )
    decode_key, decode_value = registry._decode_kv[session_id][layer_name].view()
    assert torch.equal(decode_key, torch.tensor([[[5.0, 6.0]]]))
    assert torch.equal(decode_value, torch.tensor([[[7.0, 8.0]]]))


def test_attention_executor_compute_existing_prefill_token_does_not_append() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import deserialize_attention_result, serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-prefix-existing",
            "conversation_id": "conv-prefix-existing",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
            "block_size": 16,
        },
    )
    client.post(
        "/v1/pap/attention/import-prefill-kv",
        json={
            "request_id": "req-prefix-existing",
            "layer_name": "model.layers.0.self_attn.attn",
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])),
            "seq_len": 2,
            "block_ids": [4],
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-prefix-existing",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
            "key": serialize_tensor(torch.tensor([[[9.0, 9.0]]])),
            "value": serialize_tensor(torch.tensor([[[9.0, 9.0]]])),
            "scale": 1.0,
            "block_id": 4,
            "slot": 4 * 16 + 1,
            "seq_len": 2,
        },
    )

    assert response.status_code == 200
    output = deserialize_attention_result(response.json()["output"])
    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])
    query = torch.tensor([[[0.0, 1.0]]])
    scores = torch.einsum("qhd,khd->qhk", query, key)
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probs, value)
    assert torch.allclose(output, expected)

    session = client.get("/v1/pap/attention/sessions/req-prefix-existing").json()
    assert session["block_ids"] == [4]
    assert session["seq_len"] == 2
    assert session["decode_seq_lens"] == {"model.layers.0.self_attn.attn": 2}


def test_attention_executor_tracks_decode_progress_per_layer() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-layer-progress",
            "conversation_id": "conv-layer-progress",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
            "block_size": 16,
        },
    )
    for layer_name in (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ):
        imported = client.post(
            "/v1/pap/attention/import-prefill-kv",
            json={
                "request_id": "req-layer-progress",
                "layer_name": layer_name,
                "key": serialize_tensor(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])),
                "value": serialize_tensor(torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])),
                "seq_len": 2,
                "block_ids": [4],
            },
        )
        assert imported.status_code == 200

    for layer_name in (
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ):
        response = client.post(
            "/v1/pap/attention/append-and-compute",
            json={
                "request_id": "req-layer-progress",
                "layer_name": layer_name,
                "query": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
                "key": serialize_tensor(torch.tensor([[[0.0, 1.0]]])),
                "value": serialize_tensor(torch.tensor([[[0.0, 8.0]]])),
                "scale": 1.0,
                "block_id": 99,
                "slot": 99 * 16 + 2,
                "seq_len": 3,
            },
        )
        assert response.status_code == 200
        assert response.json()["seq_len"] == 3

    session = client.get("/v1/pap/attention/sessions/req-layer-progress").json()
    assert session["seq_len"] == 3
    assert session["block_ids"] == [4, 99]
    assert session["decode_seq_lens"] == {
        "model.layers.0.self_attn.attn": 3,
        "model.layers.1.self_attn.attn": 3,
    }


def test_attention_executor_records_scheduler_descriptor_state() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    registered = client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-descriptor",
            "conversation_id": "conv-descriptor",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 0,
            "block_size": 16,
            "max_seq_len": 64,
        },
    )
    assert registered.status_code == 200
    assert registered.json()["block_size"] == 16
    assert registered.json()["max_seq_len"] == 64

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-descriptor",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
            "block_id": 4,
            "slot": 4 * 16,
            "seq_len": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["seq_len"] == 1
    session = client.get("/v1/pap/attention/sessions/req-descriptor").json()
    assert session["block_ids"] == [4]
    assert session["seq_len"] == 1


def test_attention_executor_rejects_bad_scheduler_descriptor() -> None:
    import torch

    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import serialize_tensor

    app = create_app()
    client = _ASGITestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-bad-descriptor",
            "conversation_id": "conv-bad-descriptor",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 0,
            "block_size": 16,
            "max_seq_len": 64,
        },
    )

    response = client.post(
        "/v1/pap/attention/append-and-compute",
        json={
            "request_id": "req-bad-descriptor",
            "layer_name": "model.layers.0.self_attn.attn",
            "query": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "key": serialize_tensor(torch.tensor([[[1.0, 0.0]]])),
            "value": serialize_tensor(torch.tensor([[[2.0, 0.0]]])),
            "scale": 1.0,
            "block_id": 4,
            "slot": 4 * 16 + 1,
            "seq_len": 1,
        },
    )

    assert response.status_code == 400
    assert "slot" in response.json()["detail"]
    session = client.get("/v1/pap/attention/sessions/req-bad-descriptor").json()
    assert session["block_ids"] == []
    assert session["seq_len"] == 0
