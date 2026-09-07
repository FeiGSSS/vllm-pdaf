# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.pap.integration.engine import PAPEngineControl
from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.kv import lease as pap_lease
from vllm.pap.kv_connector import PAPPrefillConnector
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus


class _Request:
    def __init__(self) -> None:
        self.request_id = "req-1"
        self.num_computed_tokens = 2
        self.all_token_ids = [10, 11, 12]

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    def append_output_token_ids(self, tokens: tuple[int, ...]) -> None:
        self.all_token_ids.extend(tokens)


class _Coordinator:
    def __init__(self) -> None:
        self.cached: list[tuple[str, int]] = []

    def cache_blocks(self, request: _Request, seq_len: int) -> None:
        self.cached.append((request.request_id, seq_len))


def _control(monkeypatch: pytest.MonkeyPatch) -> tuple[PAPEngineControl, _Request]:
    request = _Request()
    manager = SimpleNamespace(enable_caching=True, coordinator=_Coordinator())
    scheduler = SimpleNamespace(
        requests={request.request_id: request},
        finished_req_ids=set(),
        kv_cache_manager=manager,
    )
    monkeypatch.setattr(
        "vllm.pap.integration.engine.pap_lease.pap_refresh_lease",
        lambda _: None,
    )
    monkeypatch.setattr(
        "vllm.pap.integration.engine.pap_lease.pap_update_kv_seq_len",
        lambda *_: True,
    )
    return PAPEngineControl(scheduler), request


def test_control_applies_contiguous_commits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    payload = {
        "request_id": request.request_id,
        "commit_seq": 1,
        "new_seq_len": 4,
        "new_token_ids": [12, 13],
    }

    applied = control.apply("decode_commit", payload)
    duplicate = control.apply("decode_commit", payload)

    assert applied["applied"] is True
    assert applied["acked_commit_seq"] == 1
    assert request.num_computed_tokens == 4
    assert request.all_token_ids == [10, 11, 12, 13]
    assert duplicate["idempotent"] is True


def test_control_rejects_gap_and_conflicting_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    first = {
        "request_id": request.request_id,
        "commit_seq": 1,
        "new_seq_len": 3,
        "new_token_ids": [12],
    }
    control.apply("decode_commit", first)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        control.apply("decode_commit", {**first, "new_token_ids": [99]})
    with pytest.raises(ValueError, match="non-contiguous"):
        control.apply(
            "decode_commit",
            {
                "request_id": request.request_id,
                "commit_seq": 3,
                "new_seq_len": 4,
                "new_token_ids": [13],
            },
        )


def test_commit_resolves_session_handle_and_release_clears_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attention handles and Gateway IDs must resolve to the leased owner."""
    control, request = _control(monkeypatch)
    handle = "conversation@pap-session-5"
    control._scheduler.connector = SimpleNamespace(
        _request_metadata={
            request.request_id: PAPRequestMetadata(prefill_kv_handle=handle)
        }
    )
    registry = pap_lease.PAPKVLeaseRegistry()
    monkeypatch.setattr(pap_lease, "get_global_kv_lease_registry", lambda: registry)
    lease_id = registry.pin_blocks(
        request_id=request.request_id, block_ids=[1], ttl_seconds=0
    )
    payload = {
        "request_id": handle,
        "commit_seq": 1,
        "new_seq_len": 4,
        "new_token_ids": [12, 13],
    }
    result = control.apply("decode_commit", payload)
    assert result["applied"]
    assert result["request_id"] == request.request_id
    assert request.all_token_ids == [10, 11, 12, 13]
    assert control._scheduler.kv_cache_manager.coordinator.cached == [("req-1", 4)]
    assert control.apply("decode_commit", payload)["idempotent"]
    release = {
        "request_id": "conversation",
        "lease_id": lease_id,
        "final_commit_seq": 1,
    }
    result = control.apply("lease_release", release)
    assert result["released"]
    assert registry.active_entry(lease_id) is None
    assert not control._commits
    retry = control.apply("lease_release", release)
    assert not retry["released"]
    assert retry["reason"] == "unknown_or_released_lease"


def test_stale_session_handle_cannot_commit_to_new_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    control._scheduler.connector = SimpleNamespace(
        _request_metadata={
            request.request_id: PAPRequestMetadata(
                prefill_kv_handle="conversation@pap-session-6"
            )
        }
    )
    result = control.apply(
        "decode_commit",
        {
            "request_id": "conversation@pap-session-5",
            "commit_seq": 1,
            "new_seq_len": 4,
            "new_token_ids": [12, 13],
        },
    )
    assert result["applied"] is False
    assert request.num_computed_tokens == 2
    assert not control._scheduler.kv_cache_manager.coordinator.cached


def test_projection_commit_replaces_only_private_prefill_sample(monkeypatch):
    """A discarded Prefill sample must not poison accepted tokens or hashes."""
    control, _ = _control(monkeypatch)

    def hash_blocks(request):
        tokens = list(request.all_token_ids)
        return [
            tuple(tokens[i : i + 2])
            for i in range(2 * len(request.block_hashes), len(tokens) - 1, 2)
        ]

    request = Request(
        request_id="prefill-owner",
        prompt_token_ids=[10, 11, 12],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=hash_blocks,
    )
    request.kv_transfer_params = {"pap_import_prefill_kv_to_attention": True}
    request.num_computed_tokens = 3
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    request.append_output_token_ids(99)
    assert request.block_hashes == [(10, 11), (12, 99)]
    control._scheduler.requests = {request.request_id: request}
    result = control.apply(
        "decode_commit",
        {
            "request_id": request.request_id,
            "commit_seq": 1,
            "new_seq_len": 5,
            "new_token_ids": [13, 14],
        },
    )
    assert result["applied"]
    assert list(request.all_token_ids) == [10, 11, 12, 13, 14]
    assert list(request.output_token_ids) == [13, 14]
    assert request.block_hashes == [(10, 11), (12, 13)]
    assert request.num_computed_tokens == 5
    assert control._scheduler.kv_cache_manager.coordinator.cached == [
        ("prefill-owner", 5)
    ]


def test_release_checks_final_commit_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    registry = pap_lease.PAPKVLeaseRegistry()
    monkeypatch.setattr(pap_lease, "get_global_kv_lease_registry", lambda: registry)
    lease_id = registry.pin_blocks(
        request_id=request.request_id, block_ids=[1], ttl_seconds=0
    )
    control.apply(
        "decode_commit",
        {
            "request_id": request.request_id,
            "commit_seq": 1,
            "new_seq_len": 3,
            "new_token_ids": [12],
        },
    )
    with pytest.raises(ValueError, match="raced"):
        control.apply(
            "lease_release",
            {
                "request_id": request.request_id,
                "lease_id": lease_id,
                "final_commit_seq": 0,
            },
        )


def test_attention_allocation_extends_owned_blocks_without_advancing_tokens() -> None:
    """An allocation RPC changes capacity, not the committed Decode position."""
    pap_lease.reset_global_kv_lease_registry()
    owned = [1, 2]

    class Manager:
        def get_block_ids(self, request_id):
            assert request_id == "req"
            return (owned.copy(),)

        def allocate_slots(self, request, num_new_tokens, **kwargs):
            assert request.request_id == "req"
            assert num_new_tokens == 32
            assert kwargs == {"delay_cache_blocks": True}
            owned.extend((3, 4))
            return object()

    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=17,
        num_computed_tokens=17,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    connector = object.__new__(PAPPrefillConnector)
    connector._request_metadata = {
        "req": PAPRequestMetadata(
            prefill_kv_handle="session",
            decode_capacity_tokens=32,
            import_prefill_kv_to_attention=True,
        )
    }
    connector._generations = {"req": 0}
    allocation_stats = SimpleNamespace(
        decode_allocation_requests=0,
        decode_allocation_blocks=0,
        decode_allocation_failures=0,
    )

    def record_allocation(*, blocks: int, failed: bool) -> None:
        allocation_stats.decode_allocation_requests += 1
        allocation_stats.decode_allocation_blocks += blocks
        allocation_stats.decode_allocation_failures += int(failed)

    allocation_stats.record_decode_allocation = record_allocation
    scheduler = SimpleNamespace(
        connector=connector,
        requests={"req": request},
        kv_cache_manager=Manager(),
        block_size=16,
        max_model_len=128,
        pap_scheduler=allocation_stats,
    )
    registry = pap_lease.get_global_kv_lease_registry()
    lease_id = registry.pin_blocks(request_id="req", block_ids=owned, ttl_seconds=0)

    result = PAPEngineControl(scheduler).apply(
        "decode_allocate",
        {
            "session_handle": "session",
            "lease_id": lease_id,
            "generation": 0,
            "required_tokens": 33,
        },
    )

    assert result["allocated"] is True
    assert result["block_ids"] == [1, 2, 3, 4]
    assert result["writable_end_token"] == 49
    assert request.num_computed_tokens == 17
    assert registry.active_entry(lease_id).block_ids == (1, 2, 3, 4)
    assert allocation_stats.decode_allocation_requests == 1
    assert allocation_stats.decode_allocation_blocks == 2
    pap_lease.reset_global_kv_lease_registry()


def test_control_reports_non_evictable_kv_capacity() -> None:
    block_pool = SimpleNamespace(
        num_gpu_blocks=101,
        get_num_free_blocks=lambda: 60,
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(block_pool=block_pool),
        block_size=16,
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
    )

    snapshot = PAPEngineControl(scheduler).apply("kv_load_snapshot", {})

    assert snapshot == {
        "non_evictable_kv_blocks": 40,
        "non_evictable_kv_tokens": 640,
        "running_prefill_tokens": 0,
        "queued_prefill_tokens": 0,
        "outstanding_prefill_tokens": 0,
        "running_decode_reservation_tokens": 0,
        "queued_decode_reservation_tokens": 0,
        "outstanding_decode_reservation_tokens": 0,
        "running_prefill_requests": 0,
        "queued_prefill_requests": 0,
        "projected_kv_tokens": 640,
        "routing_kv_tokens": 640,
        "free_kv_blocks": 60,
        "total_kv_blocks": 100,
        "total_kv_tokens": 1600,
        "kv_block_size": 16,
        "kv_load_fraction": 0.4,
        "decode_allocation_requests": 0,
        "decode_allocation_blocks": 0,
        "decode_allocation_failures": 0,
        "prefill_revocations": 0,
    }


def test_control_includes_running_and_queued_prefill_tokens() -> None:
    block_pool = SimpleNamespace(
        num_gpu_blocks=101,
        get_num_free_blocks=lambda: 80,
    )
    running = SimpleNamespace(
        request_id="running",
        num_prompt_tokens=1000,
        num_computed_tokens=400,
        kv_transfer_params={"pap_decode_capacity_tokens": 50},
    )
    waiting = SimpleNamespace(
        request_id="waiting",
        num_prompt_tokens=2000,
        num_computed_tokens=0,
        kv_transfer_params={"pap_decode_capacity_tokens": 100},
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(block_pool=block_pool),
        block_size=16,
        requests={"running": running, "waiting": waiting},
        running=[running],
        waiting=[waiting],
        skipped_waiting=[],
    )

    snapshot = PAPEngineControl(scheduler).apply("kv_load_snapshot", {})

    assert snapshot["non_evictable_kv_tokens"] == 320
    assert snapshot["running_prefill_tokens"] == 600
    assert snapshot["queued_prefill_tokens"] == 2000
    assert snapshot["outstanding_prefill_tokens"] == 2600
    assert snapshot["outstanding_decode_reservation_tokens"] == 150
    assert snapshot["projected_kv_tokens"] == 3070


@pytest.mark.parametrize("batch_queue", [None, []])
@pytest.mark.parametrize("acknowledged", [True, False])
def test_released_references_are_reclaimed_only_at_a_synchronous_boundary(
    batch_queue, acknowledged
):
    from vllm.pap.integration.engine import PAPEngineAdapter
    from vllm.v1.core.sched.scheduler import Scheduler

    freed = []
    calls = []
    request = SimpleNamespace(request_id="owner", is_finished=lambda: True)
    scheduler = object.__new__(Scheduler)
    scheduler.requests = {"owner": request}
    scheduler.finished_req_ids = {"owner"}
    scheduler.connector = None
    scheduler.defer_block_free = False
    scheduler.kv_cache_manager = SimpleNamespace(
        free=lambda req: freed.append(req.request_id)
    )

    def collective_rpc(method, args):
        calls.append(args)
        return [{"owner"} if acknowledged else set()]

    core = SimpleNamespace(
        batch_queue=batch_queue,
        scheduler=scheduler,
        model_executor=SimpleNamespace(collective_rpc=collective_rpc),
    )
    if batch_queue is None and not acknowledged:
        with pytest.raises(RuntimeError, match="did not acknowledge"):
            PAPEngineAdapter.reclaim_released_requests(core, {"owner"})
        assert "owner" in scheduler.requests
        assert not freed
        return

    result = PAPEngineAdapter.reclaim_released_requests(core, {"owner"})
    if batch_queue is None:
        assert result == {"owner"}
        assert not scheduler.requests
        assert freed == ["owner"]
        assert calls == [({"owner"}, {"owner"})]
        assert scheduler.finished_req_ids == {"owner"}
    else:
        assert result == set()
        assert "owner" in scheduler.requests
        assert not calls and not freed
