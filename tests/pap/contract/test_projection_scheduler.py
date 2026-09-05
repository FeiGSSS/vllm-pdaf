# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from huggingface_hub import constants as hf_constants

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.pap.integration.engine import PAPEngineControl
from vllm.pap.integration.kv_cache import PAPKVCacheAdapter
from vllm.pap.kv import lease as pap_lease
from vllm.pap.kv_connector import PAPPrefillConnector
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


@pytest.fixture(autouse=True)
def _offline_cpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.platforms as platforms

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(platforms, "current_platform", CpuPlatform())


def _projection_request(num_tokens: int = 10):
    request = create_requests(1, num_tokens=num_tokens)[0]
    request.kv_transfer_params = {
        "pap_projection_kv_unaware": True,
        "pap_remote_prefix_len": num_tokens,
        "pap_attention_kv_installed": True,
    }
    return request


def test_prefill_allocates_only_prompt_but_exports_owned_blocks(monkeypatch) -> None:
    """Decode's requested limit must not become speculative Prefill allocation."""
    monkeypatch.setitem(
        KVConnectorFactory._registry,
        "PAPPrefillConnector",
        lambda: PAPPrefillConnector,
    )
    pap_lease.reset_global_kv_lease_registry()
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        block_size=16,
        async_scheduling=True,
        use_v2_model_runner=True,
        use_kv_connector="PAPPrefillConnector",
    )
    request = create_requests(1, num_tokens=64, max_tokens=1)[0]
    request.kv_transfer_params = {
        "pap_import_prefill_kv_to_attention": True,
        "pap_decode_capacity_tokens": 64,
        "pap_prefill_kv_handle": "session",
        "pap_attention_tcp_endpoint": "tcp://127.0.0.1:1",
    }
    scheduler.add_request(request)
    try:
        output = scheduler.schedule()
        owned = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
        published = output.kv_connector_metadata.requests[0]

        assert len(owned) == 4
        assert published.decode_capacity_tokens == 64
        assert published.allocated_block_ids == tuple(owned)
        assert pap_lease.pap_leased_block_ids(request.request_id) == tuple(owned)

        scheduler.finish_requests(
            request.request_id, RequestStatus.FINISHED_LENGTH_CAPPED
        )
        assert request.request_id in scheduler.requests
        allocation = PAPEngineControl(scheduler).apply(
            "decode_allocate",
            {
                "session_handle": "session",
                "lease_id": published.lease_id,
                "generation": 0,
                "required_tokens": 65,
            },
        )
        grown = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
        assert allocation["block_ids"] == grown
        assert len(grown) == 8
        assert allocation["writable_end_token"] == 128
        assert request.num_computed_tokens == 64
    finally:
        scheduler.shutdown()
        pap_lease.reset_global_kv_lease_registry()


def test_prefill_preemption_revokes_attention_before_recycling_blocks(
    monkeypatch,
) -> None:
    """A published Prefill layout is fenced while its blocks are still owned."""
    monkeypatch.setitem(
        KVConnectorFactory._registry,
        "PAPPrefillConnector",
        lambda: PAPPrefillConnector,
    )
    pap_lease.reset_global_kv_lease_registry()
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=11,
        enable_prefix_caching=False,
        use_kv_connector="PAPPrefillConnector",
    )
    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)
    for index, request in enumerate(requests):
        request.kv_transfer_params = {
            "pap_import_prefill_kv_to_attention": True,
            "pap_decode_capacity_tokens": 32,
            "pap_prefill_kv_handle": f"session-{index}",
            "pap_attention_tcp_endpoint": "tcp://127.0.0.1:1",
        }

    scheduler.add_request(requests[0])
    first_output = scheduler.schedule()
    scheduler.add_request(requests[1])
    scheduler.schedule()
    victim_blocks = scheduler.kv_cache_manager.get_block_ids(requests[1].request_id)[0]
    observations = []

    def observe_revoke(**kwargs) -> None:
        observations.append(
            (
                kwargs,
                scheduler.kv_cache_manager.get_block_ids(requests[1].request_id)[0],
            )
        )

    scheduler.update_from_output(
        first_output,
        ModelRunnerOutput(
            req_ids=[requests[0].request_id],
            req_id_to_index={requests[0].request_id: 0},
            sampled_token_ids=[[0]],
        ),
    )
    try:

        def fail_revoke(**kwargs) -> None:
            observe_revoke(**kwargs)
            raise RuntimeError("revoke unavailable")

        monkeypatch.setattr("vllm.pap.kv.handoff.revoke_prefill_kv", fail_revoke)
        with pytest.raises(RuntimeError, match="revoke unavailable"):
            scheduler.schedule()
        assert requests[1].status == RequestStatus.RUNNING
        assert requests[1] in scheduler.running
        assert (
            scheduler.kv_cache_manager.get_block_ids(requests[1].request_id)[0]
            == victim_blocks
        )

        observations.clear()
        monkeypatch.setattr("vllm.pap.kv.handoff.revoke_prefill_kv", observe_revoke)
        scheduler.schedule()

        assert requests[1].status == RequestStatus.PREEMPTED
        assert observations[0][0]["session_handle"] == "session-1"
        assert observations[0][1] == victim_blocks
        assert scheduler.kv_cache_manager.get_block_ids(requests[1].request_id) == ([],)
        assert scheduler.pap_scheduler.prefill_revocations == 1
    finally:
        scheduler.shutdown()
        pap_lease.reset_global_kv_lease_registry()


def test_projection_starts_at_remote_prefix_without_local_kv() -> None:
    scheduler = create_scheduler(enable_prefix_caching=True)
    request = _projection_request()
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    scheduler.add_request(request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {request.request_id: 1}
    assert output.scheduled_new_reqs[0].num_computed_tokens == 9
    assert output.scheduled_new_reqs[0].block_ids == ([],)
    assert output.scheduled_new_reqs[0].kv_transfer_params == (
        request.kv_transfer_params
    )
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id) == ([],)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks
    assert request.num_computed_tokens == 10


def test_projection_preemption_preserves_remote_progress() -> None:
    scheduler = create_scheduler()
    request = _projection_request()
    scheduler.add_request(request)
    scheduler.schedule()
    scheduler.running.remove(request)

    scheduler._preempt_request(request, 0.0)

    assert request.num_computed_tokens == 10


def test_projection_rejects_speculative_decoding() -> None:
    scheduler = create_scheduler(num_speculative_tokens=2)

    with pytest.raises(ValueError, match="does not support speculative"):
        scheduler.add_request(_projection_request())


def test_projection_worker_allocates_one_scratch_block() -> None:
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[KVCacheTensor(size=800, shared_by=["layer"])],
        kv_cache_groups=[],
    )

    scratch = PAPKVCacheAdapter.projection_scratch_config(config, enabled=True)

    assert scratch.num_blocks == 1
    assert scratch.kv_cache_tensors[0].size == 100
    assert config.num_blocks == 8
