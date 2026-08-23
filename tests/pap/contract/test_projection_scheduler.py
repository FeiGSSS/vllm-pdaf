# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
from huggingface_hub import constants as hf_constants

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.pap.integration.kv_cache import PAPKVCacheAdapter
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


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
