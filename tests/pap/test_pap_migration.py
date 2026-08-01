# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for out-of-band PAP KV migration."""

from types import SimpleNamespace

import pytest

from vllm.pap.integration.engine import PAPEngineAdapter
from vllm.pap.integration.migration import PAPMigrationStatus
from vllm.pap.integration.scheduler import PAPSchedulerAdapter
from vllm.pap.integration.settings import PAPRuntimeSettings


def _source_params() -> dict:
    return {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [[10, 11]],
        "remote_engine_id": "source-engine",
        "remote_request_id": "source-request",
        "remote_host": "127.0.0.1",
        "remote_port": 5559,
        "remote_num_tokens": 32,
        "tp_size": 1,
    }


def _prefix_identity() -> tuple[tuple[int, ...], tuple[bytes, ...]]:
    return tuple(range(32)), (b"a" * 32, b"b" * 32)


def test_engine_rejects_post_prefill_migration_for_tp2() -> None:
    class Scheduler:
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2)
        )
        pap_scheduler = SimpleNamespace(
            submit_migration=lambda **kwargs: pytest.fail(
                "unsupported migration was submitted"
            )
        )

    with pytest.raises(
        NotImplementedError,
        match=r"currently supports only TP=1; TP=2",
    ):
        PAPEngineAdapter.submit_kv_migration(
            Scheduler(),
            {
                "request_id": "pap-request",
                "source_kv_params": _source_params(),
                "prefix_len": 32,
                "prefix_token_ids": list(range(32)),
                "prefix_block_hashes": ["aa" * 32, "bb" * 32],
                "decode_capacity_tokens": 32,
                "session_handle": "attention-session",
                "attention_tcp_endpoint": "tcp://127.0.0.1:9300",
            },
        )


def test_scheduler_attaches_migration_without_vllm_request(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.pap.integration.scheduler.pap_lease.pap_pin_blocks",
        lambda request_id, block_ids: "lease-1",
    )
    monkeypatch.setattr(
        "vllm.pap.integration.scheduler.pap_lease.pap_record_kv_export",
        lambda *args, **kwargs: True,
    )
    adapter = PAPSchedulerAdapter(PAPRuntimeSettings.from_environ({}))
    prefix_token_ids, prefix_block_hashes = _prefix_identity()
    submitted = adapter.submit_migration(
        request_id="pap-request",
        source_kv_params=_source_params(),
        prefix_len=32,
        prefix_token_ids=prefix_token_ids,
        prefix_block_hashes=prefix_block_hashes,
        decode_capacity_tokens=32,
        session_handle="attention-session",
        attention_tcp_endpoint="tcp://127.0.0.1:9300",
    )

    class Blocks:
        @staticmethod
        def get_block_ids():
            return ([20, 21, 22, 23],)

    class Manager:
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=16),
                )
            ]
        )

        @staticmethod
        def allocate_external_transfer_slots(**kwargs):
            assert kwargs["prefix_tokens"] == 32
            assert kwargs["total_capacity_tokens"] == 64
            return Blocks()

        @staticmethod
        def free_external_transfer_slots(request_id):
            raise AssertionError(f"unexpected migration cleanup: {request_id}")

    class Connector:
        def __init__(self):
            self.recv = None

        def pap_add_migration_recv(self, metadata, **kwargs):
            self.recv = (metadata, kwargs)

        @staticmethod
        def pap_build_local_export_params(**kwargs):
            return {
                "remote_request_id": kwargs["request_id"],
                "remote_block_ids": kwargs["block_ids"],
                "remote_num_tokens": kwargs["num_tokens"],
            }

    connector = Connector()
    metadata = object()
    manifests = adapter.attach_next_migration(
        metadata=metadata,
        kv_cache_manager=Manager(),
        connector=connector,
        reserved_blocks=7,
    )

    assert submitted["status"] == "pending"
    assert len(manifests) == 1
    assert connector.recv[0] is metadata
    assert connector.recv[1]["local_block_ids"] == ([20, 21],)
    assert manifests[0]["block_ids"] == [[20, 21, 22, 23]]
    assert adapter.migration_status(submitted["job_id"])["status"] == ("transferring")
    assert adapter.migration_started(submitted["job_id"]).result()["status"] == (
        "transferring"
    )


def test_scheduler_fails_migration_when_target_has_no_capacity() -> None:
    adapter = PAPSchedulerAdapter(PAPRuntimeSettings.from_environ({}))
    prefix_token_ids, prefix_block_hashes = _prefix_identity()
    submitted = adapter.submit_migration(
        request_id="pap-request",
        source_kv_params=_source_params(),
        prefix_len=32,
        prefix_token_ids=prefix_token_ids,
        prefix_block_hashes=prefix_block_hashes,
        decode_capacity_tokens=32,
        session_handle="attention-session",
        attention_tcp_endpoint="tcp://127.0.0.1:9300",
    )
    manager = SimpleNamespace(
        allocate_external_transfer_slots=lambda **kwargs: None,
    )

    manifests = adapter.attach_next_migration(
        metadata=object(),
        kv_cache_manager=manager,
        connector=object(),
        reserved_blocks=7,
    )

    assert manifests == []
    status = adapter.migration_status(submitted["job_id"])
    assert status["status"] == "failed"
    assert "insufficient target KV capacity" in status["error"]
    assert adapter.migration_completion(submitted["job_id"]).result() == status
    assert not adapter.has_migration_work()


def test_scheduler_bounds_terminal_migration_history() -> None:
    adapter = PAPSchedulerAdapter(
        PAPRuntimeSettings.from_environ({}),
        migration_terminal_history_limit=2,
    )
    manager = SimpleNamespace(
        allocate_external_transfer_slots=lambda **kwargs: None,
    )
    prefix_token_ids, prefix_block_hashes = _prefix_identity()
    job_ids = []

    for index in range(3):
        submitted = adapter.submit_migration(
            request_id=f"pap-request-{index}",
            source_kv_params=_source_params(),
            prefix_len=32,
            prefix_token_ids=prefix_token_ids,
            prefix_block_hashes=prefix_block_hashes,
            decode_capacity_tokens=32,
            session_handle=f"attention-session-{index}",
            attention_tcp_endpoint="tcp://127.0.0.1:9300",
        )
        job_ids.append(submitted["job_id"])
        assert (
            adapter.attach_next_migration(
                metadata=object(),
                kv_cache_manager=manager,
                connector=object(),
                reserved_blocks=7,
            )
            == []
        )

    assert tuple(adapter.migration_jobs) == tuple(job_ids[1:])
    assert adapter.migration_status(job_ids[0])["status"] == "unknown"
    assert adapter.migration_status(job_ids[1])["status"] == "failed"
    assert adapter.migration_status(job_ids[2])["status"] == "failed"
    assert not adapter.has_migration_work()


def test_scheduler_rejects_empty_terminal_migration_history() -> None:
    with pytest.raises(ValueError, match="history limit must be positive"):
        PAPSchedulerAdapter(
            PAPRuntimeSettings.from_environ({}),
            migration_terminal_history_limit=0,
        )


def test_nixl_migration_export_uses_connector_config() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlBaseConnector,
    )

    connector = SimpleNamespace(
        connector_scheduler=SimpleNamespace(
            engine_id="target-engine",
            side_channel_host="127.0.0.1",
            side_channel_port=5560,
        ),
        _vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2)
        ),
    )
    params = NixlBaseConnector.pap_build_local_export_params(
        connector,
        request_id="pap-request",
        block_ids=([20, 21],),
        num_tokens=32,
    )

    assert params["remote_engine_id"] == "target-engine"
    assert params["remote_request_id"] == "pap-request"
    assert params["remote_block_ids"] == ([20, 21],)
    assert params["remote_num_tokens"] == 32
    assert params["tp_size"] == 2


def test_finished_migration_exports_original_request_id(monkeypatch) -> None:
    adapter = PAPSchedulerAdapter(PAPRuntimeSettings.from_environ({}))
    prefix_token_ids, prefix_block_hashes = _prefix_identity()
    submitted = adapter.submit_migration(
        request_id="pap-request",
        source_kv_params=_source_params(),
        prefix_len=32,
        prefix_token_ids=prefix_token_ids,
        prefix_block_hashes=prefix_block_hashes,
        decode_capacity_tokens=32,
        session_handle="attention-session",
        attention_tcp_endpoint="tcp://127.0.0.1:9300",
    )
    job = adapter.migration_jobs[submitted["job_id"]]
    job.block_ids = ((20, 21, 22, 23),)
    job.kv_transfer_params = {
        "remote_request_id": "pap-request",
        "remote_block_ids": ([20, 21, 22, 23],),
        "remote_num_tokens": 32,
    }
    job.status = PAPMigrationStatus.TRANSFERRING

    monkeypatch.setattr(
        "vllm.pap.integration.scheduler.pap_lease.pap_record_kv_export",
        lambda request_id, seq_len, kv_transfer_params, *args: (
            request_id == "pap-request"
            and seq_len == 32
            and kv_transfer_params["remote_request_id"] == "pap-request"
        ),
    )
    monkeypatch.setattr(
        "vllm.pap.integration.scheduler.pap_lease.pap_has_active_lease",
        lambda request_id: request_id == "pap-request",
    )
    monkeypatch.setattr(
        "vllm.pap.integration.scheduler.pap_lease.pap_active_lease_id",
        lambda request_id: "lease-1" if request_id == "pap-request" else None,
    )
    monkeypatch.setattr(
        "vllm.pap.integration.scheduler.pap_lease.pap_stash_deferred_blocks",
        lambda **kwargs: None,
    )

    cached_prefixes = []
    manager = SimpleNamespace(
        cache_external_transfer_prefix=lambda **kwargs: cached_prefixes.append(kwargs),
        pop_external_transfer_slots=lambda job_id: [object()],
        block_pool=SimpleNamespace(free_blocks=lambda blocks: None),
    )

    assert adapter.finish_migration(
        job_id=job.job_id,
        kv_cache_manager=manager,
        connector=object(),
    )
    assert job.kv_transfer_params["remote_request_id"] == "pap-request"
    assert cached_prefixes == [
        {
            "request_id": job.job_id,
            "prefix_token_ids": prefix_token_ids,
            "prefix_block_hashes": prefix_block_hashes,
        }
    ]
    assert job.status is PAPMigrationStatus.READY
    assert adapter.migration_completion(job.job_id).result()["status"] == "ready"


def test_migrated_blocks_are_reachable_by_next_prefill() -> None:
    import torch

    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request

    block_size = 16
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(block_size, sha256)
    source = Request(
        request_id="source",
        prompt_token_ids=list(range(32)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=block_hasher,
    )
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=128,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )
    assert manager.allocate_external_transfer_slots(
        request_id="migration",
        prefix_tokens=32,
        total_capacity_tokens=32,
    )
    assert (
        manager.cache_external_transfer_prefix(
            request_id="migration",
            prefix_token_ids=tuple(source.all_token_ids),
            prefix_block_hashes=tuple(bytes(value) for value in source.block_hashes),
        )
        == 32
    )
    manager.pop_external_transfer_slots("migration")

    continuation = Request(
        request_id="continuation",
        prompt_token_ids=list(range(48)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=block_hasher,
    )
    _, hit_tokens = manager.get_computed_blocks(continuation)

    assert hit_tokens == 32


def test_prefill_control_router_submits_and_polls_migration() -> None:
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class EngineClient:
        async def pap_submit_kv_migration_async(self, migration):
            assert migration["prefix_len"] == 32
            return {"job_id": "job-1", "status": "pending"}

        async def pap_kv_migration_status_async(self, job_id):
            assert job_id == "job-1"
            return {"job_id": job_id, "status": "ready"}

    async def run_requests():
        app = FastAPI()
        app.state.engine_client = EngineClient()
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            submitted = await client.post(
                "/v1/pap/prefill/kv-import",
                json={
                    "request_id": "pap-request",
                    "source_kv_params": _source_params(),
                    "prefix_len": 32,
                    "prefix_token_ids": list(range(32)),
                    "prefix_block_hashes": ["aa" * 32, "bb" * 32],
                    "decode_capacity_tokens": 32,
                    "session_handle": "attention-session",
                    "attention_tcp_endpoint": "tcp://127.0.0.1:9300",
                },
            )
            status = await client.post(
                "/v1/pap/prefill/kv-import/status",
                json={"job_id": "job-1"},
            )
        return submitted, status

    submitted, status = anyio.run(run_requests)
    assert submitted.json() == {"job_id": "job-1", "status": "pending"}
    assert status.json() == {"job_id": "job-1", "status": "ready"}
