# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest

from vllm.pap.lifecycle.commit import DecodeCommitClient
from vllm.pap.lifecycle.lease_release import LeaseReleaseClient
from vllm.v1.request import Request


def test_commit_endpoint_applies_to_manager():
    """Decode-commit POST invokes EngineCore through app.state.engine_client."""
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class StubEngineClient:
        def __init__(self):
            self.calls = []

        async def pap_apply_decode_commit_async(
            self, request_id, new_seq_len, new_token_ids
        ):
            self.calls.append((request_id, new_seq_len, list(new_token_ids)))
            return {"request_id": request_id, "applied": True}

    async def run_request():
        engine_client = StubEngineClient()
        app = FastAPI()
        app.state.engine_client = engine_client
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/v1/pap/prefill/decode-commit",
                json={
                    "request_id": "req-1",
                    "commit_seq": 1,
                    "new_seq_len": 17,
                    "new_token_ids": [1, 2, 3],
                    "layer_complete": True,
                },
            )
        return resp, engine_client.calls

    resp, calls = anyio.run(run_request)

    assert resp.status_code == 200
    assert resp.json()["acked_commit_seq"] == 1
    assert calls == [("req-1", 17, [1, 2, 3])]


def test_commit_endpoint_can_ack_engine_queue_submission():
    """Submit-only commit returns before EngineCore executes the utility."""
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    async def run_request():
        execution = asyncio.get_running_loop().create_future()

        class StubEngineClient:
            async def pap_submit_decode_commit_async(
                self, request_id, new_seq_len, new_token_ids
            ):
                return execution

        app = FastAPI()
        app.state.engine_client = StubEngineClient()
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            with anyio.fail_after(0.5):
                response = await client.post(
                    "/v1/pap/prefill/decode-commit",
                    json={
                        "request_id": "req-1",
                        "commit_seq": 1,
                        "new_seq_len": 17,
                        "new_token_ids": [1, 2, 3],
                        "submit_only": True,
                    },
                )
            assert not execution.done()
            execution.set_result({"request_id": "req-1", "applied": True})
            await asyncio.sleep(0)
            return response

    response = anyio.run(run_request)

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert response.json()["accepted_commit_seq"] == 1


def test_commit_endpoint_falls_back_to_sync_engine_client():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class StubEngineClient:
        def __init__(self):
            self.calls = []

        async def pap_apply_decode_commit_async(
            self, request_id, new_seq_len, new_token_ids
        ):
            raise NotImplementedError

        def pap_apply_decode_commit(self, request_id, new_seq_len, new_token_ids):
            self.calls.append((request_id, new_seq_len, tuple(new_token_ids)))
            return {"request_id": request_id, "applied": True}

    async def run_request():
        engine_client = StubEngineClient()
        app = FastAPI()
        app.state.engine_client = engine_client
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/v1/pap/prefill/decode-commit",
                json={
                    "request_id": "req-1",
                    "commit_seq": 1,
                    "new_seq_len": 18,
                    "new_token_ids": [4],
                    "layer_complete": True,
                },
            )
        return resp, engine_client.calls

    resp, calls = anyio.run(run_request)

    assert resp.status_code == 200
    assert resp.json()["acked_commit_seq"] == 1
    assert calls == [("req-1", 18, (4,))]


def test_commit_endpoint_rejects_unacknowledged_unknown_request():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class StubEngineClient:
        async def pap_apply_decode_commit_async(
            self, request_id, new_seq_len, new_token_ids
        ):
            return {
                "request_id": request_id,
                "applied": False,
                "reason": "unknown_request",
            }

    async def run_request():
        app = FastAPI()
        app.state.engine_client = StubEngineClient()
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/v1/pap/prefill/decode-commit",
                json={
                    "request_id": "req-finished",
                    "commit_seq": 1,
                    "new_seq_len": 18,
                    "new_token_ids": [4],
                    "layer_complete": True,
                },
            )

    resp = anyio.run(run_request)

    assert resp.status_code == 409
    assert resp.json()["detail"]["applied"] is False
    assert resp.json()["detail"]["reason"] == "unknown_request"


def test_commit_endpoint_deduplicates_acknowledged_sequence():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class StubEngineClient:
        def __init__(self):
            self.calls = []

        async def pap_apply_decode_commit_async(
            self, request_id, new_seq_len, new_token_ids
        ):
            self.calls.append((request_id, new_seq_len, tuple(new_token_ids)))
            return {"request_id": request_id, "applied": True}

    async def run_requests():
        engine_client = StubEngineClient()
        app = FastAPI()
        app.state.engine_client = engine_client
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        payload = {
            "request_id": "req-1",
            "commit_seq": 1,
            "new_seq_len": 17,
            "new_token_ids": [1, 2, 3],
            "layer_complete": True,
        }
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            first = await client.post("/v1/pap/prefill/decode-commit", json=payload)
            duplicate = await client.post("/v1/pap/prefill/decode-commit", json=payload)
        return first, duplicate, engine_client.calls

    first, duplicate, calls = anyio.run(run_requests)

    assert first.status_code == 200
    assert duplicate.status_code == 200
    assert duplicate.json()["acked_commit_seq"] == 1
    assert duplicate.json()["idempotent"] is True
    assert calls == [("req-1", 17, (1, 2, 3))]


def test_commit_endpoint_does_not_serialize_unrelated_sessions():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    async def run_requests():
        first_started = anyio.Event()
        second_started = anyio.Event()
        release_first = anyio.Event()
        responses = {}

        class StubEngineClient:
            async def pap_apply_decode_commit_async(
                self, request_id, new_seq_len, new_token_ids
            ):
                if request_id == "wrapped-a":
                    first_started.set()
                    await release_first.wait()
                else:
                    second_started.set()
                return {"request_id": request_id, "applied": True}

        app = FastAPI()
        app.state.engine_client = StubEngineClient()
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:

            async def post(name, payload):
                responses[name] = await client.post(
                    "/v1/pap/prefill/decode-commit",
                    json=payload,
                )

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(
                    post,
                    "first",
                    {
                        "request_id": "wrapped-a",
                        "session_request_id": "session-a",
                        "commit_seq": 1,
                        "new_seq_len": 10,
                        "new_token_ids": [1],
                    },
                )
                await first_started.wait()
                task_group.start_soon(
                    post,
                    "second",
                    {
                        "request_id": "wrapped-b",
                        "session_request_id": "session-b",
                        "commit_seq": 1,
                        "new_seq_len": 11,
                        "new_token_ids": [2],
                    },
                )
                with anyio.fail_after(0.5):
                    await second_started.wait()
                release_first.set()

        return responses

    responses = anyio.run(run_requests)

    assert responses["first"].status_code == 200
    assert responses["second"].status_code == 200


def test_prefill_control_router_releases_lease():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class StubEngineClient:
        def __init__(self):
            self.calls = []

        async def pap_release_kv_lease_async(self, request_id, lease_id):
            self.calls.append((request_id, lease_id))
            return {
                "request_id": request_id,
                "lease_id": lease_id,
                "released": True,
                "block_count": 2,
            }

    async def run_request():
        engine_client = StubEngineClient()
        app = FastAPI()
        app.state.engine_client = engine_client
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/v1/pap/prefill/lease-release",
                json={"request_id": "req-1", "lease_id": "lease-1"},
            )
        return resp, engine_client.calls

    resp, calls = anyio.run(run_request)

    assert resp.status_code == 200
    assert resp.json()["block_count"] == 2
    assert calls == [("req-1", "lease-1")]


def test_lease_release_endpoint_can_ack_engine_queue_submission():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    async def run_request():
        execution = asyncio.get_running_loop().create_future()

        class StubEngineClient:
            async def pap_submit_release_kv_lease_async(self, request_id, lease_id):
                return execution

        app = FastAPI()
        app.state.engine_client = StubEngineClient()
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            with anyio.fail_after(0.5):
                response = await client.post(
                    "/v1/pap/prefill/lease-release",
                    json={
                        "request_id": "req-1",
                        "lease_id": "lease-1",
                        "submit_only": True,
                    },
                )
            assert not execution.done()
            execution.set_result(
                {
                    "request_id": "req-1",
                    "lease_id": "lease-1",
                    "released": True,
                }
            )
            await asyncio.sleep(0)
            return response

    response = anyio.run(run_request)

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "req-1",
        "lease_id": "lease-1",
        "accepted": True,
    }


def test_submit_release_accepts_idempotent_completion(caplog):
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    async def run_request():
        execution = asyncio.get_running_loop().create_future()

        class StubEngineClient:
            async def pap_submit_release_kv_lease_async(self, request_id, lease_id):
                return execution

        app = FastAPI()
        app.state.engine_client = StubEngineClient()
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/v1/pap/prefill/lease-release",
                json={
                    "request_id": "req-1",
                    "lease_id": "lease-1",
                    "submit_only": True,
                },
            )
            execution.set_result(
                {
                    "request_id": "req-1",
                    "lease_id": "lease-1",
                    "released": False,
                    "reason": "unknown_or_released_lease",
                }
            )
            await asyncio.sleep(0)
            return response

    response = anyio.run(run_request)

    assert response.status_code == 200
    assert "submitted control was not applied" not in caplog.text


def test_prefill_control_router_release_falls_back_to_sync_engine_client():
    import anyio
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from vllm.pap.prefill_control_router import build_prefill_control_router

    class StubEngineClient:
        def __init__(self):
            self.calls = []

        async def pap_release_kv_lease_async(self, request_id, lease_id):
            raise NotImplementedError

        def pap_release_kv_lease(self, request_id, lease_id):
            self.calls.append((request_id, lease_id))
            return {
                "request_id": request_id,
                "lease_id": lease_id,
                "released": True,
                "block_count": 1,
            }

    async def run_request():
        engine_client = StubEngineClient()
        app = FastAPI()
        app.state.engine_client = engine_client
        app.include_router(build_prefill_control_router())
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/v1/pap/prefill/lease-release",
                json={"request_id": "req-1", "lease_id": "lease-1"},
            )
        return resp, engine_client.calls

    resp, calls = anyio.run(run_request)

    assert resp.status_code == 200
    assert resp.json()["block_count"] == 1
    assert calls == [("req-1", "lease-1")]


def test_async_llm_pap_control_methods_delegate_to_engine_core():
    import anyio

    from vllm.v1.engine.async_llm import AsyncLLM

    class StubEngineCore:
        def __init__(self):
            self.commit_calls = []
            self.release_calls = []

        async def pap_apply_decode_commit_async(
            self, request_id, new_seq_len, new_token_ids
        ):
            self.commit_calls.append((request_id, new_seq_len, new_token_ids))
            return {
                "request_id": request_id,
                "applied": True,
                "new_seq_len": new_seq_len,
            }

        async def pap_release_kv_lease_async(self, request_id, lease_id):
            self.release_calls.append((request_id, lease_id))
            return {
                "request_id": request_id,
                "lease_id": lease_id,
                "released": True,
            }

        def shutdown(self, timeout=None):
            pass

    async def run_calls():
        engine_core = StubEngineCore()
        llm = object.__new__(AsyncLLM)
        llm.engine_core = engine_core
        commit = await llm.pap_apply_decode_commit_async("req-1", "7", ["42"])
        release = await llm.pap_release_kv_lease_async("req-1", "lease-1")
        return commit, release, engine_core

    commit, release, engine_core = anyio.run(run_calls)

    assert commit == {"request_id": "req-1", "applied": True, "new_seq_len": 7}
    assert release == {
        "request_id": "req-1",
        "lease_id": "lease-1",
        "released": True,
    }
    assert engine_core.commit_calls == [("req-1", 7, (42,))]
    assert engine_core.release_calls == [("req-1", "lease-1")]


def test_async_mp_submit_utility_returns_before_execution():
    from types import SimpleNamespace

    import anyio

    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core_client import AsyncMPClient

    async def run_submission():
        sent = []
        client = object.__new__(AsyncMPClient)
        client.client_index = 0
        client.core_engine = b"engine-0"
        client.utility_results = {}

        class StubEncoder:
            @staticmethod
            def encode(value):
                return (b"encoded",)

        async def send_input(message, engine, objects):
            sent.append((message, engine, objects))

        client.encoder = StubEncoder()
        client._send_input_message = send_input
        client._ensure_output_queue_task = lambda: None

        execution = await client._submit_utility_async(
            "pap_apply_decode_commit",
            "req-1",
            17,
            (1, 2, 3),
            engine=b"engine-0",
        )
        request = SimpleNamespace(client_index=None)
        await client.add_request_async(request)

        assert not execution.done()
        assert [message[0][0] for message in sent] == [
            EngineCoreRequestType.UTILITY.value,
            EngineCoreRequestType.ADD.value,
        ]
        execution.set_result({"request_id": "req-1", "applied": True})
        return await execution

    assert anyio.run(run_submission) == {
        "request_id": "req-1",
        "applied": True,
    }


def test_apply_decode_commit_advances_tokens():
    """PAP decode commit appends tokens and advances num_computed."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    sampling_params = SamplingParams(max_tokens=10)
    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=lambda req: [b"h"],
    )
    # Verify initial state
    assert request.num_computed_tokens == 0
    assert request.num_tokens == 4

    request.num_computed_tokens = 4
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False

    manager.apply_decode_commit(
        request,
        new_seq_len=7,
        new_token_ids=[100, 101, 102],
    )

    assert request.num_tokens == 7
    assert request.num_computed_tokens == 7
    assert len(request.block_hashes) > 0  # appended tokens trigger block_hashes update


def test_apply_decode_commit_consumes_existing_sampled_token():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_hasher=lambda req: [b"h"],
    )
    request.num_computed_tokens = 4
    request.append_output_token_ids(100)
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False

    manager.apply_decode_commit(
        request,
        new_seq_len=5,
        new_token_ids=[100],
    )

    assert list(request.all_token_ids) == [1, 2, 3, 4, 100]
    assert request.num_computed_tokens == 5


def test_apply_decode_commit_appends_after_existing_sampled_token():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_hasher=lambda req: [b"h"],
    )
    request.num_computed_tokens = 4
    request.append_output_token_ids(100)
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False

    manager.apply_decode_commit(
        request,
        new_seq_len=6,
        new_token_ids=[100, 101],
    )

    assert list(request.all_token_ids) == [1, 2, 3, 4, 100, 101]
    assert request.num_computed_tokens == 6


def test_apply_decode_commit_rejects_existing_sampled_token_mismatch():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_hasher=lambda req: [b"h"],
    )
    request.num_computed_tokens = 4
    request.append_output_token_ids(100)
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False

    with pytest.raises(ValueError, match="existing uncomputed token"):
        manager.apply_decode_commit(
            request,
            new_seq_len=5,
            new_token_ids=[999],
        )

    assert list(request.all_token_ids) == [1, 2, 3, 4, 100]
    assert request.num_computed_tokens == 4


def test_apply_decode_commit_rejects_token_delta_mismatch():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_hasher=lambda req: [b"h"],
    )
    request.num_computed_tokens = 4
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False

    with pytest.raises(ValueError, match="new_token_ids"):
        manager.apply_decode_commit(request, new_seq_len=6, new_token_ids=[100])


def test_prefix_cache_audit_state_reports_safe_block_counts():
    from types import SimpleNamespace

    from vllm.pap.prefix_cache_audit import build_prefix_cache_audit_state

    request = SimpleNamespace(
        request_id="req-1",
        num_tokens=33,
        num_computed_tokens=32,
        block_hashes=[bytes.fromhex("11" * 32), bytes.fromhex("22" * 32)],
    )
    blocks = [
        SimpleNamespace(block_hash=bytes.fromhex("aa" * 32) + b"\x00" * 4),
        SimpleNamespace(block_hash=bytes.fromhex("bb" * 32) + b"\x00" * 4),
        SimpleNamespace(block_hash=None),
    ]
    group = SimpleNamespace(
        kv_cache_group_id=0,
        req_to_blocks={"req-1": blocks},
        num_cached_block={"req-1": 2},
    )
    manager = SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[group]))

    state = build_prefix_cache_audit_state(manager, request)

    assert state == {
        "request_id": "req-1",
        "num_tokens": 33,
        "num_computed_tokens": 32,
        "request_hash_count": 2,
        "request_hash_tail": ["1111111111111111", "2222222222222222"],
        "groups": [
            {
                "group_id": 0,
                "allocated_blocks": 3,
                "cached_blocks": 2,
                "hashed_blocks": 2,
                "allocated_hash_tail": [
                    "aaaaaaaaaaaaaaaa",
                    "bbbbbbbbbbbbbbbb",
                ],
            }
        ],
    }


def test_apply_decode_commit_emits_prefix_cache_audit(monkeypatch):
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    monkeypatch.setenv("PAP_PREFIX_CACHE_AUDIT", "1")
    request = Request(
        request_id="req-audit",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_hasher=lambda _request: [],
    )
    request.num_computed_tokens = 4
    request.append_output_token_ids(100)
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False
    manager.coordinator = SimpleNamespace(single_type_managers=[])
    logs = []
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_manager.logger.info",
        lambda message, *args: logs.append((message, args)),
    )

    manager.apply_decode_commit(request, new_seq_len=5, new_token_ids=[100])

    assert logs[0][0] == "PAP prefix cache commit audit %s"
    assert logs[0][1][0]["request_id"] == "req-audit"


def test_prefix_cache_lookup_emits_audit(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager

    class Coordinator:
        single_type_managers = []

        @staticmethod
        def find_longest_cache_hit(_block_hashes, _max_length):
            return ([],), 0

    monkeypatch.setenv("PAP_PREFIX_CACHE_AUDIT", "1")
    manager = object.__new__(KVCacheManager)
    manager.enable_caching = True
    manager.log_stats = False
    manager.empty_kv_cache_blocks = KVCacheBlocks(((),))
    manager.coordinator = Coordinator()
    logs = []
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_manager.logger.info",
        lambda message, *args: logs.append((message, args)),
    )
    request = SimpleNamespace(
        request_id="req-lookup-audit",
        skip_reading_prefix_cache=False,
        block_hashes=[b"request-hash"],
        num_tokens=17,
        num_computed_tokens=0,
        num_preemptions=0,
    )

    _, hit_tokens = manager.get_computed_blocks(request)

    assert hit_tokens == 0
    assert logs[0][0] == "PAP prefix cache lookup audit %s"
    assert logs[0][1][0]["request_id"] == "req-lookup-audit"
    assert logs[0][1][0]["hit_tokens"] == 0


def test_engine_core_pap_apply_decode_commit_uses_scheduler_manager(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    class StubManager:
        def __init__(self):
            self.calls = []

        def apply_decode_commit(self, *, request, new_seq_len, new_token_ids):
            self.calls.append((request.request_id, new_seq_len, tuple(new_token_ids)))
            request.num_computed_tokens = new_seq_len

    request = type(
        "StubReq",
        (),
        {
            "request_id": "req-1",
            "num_computed_tokens": 4,
        },
    )()
    manager = StubManager()
    refreshed = []
    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_refresh_lease",
        refreshed.append,
    )
    core = object.__new__(EngineCore)
    core.scheduler = type(
        "StubScheduler",
        (),
        {
            "requests": {"req-1": request},
            "kv_cache_manager": manager,
        },
    )()

    result = core.pap_apply_decode_commit("req-1", 5, (42,))

    assert result == {
        "request_id": "req-1",
        "applied": True,
        "old_seq_len": 4,
        "new_seq_len": 5,
    }
    assert manager.calls == [("req-1", 5, (42,))]
    assert refreshed == ["req-1"]


def test_engine_core_pap_apply_decode_commit_reports_unknown_request():
    from vllm.v1.engine.core import EngineCore

    core = object.__new__(EngineCore)
    core.scheduler = type(
        "StubScheduler",
        (),
        {
            "requests": {},
        },
    )()

    result = core.pap_apply_decode_commit("missing", 5, (42,))

    assert result == {
        "request_id": "missing",
        "applied": False,
        "reason": "unknown_request",
    }


def test_engine_core_pap_apply_decode_commit_updates_detached_lease(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.engine.core import EngineCore

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_has_active_lease",
        lambda request_id: request_id == "detached",
    )
    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_kv_seq_len",
        lambda request_id: 32,
    )
    refreshed = []
    updated = []
    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_refresh_lease",
        refreshed.append,
    )
    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_update_kv_seq_len",
        lambda request_id, seq_len: not updated.append((request_id, seq_len)),
    )
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(requests={})

    result = core.pap_apply_decode_commit("detached", 33, (42,))

    assert result == {
        "request_id": "detached",
        "applied": True,
        "old_seq_len": 32,
        "new_seq_len": 33,
        "direct_lease_commit": True,
    }
    assert refreshed == ["detached"]
    assert updated == [("detached", 33)]


def test_engine_core_pap_release_kv_lease(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    released = []

    def fake_release_lease(lease_id):
        released.append(lease_id)
        return (3, 4)

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_release_lease",
        fake_release_lease,
    )

    core = object.__new__(EngineCore)
    result = core.pap_release_kv_lease("req-1", "lease-1")

    assert result == {
        "request_id": "req-1",
        "lease_id": "lease-1",
        "released": True,
        "block_count": 2,
    }
    assert released == ["lease-1"]


def test_engine_core_pap_release_kv_lease_reports_miss(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease.pap_release_lease",
        lambda lease_id: (),
    )

    core = object.__new__(EngineCore)
    result = core.pap_release_kv_lease("req-1", "lease-missing")

    assert result == {
        "request_id": "req-1",
        "lease_id": "lease-missing",
        "released": False,
        "reason": "unknown_or_released_lease",
        "block_count": 0,
    }


def test_pap_lease_remembers_recently_released_request():
    from vllm.pap.lifecycle import lease as pap_lease

    pap_lease.reset_global_kv_lease_registry()
    lease_id = pap_lease.pap_pin_blocks("request", [1, 2])

    assert not pap_lease.pap_was_recently_released("request")
    assert pap_lease.pap_release_lease(lease_id) == (1, 2)
    assert pap_lease.pap_was_recently_released("request")

    pap_lease.reset_global_kv_lease_registry()


# --- DecodeCommitClient tests -------------------------------------------------


class _CommitAckResponse:
    status_code = 200

    def __init__(self, acked_commit_seq: int):
        self.acked_commit_seq = acked_commit_seq

    def raise_for_status(self):
        pass

    def json(self):
        return {"acked_commit_seq": self.acked_commit_seq}


def test_commit_client_posts_to_endpoint(monkeypatch):
    """Verify the client POSTs correct JSON to the configured endpoint."""
    from threading import Event

    posted = {}
    posted_event = Event()

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted_event.set()
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit"
    )
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1, 2))
    assert posted_event.wait(timeout=1.0)
    assert posted["url"].endswith("/v1/pap/prefill/decode-commit")
    assert posted["json"]["request_id"] == "r"
    assert posted["json"]["commit_seq"] == 1
    assert posted["json"]["new_seq_len"] == 10
    assert posted["json"]["new_token_ids"] == [1, 2]
    assert posted["json"]["layer_complete"] is True
    assert posted["json"]["submit_only"] is True


def test_commit_client_can_route_each_request_to_its_prefill(monkeypatch):
    """A process-wide client must not pin every PA session to PA0."""
    from threading import Event

    monkeypatch.delenv("PAP_DECODE_COMMIT_ENDPOINT", raising=False)
    posted = {}
    posted_event = Event()

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted_event.set()
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(endpoint=None)
    endpoint = "http://127.0.0.1:8103/v1/pap/prefill/decode-commit"

    client.commit(
        request_id="pa3-request",
        new_seq_len=10,
        new_token_ids=(7,),
        endpoint=endpoint,
    )

    assert client.flush_request("pa3-request", timeout_s=1.0)
    assert posted_event.wait(timeout=1.0)
    assert posted["url"] == endpoint
    assert posted["json"]["request_id"] == "pa3-request"


def test_commit_client_disabled_when_no_endpoint():
    """client.enabled is False when no endpoint is configured."""
    client = DecodeCommitClient(endpoint=None)
    assert not client.enabled
    # commit() should be a no-op, not raise
    client.commit(request_id="r", new_seq_len=1, new_token_ids=(1,))


def test_commit_client_env_var(monkeypatch):
    """Endpoint can be set via PAP_DECODE_COMMIT_ENDPOINT env var."""
    monkeypatch.setenv("PAP_DECODE_COMMIT_ENDPOINT", "http://localhost:1/x")
    client = DecodeCommitClient()
    assert client.enabled
    assert client.endpoint == "http://localhost:1/x"


def test_commit_client_commit_does_not_block_on_slow_post(monkeypatch):
    import time
    from threading import Event, Timer

    post_started = Event()
    release_post = Event()

    def fake_post(url, json=None, timeout=None):
        post_started.set()
        release_post.wait(timeout=1.0)
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit"
    )
    timer = Timer(0.2, release_post.set)
    timer.start()
    start = time.perf_counter()

    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))
    elapsed = time.perf_counter() - start
    release_post.set()
    timer.cancel()

    assert elapsed < 0.05
    assert post_started.wait(timeout=1.0)


def test_commit_client_deduplicates_pending_payloads(monkeypatch):
    from threading import Event

    posted = []
    posted_event = Event()

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        posted_event.set()
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit"
    )

    for _ in range(8):
        client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))

    assert client.flush_request("r", timeout_s=1.0)
    assert posted_event.wait(timeout=1.0)
    assert len(posted) == 1
    assert posted[0]["request_id"] == "r"
    assert posted[0]["new_seq_len"] == 10


def test_commit_client_coalesces_queued_request_to_latest_state(monkeypatch):
    from threading import Event

    first_post_started = Event()
    release_first_post = Event()
    posted = []

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        first_post_started.set()
        release_first_post.wait(timeout=1.0)
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit"
    )

    client.commit(request_id="blocker", new_seq_len=1, new_token_ids=(1,))
    assert first_post_started.wait(timeout=1.0)
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(10,))
    client.commit(request_id="r", new_seq_len=11, new_token_ids=(11,))
    client.commit(request_id="r", new_seq_len=12, new_token_ids=(12,))

    release_first_post.set()

    assert client.flush_request("r", timeout_s=1.0)
    assert len(posted) == 2
    assert posted[1]["request_id"] == "r"
    assert posted[1]["commit_seq"] == 3
    assert posted[1]["new_seq_len"] == 12
    assert posted[1]["new_token_ids"] == [10, 11, 12]


def test_commit_client_flush_request_waits_for_pending(monkeypatch):
    import time
    from threading import Event, Thread

    post_started = Event()
    release_post = Event()
    flush_result = []

    def fake_post(url, json=None, timeout=None):
        post_started.set()
        release_post.wait(timeout=1.0)
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit"
    )
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))
    assert post_started.wait(timeout=1.0)

    thread = Thread(
        target=lambda: flush_result.append(client.flush_request("r", timeout_s=1.0))
    )
    thread.start()
    time.sleep(0.05)
    assert flush_result == []

    release_post.set()
    thread.join(timeout=1.0)
    assert flush_result == [True]


def test_commit_client_flushes_wrapped_targets_by_session(monkeypatch):
    import time
    from threading import Event, Thread

    post_started = Event()
    release_post = Event()
    posted = []
    flush_result = []

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        post_started.set()
        release_post.wait(timeout=1.0)
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit"
    )
    client.commit(
        request_id="chatcmpl-session-a-turn-1",
        session_request_id="session-a",
        new_seq_len=10,
        new_token_ids=(1,),
    )
    assert post_started.wait(timeout=1.0)

    thread = Thread(
        target=lambda: flush_result.append(
            client.flush_request("session-a", timeout_s=1.0)
        )
    )
    thread.start()
    time.sleep(0.05)
    assert flush_result == []

    release_post.set()
    thread.join(timeout=1.0)
    assert flush_result == [True]
    assert posted[0]["request_id"] == "chatcmpl-session-a-turn-1"
    assert posted[0]["session_request_id"] == "session-a"


def test_commit_client_retries_until_ack(monkeypatch):
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(json["commit_seq"])
        if len(attempts) < 3:
            raise RuntimeError("temporary failure")
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit",
        max_attempts=3,
        retry_initial_s=0,
    )

    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))

    assert client.flush_request("r", timeout_s=1.0)
    assert attempts == [1, 1, 1]


def test_commit_client_flush_fails_without_ack(monkeypatch):
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(json["commit_seq"])
        raise RuntimeError("persistent failure")

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit",
        max_attempts=2,
        retry_initial_s=0,
    )

    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))

    assert not client.flush_request("r", timeout_s=1.0)
    assert attempts == [1, 1]


def test_commit_client_flush_reports_failure_on_queue_full(monkeypatch):
    from threading import Event

    post_started = Event()
    release_post = Event()

    def fake_post(url, json=None, timeout=None):
        post_started.set()
        release_post.wait(timeout=5.0)
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.lifecycle.commit.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit",
        queue_size=1,
    )

    client.commit(request_id="blocker", new_seq_len=1, new_token_ids=(1,))
    assert post_started.wait(timeout=1.0)

    client.commit(request_id="queued", new_seq_len=1, new_token_ids=(1,))
    client.commit(request_id="dropped", new_seq_len=1, new_token_ids=(1,))

    assert not client.flush_request("dropped", timeout_s=1.0)

    release_post.set()


class _LeaseReleaseResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_lease_release_client_default_timeout_covers_commit_lock_wait(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PAP_LEASE_RELEASE_TIMEOUT", raising=False)

    assert LeaseReleaseClient().timeout_s == 5.0


def test_lease_release_client_retries_until_ack(monkeypatch):
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(json)
        if len(attempts) == 1:
            raise RuntimeError("temporary failure")
        return _LeaseReleaseResponse({"released": True})

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease_release.httpx.post",
        fake_post,
    )
    client = LeaseReleaseClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/lease-release",
        max_attempts=2,
        retry_initial_s=0,
    )

    assert client.release(request_id="r", lease_id="lease-1")
    assert len(attempts) == 2


def test_lease_release_client_can_route_to_session_prefill(monkeypatch):
    monkeypatch.delenv("PAP_LEASE_RELEASE_ENDPOINT", raising=False)
    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return _LeaseReleaseResponse({"released": True})

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease_release.httpx.post",
        fake_post,
    )
    client = LeaseReleaseClient(endpoint=None, max_attempts=1)
    endpoint = "http://127.0.0.1:8103/v1/pap/prefill/lease-release"

    assert client.release(
        request_id="pa3-request",
        lease_id="lease-pa3",
        endpoint=endpoint,
    )
    assert posted == {
        "url": endpoint,
        "json": {
            "request_id": "pa3-request",
            "lease_id": "lease-pa3",
            "submit_only": True,
        },
    }


def test_lease_release_client_accepts_idempotent_release(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return _LeaseReleaseResponse(
            {"released": False, "reason": "unknown_or_released_lease"}
        )

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease_release.httpx.post",
        fake_post,
    )
    client = LeaseReleaseClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/lease-release",
        max_attempts=1,
    )

    assert client.release(request_id="r", lease_id="lease-1")


def test_lease_release_client_reports_terminal_failure(monkeypatch):
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(json)
        raise RuntimeError("persistent failure")

    monkeypatch.setattr(
        "vllm.pap.lifecycle.lease_release.httpx.post",
        fake_post,
    )
    client = LeaseReleaseClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/lease-release",
        max_attempts=2,
        retry_initial_s=0,
    )

    assert not client.release(request_id="r", lease_id="lease-1")
    assert len(attempts) == 2


def test_kv_lease_default_ttl_is_finite(monkeypatch):
    from vllm.pap.lifecycle.lease import PAPKVLeaseRegistry

    monkeypatch.delenv("PAP_KV_LEASE_TTL_SECONDS", raising=False)

    assert PAPKVLeaseRegistry()._ttl_seconds == 300.0


def test_kv_lease_refresh_extends_expiry(monkeypatch):
    from vllm.pap.lifecycle.lease import PAPKVLeaseRegistry

    now = [100.0]
    monkeypatch.setattr("vllm.pap.lifecycle.lease.time.time", lambda: now[0])
    registry = PAPKVLeaseRegistry(_ttl_seconds=10.0)
    lease_id = registry.pin_blocks(request_id="r", block_ids=(1, 2))
    assert registry._by_lease[lease_id].expires_at == 110.0

    now[0] = 105.0

    assert registry.refresh_lease("r")
    assert registry._by_lease[lease_id].expires_at == 115.0


def test_kv_lease_tracks_decode_sequence_length() -> None:
    from vllm.pap.lifecycle.lease import PAPKVLeaseRegistry

    registry = PAPKVLeaseRegistry(_ttl_seconds=10.0)
    lease_id = registry.pin_blocks(request_id="r", block_ids=(1, 2, 3))

    assert registry.record_seq_len(request_id="r", seq_len=32)
    assert registry.update_seq_len("r", 40)
    assert registry.seq_len("r") == 40

    registry.release_lease(lease_id)
    assert registry.seq_len("r") is None


def test_kv_lease_binds_sequence_length_recorded_before_manifest_pin() -> None:
    from vllm.pap.lifecycle.lease import PAPKVLeaseRegistry

    registry = PAPKVLeaseRegistry(_ttl_seconds=10.0)

    assert registry.record_seq_len(request_id="r", seq_len=32)
    assert registry.seq_len("r") is None

    registry.pin_blocks(request_id="r", block_ids=(1, 2))
    assert registry.seq_len("r") == 32


def test_kv_lease_sweeps_replaced_expired_lease(monkeypatch):
    from vllm.pap.lifecycle.lease import PAPKVLeaseRegistry

    now = [100.0]
    monkeypatch.setattr("vllm.pap.lifecycle.lease.time.time", lambda: now[0])
    registry = PAPKVLeaseRegistry(_ttl_seconds=10.0)
    old_lease = registry.pin_blocks(request_id="r", block_ids=(1, 2))
    freed = []
    registry.stash_deferred_blocks(
        lease_id=old_lease,
        blocks=(1, 2),
        free_callback=freed.append,
    )

    now[0] = 105.0
    new_lease = registry.pin_blocks(request_id="r", block_ids=(3, 4))
    now[0] = 111.0

    assert registry.sweep_expired_leases() == [old_lease]
    assert freed == [(1, 2)]
    assert registry.active_lease_id("r") == new_lease


def test_kv_lease_release_does_not_retain_tombstone_entries() -> None:
    from vllm.pap.lifecycle.lease import PAPKVLeaseRegistry

    registry = PAPKVLeaseRegistry(_ttl_seconds=10.0)
    for i in range(100):
        lease_id = registry.pin_blocks(request_id=f"r{i}", block_ids=(i,))
        registry.release_lease(lease_id)

    assert registry._by_lease == {}


def test_scheduler_stashes_leased_blocks_tail_first(monkeypatch):
    from vllm.pap.lifecycle import lease as kv_lease
    from vllm.v1.core.sched.scheduler import Scheduler

    captured = {}

    monkeypatch.setattr(kv_lease, "pap_has_active_lease", lambda _request_id: True)
    monkeypatch.setattr(
        kv_lease,
        "pap_active_lease_id",
        lambda _request_id: "lease-1",
    )

    def capture_stash(*, lease_id, blocks, free_callback):
        captured["lease_id"] = lease_id
        captured["blocks"] = list(blocks)
        captured["free_callback"] = free_callback

    monkeypatch.setattr(kv_lease, "pap_stash_deferred_blocks", capture_stash)

    class StubBlockPool:
        def free_blocks(self, _blocks):
            raise AssertionError("lease release must own deferred block freeing")

    class StubKVCacheManager:
        def __init__(self):
            self.block_pool = StubBlockPool()

        def pop_blocks_for_free(self, _request):
            return ["prefix", "middle", "tail"]

    scheduler = object.__new__(Scheduler)
    scheduler.kv_cache_manager = StubKVCacheManager()
    request = type("StubRequest", (), {"request_id": "req-1"})()

    scheduler._free_request_blocks(request)

    assert captured["lease_id"] == "lease-1"
    assert captured["blocks"] == ["tail", "middle", "prefix"]
