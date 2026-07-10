import pytest

from vllm.pap.decode_commit import PAPDecodeCommit, deserialize_commit, serialize_commit
from vllm.pap.decode_commit_client import DecodeCommitClient
from vllm.pap.lease_release_client import LeaseReleaseClient
from vllm.v1.request import Request


def test_commit_roundtrip():
    commit = PAPDecodeCommit(
        request_id="req-1",
        commit_seq=3,
        new_seq_len=17,
        new_token_ids=[42, 7, 99],
        layer_complete=True,
    )
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored == commit


def test_commit_tuple_input():
    commit = PAPDecodeCommit(
        request_id="r",
        commit_seq=1,
        new_seq_len=5,
        new_token_ids=(1, 2),
        layer_complete=True,
    )
    assert isinstance(commit.new_token_ids, tuple)
    assert commit.new_token_ids == (1, 2)


def test_commit_layer_incomplete():
    commit = PAPDecodeCommit(
        request_id="r",
        commit_seq=1,
        new_seq_len=1,
        new_token_ids=(),
        layer_complete=False,
    )
    assert commit.layer_complete is False
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored.layer_complete is False


def test_commit_empty_tokens():
    commit = PAPDecodeCommit(
        request_id="r",
        commit_seq=1,
        new_seq_len=0,
        new_token_ids=(),
        layer_complete=True,
    )
    assert commit.new_token_ids == ()
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored.new_token_ids == ()


def test_from_dict_missing_layer_complete_raises():
    with pytest.raises(KeyError):
        PAPDecodeCommit.from_dict(
            {
                "request_id": "x",
                "commit_seq": 1,
                "new_seq_len": 1,
                "new_token_ids": [],
            }
        )


def test_from_dict_missing_request_id_raises():
    with pytest.raises(KeyError):
        PAPDecodeCommit.from_dict(
            {
                "commit_seq": 1,
                "new_seq_len": 1,
                "new_token_ids": [],
                "layer_complete": True,
            }
        )


def test_commit_rejects_non_positive_commit_seq():
    with pytest.raises(ValueError, match="commit_seq"):
        PAPDecodeCommit(
            request_id="r",
            commit_seq=0,
            new_seq_len=1,
            new_token_ids=(1,),
            layer_complete=True,
        )


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
    monkeypatch.setattr("vllm.pap.kv_lease.pap_refresh_lease", refreshed.append)
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


def test_engine_core_pap_release_kv_lease(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    released = []

    def fake_release_lease(lease_id):
        released.append(lease_id)
        return (3, 4)

    monkeypatch.setattr("vllm.pap.kv_lease.pap_release_lease", fake_release_lease)

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

    monkeypatch.setattr("vllm.pap.kv_lease.pap_release_lease", lambda lease_id: ())

    core = object.__new__(EngineCore)
    result = core.pap_release_kv_lease("req-1", "lease-missing")

    assert result == {
        "request_id": "req-1",
        "lease_id": "lease-missing",
        "released": False,
        "reason": "unknown_or_released_lease",
        "block_count": 0,
    }


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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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


def test_commit_client_retries_until_ack(monkeypatch):
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(json["commit_seq"])
        if len(attempts) < 3:
            raise RuntimeError("temporary failure")
        return _CommitAckResponse(json["commit_seq"])

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.decode_commit_client.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit",
        max_attempts=2,
        retry_initial_s=0,
    )

    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))

    assert not client.flush_request("r", timeout_s=1.0)
    assert attempts == [1, 1]


class _LeaseReleaseResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_lease_release_client_retries_until_ack(monkeypatch):
    attempts = []

    def fake_post(url, json=None, timeout=None):
        attempts.append(json)
        if len(attempts) == 1:
            raise RuntimeError("temporary failure")
        return _LeaseReleaseResponse({"released": True})

    monkeypatch.setattr("vllm.pap.lease_release_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.lease_release_client.httpx.post", fake_post)
    client = LeaseReleaseClient(endpoint=None, max_attempts=1)
    endpoint = "http://127.0.0.1:8103/v1/pap/prefill/lease-release"

    assert client.release(
        request_id="pa3-request",
        lease_id="lease-pa3",
        endpoint=endpoint,
    )
    assert posted == {
        "url": endpoint,
        "json": {"request_id": "pa3-request", "lease_id": "lease-pa3"},
    }


def test_lease_release_client_accepts_idempotent_release(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return _LeaseReleaseResponse(
            {"released": False, "reason": "unknown_or_released_lease"}
        )

    monkeypatch.setattr("vllm.pap.lease_release_client.httpx.post", fake_post)
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

    monkeypatch.setattr("vllm.pap.lease_release_client.httpx.post", fake_post)
    client = LeaseReleaseClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/lease-release",
        max_attempts=2,
        retry_initial_s=0,
    )

    assert not client.release(request_id="r", lease_id="lease-1")
    assert len(attempts) == 2


def test_kv_lease_default_ttl_is_finite(monkeypatch):
    from vllm.pap.kv_lease import PAPKVLeaseRegistry

    monkeypatch.delenv("PAP_KV_LEASE_TTL_SECONDS", raising=False)

    assert PAPKVLeaseRegistry()._ttl_seconds == 300.0


def test_kv_lease_refresh_extends_expiry(monkeypatch):
    from vllm.pap.kv_lease import PAPKVLeaseRegistry

    now = [100.0]
    monkeypatch.setattr("vllm.pap.kv_lease.time.time", lambda: now[0])
    registry = PAPKVLeaseRegistry(_ttl_seconds=10.0)
    lease_id = registry.pin_blocks(request_id="r", block_ids=(1, 2))
    assert registry._by_lease[lease_id].expires_at == 110.0

    now[0] = 105.0

    assert registry.refresh_lease("r")
    assert registry._by_lease[lease_id].expires_at == 115.0


def test_kv_lease_sweeps_replaced_expired_lease(monkeypatch):
    from vllm.pap.kv_lease import PAPKVLeaseRegistry

    now = [100.0]
    monkeypatch.setattr("vllm.pap.kv_lease.time.time", lambda: now[0])
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


# --- Descriptor integration tests ---------------------------------------------


def test_offload_exec_descriptor_supports_decode_token_ids():
    """PAPOffloadExecDescriptor carries optional decode_token_ids."""
    from vllm.pap.data_plane import PAPOffloadExecDescriptor

    # Default: empty tuple, backward-compatible
    desc = PAPOffloadExecDescriptor(
        request_id="r",
        layer_name="l",
        step=10,
        scale=0.5,
    )
    assert desc.decode_token_ids == ()

    # With token IDs
    desc2 = PAPOffloadExecDescriptor(
        request_id="r",
        layer_name="l",
        step=10,
        scale=0.5,
        decode_token_ids=(42, 7),
    )
    assert desc2.decode_token_ids == (42, 7)
