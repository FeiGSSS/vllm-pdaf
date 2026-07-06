import pytest
from vllm.pap.decode_commit import PAPDecodeCommit, serialize_commit, deserialize_commit
from vllm.pap.decode_commit_client import DecodeCommitClient
from vllm.v1.request import Request


def test_commit_roundtrip():
    commit = PAPDecodeCommit(
        request_id="req-1",
        new_seq_len=17,
        new_token_ids=[42, 7, 99],
        layer_complete=True,
    )
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored == commit


def test_commit_tuple_input():
    commit = PAPDecodeCommit(
        request_id="r", new_seq_len=5, new_token_ids=(1, 2), layer_complete=True
    )
    assert isinstance(commit.new_token_ids, tuple)
    assert commit.new_token_ids == (1, 2)


def test_commit_layer_incomplete():
    commit = PAPDecodeCommit(
        request_id="r", new_seq_len=1, new_token_ids=(), layer_complete=False
    )
    assert commit.layer_complete is False
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored.layer_complete is False


def test_commit_empty_tokens():
    commit = PAPDecodeCommit(
        request_id="r", new_seq_len=0, new_token_ids=(), layer_complete=True
    )
    assert commit.new_token_ids == ()
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored.new_token_ids == ()


def test_from_dict_missing_layer_complete_raises():
    with pytest.raises(KeyError):
        PAPDecodeCommit.from_dict(
            {"request_id": "x", "new_seq_len": 1, "new_token_ids": []}
        )


def test_from_dict_missing_request_id_raises():
    with pytest.raises(KeyError):
        PAPDecodeCommit.from_dict(
            {"new_seq_len": 1, "new_token_ids": [], "layer_complete": True}
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
                    "new_seq_len": 17,
                    "new_token_ids": [1, 2, 3],
                    "layer_complete": True,
                },
            )
        return resp, engine_client.calls

    resp, calls = anyio.run(run_request)

    assert resp.status_code == 200
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
                    "new_seq_len": 18,
                    "new_token_ids": [4],
                    "layer_complete": True,
                },
            )
        return resp, engine_client.calls

    resp, calls = anyio.run(run_request)

    assert resp.status_code == 200
    assert calls == [("req-1", 18, (4,))]


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


def test_engine_core_pap_apply_decode_commit_uses_scheduler_manager():
    from vllm.v1.engine.core import EngineCore

    class StubManager:
        def __init__(self):
            self.calls = []

        def apply_decode_commit(self, *, request, new_seq_len, new_token_ids):
            self.calls.append((request.request_id, new_seq_len, tuple(new_token_ids)))
            request.num_computed_tokens = new_seq_len

    request = type("StubReq", (), {
        "request_id": "req-1",
        "num_computed_tokens": 4,
    })()
    manager = StubManager()
    core = object.__new__(EngineCore)
    core.scheduler = type("StubScheduler", (), {
        "requests": {"req-1": request},
        "kv_cache_manager": manager,
    })()

    result = core.pap_apply_decode_commit("req-1", 5, (42,))

    assert result == {
        "request_id": "req-1",
        "applied": True,
        "old_seq_len": 4,
        "new_seq_len": 5,
    }
    assert manager.calls == [("req-1", 5, (42,))]


def test_engine_core_pap_apply_decode_commit_reports_unknown_request():
    from vllm.v1.engine.core import EngineCore

    core = object.__new__(EngineCore)
    core.scheduler = type("StubScheduler", (), {
        "requests": {},
    })()

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


# --- DecodeCommitClient tests -------------------------------------------------


def test_commit_client_posts_to_endpoint(monkeypatch):
    """Verify the client POSTs correct JSON to the configured endpoint."""
    from threading import Event

    posted = {}
    posted_event = Event()

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted_event.set()
        return FakeResp()

    monkeypatch.setattr(
        "vllm.pap.decode_commit_client.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit")
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1, 2))
    assert posted_event.wait(timeout=1.0)
    assert posted["url"].endswith("/v1/pap/prefill/decode-commit")
    assert posted["json"]["request_id"] == "r"
    assert posted["json"]["new_seq_len"] == 10
    assert posted["json"]["new_token_ids"] == [1, 2]
    assert posted["json"]["layer_complete"] is True


def test_commit_client_disabled_when_no_endpoint():
    """client.enabled is False when no endpoint is configured."""
    client = DecodeCommitClient(endpoint=None)
    assert not client.enabled
    # commit() should be a no-op, not raise
    client.commit(request_id="r", new_seq_len=1, new_token_ids=(1,))


def test_commit_client_env_var(monkeypatch):
    """Endpoint can be set via PAP_DECODE_COMMIT_ENDPOINT env var."""
    monkeypatch.setenv("PAP_DECODE_COMMIT_ENDPOINT",
                       "http://localhost:1/x")
    client = DecodeCommitClient()
    assert client.enabled
    assert client.endpoint == "http://localhost:1/x"


def test_commit_client_commit_does_not_block_on_slow_post(monkeypatch):
    import time
    from threading import Event, Timer

    post_started = Event()
    release_post = Event()

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        post_started.set()
        release_post.wait(timeout=1.0)
        return FakeResp()

    monkeypatch.setattr(
        "vllm.pap.decode_commit_client.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit")
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

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        posted.append(json)
        posted_event.set()
        return FakeResp()

    monkeypatch.setattr(
        "vllm.pap.decode_commit_client.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit")

    for _ in range(8):
        client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))

    assert client.flush_request("r", timeout_s=1.0)
    assert posted_event.wait(timeout=1.0)
    assert len(posted) == 1
    assert posted[0]["request_id"] == "r"
    assert posted[0]["new_seq_len"] == 10


def test_commit_client_flush_request_waits_for_pending(monkeypatch):
    import time
    from threading import Event, Thread

    post_started = Event()
    release_post = Event()
    flush_result = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        post_started.set()
        release_post.wait(timeout=1.0)
        return FakeResp()

    monkeypatch.setattr(
        "vllm.pap.decode_commit_client.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit")
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1,))
    assert post_started.wait(timeout=1.0)

    thread = Thread(
        target=lambda: flush_result.append(
            client.flush_request("r", timeout_s=1.0)
        )
    )
    thread.start()
    time.sleep(0.05)
    assert flush_result == []

    release_post.set()
    thread.join(timeout=1.0)
    assert flush_result == [True]


# --- Descriptor integration tests ---------------------------------------------


def test_offload_exec_descriptor_supports_decode_token_ids():
    """PAPOffloadExecDescriptor carries optional decode_token_ids."""
    from vllm.pap.data_plane import PAPOffloadExecDescriptor

    # Default: empty tuple, backward-compatible
    desc = PAPOffloadExecDescriptor(
        request_id="r", layer_name="l", step=10, scale=0.5,
    )
    assert desc.decode_token_ids == ()

    # With token IDs
    desc2 = PAPOffloadExecDescriptor(
        request_id="r", layer_name="l", step=10, scale=0.5,
        decode_token_ids=(42, 7),
    )
    assert desc2.decode_token_ids == (42, 7)
