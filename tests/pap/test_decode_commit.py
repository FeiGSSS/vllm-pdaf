# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.pap.lifecycle.commit import DecodeCommitClient
from vllm.pap.lifecycle.lease_release import LeaseReleaseClient


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
    freed: list[tuple[int, ...]] = []
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
