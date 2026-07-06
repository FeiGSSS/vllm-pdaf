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
    """Decode-commit POST invokes manager.apply_decode_commit with correct args."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.pap.decode_commit_router import build_commit_router

    class StubManager:
        def __init__(self):
            self.calls = []

        def apply_decode_commit(self, request, new_seq_len, new_token_ids):
            self.calls.append((request.request_id, new_seq_len,
                               list(new_token_ids)))

    manager = StubManager()
    stub_req = type("StubReq", (), {"request_id": "req-1"})()
    requests = {"req-1": stub_req}
    router = build_commit_router(manager=manager, requests=requests)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    resp = client.post("/v1/pap/prefill/decode-commit",
                       json={
                           "request_id": "req-1",
                           "new_seq_len": 17,
                           "new_token_ids": [1, 2, 3],
                           "layer_complete": True,
                       })
    assert resp.status_code == 200
    assert manager.calls == [("req-1", 17, [1, 2, 3])]


def test_apply_decode_commit_advances_tokens():
    """Simulate a PAP decode commit: tokens appended, num_computed updated."""
    from vllm.sampling_params import SamplingParams

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

    # Apply decode commit: 3 new tokens at seq positions 4,5,6
    request.append_output_token_ids([100, 101, 102])
    request.num_computed_tokens = 7
    assert request.num_tokens == 7
    assert request.num_computed_tokens == 7
    assert len(request.block_hashes) > 0  # appended tokens trigger block_hashes update


# --- DecodeCommitClient tests -------------------------------------------------


def test_commit_client_posts_to_endpoint(monkeypatch):
    """Verify the client POSTs correct JSON to the configured endpoint."""
    posted = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return FakeResp()

    monkeypatch.setattr(
        "vllm.pap.decode_commit_client.httpx.post", fake_post)
    client = DecodeCommitClient(
        endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit")
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1, 2))
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
