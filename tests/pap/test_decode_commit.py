import pytest
from vllm.pap.decode_commit import PAPDecodeCommit, serialize_commit, deserialize_commit
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
