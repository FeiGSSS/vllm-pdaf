import pytest
from vllm.pap.decode_commit import PAPDecodeCommit, serialize_commit, deserialize_commit


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
