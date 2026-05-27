import pytest

from vllm.pap.attention_session import (
    AttentionDecodeDescriptor,
    AttentionSessionStore,
)


def test_attention_session_lifecycle_tracks_prefill_and_decode_blocks() -> None:
    store = AttentionSessionStore()

    session = store.create_session(
        request_id="req-1",
        conversation_id="conv-1",
        block_size=16,
        max_seq_len=64,
    )
    store.import_prefill_kv("req-1", block_ids=[1, 2], seq_len=24)
    store.append_decode_token("req-1", block_id=2, seq_len=25)

    assert session.request_id == "req-1"
    assert store.get_session("req-1").conversation_id == "conv-1"
    assert store.get_session("req-1").block_ids == (1, 2)
    assert store.get_session("req-1").seq_len == 25

    store.free_session("req-1")
    assert store.get_session("req-1") is None


def test_append_decode_token_does_not_reenter_public_record_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)
    store.import_prefill_kv("req-1", block_ids=[2], seq_len=24)

    def fail_if_called(_descriptor: AttentionDecodeDescriptor) -> None:
        raise AssertionError("append_decode_token must not reenter public record")

    monkeypatch.setattr(store, "record_decode_descriptor", fail_if_called)

    updated = store.append_decode_token("req-1", block_id=2, seq_len=25)

    assert updated.block_ids == (2,)
    assert updated.seq_len == 25


def test_attention_session_rejects_duplicate_request_id() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)

    with pytest.raises(ValueError, match="already exists"):
        store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)


def test_attention_session_rejects_seq_len_beyond_capacity() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=32)

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        store.import_prefill_kv("req-1", block_ids=[1, 2, 3], seq_len=33)

def test_attention_session_appends_with_scheduler_decode_descriptor() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)
    store.import_prefill_kv("req-1", block_ids=[3, 4], seq_len=24)

    updated = store.append_decode_descriptor(
        AttentionDecodeDescriptor(
            request_id="req-1",
            block_id=4,
            slot=4 * 16 + 8,
            seq_len=25,
        )
    )

    assert updated.block_ids == (3, 4)
    assert updated.seq_len == 25


def test_attention_session_records_existing_scheduler_descriptor() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)
    store.import_prefill_kv("req-1", block_ids=[3, 4], seq_len=24)

    updated, appended = store.record_decode_descriptor(
        AttentionDecodeDescriptor(
            request_id="req-1",
            block_id=4,
            slot=4 * 16 + 7,
            seq_len=24,
        )
    )

    assert appended is False
    assert updated.block_ids == (3, 4)
    assert updated.seq_len == 24


def test_attention_session_appends_new_block_from_scheduler_descriptor() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)
    store.import_prefill_kv("req-1", block_ids=[3, 4], seq_len=32)

    updated = store.append_decode_descriptor(
        AttentionDecodeDescriptor(
            request_id="req-1",
            block_id=9,
            slot=9 * 16,
            seq_len=33,
        )
    )

    assert updated.block_ids == (3, 4, 9)
    assert updated.seq_len == 33


def test_attention_session_rejects_descriptor_that_skips_a_token() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)
    store.import_prefill_kv("req-1", block_ids=[3, 4], seq_len=24)

    with pytest.raises(ValueError, match="expected seq_len 25"):
        store.append_decode_descriptor(
            AttentionDecodeDescriptor(
                request_id="req-1",
                block_id=4,
                slot=4 * 16 + 9,
                seq_len=26,
            )
        )


def test_attention_session_rejects_descriptor_with_wrong_slot() -> None:
    store = AttentionSessionStore()
    store.create_session("req-1", "conv-1", block_size=16, max_seq_len=64)
    store.import_prefill_kv("req-1", block_ids=[3, 4], seq_len=24)

    with pytest.raises(ValueError, match="slot"):
        store.append_decode_descriptor(
            AttentionDecodeDescriptor(
                request_id="req-1",
                block_id=4,
                slot=4 * 16 + 7,
                seq_len=25,
            )
        )
