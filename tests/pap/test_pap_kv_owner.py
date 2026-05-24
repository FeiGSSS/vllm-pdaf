import pytest

from vllm.pap.kv_owner import PAKVOwner


def test_pa_kv_owner_reserves_and_materializes_decode_slot() -> None:
    owner = PAKVOwner()

    owner.register_session(
        session_id="session-1",
        block_size=4,
        max_seq_len=16,
    )
    owner.register_layer_blocks(
        session_id="session-1",
        layer_name="model.layers.0.self_attn.attn",
        block_ids=[0],
        seq_len=4,
        num_blocks=2,
    )

    slot = owner.reserve_decode_slot(
        session_id="session-1",
        layer_name="model.layers.0.self_attn.attn",
        block_id=1,
        seq_len=5,
    )

    assert slot.block_id == 1
    assert slot.slot == 4
    assert slot.seq_len == 5
    assert slot.materialized is False
    assert owner.get_layer_state(
        "session-1", "model.layers.0.self_attn.attn"
    ).block_ids == (0, 1)

    materialized = owner.materialize_decode_slot(
        session_id="session-1",
        layer_name="model.layers.0.self_attn.attn",
        block_id=1,
        seq_len=5,
    )

    assert materialized.materialized is True
    assert owner.get_layer_state(
        "session-1", "model.layers.0.self_attn.attn"
    ).seq_len == 5


def test_pa_kv_owner_rejects_unbacked_decode_block() -> None:
    owner = PAKVOwner()
    owner.register_session(
        session_id="session-1",
        block_size=4,
        max_seq_len=16,
    )
    owner.register_layer_blocks(
        session_id="session-1",
        layer_name="model.layers.0.self_attn.attn",
        block_ids=[0],
        seq_len=4,
        num_blocks=1,
    )

    with pytest.raises(ValueError, match="not backed"):
        owner.reserve_decode_slot(
            session_id="session-1",
            layer_name="model.layers.0.self_attn.attn",
            block_id=1,
            seq_len=5,
        )


def test_pa_kv_owner_lease_refcount_controls_release() -> None:
    owner = PAKVOwner()
    owner.register_session(
        session_id="session-1",
        block_size=4,
        max_seq_len=16,
    )
    owner.acquire_lease("session-1")
    owner.acquire_lease("session-1")

    assert owner.release_lease("session-1") is False
    assert owner.get_session("session-1").lease_count == 1
    assert owner.release_lease("session-1") is True
    assert owner.get_session("session-1") is None
