import pytest

from vllm.pap.mode import PAPMode, parse_pap_mode, is_debug_remote_attention


def test_parse_pap_mode_defaults_to_debug_remote_attention() -> None:
    assert parse_pap_mode(None) is PAPMode.DEBUG_REMOTE_ATTENTION
    assert parse_pap_mode("") is PAPMode.DEBUG_REMOTE_ATTENTION


def test_parse_pap_mode_accepts_true_split() -> None:
    assert parse_pap_mode("true_split") is PAPMode.TRUE_SPLIT


def test_parse_pap_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unsupported PAP mode"):
        parse_pap_mode("shadow-but-fast")


def test_debug_remote_attention_helper() -> None:
    assert is_debug_remote_attention("debug_remote_attention")
    assert not is_debug_remote_attention("true_split")


def test_parse_pap_mode_accepts_true_split_performance() -> None:
    assert parse_pap_mode("true_split_performance") is PAPMode.TRUE_SPLIT_PERFORMANCE


def test_debug_remote_attention_helper_rejects_performance_mode() -> None:
    assert not is_debug_remote_attention("true_split_performance")
