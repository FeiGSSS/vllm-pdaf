from vllm.pap.mode import is_pap_enabled


def test_is_pap_enabled_false_for_none() -> None:
    assert not is_pap_enabled(None)


def test_is_pap_enabled_false_for_empty_dict() -> None:
    assert not is_pap_enabled({})


def test_is_pap_enabled_false_when_missing() -> None:
    assert not is_pap_enabled({})


def test_is_pap_enabled_false_when_false() -> None:
    assert not is_pap_enabled({"pap_enabled": False})


def test_is_pap_enabled_true() -> None:
    assert is_pap_enabled({"pap_enabled": True})
