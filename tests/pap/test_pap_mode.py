# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.pap.mode import (
    is_pap_request_id,
    pap_request_ids_are_routable,
)


def test_is_pap_request_id_accepts_openai_completion_ids() -> None:
    assert is_pap_request_id("cmpl-123")
    assert is_pap_request_id("chatcmpl-123")


def test_is_pap_request_id_rejects_non_openai_completion_ids() -> None:
    assert not is_pap_request_id("req-123")
    assert not is_pap_request_id(None)


def test_pap_request_ids_are_routable_rejects_mixed_scheduled_batch() -> None:
    assert not pap_request_ids_are_routable(("cmpl-1", "req-2"), 2)


def test_pap_request_ids_are_routable_accepts_all_scheduled_requests() -> None:
    assert pap_request_ids_are_routable(("cmpl-1", "chatcmpl-2"), 2)


def test_pap_request_ids_are_routable_checks_only_scheduled_requests() -> None:
    assert pap_request_ids_are_routable(("cmpl-1", "req-2"), 1)


def test_pap_request_ids_are_routable_rejects_short_or_empty_inputs() -> None:
    assert not pap_request_ids_are_routable(("cmpl-1",), 2)
    assert not pap_request_ids_are_routable((), 0)
    assert not pap_request_ids_are_routable(None, 1)
