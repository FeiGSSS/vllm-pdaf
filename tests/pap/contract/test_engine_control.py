# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.pap.integration.control import PAPEngineControl


class _Request:
    def __init__(self) -> None:
        self.request_id = "req-1"
        self.num_computed_tokens = 2
        self.all_token_ids = [10, 11, 12]

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    def append_output_token_ids(self, tokens: tuple[int, ...]) -> None:
        self.all_token_ids.extend(tokens)


class _Coordinator:
    def __init__(self) -> None:
        self.cached: list[tuple[str, int]] = []

    def cache_blocks(self, request: _Request, seq_len: int) -> None:
        self.cached.append((request.request_id, seq_len))


def _control(monkeypatch: pytest.MonkeyPatch) -> tuple[PAPEngineControl, _Request]:
    request = _Request()
    manager = SimpleNamespace(enable_caching=True, coordinator=_Coordinator())
    scheduler = SimpleNamespace(
        requests={request.request_id: request},
        finished_req_ids=set(),
        kv_cache_manager=manager,
    )
    monkeypatch.setattr(
        "vllm.pap.integration.engine.pap_lease.pap_refresh_lease",
        lambda _: None,
    )
    monkeypatch.setattr(
        "vllm.pap.integration.engine.pap_lease.pap_update_kv_seq_len",
        lambda *_: True,
    )
    return PAPEngineControl(scheduler), request


def test_control_applies_contiguous_commits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    payload = {
        "request_id": request.request_id,
        "commit_seq": 1,
        "new_seq_len": 4,
        "new_token_ids": [12, 13],
    }

    applied = control.apply("decode_commit", payload)
    duplicate = control.apply("decode_commit", payload)

    assert applied["applied"] is True
    assert applied["acked_commit_seq"] == 1
    assert request.num_computed_tokens == 4
    assert request.all_token_ids == [10, 11, 12, 13]
    assert duplicate["idempotent"] is True


def test_control_rejects_gap_and_conflicting_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    first = {
        "request_id": request.request_id,
        "commit_seq": 1,
        "new_seq_len": 3,
        "new_token_ids": [12],
    }
    control.apply("decode_commit", first)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        control.apply("decode_commit", {**first, "new_token_ids": [99]})
    with pytest.raises(ValueError, match="non-contiguous"):
        control.apply(
            "decode_commit",
            {
                "request_id": request.request_id,
                "commit_seq": 3,
                "new_seq_len": 4,
                "new_token_ids": [13],
            },
        )


def test_release_checks_final_commit_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, request = _control(monkeypatch)
    control.apply(
        "decode_commit",
        {
            "request_id": request.request_id,
            "commit_seq": 1,
            "new_seq_len": 3,
            "new_token_ids": [12],
        },
    )
    with pytest.raises(ValueError, match="raced"):
        control.apply(
            "lease_release",
            {
                "request_id": request.request_id,
                "lease_id": "lease-1",
                "final_commit_seq": 0,
            },
        )
