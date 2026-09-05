# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.kv_connector import PAPPrefillConnector
from vllm.pap.model.prefill import PAPPrefillKVPublisher
from vllm.v1.request import RequestStatus


def _connector() -> PAPPrefillConnector:
    connector = object.__new__(PAPPrefillConnector)
    connector._request_metadata = {
        "decode": PAPRequestMetadata(),
        "prefill": PAPRequestMetadata(
            attention_tcp_endpoint="127.0.0.1:8200",
            decode_capacity_tokens=200,
            prefill_kv_handle="session-1",
            import_prefill_kv_to_attention=True,
        ),
    }
    connector._generations = {"prefill": 0}
    connector._decode_query_len = 1
    connector._pending_finished = set()
    connector._publishers = {}
    return connector


def test_connector_metadata_matches_mrv2_batch_order() -> None:
    connector = _connector()
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"prefill": 128, "decode": 1},
        finished_req_ids={"old"},
    )

    metadata = connector.build_connector_meta(
        scheduler_output, allocated_blocks={"prefill": ([3, 4],)}
    )

    assert [request.request_id for request in metadata.requests] == [
        "decode",
        "prefill",
    ]
    assert metadata.requests[1].prefill_kv_handle == "session-1"
    assert metadata.requests[1].allocated_block_ids == (3, 4)
    assert metadata.finished_request_ids == ("old",)


def test_prefill_preemption_revokes_before_releasing_ownership(monkeypatch) -> None:
    connector = _connector()
    connector._request_metadata["prefill"] = PAPRequestMetadata(
        attention_tcp_endpoint="tcp://127.0.0.1:8300",
        prefill_kv_handle="session-1",
        import_prefill_kv_to_attention=True,
    )
    calls = []
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_active_lease_id",
        lambda _request_id: "lease-1",
    )
    monkeypatch.setattr(
        "vllm.pap.kv.handoff.revoke_prefill_kv",
        lambda **kwargs: calls.append(("revoke", kwargs)),
    )
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_release_lease",
        lambda lease_id: calls.append(("release", lease_id)),
    )

    connector.preempt_request(SimpleNamespace(request_id="prefill"))

    assert [call[0] for call in calls] == ["revoke", "release"]
    assert connector._generations["prefill"] == 1


def test_failed_prefill_revocation_preserves_local_generation(monkeypatch) -> None:
    connector = _connector()

    def fail_revoke(**_kwargs) -> None:
        raise RuntimeError("unreachable")

    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_active_lease_id",
        lambda _request_id: "lease-1",
    )
    monkeypatch.setattr(
        "vllm.pap.kv.handoff.revoke_prefill_kv",
        fail_revoke,
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        connector.preempt_request(SimpleNamespace(request_id="prefill"))

    assert connector._generations["prefill"] == 0


def test_publisher_uses_connector_batch_rows(monkeypatch) -> None:
    publisher = PAPPrefillKVPublisher("layer.0", 8, False, 36)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        PAPPrefillKVPublisher,
        "_publish_manifests",
        lambda self, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "vllm.pap.model.prefill.get_kv_cache_layout",
        lambda: "NHD",
    )
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([5, 9]),
        block_table=torch.tensor([[1, 2], [3, 4]]),
    )

    publisher.publish_connector_batch(
        request_ids=("decode", "prefill"),
        num_scheduled_tokens=(1, 4),
        prefill_kv_handle_by_request={"prefill": "session-1"},
        import_request_ids={"prefill"},
        tcp_endpoint_by_request={"prefill": "127.0.0.1:8200"},
        attn_metadata=metadata,
        kv_cache=torch.empty(1),
        block_size=16,
        allocated_block_ids_by_request={"prefill": (3, 4)},
        lease_ids_by_request={"prefill": "lease-1"},
        generations_by_request={"prefill": 0},
    )

    assert captured["request_ids"] == ("decode", "prefill")
    assert captured["num_reqs"] == 2
    assert captured["block_table"] is metadata.block_table


def test_connector_reports_finished_only_after_lease_release(monkeypatch) -> None:
    connector = _connector()
    active = {"req"}
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_has_active_lease",
        lambda request_id: request_id in active,
    )

    assert connector.get_finished({"req"}) == (None, None)
    active.clear()
    assert connector.get_finished(set()) == ({"req"}, None)


def test_connector_does_not_report_abort_without_delayed_free(monkeypatch) -> None:
    connector = _connector()
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_has_active_lease",
        lambda _request_id: False,
    )
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_was_recently_released",
        lambda _request_id: False,
    )

    assert connector.get_finished({"req"}) == (None, None)
    assert connector._pending_finished == set()


def test_connector_reports_lease_released_before_first_poll(monkeypatch) -> None:
    connector = _connector()
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_has_active_lease",
        lambda _request_id: False,
    )
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_was_recently_released",
        lambda _request_id: True,
    )

    assert connector.get_finished({"req"}) == ({"req"}, None)


def test_connector_allows_abort_before_kv_lease(monkeypatch) -> None:
    connector = _connector()
    connector._pending_finished.add("prefill")
    finished: list[set[str]] = []
    connector._publishers = {
        "layer.0": SimpleNamespace(
            finish_requests=lambda request_ids: finished.append(request_ids)
        )
    }
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_active_lease_id",
        lambda _request_id: None,
    )
    request = SimpleNamespace(
        request_id="prefill",
        status=RequestStatus.FINISHED_ABORTED,
        num_computed_tokens=1024,
    )

    assert connector.request_finished(request, []) == (False, None)
    assert "prefill" not in connector._pending_finished
    assert finished == [{"prefill"}]


def test_connector_rejects_non_abort_without_kv_lease(monkeypatch) -> None:
    connector = _connector()
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_active_lease_id",
        lambda _request_id: None,
    )
    request = SimpleNamespace(
        request_id="prefill",
        status=RequestStatus.FINISHED_STOPPED,
        num_computed_tokens=1024,
    )

    with pytest.raises(RuntimeError, match="finished without a KV lease"):
        connector.request_finished(request, [])
