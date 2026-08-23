# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from vllm.pap.integration.request import PAPRequestMetadata
from vllm.pap.kv_connector import PAPPrefillConnector
from vllm.pap.model.prefill import PAPPrefillKVPublisher


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

    metadata = connector.build_connector_meta(scheduler_output)

    assert [request.request_id for request in metadata.requests] == [
        "decode",
        "prefill",
    ]
    assert metadata.requests[1].prefill_kv_handle == "session-1"
    assert metadata.finished_request_ids == ("old",)


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
        decode_capacity_tokens_by_request={"prefill": 200},
        import_request_ids={"prefill"},
        tcp_endpoint_by_request={"prefill": "127.0.0.1:8200"},
        attn_metadata=metadata,
        kv_cache=torch.empty(1),
        block_size=16,
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
