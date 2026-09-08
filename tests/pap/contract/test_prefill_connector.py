# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.pap.integration.request import PAPKVCacheMigration, PAPRequestMetadata
from vllm.pap.kv_connector import (
    PAPPrefillConnector,
    _contiguous_block_copy_ranges,
)
from vllm.pap.model.prefill import PAPPrefillKVPublisher
from vllm.pap.protocol import (
    PAPCudaIPCTensorHandle,
    PAPPrefillKVCacheCatalogDescriptor,
)
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
    connector._control_finished = set()
    connector._publishers = {}
    connector._migration_start_blocks = {}
    connector._migration_loads = {}
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


def test_migration_copy_ranges_require_both_sides_to_be_contiguous() -> None:
    assert _contiguous_block_copy_ranges(
        (1, 2, 8, 9, 10),
        (20, 21, 22, 30, 31),
    ) == ((1, 20, 2), (8, 22, 1), (9, 30, 2))


def test_connector_allocates_only_missing_migration_blocks() -> None:
    connector = _connector()
    connector._block_size = 16
    migration = PAPKVCacheMigration.from_mapping(
        {
            "transfer_id": "transfer",
            "source_worker_id": 1,
            "prefix_tokens": 64,
            "block_size": 16,
            "source_block_ids": [10, 11, 12, 13],
            "catalogs": [
                PAPPrefillKVCacheCatalogDescriptor(
                    catalog_id="source",
                    layer_name="layer.0",
                    block_size=16,
                    num_kv_heads=8,
                    layout="NHD",
                    kv_cache=PAPCudaIPCTensorHandle(
                        dtype="float16",
                        shape=(32, 8, 16, 16),
                        ipc_handle={"GPU-source": (b"handle",)},
                    ),
                ).to_dict()
            ],
        }
    )
    connector._request_metadata["migration"] = PAPRequestMetadata(
        kv_migration=migration
    )
    request = SimpleNamespace(request_id="migration", num_tokens=81)

    assert connector.get_num_new_matched_tokens(request, 16) == (48, False)
    blocks = SimpleNamespace(
        blocks=(tuple(SimpleNamespace(block_id=value) for value in range(20, 26)),)
    )
    connector.update_state_after_alloc(request, blocks, 48)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"migration": 17},
        finished_req_ids=set(),
    )
    metadata = connector.build_connector_meta(
        scheduler_output,
        allocated_blocks={"migration": ([20, 21, 22, 23, 24, 25],)},
    )

    migrated = metadata.requests[0]
    assert migrated.kv_migration == migration
    assert migrated.migration_source_block_ids == (11, 12, 13)
    assert migrated.migration_destination_block_ids == (21, 22, 23)


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
        decode_capacity_tokens_by_request={"prefill": 200},
        lease_ids_by_request={"prefill": "lease-1"},
        generations_by_request={"prefill": 0},
    )

    assert captured["request_ids"] == ("decode", "prefill")
    assert captured["num_reqs"] == 2
    assert captured["block_table"] is metadata.block_table
    assert captured["decode_capacity_tokens_by_request"] == {"prefill": 200}


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


def test_control_reclamation_does_not_duplicate_worker_completion(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_has_active_lease", lambda _: False
    )
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_was_recently_released", lambda _: True
    )
    assert connector.get_finished_for_control({"req"}, {"req"}) == {"req"}
    assert connector.get_finished_for_control({"req"}, {"req"}) == set()
    assert connector.get_finished({"req"}) == (None, None)


def test_control_reclamation_leaves_unselected_requests_pending(monkeypatch):
    connector = _connector()
    connector._pending_finished.update({"selected", "other"})
    monkeypatch.setattr(
        "vllm.pap.kv_connector.pap_lease.pap_has_active_lease", lambda _: False
    )
    assert connector.get_finished_for_control(set(), {"selected"}) == {"selected"}
    assert connector.get_finished(set()) == ({"other"}, None)


def test_finished_worker_notification_removes_scheduler_ownership():
    from vllm.v1.outputs import KVConnectorOutput

    connector = _connector()
    connector.update_connector_output(KVConnectorOutput(finished_sending={"prefill"}))
    assert "prefill" not in connector._request_metadata
    assert "prefill" not in connector._generations
    assert "decode" in connector._request_metadata
