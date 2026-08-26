# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.pap.model.context import pap_endpoint_for_tp_rank
from vllm.pap.model.projection_routing import (
    _pap_offload_exec_session_request_id,
    _pap_offload_exec_step_groups,
)
from vllm.pap.transport.projection import (
    _pap_bind_offload_exec_nvshmem_peer,
    _pap_cached_offload_exec_transport,
)


def test_pap_endpoint_for_tp_rank_selects_csv_entry() -> None:
    assert (
        pap_endpoint_for_tp_rank(
            "http://127.0.0.1:8300,http://127.0.0.1:8301",
            tp_rank=1,
        )
        == "http://127.0.0.1:8301"
    )


def test_projection_transport_identity_includes_tp_rank(monkeypatch) -> None:
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "1")
    monkeypatch.setattr(
        "vllm.pap.transport.factory.build_offload_exec_transport",
        fake_build,
    )
    _pap_cached_offload_exec_transport.cache_clear()
    try:
        _pap_cached_offload_exec_transport("http://127.0.0.1:8302")
    finally:
        _pap_cached_offload_exec_transport.cache_clear()

    assert calls[0]["local_rank"] == 1
    assert calls[0]["actor_id"].startswith("projection-r1-")


def test_nvshmem_bind_uses_stable_projection_source_id(monkeypatch) -> None:
    calls = []

    class FakeTransport:
        local_agent_metadata = b"projection-metadata"

        def bind_peer(self, metadata):
            calls.append(("bind", metadata))

    def fake_bind(**kwargs):
        calls.append(("request", kwargs))
        return b"attention-metadata"

    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "1")
    monkeypatch.setattr(
        "vllm.pap.attention.client.bind_offload_exec_nvshmem",
        fake_bind,
    )
    transport = FakeTransport()
    _pap_bind_offload_exec_nvshmem_peer(
        transport,
        "http://127.0.0.1:8302",
    )

    assert calls[0][1]["source_id"] == "projection-r1"
    assert calls[1] == ("bind", b"attention-metadata")


def test_step_groups_use_only_attention_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("PAP_OFFLOAD_EXEC_LOCAL_RANK", "0")
    request_ids = ("cmpl-projection-0", "cmpl-projection-1")
    kwargs = {
        "pap_request_ids": request_ids,
        "pap_prefill_kv_handle_by_request": {
            request_ids[0]: "cmpl-prefill-0",
            request_ids[1]: "cmpl-prefill-1",
        },
        "pap_attention_kv_installed_by_request": request_ids,
        "pap_prefill_prefix_len_by_request": dict.fromkeys(request_ids, 50),
        "pap_offload_exec_route_groups": (
            {
                "attention_endpoint": "http://127.0.0.1:8300",
                "req_indices": (0, 1),
                "request_ids": request_ids,
                "steps": (51, 52),
            },
        ),
    }

    groups = _pap_offload_exec_step_groups(kwargs, num_reqs=2, scaling=0.125)

    assert groups[0].attention_endpoint == "http://127.0.0.1:8300"
    assert groups[0].req_indices == (0, 1)
    assert groups[0].metadata_template["r"] == (
        "cmpl-prefill-0",
        "cmpl-prefill-1",
    )
    assert _pap_offload_exec_session_request_id("request", None) == "request"
