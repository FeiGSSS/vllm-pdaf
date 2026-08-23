# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.pap.transport import factory as factory_module


def test_transport_factory_always_builds_nvshmem_graph(monkeypatch) -> None:
    calls = []
    expected = object()

    def fake_build(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        factory_module,
        "build_nvshmem_offload_exec_transport",
        fake_build,
    )
    result = factory_module.build_offload_exec_transport(
        actor_id="attention",
        local_rank=1,
        buffer_bytes=8192,
    )

    assert result is expected
    assert calls == [{"actor_id": "attention", "local_rank": 1, "buffer_bytes": 8192}]


def test_transport_factory_has_no_transport_selector() -> None:
    assert "transport" not in (
        factory_module.build_offload_exec_transport.__annotations__
    )
