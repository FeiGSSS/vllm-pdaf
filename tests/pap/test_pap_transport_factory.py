# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm.pap.config import PAPConfigError, PAPOffloadExecTransport
from vllm.pap.transport import factory as factory_module


@pytest.mark.parametrize(
    "transport",
    ["nixl_mailbox", PAPOffloadExecTransport.NIXL_MAILBOX],
)
def test_transport_factory_selects_nixl_mailbox(monkeypatch, transport) -> None:
    calls = []
    expected = object()

    def fake_build(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        factory_module,
        "build_nixl_mailbox_offload_exec_transport",
        fake_build,
    )

    result = factory_module.build_offload_exec_transport(
        transport=transport,
        actor_id="projection-0",
        local_rank=2,
        buffer_bytes=4096,
    )

    assert result is expected
    assert calls == [
        {
            "actor_id": "projection-0",
            "local_rank": 2,
            "buffer_bytes": 4096,
        }
    ]


@pytest.mark.parametrize(
    "transport",
    ["local_fast", PAPOffloadExecTransport.LOCAL_FAST],
)
def test_transport_factory_selects_local_fast(monkeypatch, transport) -> None:
    calls = []
    expected = object()

    def fake_build(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        factory_module,
        "build_local_fast_offload_exec_transport",
        fake_build,
    )

    result = factory_module.build_offload_exec_transport(
        transport=transport,
        actor_id="attention",
        local_rank=1,
    )

    assert result is expected
    assert calls == [
        {
            "actor_id": "attention",
            "local_rank": 1,
            "buffer_bytes": None,
        }
    ]


def test_transport_factory_rejects_unknown_backend() -> None:
    with pytest.raises(PAPConfigError, match="PAP_OFFLOAD_EXEC_TRANSPORT"):
        factory_module.build_offload_exec_transport(
            transport="nccl",
            actor_id="projection-0",
            local_rank=0,
        )
