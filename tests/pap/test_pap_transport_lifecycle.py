# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from vllm.pap.attention.execution import run_offload_exec_nvshmem_graph_loop
from vllm.pap.attention.peers import (
    PAPAttentionPeerConflict,
    PAPAttentionPeerManager,
)
from vllm.pap.protocol import PAPOffloadExecTransportClosed


def test_graph_receive_loop_treats_transport_stop_as_normal() -> None:
    class StoppedTransport:
        actor_id = "attention"

        def recv_graph_step_plan(self):
            raise PAPOffloadExecTransportClosed

    run_offload_exec_nvshmem_graph_loop(
        registry=SimpleNamespace(),
        transport=StoppedTransport(),
    )


def test_peer_manager_stops_receiver_before_runtime_and_transport() -> None:
    events: list[str] = []
    receive_stopped = Event()

    class FakeTransport:
        local_agent_metadata = b"local"

        def stop_receiving(self) -> None:
            events.append("stop_receiving")
            receive_stopped.set()

        def close(self) -> None:
            assert not receiver.is_alive()
            events.append("close")

    class FakeRuntime:
        def stop(self) -> None:
            assert not receiver.is_alive()
            events.append("runtime_stop")

    transport = FakeTransport()
    manager = PAPAttentionPeerManager(
        runtime=FakeRuntime(),
        config=SimpleNamespace(
            attention=SimpleNamespace(local_rank=0, actor_id="attention")
        ),
    )
    receiver = Thread(target=receive_stopped.wait)
    receiver.start()
    manager.initial_transport = transport
    manager.transports["peer"] = transport
    manager.receiver_threads["peer"] = receiver

    manager.stop()

    assert events == ["stop_receiving", "runtime_stop", "close"]
    with pytest.raises(PAPAttentionPeerConflict, match="stopping"):
        manager.bind(peer_metadata=b"peer", source_id=None)
