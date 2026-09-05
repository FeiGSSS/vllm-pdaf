# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

import vllm.pap.attention.step_graph as attention_step_graph
from vllm.pap.attention.execution import run_offload_exec_nvshmem_graph_loop
from vllm.pap.attention.peers import (
    PAPAttentionPeerConflict,
    PAPAttentionPeerManager,
)
from vllm.pap.protocol import PAPOffloadExecTransportClosed
from vllm.pap.transport.nvshmem.transport import PAPNVSHMEMTransport


def test_graph_receive_loop_treats_transport_stop_as_normal() -> None:
    class StoppedTransport:
        actor_id = "attention"

        def recv_graph_step_plan(self):
            raise PAPOffloadExecTransportClosed

    run_offload_exec_nvshmem_graph_loop(
        registry=SimpleNamespace(),
        transport=StoppedTransport(),
    )


def test_attention_graph_cache_evicts_oldest_entry_safely() -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.synchronizations = 0

        def synchronize(self) -> None:
            self.synchronizations += 1

    def entry(plan):
        return attention_step_graph._PAPAttentionGraphEntry(
            graph=object(),
            stream=FakeStream(),
            bound_tensors=(),
            attention_plan=plan,
        )

    executor = attention_step_graph.PAPAttentionStepGraphExecutor(
        registry=None,
        transport=None,
        max_entries=2,
    )
    first_plan = object()
    first = entry(first_plan)
    second = entry(object())
    third = entry(object())

    executor._store_entry((1,), first)
    executor._store_entry((2,), second)
    executor._store_entry((3,), third)

    assert list(executor._entries) == [(2,), (3,)]
    assert first.stream.synchronizations == 1
    assert first.attention_plan is first_plan
    executor.shutdown()
    assert second.stream.synchronizations == 1
    assert third.stream.synchronizations == 1
    assert not executor._entries


def test_peer_manager_stops_receiver_before_runtime_and_transport() -> None:
    events: list[str] = []
    receive_stopped = Event()

    class FakeTraceRecorder:
        def export_attention_kernel_trace(self) -> None:
            assert not receiver.is_alive()
            events.append("export_trace")

    class FakeTransport:
        local_agent_metadata = b"local"
        tracing = FakeTraceRecorder()

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

    assert events == ["stop_receiving", "export_trace", "runtime_stop", "close"]
    with pytest.raises(PAPAttentionPeerConflict, match="stopping"):
        manager.bind(peer_metadata=b"peer", source_id=None)


def test_transport_does_not_commit_after_receive_stop() -> None:
    transport = object.__new__(PAPNVSHMEMTransport)
    transport._lifecycle_lock = Lock()
    transport._stopped = Event()
    transport._closed = False
    commits = []

    assert transport.commit_received_step(lambda: commits.append("first"))
    transport._stopped.set()
    assert not transport.commit_received_step(lambda: commits.append("late"))
    assert commits == ["first"]
