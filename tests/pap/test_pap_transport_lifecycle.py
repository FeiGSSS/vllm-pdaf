# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP OFFLOAD_EXEC transport lifecycle tests."""

from collections import OrderedDict
from threading import Condition, Event, RLock, Thread
from types import SimpleNamespace

import pytest

from vllm.pap.attention.execution import (
    run_offload_exec_mailbox_loop,
    run_offload_exec_mailbox_receiver_loop,
)
from vllm.pap.attention.peers import (
    PAPAttentionPeerConflict,
    PAPAttentionPeerManager,
)
from vllm.pap.protocol import PAPOffloadExecTransportClosed
from vllm.pap.transport.local.protocol import DIR_QKV, DOORBELL_BYTES
from vllm.pap.transport.local.transport import PAPLocalFastTransport
from vllm.pap.transport.nixl.endpoint import PAPNixlMailboxEndpoint


def test_local_receive_wait_stops_without_closing_doorbell() -> None:
    transport = object.__new__(PAPLocalFastTransport)
    transport._peer = SimpleNamespace(expected_qkv_seq=1)
    transport._doorbell_mm = bytearray(DOORBELL_BYTES)
    transport._receive_stopped = Event()
    transport._deferred_cuda_trace = False
    stopped = Event()

    def receive() -> None:
        with pytest.raises(PAPOffloadExecTransportClosed):
            transport._recv_from_peer(direction=DIR_QKV)
        stopped.set()

    thread = Thread(target=receive)
    thread.start()
    transport.stop_receiving()
    thread.join(timeout=1.0)

    assert stopped.is_set()
    assert transport._doorbell_mm is not None


def test_nixl_receive_wait_is_woken_by_stop() -> None:
    endpoint = object.__new__(PAPNixlMailboxEndpoint)
    endpoint._lock = RLock()
    endpoint._cv = Condition(endpoint._lock)
    endpoint._incoming = OrderedDict()
    endpoint._receive_stopped = Event()
    endpoint._closed = Event()
    endpoint._inline_poll_enabled = False
    endpoint._trace_enabled = False
    stopped = Event()

    def receive() -> None:
        with pytest.raises(PAPOffloadExecTransportClosed):
            endpoint.recv()
        stopped.set()

    thread = Thread(target=receive)
    thread.start()
    endpoint.stop_receiving()
    thread.join(timeout=1.0)

    assert stopped.is_set()


@pytest.mark.parametrize(
    "loop",
    [run_offload_exec_mailbox_loop, run_offload_exec_mailbox_receiver_loop],
)
def test_attention_receive_loops_treat_transport_stop_as_normal(loop) -> None:
    class StoppedTransport:
        def recv_next_qkv_batch_message(self):
            raise PAPOffloadExecTransportClosed

    kwargs = {
        "registry": SimpleNamespace(),
        "transport": StoppedTransport(),
        "peer_id": "projection-a",
    }
    if loop is run_offload_exec_mailbox_receiver_loop:
        kwargs["dispatcher"] = SimpleNamespace()

    loop(**kwargs)


def test_peer_manager_stops_receivers_before_runtime_and_transport() -> None:
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
        dispatch_mode = "direct"
        dispatcher = None

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
    assert manager.initial_transport is None
    with pytest.raises(PAPAttentionPeerConflict, match="stopping"):
        manager.bind(peer_metadata=b"peer", source_id=None)
