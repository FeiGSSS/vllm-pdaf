# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Event

import pytest

from vllm.pap.attention_scheduler import (
    PAPAttentionDispatcher,
    PAPAttentionWorkItem,
)


class _FakeMessage:
    def __init__(self, peer_id: str, released: list[str]) -> None:
        self.peer_id = peer_id
        self.released = released

    def release(self) -> None:
        self.released.append(self.peer_id)


def _make_item(peer_id: str, released: list[str]) -> PAPAttentionWorkItem:
    return PAPAttentionWorkItem(
        descriptor=object(),
        qkv_batch=object(),
        transport=object(),
        peer_id=peer_id,
        arrival_ns=1,
        input_message=_FakeMessage(peer_id, released),
    )


def test_dispatcher_preserves_fifo_and_releases_each_input_once() -> None:
    handled: list[str] = []
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(
        handler=lambda item: handled.append(item.peer_id)
    )
    first = _make_item("p0", released)
    second = _make_item("p1", released)
    dispatcher.enqueue(first)
    dispatcher.enqueue(second)

    assert dispatcher.dispatch_next(timeout=0.1)
    assert dispatcher.dispatch_next(timeout=0.1)
    first.release_input()

    assert handled == ["p0", "p1"]
    assert released == ["p0", "p1"]
    assert first.input_released
    assert second.input_released
    assert first.wait_completed(timeout=0)
    assert second.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_enqueued"] == 2
    assert stats["dispatcher_dispatched"] == 2
    assert stats["dispatcher_failures"] == 0
    assert stats["dispatcher_max_queue_depth"] == 2


def test_dispatcher_releases_input_and_records_handler_error() -> None:
    released: list[str] = []
    item = _make_item("p0", released)

    def fail(_item: PAPAttentionWorkItem) -> None:
        raise RuntimeError("boom")

    dispatcher = PAPAttentionDispatcher(handler=fail)
    dispatcher.enqueue(item)

    with pytest.raises(RuntimeError, match="boom"):
        dispatcher.dispatch_next(timeout=0.1)

    assert released == ["p0"]
    assert item.input_released
    assert item.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_dispatched"] == 0
    assert stats["dispatcher_failures"] == 1
    assert stats["dispatcher_fatal_error"] == "RuntimeError: boom"


def test_dispatcher_records_input_release_error() -> None:
    class FailingMessage:
        def release(self) -> None:
            raise RuntimeError("release failed")

    item = PAPAttentionWorkItem(
        descriptor=object(),
        qkv_batch=object(),
        transport=object(),
        peer_id="p0",
        arrival_ns=1,
        input_message=FailingMessage(),
    )
    dispatcher = PAPAttentionDispatcher(handler=lambda _item: None)
    dispatcher.enqueue(item)

    with pytest.raises(RuntimeError, match="release failed"):
        dispatcher.dispatch_next(timeout=0.1)

    assert item.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_dispatched"] == 0
    assert stats["dispatcher_failures"] == 1
    assert stats["dispatcher_fatal_error"] == "RuntimeError: release failed"


def test_dispatcher_worker_starts_and_stops_after_drain() -> None:
    handled = Event()
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(handler=lambda _item: handled.set())
    dispatcher.start()
    dispatcher.enqueue(_make_item("p0", released))

    assert handled.wait(timeout=1.0)
    dispatcher.stop(drain=True, timeout=1.0)

    assert released == ["p0"]
    stats = dispatcher.stats()
    assert stats["dispatcher_dispatched"] == 1
    assert stats["dispatcher_running"] is False
    assert stats["dispatcher_accepting"] is False


def test_enqueue_publishes_depth_before_waking_dispatcher() -> None:
    handler_started = Event()
    allow_handler_to_finish = Event()
    released: list[str] = []

    def handle(_item: PAPAttentionWorkItem) -> None:
        handler_started.set()
        assert allow_handler_to_finish.wait(timeout=1.0)

    dispatcher = PAPAttentionDispatcher(handler=handle)
    original_put = dispatcher._queue.put_nowait
    consumer_started_early = []

    def put_and_probe_consumer(item) -> None:
        original_put(item)
        consumer_started_early.append(handler_started.wait(timeout=0.05))

    dispatcher._queue.put_nowait = put_and_probe_consumer
    dispatcher.start()
    item = _make_item("p0", released)
    dispatcher.enqueue(item)
    allow_handler_to_finish.set()
    assert item.wait_completed(timeout=1.0)
    dispatcher.stop(drain=True, timeout=1.0)

    stats = dispatcher.stats()
    assert consumer_started_early == [False]
    assert stats["dispatcher_queue_depth"] == 0
    assert stats["dispatcher_max_queue_depth"] == 1


def test_dispatcher_stop_without_drain_releases_queued_inputs() -> None:
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(handler=lambda _item: None)
    first = _make_item("p0", released)
    second = _make_item("p1", released)
    dispatcher.enqueue(first)
    dispatcher.enqueue(second)

    dispatcher.stop(drain=False, timeout=1.0)

    assert released == ["p0", "p1"]
    assert first.wait_completed(timeout=0)
    assert second.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_dropped"] == 2
    assert stats["dispatcher_queue_depth"] == 0
    with pytest.raises(RuntimeError, match="not accepting"):
        dispatcher.enqueue(_make_item("p2", released))
    assert released == ["p0", "p1"]


def test_dispatcher_drop_continues_after_release_error() -> None:
    class FailingMessage:
        def release(self) -> None:
            raise RuntimeError("release failed")

    released: list[str] = []
    failing = PAPAttentionWorkItem(
        descriptor=object(),
        qkv_batch=object(),
        transport=object(),
        peer_id="p0",
        arrival_ns=1,
        input_message=FailingMessage(),
    )
    healthy = _make_item("p1", released)
    dispatcher = PAPAttentionDispatcher(handler=lambda _item: None)
    dispatcher.enqueue(failing)
    dispatcher.enqueue(healthy)

    dispatcher.stop(drain=False, timeout=1.0)

    assert failing.wait_completed(timeout=0)
    assert healthy.wait_completed(timeout=0)
    assert released == ["p1"]
    stats = dispatcher.stats()
    assert stats["dispatcher_dropped"] == 2
    assert stats["dispatcher_failures"] == 1
    assert stats["dispatcher_fatal_error"] == "RuntimeError: release failed"
