# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Event, Thread

import pytest
import torch

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


def test_dispatcher_combines_all_ready_compatible_items() -> None:
    handled: list[tuple[str, ...]] = []
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda items: handled.append(
            tuple(item.peer_id for item in items)
        ),
        compatibility_key=lambda item: item.descriptor,
    )
    first = _make_item("p0", released)
    first.descriptor = "layer0"
    incompatible = _make_item("p1", released)
    incompatible.descriptor = "layer1"
    compatible = _make_item("p2", released)
    compatible.descriptor = "layer0"
    dispatcher.enqueue(first)
    dispatcher.enqueue(incompatible)
    dispatcher.enqueue(compatible)

    assert dispatcher.dispatch_next(timeout=0.1)

    assert handled == [("p0", "p2")]
    assert released == ["p0", "p2"]
    assert first.wait_completed(timeout=0)
    assert compatible.wait_completed(timeout=0)
    assert not incompatible.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_dispatched"] == 2
    assert stats["dispatcher_dispatch_groups"] == 1
    assert stats["dispatcher_combined_groups"] == 1
    assert stats["dispatcher_max_items_per_group"] == 2
    assert stats["dispatcher_queue_depth"] == 1
    assert stats["dispatcher_ready_candidates"] == 2
    assert stats["dispatcher_compatible_candidates"] == 1
    assert stats["dispatcher_incompatible_candidates"] == 1

    assert dispatcher.dispatch_next(timeout=0.1)

    assert handled == [("p0", "p2"), ("p1",)]
    assert released == ["p0", "p2", "p1"]
    assert incompatible.wait_completed(timeout=0)


def test_dispatcher_batch_failure_releases_and_completes_whole_group() -> None:
    released: list[str] = []

    def fail(_items: tuple[PAPAttentionWorkItem, ...]) -> None:
        raise RuntimeError("combined boom")

    dispatcher = PAPAttentionDispatcher(
        batch_handler=fail,
        compatibility_key=lambda item: item.descriptor,
    )
    first = _make_item("p0", released)
    first.descriptor = "layer0"
    second = _make_item("p1", released)
    second.descriptor = "layer0"
    dispatcher.enqueue(first)
    dispatcher.enqueue(second)

    with pytest.raises(RuntimeError, match="combined boom"):
        dispatcher.dispatch_next(timeout=0.1)

    assert released == ["p0", "p1"]
    assert first.wait_completed(timeout=0)
    assert second.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_dispatched"] == 0
    assert stats["dispatcher_failures"] == 1
    assert stats["dispatcher_queue_depth"] == 0


def test_bounded_dispatcher_counts_deferred_items_toward_capacity() -> None:
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda _items: None,
        compatibility_key=lambda item: item.descriptor,
        max_queue_size=3,
    )
    for peer_id, layer_name in (
        ("p0", "layer0"),
        ("p1", "layer1"),
        ("p2", "layer0"),
    ):
        item = _make_item(peer_id, released)
        item.descriptor = layer_name
        dispatcher.enqueue(item)
    assert dispatcher.dispatch_next(timeout=0.1)
    for peer_id in ("p3", "p4"):
        item = _make_item(peer_id, released)
        item.descriptor = "layer2"
        dispatcher.enqueue(item)

    overflow = _make_item("p5", released)
    overflow.descriptor = "layer2"
    with pytest.raises(RuntimeError, match="queue is full"):
        dispatcher.enqueue(overflow)

    assert dispatcher.stats()["dispatcher_queue_depth"] == 3


def test_compatibility_key_failure_releases_first_item() -> None:
    released: list[str] = []

    def fail_key(_item: PAPAttentionWorkItem) -> str:
        raise RuntimeError("bad compatibility key")

    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda _items: None,
        compatibility_key=fail_key,
    )
    item = _make_item("p0", released)
    dispatcher.enqueue(item)

    with pytest.raises(RuntimeError, match="bad compatibility key"):
        dispatcher.dispatch_next(timeout=0.1)

    assert released == ["p0"]
    assert item.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_queue_depth"] == 0
    assert stats["dispatcher_failures"] == 1


def test_dispatcher_coalesces_compatible_item_arriving_within_window() -> None:
    handled: list[tuple[str, ...]] = []
    released: list[str] = []
    enqueue_second = Event()

    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda items: handled.append(
            tuple(item.peer_id for item in items)
        ),
        compatibility_key=lambda item: str(item.descriptor),
        coalesce_timeout_s=0.1,
        expected_group_size=2,
    )
    first = _make_item("p0", released)
    first.descriptor = "layer0"
    first.arrival_ns = 1_000_000
    first.qkv_batch = torch.tensor([0, 1])
    second = _make_item("p1", released)
    second.descriptor = "layer0"
    second.arrival_ns = 1_075_000
    second.qkv_batch = torch.tensor([2, 3])
    dispatcher.enqueue(first)
    original_get = dispatcher._queue.get
    blocking_get_calls = 0

    def get_and_signal(block=True, timeout=None):
        nonlocal blocking_get_calls
        if block:
            blocking_get_calls += 1
            if blocking_get_calls == 2:
                enqueue_second.set()
        return original_get(block=block, timeout=timeout)

    dispatcher._queue.get = get_and_signal

    def enqueue_later() -> None:
        assert enqueue_second.wait(timeout=1.0)
        dispatcher.enqueue(second)

    producer = Thread(target=enqueue_later)
    producer.start()
    assert dispatcher.dispatch_next(timeout=0.1)
    producer.join(timeout=1.0)

    assert not producer.is_alive()
    assert handled == [("p0", "p1")]
    assert released == ["p0", "p1"]
    stats = dispatcher.stats()
    assert stats["dispatcher_coalesce_waits"] == 1
    assert stats["dispatcher_coalesce_timeouts"] == 0
    assert stats["dispatcher_waited_compatible_candidates"] == 1
    assert stats["dispatcher_coalesce_wait_outcomes"] == {
        "compatible": 1,
        "incompatible": 0,
        "timeout": 0,
        "stopped": 0,
    }
    assert sum(stats["dispatcher_coalesce_wait_us_histogram"].values()) == 1
    assert stats["dispatcher_compatible_arrival_skew_samples"] == 1
    assert stats["dispatcher_compatible_arrival_skew_ns_sum"] == 75_000
    assert stats["dispatcher_compatible_arrival_skew_ns_max"] == 75_000
    assert stats["dispatcher_compatible_arrival_skew_us_histogram"] == {
        "le_50": 0,
        "le_100": 1,
        "le_200": 0,
        "le_500": 0,
        "le_1000": 0,
        "gt_1000": 0,
    }
    assert stats["dispatcher_expected_group_size"] == 2


def test_dispatcher_records_coalesce_timeout_outcome() -> None:
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda _items: None,
        compatibility_key=lambda item: item.descriptor,
        coalesce_timeout_s=0.001,
        expected_group_size=2,
    )
    item = _make_item("p0", released)
    item.descriptor = "layer0"
    dispatcher.enqueue(item)

    assert dispatcher.dispatch_next(timeout=0.1)

    stats = dispatcher.stats()
    assert stats["dispatcher_coalesce_waits"] == 1
    assert stats["dispatcher_coalesce_timeouts"] == 1
    assert stats["dispatcher_coalesce_wait_outcomes"] == {
        "compatible": 0,
        "incompatible": 0,
        "timeout": 1,
        "stopped": 0,
    }
    assert sum(stats["dispatcher_coalesce_wait_us_histogram"].values()) == 1
    assert stats["dispatcher_compatible_arrival_skew_samples"] == 0


def test_dispatcher_prefers_designated_peer_when_layers_mismatch() -> None:
    handled: list[tuple[str, ...]] = []
    released: list[str] = []
    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda items: handled.append(
            tuple(item.peer_id for item in items)
        ),
        compatibility_key=lambda item: item.descriptor,
    )
    dispatcher.set_preferred_peer_id("projection-0")
    lagging = _make_item("projection-1", released)
    lagging.descriptor = "layer5"
    leader = _make_item("projection-0", released)
    leader.descriptor = "layer2"
    dispatcher.enqueue(lagging)
    dispatcher.enqueue(leader)

    assert dispatcher.dispatch_next(timeout=0.1)

    assert handled == [("projection-0",)]
    assert released == ["projection-0"]
    assert leader.wait_completed(timeout=0)
    assert not lagging.wait_completed(timeout=0)
    assert dispatcher.stats()["dispatcher_queue_depth"] == 1


def test_dispatcher_waits_for_preferred_peer_before_selecting_group() -> None:
    handled: list[tuple[str, ...]] = []
    released: list[str] = []
    request_leader = Event()
    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda items: handled.append(
            tuple(item.peer_id for item in items)
        ),
        compatibility_key=lambda item: str(item.descriptor),
        coalesce_timeout_s=0.1,
        expected_group_size=2,
    )
    dispatcher.set_preferred_peer_id("projection-0")
    lagging = _make_item("projection-1", released)
    lagging.descriptor = "layer5"
    leader = _make_item("projection-0", released)
    leader.descriptor = "layer2"
    dispatcher.enqueue(lagging)
    original_get = dispatcher._queue.get
    blocking_get_calls = 0

    def get_and_signal(block=True, timeout=None):
        nonlocal blocking_get_calls
        if block:
            blocking_get_calls += 1
            if blocking_get_calls == 2:
                request_leader.set()
        return original_get(block=block, timeout=timeout)

    dispatcher._queue.get = get_and_signal

    def enqueue_later() -> None:
        assert request_leader.wait(timeout=1.0)
        dispatcher.enqueue(leader)

    producer = Thread(target=enqueue_later)
    producer.start()
    assert dispatcher.dispatch_next(timeout=0.1)
    producer.join(timeout=1.0)

    assert not producer.is_alive()
    assert handled == [("projection-0",)]
    assert released == ["projection-0"]
    assert not lagging.wait_completed(timeout=0)
    stats = dispatcher.stats()
    assert stats["dispatcher_waited_compatible_candidates"] == 0
    assert stats["dispatcher_coalesce_wait_outcomes"] == {
        "compatible": 0,
        "incompatible": 1,
        "timeout": 0,
        "stopped": 0,
    }


def test_invalid_preferred_key_falls_back_without_losing_ready_item() -> None:
    handled: list[str] = []
    released: list[str] = []

    def compatibility_key(item: PAPAttentionWorkItem) -> str:
        if item.peer_id == "projection-0":
            raise RuntimeError("invalid leader")
        return str(item.descriptor)

    dispatcher = PAPAttentionDispatcher(
        batch_handler=lambda items: handled.extend(item.peer_id for item in items),
        compatibility_key=compatibility_key,
    )
    dispatcher.set_preferred_peer_id("projection-0")
    healthy = _make_item("projection-1", released)
    healthy.descriptor = "layer5"
    invalid = _make_item("projection-0", released)
    invalid.descriptor = "layer2"
    dispatcher.enqueue(healthy)
    dispatcher.enqueue(invalid)

    assert dispatcher.dispatch_next(timeout=0.1)

    assert handled == ["projection-1"]
    assert released == ["projection-1"]
    assert dispatcher.stats()["dispatcher_queue_depth"] == 1

    with pytest.raises(RuntimeError, match="invalid leader"):
        dispatcher.dispatch_next(timeout=0.1)

    assert released == ["projection-1", "projection-0"]
    assert invalid.wait_completed(timeout=0)
