from __future__ import annotations

import threading
from typing import Any

from vllm.pap.deferred_cuda_trace import DeferredCudaTraceCollector


class _FakeEvent:
    def __init__(self) -> None:
        self.ready = False
        self.duration_ms = 0.0
        self.record_calls = 0
        self.query_calls = 0
        self.synchronize_calls = 0
        self.elapsed_time_calls = 0

    def record(self, stream: Any) -> None:
        del stream
        self.record_calls += 1

    def query(self) -> bool:
        self.query_calls += 1
        return self.ready

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.ready = True

    def elapsed_time(self, end_event: _FakeEvent) -> float:
        self.elapsed_time_calls += 1
        assert end_event.ready
        return end_event.duration_ms


class _FakeEventFactory:
    def __init__(self) -> None:
        self.events: list[_FakeEvent] = []

    def __call__(self) -> _FakeEvent:
        event = _FakeEvent()
        self.events.append(event)
        return event


def test_deferred_cuda_trace_nonblocking_collection_reuses_events() -> None:
    factory = _FakeEventFactory()
    collector = DeferredCudaTraceCollector(
        max_pending=1,
        event_factory=factory,
    )

    span = collector.begin("paged_fa_gpu_ms", object())
    assert span is not None
    span.end_event.duration_ms = 0.25
    collector.end(span)

    pending = collector.raw_snapshot(blocking=False)
    assert pending["pending_records"] == 1
    assert pending["durations"] == {}
    assert span.end_event.synchronize_calls == 0
    assert span.start_event.elapsed_time_calls == 0

    span.end_event.ready = True
    ready = collector.raw_snapshot(blocking=False)
    assert ready["pending_records"] == 0
    assert ready["durations"] == {"paged_fa_gpu_ms": [0.25]}
    assert span.end_event.synchronize_calls == 0
    assert span.start_event.elapsed_time_calls == 1

    reused = collector.begin("kv_append_gpu_ms", object())
    assert reused is not None
    assert len(factory.events) == 2


def test_deferred_cuda_trace_blocks_only_when_explicitly_flushed() -> None:
    factory = _FakeEventFactory()
    collector = DeferredCudaTraceCollector(
        max_pending=1,
        event_factory=factory,
    )

    span = collector.begin("qkv_ready_wait_gpu_ms", object())
    assert span is not None
    span.end_event.duration_ms = 0.125
    collector.end(span)

    snapshot = collector.raw_snapshot(blocking=True)

    assert span.end_event.synchronize_calls == 1
    assert snapshot["pending_records"] == 0
    assert snapshot["durations"] == {"qkv_ready_wait_gpu_ms": [0.125]}


def test_deferred_cuda_trace_drops_when_event_pool_is_saturated() -> None:
    factory = _FakeEventFactory()
    collector = DeferredCudaTraceCollector(
        max_pending=1,
        event_factory=factory,
    )

    first = collector.begin("kv_append_gpu_ms", object())
    assert first is not None
    collector.end(first)

    second = collector.begin("paged_fa_gpu_ms", object())

    assert second is None
    snapshot = collector.raw_snapshot(blocking=False)
    assert snapshot["pending_records"] == 1
    assert snapshot["dropped_records"] == 1
    assert first.end_event.synchronize_calls == 0


def test_deferred_cuda_trace_end_is_idempotent() -> None:
    factory = _FakeEventFactory()
    collector = DeferredCudaTraceCollector(
        max_pending=1,
        event_factory=factory,
    )

    span = collector.begin("output_p2p_copy_gpu_ms", object())
    assert span is not None
    collector.end(span)
    collector.end(span)

    assert span.end_event.record_calls == 1
    assert collector.raw_snapshot(blocking=False)["pending_records"] == 1


def test_deferred_cuda_trace_snapshot_is_thread_safe() -> None:
    factory = _FakeEventFactory()
    collector = DeferredCudaTraceCollector(
        max_pending=16,
        event_factory=factory,
    )
    producer_done = threading.Event()

    def produce() -> None:
        for _ in range(200):
            span = collector.begin("paged_fa_gpu_ms", object())
            assert span is not None
            span.end_event.duration_ms = 0.5
            span.end_event.ready = True
            collector.end(span)
        producer_done.set()

    def snapshot() -> None:
        while not producer_done.is_set():
            collector.raw_snapshot(blocking=False)

    producer = threading.Thread(target=produce)
    reader = threading.Thread(target=snapshot)
    producer.start()
    reader.start()
    producer.join()
    reader.join()

    result = collector.raw_snapshot(blocking=True)
    assert result["pending_records"] == 0
    assert result["dropped_records"] == 0
    assert result["error_records"] == 0
    assert len(result["durations"]["paged_fa_gpu_ms"]) == 200
