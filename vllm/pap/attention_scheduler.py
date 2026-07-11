# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling primitives for PAP Attention work."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)

_STOP = object()
_LATENCY_HISTOGRAM_BUCKETS_US = (50, 100, 200, 500, 1000)


def _latency_histogram() -> dict[str, int]:
    return {
        **{f"le_{upper_us}": 0 for upper_us in _LATENCY_HISTOGRAM_BUCKETS_US},
        "gt_1000": 0,
    }


def _latency_histogram_bucket(duration_ns: int) -> str:
    duration_us = max(0, int(duration_ns)) / 1000.0
    for upper_us in _LATENCY_HISTOGRAM_BUCKETS_US:
        if duration_us <= upper_us:
            return f"le_{upper_us}"
    return "gt_1000"


@dataclass
class PAPAttentionWorkItem:
    """One peer batch whose input lifetime transfers to a dispatcher."""

    descriptor: Any
    qkv_batch: Any
    transport: Any
    peer_id: str
    arrival_ns: int
    input_message: Any | None = None
    ready_event: Any | None = None
    trace_context: dict[str, Any] = field(default_factory=dict)
    dispatch_started_ns: int = 0
    queue_wait_ns: int = 0
    _release_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _input_released: bool = field(default=False, init=False, repr=False)
    _completed: Event = field(default_factory=Event, init=False, repr=False)

    @property
    def input_released(self) -> bool:
        with self._release_lock:
            return self._input_released

    def release_input(self) -> None:
        with self._release_lock:
            if self._input_released:
                return
            self._input_released = True
        if self.input_message is not None:
            self.input_message.release()

    def mark_completed(self) -> None:
        self._completed.set()

    def wait_completed(self, timeout: float | None = None) -> bool:
        return self._completed.wait(timeout=timeout)


class PAPAttentionDispatcher:
    """Own a FIFO queue and at most one Attention compute thread."""

    def __init__(
        self,
        *,
        handler: Callable[[PAPAttentionWorkItem], None] | None = None,
        batch_handler: (
            Callable[[tuple[PAPAttentionWorkItem, ...]], None] | None
        ) = None,
        compatibility_key: Callable[[PAPAttentionWorkItem], Any] | None = None,
        max_queue_size: int = 0,
        coalesce_timeout_s: float = 0.0,
        expected_group_size: int = 1,
    ) -> None:
        if (handler is None) == (batch_handler is None):
            raise ValueError("PAP Attention dispatcher requires exactly one handler")
        if compatibility_key is not None and batch_handler is None:
            raise ValueError("PAP Attention compatibility key requires a batch handler")
        if float(coalesce_timeout_s) < 0:
            raise ValueError("PAP Attention coalesce timeout must be non-negative")
        if int(expected_group_size) <= 0:
            raise ValueError("PAP Attention expected group size must be positive")
        self._handler = handler
        self._batch_handler = batch_handler
        self._compatibility_key = compatibility_key
        self._coalesce_timeout_s = float(coalesce_timeout_s)
        self._expected_group_size = int(expected_group_size)
        self._preferred_peer_id: str | None = None
        self._max_queue_size = max(0, int(max_queue_size))
        self._queue: Queue[PAPAttentionWorkItem | object] = Queue(
            maxsize=self._max_queue_size
        )
        self._deferred: deque[PAPAttentionWorkItem | object] = deque()
        self._deferred_lock = Lock()
        self._state_lock = Lock()
        self._stats_lock = Lock()
        self._thread: Thread | None = None
        self._accepting = True
        self._running = False
        self._fatal_error: BaseException | None = None
        self._pending_items = 0
        self._enqueued = 0
        self._dispatched = 0
        self._failures = 0
        self._dropped = 0
        self._max_queue_depth = 0
        self._queue_wait_ns_sum = 0
        self._queue_wait_ns_max = 0
        self._dispatch_groups = 0
        self._combined_groups = 0
        self._max_items_per_group = 0
        self._ready_candidates = 0
        self._compatible_candidates = 0
        self._incompatible_candidates = 0
        self._coalesce_waits = 0
        self._coalesce_timeouts = 0
        self._waited_compatible_candidates = 0
        self._coalesce_wait_ns_sum = 0
        self._coalesce_wait_ns_max = 0
        self._coalesce_wait_outcomes = {
            "compatible": 0,
            "incompatible": 0,
            "timeout": 0,
            "stopped": 0,
        }
        self._coalesce_wait_us_histogram = _latency_histogram()
        self._compatible_arrival_skew_samples = 0
        self._compatible_arrival_skew_ns_sum = 0
        self._compatible_arrival_skew_ns_max = 0
        self._compatible_arrival_skew_us_histogram = _latency_histogram()
        self._preferred_peer_selections = 0

    def start(self) -> None:
        with self._state_lock:
            if self._running:
                return
            if not self._accepting:
                raise RuntimeError("PAP Attention dispatcher is not accepting work")
            self._running = True
            self._thread = Thread(
                target=self._run,
                name="pap-attention-dispatcher",
                daemon=True,
            )
            self._thread.start()

    def enqueue(self, item: PAPAttentionWorkItem) -> None:
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("PAP Attention dispatcher is not accepting work")
            with self._stats_lock:
                if (
                    self._max_queue_size > 0
                    and self._pending_items >= self._max_queue_size
                ):
                    raise RuntimeError("PAP Attention dispatcher queue is full")
                previous_max_queue_depth = self._max_queue_depth
                self._pending_items += 1
                self._enqueued += 1
                self._max_queue_depth = max(
                    self._max_queue_depth,
                    self._pending_items,
                )
                try:
                    self._queue.put_nowait(item)
                except Full as exc:
                    self._pending_items -= 1
                    self._enqueued -= 1
                    self._max_queue_depth = previous_max_queue_depth
                    raise RuntimeError(
                        "PAP Attention dispatcher queue is full"
                    ) from exc

    def set_expected_group_size(self, expected_group_size: int) -> None:
        if int(expected_group_size) <= 0:
            raise ValueError("PAP Attention expected group size must be positive")
        with self._state_lock:
            self._expected_group_size = int(expected_group_size)

    def set_preferred_peer_id(self, peer_id: str | None) -> None:
        with self._state_lock:
            self._preferred_peer_id = None if peer_id is None else str(peer_id)

    def dispatch_next(self, timeout: float | None = None) -> bool:
        entry = self._next_entry(timeout=timeout)
        if entry is _STOP:
            self._queue.task_done()
            return False
        assert isinstance(entry, PAPAttentionWorkItem)
        try:
            items = self._collect_compatible(entry)
        except BaseException as exc:
            with self._stats_lock:
                self._pending_items -= 1
            self._release_inputs((entry,))
            entry.mark_completed()
            self._queue.task_done()
            self._record_failure(exc)
            raise
        dispatch_started_ns = time.perf_counter_ns()
        for item in items:
            item.dispatch_started_ns = dispatch_started_ns
            item.queue_wait_ns = max(
                0,
                dispatch_started_ns - int(item.arrival_ns),
            )
        with self._stats_lock:
            self._pending_items -= len(items)
            self._queue_wait_ns_sum += sum(item.queue_wait_ns for item in items)
            self._queue_wait_ns_max = max(
                self._queue_wait_ns_max,
                *(item.queue_wait_ns for item in items),
            )
        try:
            try:
                if self._batch_handler is not None:
                    self._batch_handler(items)
                else:
                    assert self._handler is not None
                    assert len(items) == 1
                    self._handler(items[0])
            except BaseException as exc:
                self._release_inputs(items)
                self._record_failure(exc)
                raise
            release_error = self._release_inputs(items)
            if release_error is not None:
                self._record_failure(release_error)
                raise release_error
            with self._stats_lock:
                self._dispatched += len(items)
                self._dispatch_groups += 1
                if len(items) > 1:
                    self._combined_groups += 1
                self._max_items_per_group = max(
                    self._max_items_per_group,
                    len(items),
                )
            return True
        finally:
            for item in items:
                item.mark_completed()
                self._queue.task_done()

    def stop(self, *, drain: bool, timeout: float | None = None) -> None:
        with self._state_lock:
            self._accepting = False
            thread = self._thread
        if thread is None:
            if drain:
                while True:
                    try:
                        self.dispatch_next(timeout=0)
                    except Empty:
                        break
            else:
                self._discard_pending()
            return

        if not drain:
            self._discard_pending()
        try:
            self._queue.put(_STOP, timeout=timeout)
        except Full as exc:
            raise TimeoutError("PAP Attention dispatcher stop timed out") from exc
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("PAP Attention dispatcher did not stop")

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            running = self._running
            accepting = self._accepting
            fatal_error = self._fatal_error
            expected_group_size = self._expected_group_size
            preferred_peer_id = self._preferred_peer_id
        with self._stats_lock:
            return {
                "dispatcher_enqueued": self._enqueued,
                "dispatcher_dispatched": self._dispatched,
                "dispatcher_failures": self._failures,
                "dispatcher_dropped": self._dropped,
                "dispatcher_queue_depth": self._pending_items,
                "dispatcher_max_queue_depth": self._max_queue_depth,
                "dispatcher_queue_wait_ns_sum": self._queue_wait_ns_sum,
                "dispatcher_queue_wait_ns_max": self._queue_wait_ns_max,
                "dispatcher_dispatch_groups": self._dispatch_groups,
                "dispatcher_combined_groups": self._combined_groups,
                "dispatcher_max_items_per_group": self._max_items_per_group,
                "dispatcher_ready_candidates": self._ready_candidates,
                "dispatcher_compatible_candidates": (self._compatible_candidates),
                "dispatcher_incompatible_candidates": (self._incompatible_candidates),
                "dispatcher_coalesce_waits": self._coalesce_waits,
                "dispatcher_coalesce_timeouts": self._coalesce_timeouts,
                "dispatcher_waited_compatible_candidates": (
                    self._waited_compatible_candidates
                ),
                "dispatcher_coalesce_wait_ns_sum": (self._coalesce_wait_ns_sum),
                "dispatcher_coalesce_wait_ns_max": (self._coalesce_wait_ns_max),
                "dispatcher_coalesce_wait_outcomes": dict(self._coalesce_wait_outcomes),
                "dispatcher_coalesce_wait_us_histogram": dict(
                    self._coalesce_wait_us_histogram
                ),
                "dispatcher_compatible_arrival_skew_samples": (
                    self._compatible_arrival_skew_samples
                ),
                "dispatcher_compatible_arrival_skew_ns_sum": (
                    self._compatible_arrival_skew_ns_sum
                ),
                "dispatcher_compatible_arrival_skew_ns_max": (
                    self._compatible_arrival_skew_ns_max
                ),
                "dispatcher_compatible_arrival_skew_us_histogram": dict(
                    self._compatible_arrival_skew_us_histogram
                ),
                "dispatcher_coalesce_timeout_us": int(
                    self._coalesce_timeout_s * 1_000_000
                ),
                "dispatcher_expected_group_size": expected_group_size,
                "dispatcher_preferred_peer_id": preferred_peer_id,
                "dispatcher_preferred_peer_selections": (
                    self._preferred_peer_selections
                ),
                "dispatcher_running": running,
                "dispatcher_accepting": accepting,
                "dispatcher_fatal_error": (
                    None
                    if fatal_error is None
                    else f"{type(fatal_error).__name__}: {fatal_error}"
                ),
            }

    def _run(self) -> None:
        try:
            while self.dispatch_next():
                pass
        except BaseException:
            logger.exception("PAP Attention dispatcher stopped after a fatal error")
            self._discard_pending()
        finally:
            with self._state_lock:
                self._running = False
                self._accepting = False

    def _record_failure(self, exc: BaseException) -> None:
        with self._state_lock:
            self._accepting = False
            self._fatal_error = exc
        with self._stats_lock:
            self._failures += 1

    def _next_entry(
        self,
        *,
        timeout: float | None,
    ) -> PAPAttentionWorkItem | object:
        with self._deferred_lock:
            if self._deferred:
                return self._deferred.popleft()
        return self._queue.get(timeout=timeout)

    def _collect_compatible(
        self,
        first: PAPAttentionWorkItem,
    ) -> tuple[PAPAttentionWorkItem, ...]:
        if self._compatibility_key is None:
            return (first,)
        ready = [first]
        stop_entries: deque[object] = deque()
        waited_items: list[PAPAttentionWorkItem] = []
        coalesce_waits = 0
        coalesce_timeouts = 0
        coalesce_wait_ns = 0
        with self._state_lock:
            expected_group_size = self._expected_group_size
            preferred_peer_id = self._preferred_peer_id
        with self._deferred_lock:
            candidates = self._deferred
            self._deferred = deque()
            while True:
                try:
                    candidates.append(self._queue.get_nowait())
                except Empty:
                    break
            while candidates:
                candidate = candidates.popleft()
                if candidate is _STOP:
                    stop_entries.append(candidate)
                    continue
                assert isinstance(candidate, PAPAttentionWorkItem)
                ready.append(candidate)
            if (
                self._coalesce_timeout_s > 0
                and len(ready) < expected_group_size
                and not stop_entries
            ):
                coalesce_waits = 1
                wait_started_ns = time.perf_counter_ns()
                deadline = time.perf_counter() + self._coalesce_timeout_s
                while len(ready) < expected_group_size:
                    remaining_s = deadline - time.perf_counter()
                    if remaining_s <= 0:
                        coalesce_timeouts = 1
                        break
                    try:
                        candidate = self._queue.get(timeout=remaining_s)
                    except Empty:
                        coalesce_timeouts = 1
                        break
                    if candidate is _STOP:
                        stop_entries.append(candidate)
                        break
                    assert isinstance(candidate, PAPAttentionWorkItem)
                    ready.append(candidate)
                    waited_items.append(candidate)
                coalesce_wait_ns = max(
                    0,
                    time.perf_counter_ns() - wait_started_ns,
                )
            selected = first
            if preferred_peer_id is not None:
                selected = next(
                    (item for item in ready if item.peer_id == preferred_peer_id),
                    first,
                )
            try:
                compatibility_key = self._compatibility_key(selected)
            except BaseException:
                if selected is not first:
                    selected = first
                    try:
                        compatibility_key = self._compatibility_key(selected)
                    except BaseException:
                        self._deferred = deque(
                            item for item in ready if item is not first
                        )
                        self._deferred.extend(stop_entries)
                        raise
                else:
                    self._deferred = deque(item for item in ready if item is not first)
                    self._deferred.extend(stop_entries)
                    raise
            compatible = [selected]
            deferred: deque[PAPAttentionWorkItem | object] = deque()
            for candidate in ready:
                if candidate is selected:
                    continue
                try:
                    is_compatible = (
                        self._compatibility_key(candidate) == compatibility_key
                    )
                except BaseException:
                    is_compatible = False
                if is_compatible:
                    compatible.append(candidate)
                else:
                    deferred.append(candidate)
            deferred.extend(stop_entries)
            self._deferred = deferred
        ready_candidates = len(ready) - 1
        compatible_candidates = len(compatible) - 1
        incompatible_candidates = len(ready) - len(compatible)
        compatible_ids = {id(item) for item in compatible}
        compatible_candidate_ids = {
            id(item) for item in compatible if item is not selected
        }
        waited_compatible_candidates = sum(
            id(item) in compatible_candidate_ids for item in waited_items
        )
        compatible_arrival_skews_ns = [
            abs(int(item.arrival_ns) - int(selected.arrival_ns))
            for item in compatible
            if item is not selected
        ]
        coalesce_wait_outcome: str | None = None
        if coalesce_waits:
            if coalesce_timeouts:
                coalesce_wait_outcome = "timeout"
            elif (
                any(id(item) in compatible_ids for item in waited_items)
                and len(compatible) > 1
            ):
                coalesce_wait_outcome = "compatible"
            elif stop_entries and not waited_items:
                coalesce_wait_outcome = "stopped"
            else:
                coalesce_wait_outcome = "incompatible"
        preferred_selection = int(
            preferred_peer_id is not None and selected.peer_id == preferred_peer_id
        )
        with self._stats_lock:
            self._ready_candidates += ready_candidates
            self._compatible_candidates += compatible_candidates
            self._incompatible_candidates += incompatible_candidates
            self._waited_compatible_candidates += waited_compatible_candidates
            self._coalesce_waits += coalesce_waits
            self._coalesce_timeouts += coalesce_timeouts
            self._coalesce_wait_ns_sum += coalesce_wait_ns
            self._coalesce_wait_ns_max = max(
                self._coalesce_wait_ns_max,
                coalesce_wait_ns,
            )
            if coalesce_wait_outcome is not None:
                self._coalesce_wait_outcomes[coalesce_wait_outcome] += 1
                wait_bucket = _latency_histogram_bucket(coalesce_wait_ns)
                self._coalesce_wait_us_histogram[wait_bucket] += 1
            for skew_ns in compatible_arrival_skews_ns:
                self._compatible_arrival_skew_samples += 1
                self._compatible_arrival_skew_ns_sum += skew_ns
                self._compatible_arrival_skew_ns_max = max(
                    self._compatible_arrival_skew_ns_max,
                    skew_ns,
                )
                skew_bucket = _latency_histogram_bucket(skew_ns)
                self._compatible_arrival_skew_us_histogram[skew_bucket] += 1
            self._preferred_peer_selections += preferred_selection
        return tuple(compatible)

    @staticmethod
    def _release_inputs(
        items: tuple[PAPAttentionWorkItem, ...],
    ) -> BaseException | None:
        first_error: BaseException | None = None
        for item in items:
            try:
                item.release_input()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                logger.exception("PAP Attention input release failed")
        return first_error

    def _discard_pending(self) -> None:
        with self._deferred_lock:
            deferred = tuple(self._deferred)
            self._deferred.clear()
        for entry in deferred:
            self._discard_entry(entry)
        while True:
            try:
                entry = self._queue.get_nowait()
            except Empty:
                return
            self._discard_entry(entry)

    def _discard_entry(self, entry: PAPAttentionWorkItem | object) -> None:
        try:
            if isinstance(entry, PAPAttentionWorkItem):
                try:
                    entry.release_input()
                except BaseException as exc:
                    logger.exception(
                        "PAP Attention input release failed while dropping work"
                    )
                    self._record_failure(exc)
                finally:
                    entry.mark_completed()
                    with self._stats_lock:
                        self._pending_items -= 1
                        self._dropped += 1
        finally:
            self._queue.task_done()
