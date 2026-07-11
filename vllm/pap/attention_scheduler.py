# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduling primitives for PAP Attention work."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)

_STOP = object()


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
        handler: Callable[[PAPAttentionWorkItem], None],
        max_queue_size: int = 0,
    ) -> None:
        self._handler = handler
        self._queue: Queue[PAPAttentionWorkItem | object] = Queue(
            maxsize=max(0, int(max_queue_size))
        )
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
            if self._queue.full():
                raise RuntimeError("PAP Attention dispatcher queue is full")
            with self._stats_lock:
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

    def dispatch_next(self, timeout: float | None = None) -> bool:
        entry = self._queue.get(timeout=timeout)
        if entry is _STOP:
            self._queue.task_done()
            return False
        assert isinstance(entry, PAPAttentionWorkItem)
        item = entry
        dispatch_started_ns = time.perf_counter_ns()
        item.dispatch_started_ns = dispatch_started_ns
        item.queue_wait_ns = max(0, dispatch_started_ns - int(item.arrival_ns))
        with self._stats_lock:
            self._pending_items -= 1
            self._queue_wait_ns_sum += item.queue_wait_ns
            self._queue_wait_ns_max = max(
                self._queue_wait_ns_max,
                item.queue_wait_ns,
            )
        try:
            try:
                self._handler(item)
            except BaseException as exc:
                try:
                    item.release_input()
                except BaseException:
                    logger.exception(
                        "PAP Attention input release failed after handler error"
                    )
                self._record_failure(exc)
                raise
            try:
                item.release_input()
            except BaseException as exc:
                self._record_failure(exc)
                raise
            with self._stats_lock:
                self._dispatched += 1
            return True
        finally:
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

    def _discard_pending(self) -> None:
        while True:
            try:
                entry = self._queue.get_nowait()
            except Empty:
                return
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
