# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-blocking CUDA event timing for PAP diagnostic runs."""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_TRUE_VALUES = {"1", "true", "yes", "on"}
_TRACE_ENABLED = os.environ.get("PAP_DEFERRED_CUDA_TRACE", "0").lower() in (
    _TRUE_VALUES
)
_MAX_PENDING = max(
    1,
    int(os.environ.get("PAP_DEFERRED_CUDA_TRACE_MAX_PENDING", "1024")),
)
_TRACE_ROLE = os.environ.get("PAP_DEFERRED_TRACE_ROLE", "").strip()
_TRACE_OUTPUT = os.environ.get("PAP_DEFERRED_TRACE_OUTPUT", "").strip()
_TRACE_SCOPES = {
    "projection": "projection_process_critical_chain",
    "pd_decode": "pd_decode_process_critical_chain",
}


@dataclass
class DeferredCudaSpan:
    """One recorded CUDA event pair awaiting non-blocking collection."""

    name: str
    start_event: Any
    end_event: Any
    stream: Any
    finished: bool = False


@dataclass
class DeferredCudaFanIn:
    """Cross-stream CUDA events for one non-blocking fan-in sample."""

    name: str
    origin_event: Any
    ready_events: tuple[Any, ...]
    layer: int
    calls: int
    ready_count: int = 0
    finished: bool = False


class DeferredCudaTraceCollector:
    """Collect CUDA event durations without synchronizing the hot path."""

    def __init__(
        self,
        *,
        max_pending: int = _MAX_PENDING,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.max_pending = max(1, int(max_pending))
        self._event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=True)
        )
        self._pending: deque[DeferredCudaSpan] = deque()
        self._free_pairs: list[tuple[Any, Any]] = []
        self._allocated_pairs = 0
        self._pending_fanins: deque[DeferredCudaFanIn] = deque()
        self._free_fanin_events: dict[
            int,
            list[tuple[Any, tuple[Any, ...]]],
        ] = defaultdict(list)
        self._allocated_fanin_groups = 0
        self._fanin_samples: dict[
            str,
            list[
                tuple[
                    int,
                    int,
                    int,
                    float,
                    float,
                    float,
                    tuple[float, ...],
                ]
            ],
        ] = defaultdict(list)
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.dropped_records = 0
        self.error_records = 0

    def _collect_ready_locked(self) -> None:
        while self._pending:
            span = self._pending[0]
            try:
                ready = bool(span.end_event.query())
            except Exception:
                self.error_records += 1
                self._pending.popleft()
                self._free_pairs.append((span.start_event, span.end_event))
                continue
            if not ready:
                break
            self._pending.popleft()
            self._finish_record_locked(span)
        while self._pending_fanins:
            fanin = self._pending_fanins[0]
            try:
                ready = all(event.query() for event in fanin.ready_events)
            except Exception:
                self.error_records += 1
                self._pending_fanins.popleft()
                self._release_fanin_events_locked(fanin)
                continue
            if not ready:
                break
            self._pending_fanins.popleft()
            self._finish_fanin_locked(fanin)

    def _finish_record_locked(self, span: DeferredCudaSpan) -> None:
        try:
            duration_ms = float(span.start_event.elapsed_time(span.end_event))
        except Exception:
            self.error_records += 1
        else:
            self._durations[span.name].append(duration_ms)
        self._free_pairs.append((span.start_event, span.end_event))

    def _release_fanin_events_locked(self, fanin: DeferredCudaFanIn) -> None:
        self._free_fanin_events[len(fanin.ready_events)].append(
            (fanin.origin_event, fanin.ready_events)
        )

    def _finish_fanin_locked(self, fanin: DeferredCudaFanIn) -> None:
        try:
            ready_times_ms = [
                float(fanin.origin_event.elapsed_time(event))
                for event in fanin.ready_events
            ]
        except Exception:
            self.error_records += 1
        else:
            first_ready_ms = min(ready_times_ms)
            last_ready_ms = max(ready_times_ms)
            self._fanin_samples[fanin.name].append(
                (
                    int(fanin.layer),
                    len(fanin.ready_events),
                    int(fanin.calls),
                    first_ready_ms,
                    last_ready_ms,
                    last_ready_ms - first_ready_ms,
                    tuple(ready_times_ms),
                )
            )
        self._release_fanin_events_locked(fanin)

    def begin(self, name: str, stream: Any) -> DeferredCudaSpan | None:
        """Record a start event or drop the diagnostic record if saturated."""

        with self._lock:
            return self._begin_locked(name, stream)

    def _begin_locked(
        self,
        name: str,
        stream: Any,
    ) -> DeferredCudaSpan | None:
        self._collect_ready_locked()
        if self._free_pairs:
            start_event, end_event = self._free_pairs.pop()
        elif self._allocated_pairs < self.max_pending:
            try:
                start_event = self._event_factory()
                end_event = self._event_factory()
            except Exception:
                self.error_records += 1
                return None
            self._allocated_pairs += 1
        else:
            self.dropped_records += 1
            return None
        try:
            start_event.record(stream)
        except Exception:
            self.error_records += 1
            self._free_pairs.append((start_event, end_event))
            return None
        return DeferredCudaSpan(
            name=str(name),
            start_event=start_event,
            end_event=end_event,
            stream=stream,
        )

    def end(self, span: DeferredCudaSpan | None) -> None:
        """Record an end event; this method never waits for the GPU."""

        with self._lock:
            if span is None or span.finished:
                return
            span.finished = True
            try:
                span.end_event.record(span.stream)
            except Exception:
                self.error_records += 1
                self._free_pairs.append((span.start_event, span.end_event))
                return
            self._pending.append(span)

    def begin_fanin(
        self,
        name: str,
        origin_stream: Any,
        *,
        peers: int,
        layer: int,
        calls: int,
    ) -> DeferredCudaFanIn | None:
        """Begin one cross-stream fan-in sample without waiting for CUDA."""
        peer_count = int(peers)
        if peer_count <= 1 or calls <= 0:
            return None
        with self._lock:
            self._collect_ready_locked()
            free_groups = self._free_fanin_events[peer_count]
            if free_groups:
                origin_event, ready_events = free_groups.pop()
            elif self._allocated_fanin_groups < self.max_pending:
                try:
                    origin_event = self._event_factory()
                    ready_events = tuple(
                        self._event_factory() for _ in range(peer_count)
                    )
                except Exception:
                    self.error_records += 1
                    return None
                self._allocated_fanin_groups += 1
            else:
                self.dropped_records += 1
                return None
            try:
                origin_event.record(origin_stream)
            except Exception:
                self.error_records += 1
                self._free_fanin_events[peer_count].append((origin_event, ready_events))
                return None
            return DeferredCudaFanIn(
                name=str(name),
                origin_event=origin_event,
                ready_events=ready_events,
                layer=int(layer),
                calls=int(calls),
            )

    def record_fanin_ready(
        self,
        fanin: DeferredCudaFanIn | None,
        *,
        index: int,
        stream: Any,
    ) -> None:
        """Record one peer-ready event for a pending fan-in sample."""
        if fanin is None:
            return
        with self._lock:
            if fanin.finished:
                return
            if index != fanin.ready_count or index >= len(fanin.ready_events):
                self.error_records += 1
                return
            try:
                fanin.ready_events[index].record(stream)
            except Exception:
                self.error_records += 1
                return
            fanin.ready_count += 1

    def end_fanin(self, fanin: DeferredCudaFanIn | None) -> None:
        """Publish a fully recorded fan-in sample for deferred collection."""
        if fanin is None:
            return
        with self._lock:
            if fanin.finished:
                return
            fanin.finished = True
            if fanin.ready_count != len(fanin.ready_events):
                self.error_records += 1
                self._release_fanin_events_locked(fanin)
                return
            self._pending_fanins.append(fanin)

    def record_duration(self, name: str, duration_ms: float) -> None:
        """Record an already measured host duration."""

        value = float(duration_ms)
        if not math.isfinite(value) or value < 0:
            with self._lock:
                self.error_records += 1
            return
        with self._lock:
            self._durations[str(name)].append(value)

    def flush(self, *, blocking: bool) -> None:
        """Collect completed spans, optionally waiting during post-run drain."""

        with self._lock:
            self._flush_locked(blocking=blocking)

    def _flush_locked(self, *, blocking: bool) -> None:
        self._collect_ready_locked()
        if not blocking:
            return
        pending = self._pending
        self._pending = deque()
        for span in pending:
            try:
                span.end_event.synchronize()
            except Exception:
                self.error_records += 1
                self._free_pairs.append((span.start_event, span.end_event))
                continue
            self._finish_record_locked(span)
        pending_fanins = self._pending_fanins
        self._pending_fanins = deque()
        for fanin in pending_fanins:
            try:
                for event in fanin.ready_events:
                    event.synchronize()
            except Exception:
                self.error_records += 1
                self._release_fanin_events_locked(fanin)
                continue
            self._finish_fanin_locked(fanin)

    def raw_snapshot(self, *, blocking: bool) -> dict[str, Any]:
        """Return raw duration lists for process-level aggregation."""

        with self._lock:
            self._flush_locked(blocking=blocking)
            return {
                "pending_records": len(self._pending) + len(self._pending_fanins),
                "dropped_records": self.dropped_records,
                "error_records": self.error_records,
                "durations": {
                    name: list(values) for name, values in self._durations.items()
                },
                "fanins": {
                    name: list(samples) for name, samples in self._fanin_samples.items()
                },
            }


class DeferredTraceFileExporter:
    """Flush one process-local trace after a post-workload trigger."""

    def __init__(
        self,
        *,
        output_path: str,
        scope: str,
        role: str,
        snapshot_fn: Callable[..., dict[str, Any]],
        poll_interval_s: float = 0.05,
    ) -> None:
        self.output_path = Path(output_path)
        self.trigger_path = Path(f"{self.output_path}.flush")
        self.scope = str(scope)
        self.role = str(role)
        self.snapshot_fn = snapshot_fn
        self.poll_interval_s = max(0.001, float(poll_interval_s))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: BaseException | None = None

    def start(self) -> None:
        """Start the daemon poller once."""

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"deferred-trace-exporter-{self.role}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the poller without forcing a trace flush."""

        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.poll_interval_s * 4))

    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval_s):
            if not self.trigger_path.exists():
                continue
            try:
                self._export()
            except BaseException as error:
                self.last_error = error
                return
            return

    def _export(self) -> None:
        payload = dict(self.snapshot_fn(blocking=True))
        payload.update(
            {
                "scope": self.scope,
                "role": self.role,
                "pid": os.getpid(),
            }
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_name(
            f".{self.output_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.output_path)
        self.trigger_path.unlink(missing_ok=True)


_thread_local = threading.local()
_registry_lock = threading.Lock()
_collectors: list[DeferredCudaTraceCollector] = []
_exporter_lock = threading.Lock()
_file_exporter: DeferredTraceFileExporter | None = None


def deferred_cuda_trace_enabled() -> bool:
    """Return whether the diagnostic-only deferred event lane is enabled."""

    return _TRACE_ENABLED


def deferred_trace_role() -> str:
    """Return the process role selected for deferred diagnostics."""

    return _TRACE_ROLE


def ensure_deferred_trace_file_exporter() -> None:
    """Lazily start the diagnostic exporter for supported process roles."""

    global _file_exporter
    if not _TRACE_ENABLED or not _TRACE_OUTPUT:
        return
    scope = _TRACE_SCOPES.get(_TRACE_ROLE)
    if scope is None:
        return
    with _exporter_lock:
        if _file_exporter is not None:
            return
        _file_exporter = DeferredTraceFileExporter(
            output_path=_TRACE_OUTPUT,
            scope=scope,
            role=_TRACE_ROLE,
            snapshot_fn=deferred_cuda_trace_snapshot,
        )
        _file_exporter.start()


def _thread_collector() -> DeferredCudaTraceCollector:
    collector = getattr(_thread_local, "pap_deferred_cuda_trace", None)
    if collector is not None:
        return collector
    collector = DeferredCudaTraceCollector()
    _thread_local.pap_deferred_cuda_trace = collector
    with _registry_lock:
        _collectors.append(collector)
    ensure_deferred_trace_file_exporter()
    return collector


def begin_deferred_cuda_span(
    name: str,
    stream: Any,
) -> tuple[DeferredCudaTraceCollector, DeferredCudaSpan] | None:
    """Begin a span on the calling thread's collector."""

    if not _TRACE_ENABLED:
        return None
    collector = _thread_collector()
    span = collector.begin(name, stream)
    if span is None:
        return None
    return collector, span


def end_deferred_cuda_span(
    handle: tuple[DeferredCudaTraceCollector, DeferredCudaSpan] | None,
) -> None:
    """End a span without synchronizing the CUDA stream."""

    if handle is None:
        return
    collector, span = handle
    collector.end(span)


def begin_deferred_cuda_fanin(
    name: str,
    origin_stream: Any,
    *,
    peers: int,
    layer: int,
    calls: int,
) -> tuple[DeferredCudaTraceCollector, DeferredCudaFanIn] | None:
    """Begin one cross-stream fan-in sample on the calling thread."""
    if not _TRACE_ENABLED:
        return None
    collector = _thread_collector()
    fanin = collector.begin_fanin(
        name,
        origin_stream,
        peers=peers,
        layer=layer,
        calls=calls,
    )
    if fanin is None:
        return None
    return collector, fanin


def record_deferred_cuda_fanin_ready(
    handle: tuple[DeferredCudaTraceCollector, DeferredCudaFanIn] | None,
    *,
    index: int,
    stream: Any,
) -> None:
    """Record one peer-ready event without synchronizing its CUDA stream."""
    if handle is None:
        return
    collector, fanin = handle
    collector.record_fanin_ready(fanin, index=index, stream=stream)


def end_deferred_cuda_fanin(
    handle: tuple[DeferredCudaTraceCollector, DeferredCudaFanIn] | None,
) -> None:
    """Queue one complete fan-in sample for deferred collection."""
    if handle is None:
        return
    collector, fanin = handle
    collector.end_fanin(fanin)


def record_deferred_host_duration(name: str, duration_ms: float) -> None:
    """Record a host-side duration only when deferred tracing is enabled."""

    if not _TRACE_ENABLED:
        return
    _thread_collector().record_duration(name, duration_ms)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return float(sorted_values[min(rank, len(sorted_values) - 1)])


def _duration_stats(values: list[float]) -> dict[str, float | int]:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "p999_ms": 0.0,
            "p9999_ms": 0.0,
            "max_ms": 0.0,
            "ge_1ms": 0,
            "ge_10ms": 0,
            "ge_100ms": 0,
            "ge_1000ms": 0,
        }
    return {
        "count": len(sorted_values),
        "mean_ms": statistics.mean(sorted_values),
        "p50_ms": statistics.median(sorted_values),
        "p90_ms": _percentile(sorted_values, 0.90),
        "p95_ms": _percentile(sorted_values, 0.95),
        "p99_ms": _percentile(sorted_values, 0.99),
        "p999_ms": _percentile(sorted_values, 0.999),
        "p9999_ms": _percentile(sorted_values, 0.9999),
        "max_ms": sorted_values[-1],
        "ge_1ms": sum(value >= 1.0 for value in sorted_values),
        "ge_10ms": sum(value >= 10.0 for value in sorted_values),
        "ge_100ms": sum(value >= 100.0 for value in sorted_values),
        "ge_1000ms": sum(value >= 1000.0 for value in sorted_values),
    }


def deferred_cuda_trace_snapshot(*, blocking: bool) -> dict[str, Any]:
    """Aggregate all thread-local collectors in the current process."""

    with _registry_lock:
        collectors = list(_collectors)
    durations: dict[str, list[float]] = defaultdict(list)
    fanins: dict[
        str,
        list[
            tuple[
                int,
                int,
                int,
                float,
                float,
                float,
                tuple[float, ...],
            ]
        ],
    ] = defaultdict(list)
    pending_records = 0
    dropped_records = 0
    error_records = 0
    for collector in collectors:
        snapshot = collector.raw_snapshot(blocking=blocking)
        pending_records += int(snapshot["pending_records"])
        dropped_records += int(snapshot["dropped_records"])
        error_records += int(snapshot["error_records"])
        for name, values in snapshot["durations"].items():
            durations[str(name)].extend(float(value) for value in values)
        for name, samples in snapshot["fanins"].items():
            fanins[str(name)].extend(tuple(sample) for sample in samples)
    return {
        "enabled": _TRACE_ENABLED,
        "collector_count": len(collectors),
        "pending_records": pending_records,
        "dropped_records": dropped_records,
        "error_records": error_records,
        "spans": {
            name: _duration_stats(values) for name, values in sorted(durations.items())
        },
        "fanins": {
            name: {
                "count": len(samples),
                "sample_fields": (
                    "layer",
                    "peers",
                    "calls",
                    "first_ready_ms",
                    "last_ready_ms",
                    "spread_ms",
                    "ready_times_ms",
                ),
                "first_ready_ms": _duration_stats(
                    [float(sample[3]) for sample in samples]
                ),
                "last_ready_ms": _duration_stats(
                    [float(sample[4]) for sample in samples]
                ),
                "spread_ms": _duration_stats([float(sample[5]) for sample in samples]),
                "samples": samples,
            }
            for name, samples in sorted(fanins.items())
        },
    }
