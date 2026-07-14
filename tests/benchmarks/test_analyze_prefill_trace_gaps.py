# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.multi_turn.analyze_prefill_trace_gaps import analyze_trace


def _launch(correlation: int, ts: float, dur: float = 1.0) -> dict:
    return {
        "ph": "X",
        "cat": "cuda_driver",
        "name": "cuLaunchKernel",
        "ts": ts,
        "dur": dur,
        "args": {"correlation": correlation},
    }


def _kernel(correlation: int, ts: float, dur: float) -> dict:
    return {
        "ph": "X",
        "cat": "kernel",
        "name": "test_kernel",
        "pid": 0,
        "tid": 7,
        "ts": ts,
        "dur": dur,
        "args": {"correlation": correlation, "stream": 7},
    }


def test_analyze_trace_decomposes_main_stream_gaps(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    events = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "execute_context_1(4096)_generation_0(0)",
            "ts": 0.0,
            "dur": 100.0,
        },
        _launch(1, 5.0),
        _kernel(1, 10.0, 10.0),
        _launch(2, 30.0),
        _kernel(2, 40.0, 10.0),
        _launch(3, 15.0),
        _kernel(3, 70.0, 5.0),
    ]
    trace.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")

    analysis = analyze_trace(trace)

    assert analysis["iterations"] == 1
    assert analysis["main_stream"] == 7
    totals = analysis["totals"]
    assert totals["gpu_main_busy_ms"] == pytest.approx(0.025)
    assert totals["gpu_main_span_ms"] == pytest.approx(0.065)
    assert totals["gpu_main_gap_ms"] == pytest.approx(0.040)
    assert totals["host_unsubmitted_gap_ms"] == pytest.approx(0.010)
    assert totals["queued_gap_ms"] == pytest.approx(0.020)
    assert totals["ambiguous_gap_ms"] == pytest.approx(0.010)
