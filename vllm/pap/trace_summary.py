# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for summarizing PAP OFFLOAD_EXEC trace logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TraceStat:
    count: int
    mean: float
    median: float
    p90: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
            "p99": self.p99,
            "max": self.max,
        }


_PROJECTION_TRACE_RE = re.compile(
    r"projection trace .* send_ms=([0-9.]+) "
    r"trigger_ms=([0-9.]+) recv_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_ATTENTION_TRACE_RE = re.compile(
    r"attention mailbox batch trace .* recv_qkv_ms=([0-9.]+) "
    r"compute_ms=([0-9.]+) send_output_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_MAILBOX_SEND_RE = re.compile(
    r"PAP NIXL mailbox send trace actor=(\w+) .* nbytes=(\d+) "
    r"queue_ms=([0-9.]+) publish_ms=([0-9.]+) pack_ms=([0-9.]+) "
    r"copy_ms=([0-9.]+) notify_ms=([0-9.]+) ack_wait_ms=([0-9.]+) "
    r"total_ms=([0-9.]+)"
)
_MAILBOX_READ_RE = re.compile(
    r"PAP NIXL mailbox read trace actor=(\w+) .* nbytes=(\d+) "
    r"prepare_ms=([0-9.]+) transfer_ms=([0-9.]+) transfer_polls=(\d+) "
    r"materialize_ms=([0-9.]+) total_ms=([0-9.]+)"
)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    index = min(
        len(sorted_values) - 1,
        int(round((len(sorted_values) - 1) * percentile / 100.0)),
    )
    return sorted_values[index]


def _stat(values: Iterable[float]) -> TraceStat:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return TraceStat(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TraceStat(
        count=len(sorted_values),
        mean=statistics.mean(sorted_values),
        median=statistics.median(sorted_values),
        p90=_percentile(sorted_values, 90),
        p99=_percentile(sorted_values, 99),
        max=max(sorted_values),
    )


def _add_grouped_value(
    grouped: dict[str, dict[str, list[float]]],
    group: str,
    field: str,
    value: float,
) -> None:
    grouped.setdefault(group, {}).setdefault(field, []).append(float(value))


def summarize_pap_trace_logs(
    log_dir: str | Path,
    *,
    max_total_ms: float | None = 10.0,
) -> dict[str, object]:
    """Summarize PAP trace timings from a benchmark service_logs directory."""

    path = Path(log_dir)
    projection: dict[str, list[float]] = {
        "send_ms": [],
        "trigger_ms": [],
        "recv_ms": [],
        "total_ms": [],
    }
    attention: dict[str, list[float]] = {
        "recv_qkv_ms": [],
        "compute_ms": [],
        "send_output_ms": [],
        "total_ms": [],
    }
    mailbox_send: dict[str, dict[str, list[float]]] = {}
    mailbox_read: dict[str, dict[str, list[float]]] = {}

    for log_path in sorted(path.glob("*.log")):
        for line in log_path.read_text(errors="ignore").splitlines():
            if match := _PROJECTION_TRACE_RE.search(line):
                send_ms, trigger_ms, recv_ms, total_ms = map(float, match.groups())
                if max_total_ms is None or total_ms <= max_total_ms:
                    projection["send_ms"].append(send_ms)
                    projection["trigger_ms"].append(trigger_ms)
                    projection["recv_ms"].append(recv_ms)
                    projection["total_ms"].append(total_ms)
                continue
            if match := _ATTENTION_TRACE_RE.search(line):
                recv_ms, compute_ms, send_ms, total_ms = map(float, match.groups())
                if max_total_ms is None or total_ms <= max_total_ms:
                    attention["recv_qkv_ms"].append(recv_ms)
                    attention["compute_ms"].append(compute_ms)
                    attention["send_output_ms"].append(send_ms)
                    attention["total_ms"].append(total_ms)
                continue
            if match := _MAILBOX_SEND_RE.search(line):
                (
                    actor,
                    nbytes,
                    queue_ms,
                    publish_ms,
                    pack_ms,
                    copy_ms,
                    notify_ms,
                    ack_wait_ms,
                    total_ms,
                ) = match.groups()
                total = float(total_ms)
                if max_total_ms is not None and total > max_total_ms:
                    continue
                for field, value in (
                    ("nbytes", nbytes),
                    ("queue_ms", queue_ms),
                    ("publish_ms", publish_ms),
                    ("pack_ms", pack_ms),
                    ("copy_ms", copy_ms),
                    ("notify_ms", notify_ms),
                    ("ack_wait_ms", ack_wait_ms),
                    ("total_ms", total_ms),
                ):
                    _add_grouped_value(mailbox_send, actor, field, float(value))
                continue
            if match := _MAILBOX_READ_RE.search(line):
                (
                    actor,
                    nbytes,
                    prepare_ms,
                    transfer_ms,
                    transfer_polls,
                    materialize_ms,
                    total_ms,
                ) = match.groups()
                total = float(total_ms)
                if max_total_ms is not None and total > max_total_ms:
                    continue
                for field, value in (
                    ("nbytes", nbytes),
                    ("prepare_ms", prepare_ms),
                    ("transfer_ms", transfer_ms),
                    ("transfer_polls", transfer_polls),
                    ("materialize_ms", materialize_ms),
                    ("total_ms", total_ms),
                ):
                    _add_grouped_value(mailbox_read, actor, field, float(value))

    return {
        "projection_trace": {field: _stat(values) for field, values in projection.items()},
        "attention_trace": {field: _stat(values) for field, values in attention.items()},
        "mailbox_send": {
            actor: {field: _stat(values) for field, values in fields.items()}
            for actor, fields in mailbox_send.items()
        },
        "mailbox_read": {
            actor: {field: _stat(values) for field, values in fields.items()}
            for actor, fields in mailbox_read.items()
        },
    }


def summary_to_jsonable(summary: dict[str, object]) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, TraceStat):
            return value.to_dict()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(summary)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--include-outliers",
        action="store_true",
        help="include trace rows above the default 10ms warmup/outlier cutoff",
    )
    args = parser.parse_args(argv)
    summary = summarize_pap_trace_logs(
        args.log_dir,
        max_total_ms=None if args.include_outliers else 10.0,
    )
    print(json.dumps(summary_to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
