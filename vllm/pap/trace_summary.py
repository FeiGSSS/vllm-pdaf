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
    r"projection trace .*?(?:batches=(\d+) )?calls=(\d+) "
    r"send_ms=([0-9.]+) trigger_ms=([0-9.]+) "
    r"(?:yield_ms=([0-9.]+) )?recv_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_ATTENTION_TRACE_RE = re.compile(
    r"attention mailbox batch trace .* calls=(\d+) recv_qkv_ms=([0-9.]+) "
    r"compute_ms=([0-9.]+) send_output_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_PROJECTION_CORRELATION_RE = re.compile(
    r"batch_keys=(\S+) send_done_ns=(\d+) yield_start_ns=(\d+) "
    r"yield_end_ns=(\d+) recv_done_ns=(\d+)"
)
_ATTENTION_CORRELATION_RE = re.compile(
    r"batch_key=(\S+) recv_done_ns=(\d+) compute_done_ns=(\d+) send_done_ns=(\d+)"
)
_MAILBOX_SEND_RE = re.compile(
    r"PAP NIXL mailbox send trace actor=(\S+) .* kind=(\S+) nbytes=(\d+) "
    r"queue_ms=([0-9.]+) publish_ms=([0-9.]+) pack_ms=([0-9.]+) "
    r"copy_ms=([0-9.]+) notify_ms=([0-9.]+) ack_wait_ms=([0-9.]+) "
    r"total_ms=([0-9.]+)"
)
_MAILBOX_READ_RE = re.compile(
    r"PAP NIXL mailbox read trace actor=(\S+) .* kind=(\S+) nbytes=(\d+) "
    r"prepare_ms=([0-9.]+) transfer_ms=([0-9.]+) transfer_polls=(\d+) "
    r"materialize_ms=([0-9.]+) total_ms=([0-9.]+)"
)
_MAILBOX_WAIT_RE = re.compile(
    r"PAP NIXL mailbox recv wait trace actor=(\S+) .* kind=(\S+) "
    r"requested_msg_id=.* wait_ms=([0-9.]+)"
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


def _mailbox_actor_group(actor: str) -> str:
    if actor.startswith("projection-"):
        return "projection"
    return actor


def _mailbox_kind_group(actor: str, kind: str) -> str:
    return f"{_mailbox_actor_group(actor)}:{kind}"


def summarize_pap_trace_logs(
    log_dir: str | Path,
    *,
    max_total_ms: float | None = 10.0,
) -> dict[str, object]:
    """Summarize PAP trace timings from a benchmark service_logs directory."""

    path = Path(log_dir)
    projection: dict[str, list[float]] = {
        "batches": [],
        "calls": [],
        "send_ms": [],
        "trigger_ms": [],
        "yield_ms": [],
        "recv_ms": [],
        "gap_ms": [],
        "total_ms": [],
    }
    attention: dict[str, list[float]] = {
        "calls": [],
        "recv_qkv_ms": [],
        "compute_ms": [],
        "send_output_ms": [],
        "total_ms": [],
    }
    mailbox_send: dict[str, dict[str, list[float]]] = {}
    mailbox_read: dict[str, dict[str, list[float]]] = {}
    mailbox_send_by_kind: dict[str, dict[str, list[float]]] = {}
    mailbox_read_by_kind: dict[str, dict[str, list[float]]] = {}
    mailbox_wait_by_kind: dict[str, dict[str, list[float]]] = {}
    projection_correlation_entries: list[tuple[list[str], int, int, int]] = []
    attention_send_done_ns_by_key: dict[str, int] = {}

    for log_path in sorted(path.glob("*.log")):
        for line in log_path.read_text(errors="ignore").splitlines():
            if match := _PROJECTION_TRACE_RE.search(line):
                (
                    batches,
                    calls,
                    send_ms,
                    trigger_ms,
                    yield_ms,
                    recv_ms,
                    total_ms,
                ) = match.groups()
                batches_value = int(batches) if batches is not None else 0
                calls_value = int(calls)
                send_ms = float(send_ms)
                trigger_ms = float(trigger_ms)
                yield_ms = float(yield_ms or 0.0)
                recv_ms = float(recv_ms)
                total_ms = float(total_ms)
                if max_total_ms is None or total_ms <= max_total_ms:
                    projection["batches"].append(batches_value)
                    projection["calls"].append(calls_value)
                    projection["send_ms"].append(send_ms)
                    projection["trigger_ms"].append(trigger_ms)
                    projection["yield_ms"].append(yield_ms)
                    projection["recv_ms"].append(recv_ms)
                    projection["gap_ms"].append(
                        max(0.0, total_ms - send_ms - trigger_ms - yield_ms - recv_ms)
                    )
                    projection["total_ms"].append(total_ms)
                    if correlation := _PROJECTION_CORRELATION_RE.search(line):
                        (
                            batch_keys,
                            send_done_ns,
                            _yield_start_ns,
                            yield_end_ns,
                            recv_done_ns,
                        ) = correlation.groups()
                        projection_correlation_entries.append(
                            (
                                batch_keys.split("|"),
                                int(send_done_ns),
                                int(yield_end_ns),
                                int(recv_done_ns),
                            )
                        )
                continue
            if match := _ATTENTION_TRACE_RE.search(line):
                (
                    calls,
                    recv_ms,
                    compute_ms,
                    send_ms,
                    total_ms,
                ) = match.groups()
                calls = int(calls)
                recv_ms, compute_ms, send_ms, total_ms = map(
                    float, (recv_ms, compute_ms, send_ms, total_ms)
                )
                if max_total_ms is None or total_ms <= max_total_ms:
                    attention["calls"].append(calls)
                    attention["recv_qkv_ms"].append(recv_ms)
                    attention["compute_ms"].append(compute_ms)
                    attention["send_output_ms"].append(send_ms)
                    attention["total_ms"].append(total_ms)
                    if correlation := _ATTENTION_CORRELATION_RE.search(line):
                        (
                            batch_key,
                            _recv_done_ns,
                            _compute_done_ns,
                            send_done_ns,
                        ) = correlation.groups()
                        attention_send_done_ns_by_key[batch_key] = int(send_done_ns)
                continue
            if match := _MAILBOX_SEND_RE.search(line):
                (
                    actor,
                    kind,
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
                actor_group = _mailbox_actor_group(actor)
                kind_group = _mailbox_kind_group(actor, kind)
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
                    _add_grouped_value(mailbox_send, actor_group, field, float(value))
                    _add_grouped_value(
                        mailbox_send_by_kind, kind_group, field, float(value)
                    )
                continue
            if match := _MAILBOX_READ_RE.search(line):
                (
                    actor,
                    kind,
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
                actor_group = _mailbox_actor_group(actor)
                kind_group = _mailbox_kind_group(actor, kind)
                for field, value in (
                    ("nbytes", nbytes),
                    ("prepare_ms", prepare_ms),
                    ("transfer_ms", transfer_ms),
                    ("transfer_polls", transfer_polls),
                    ("materialize_ms", materialize_ms),
                    ("total_ms", total_ms),
                ):
                    _add_grouped_value(mailbox_read, actor_group, field, float(value))
                    _add_grouped_value(
                        mailbox_read_by_kind, kind_group, field, float(value)
                    )
                continue
            if match := _MAILBOX_WAIT_RE.search(line):
                actor, kind, wait_ms = match.groups()
                _add_grouped_value(
                    mailbox_wait_by_kind,
                    _mailbox_kind_group(actor, kind),
                    "wait_ms",
                    float(wait_ms),
                )

    projection_attention_correlation: dict[str, list[float]] = {
        "matched_batches": [],
        "attention_path_after_projection_send_ms": [],
        "projection_resume_after_attention_ready_ms": [],
        "attention_ready_after_projection_resume_ms": [],
        "projection_resume_to_recv_done_ms": [],
    }
    for batch_keys, send_done_ns, yield_end_ns, recv_done_ns in (
        projection_correlation_entries
    ):
        attention_done_times = [
            attention_send_done_ns_by_key[key]
            for key in batch_keys
            if key in attention_send_done_ns_by_key
        ]
        if len(attention_done_times) != len(batch_keys):
            continue
        max_attention_done_ns = max(attention_done_times)
        projection_attention_correlation["matched_batches"].append(
            float(len(attention_done_times))
        )
        projection_attention_correlation[
            "attention_path_after_projection_send_ms"
        ].append((max_attention_done_ns - send_done_ns) / 1_000_000.0)
        projection_attention_correlation[
            "projection_resume_after_attention_ready_ms"
        ].append(max(0.0, (yield_end_ns - max_attention_done_ns) / 1_000_000.0))
        projection_attention_correlation[
            "attention_ready_after_projection_resume_ms"
        ].append(max(0.0, (max_attention_done_ns - yield_end_ns) / 1_000_000.0))
        projection_attention_correlation["projection_resume_to_recv_done_ms"].append(
            (recv_done_ns - yield_end_ns) / 1_000_000.0
        )

    return {
        "projection_trace": {field: _stat(values) for field, values in projection.items()},
        "attention_trace": {field: _stat(values) for field, values in attention.items()},
        "projection_attention_correlation": {
            field: _stat(values)
            for field, values in projection_attention_correlation.items()
        },
        "mailbox_send": {
            actor: {field: _stat(values) for field, values in fields.items()}
            for actor, fields in mailbox_send.items()
        },
        "mailbox_read": {
            actor: {field: _stat(values) for field, values in fields.items()}
            for actor, fields in mailbox_read.items()
        },
        "mailbox_send_by_kind": {
            group: {field: _stat(values) for field, values in fields.items()}
            for group, fields in mailbox_send_by_kind.items()
        },
        "mailbox_read_by_kind": {
            group: {field: _stat(values) for field, values in fields.items()}
            for group, fields in mailbox_read_by_kind.items()
        },
        "mailbox_wait_by_kind": {
            group: {field: _stat(values) for field, values in fields.items()}
            for group, fields in mailbox_wait_by_kind.items()
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
