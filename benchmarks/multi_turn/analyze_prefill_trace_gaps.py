# SPDX-License-Identifier: Apache-2.0
"""Decompose Prefill torch-profiler time into GPU work and launch gaps."""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
ANNOTATION_PREFIX = "execute_context_"


def _load_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"traceEvents is missing from {path}")
    return events


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


def _launches_by_correlation(
    events: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    launches: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        correlation = event.get("args", {}).get("correlation")
        if not isinstance(correlation, int):
            continue
        name = str(event.get("name", ""))
        if not any(token in name for token in ("Launch", "Memcpy", "Memset")):
            continue
        current = launches.get(correlation)
        if current is None or float(event.get("ts", 0.0)) > float(
            current.get("ts", 0.0)
        ):
            launches[correlation] = event
    return launches


def analyze_trace(path: Path) -> dict[str, Any]:
    events = _load_trace(path)
    annotations = sorted(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and str(event.get("name", "")).startswith(ANNOTATION_PREFIX)
        ),
        key=lambda event: float(event["ts"]),
    )
    if not annotations:
        raise ValueError(f"no Prefill execute annotations found in {path}")

    launches = _launches_by_correlation(events)
    starts = [float(annotation["ts"]) for annotation in annotations]
    assigned: list[list[dict[str, Any]]] = [[] for _ in annotations]
    stream_duration: dict[int, float] = defaultdict(float)
    unmatched_gpu_ops = 0

    for event in events:
        if event.get("cat") not in GPU_CATEGORIES:
            continue
        args = event.get("args", {})
        correlation = args.get("correlation")
        launch = launches.get(correlation)
        if launch is None:
            unmatched_gpu_ops += 1
            continue
        launch_ts = float(launch["ts"])
        annotation_index = bisect.bisect_right(starts, launch_ts) - 1
        if annotation_index < 0:
            unmatched_gpu_ops += 1
            continue
        annotation = annotations[annotation_index]
        annotation_end = float(annotation["ts"]) + float(annotation.get("dur", 0.0))
        if launch_ts > annotation_end:
            unmatched_gpu_ops += 1
            continue
        enriched = dict(event)
        enriched["_launch"] = launch
        assigned[annotation_index].append(enriched)
        stream = int(args.get("stream", event.get("tid", -1)))
        stream_duration[stream] += float(event.get("dur", 0.0))

    if not stream_duration:
        raise ValueError(f"no correlated GPU operations found in {path}")
    main_stream = max(stream_duration, key=stream_duration.__getitem__)

    iteration_rows: list[dict[str, Any]] = []
    host_gap_rows: list[dict[str, Any]] = []
    for index, (annotation, gpu_ops) in enumerate(
        zip(annotations, assigned, strict=True)
    ):
        main_ops = sorted(
            (
                event
                for event in gpu_ops
                if int(
                    event.get("args", {}).get("stream", event.get("tid", -1))
                )
                == main_stream
            ),
            key=lambda event: float(event["ts"]),
        )
        intervals = [
            (float(event["ts"]), float(event["ts"]) + float(event.get("dur", 0.0)))
            for event in main_ops
        ]
        busy_us = _union_duration(intervals)
        span_us = intervals[-1][1] - intervals[0][0] if intervals else 0.0
        host_unsubmitted_gap_us = 0.0
        queued_gap_us = 0.0
        ambiguous_gap_us = 0.0
        for previous, current in zip(main_ops, main_ops[1:]):
            previous_end = float(previous["ts"]) + float(previous.get("dur", 0.0))
            current_start = float(current["ts"])
            gap = max(0.0, current_start - previous_end)
            if gap == 0.0:
                continue
            launch = current["_launch"]
            launch_start = float(launch["ts"])
            launch_end = launch_start + float(launch.get("dur", 0.0))
            if launch_start >= previous_end:
                host_gap = min(gap, launch_start - previous_end)
                host_unsubmitted_gap_us += host_gap
                ambiguous_gap_us += gap - host_gap
                if host_gap > 0.0:
                    host_gap_rows.append(
                        {
                            "iteration": index,
                            "iteration_name": annotation["name"],
                            "gap_ms": gap / 1000.0,
                            "host_unsubmitted_ms": host_gap / 1000.0,
                            "gap_start_us": previous_end,
                            "next_launch_start_us": launch_start,
                            "previous_gpu_op": previous["name"],
                            "next_gpu_op": current["name"],
                            "next_launch_api": launch["name"],
                        }
                    )
            elif launch_end <= previous_end:
                queued_gap_us += gap
            else:
                ambiguous_gap_us += gap

        iteration_rows.append(
            {
                "index": index,
                "name": annotation["name"],
                "cpu_envelope_ms": float(annotation.get("dur", 0.0)) / 1000.0,
                "gpu_main_span_ms": span_us / 1000.0,
                "gpu_main_busy_ms": busy_us / 1000.0,
                "gpu_main_gap_ms": (span_us - busy_us) / 1000.0,
                "host_unsubmitted_gap_ms": host_unsubmitted_gap_us / 1000.0,
                "queued_gap_ms": queued_gap_us / 1000.0,
                "ambiguous_gap_ms": ambiguous_gap_us / 1000.0,
                "main_stream_gpu_ops": len(main_ops),
            }
        )

    totals = {
        field: sum(float(row[field]) for row in iteration_rows)
        for field in (
            "cpu_envelope_ms",
            "gpu_main_span_ms",
            "gpu_main_busy_ms",
            "gpu_main_gap_ms",
            "host_unsubmitted_gap_ms",
            "queued_gap_ms",
            "ambiguous_gap_ms",
        )
    }
    totals["main_stream_gpu_ops"] = sum(
        int(row["main_stream_gpu_ops"]) for row in iteration_rows
    )
    cpu_pid = int(annotations[0].get("pid", -1))
    cpu_tid = int(annotations[0].get("tid", -1))
    cpu_events = [
        event
        for event in events
        if event.get("pid") == cpu_pid
        and event.get("tid") == cpu_tid
        and event.get("ph") == "X"
        and event.get("cat") in {"cpu_op", "cuda_runtime", "cuda_driver"}
    ]
    top_host_gaps = sorted(
        host_gap_rows,
        key=lambda row: float(row["host_unsubmitted_ms"]),
        reverse=True,
    )[:50]
    for gap_row in top_host_gaps:
        gap_start = float(gap_row["gap_start_us"])
        gap_end = float(gap_row["next_launch_start_us"])
        overlaps = []
        for event in cpu_events:
            event_start = float(event.get("ts", 0.0))
            event_end = event_start + float(event.get("dur", 0.0))
            overlap = min(event_end, gap_end) - max(event_start, gap_start)
            if overlap <= 0.0:
                continue
            overlaps.append(
                {
                    "category": event.get("cat"),
                    "name": event.get("name"),
                    "overlap_ms": overlap / 1000.0,
                    "duration_ms": float(event.get("dur", 0.0)) / 1000.0,
                }
            )
        gap_row["top_cpu_overlaps"] = sorted(
            overlaps,
            key=lambda row: float(row["overlap_ms"]),
            reverse=True,
        )[:12]
    return {
        "trace": str(path.resolve()),
        "iterations": len(iteration_rows),
        "main_stream": main_stream,
        "unmatched_gpu_ops": unmatched_gpu_ops,
        "totals": totals,
        "iteration_rows": iteration_rows,
        "top_host_gaps": top_host_gaps,
    }


def _print_summary(label: str, analysis: dict[str, Any]) -> None:
    totals = analysis["totals"]
    print(f"[{label}]")
    print(
        "iterations={iterations} main_stream={main_stream} gpu_ops={gpu_ops} "
        "unmatched={unmatched}".format(
            iterations=analysis["iterations"],
            main_stream=analysis["main_stream"],
            gpu_ops=totals["main_stream_gpu_ops"],
            unmatched=analysis["unmatched_gpu_ops"],
        )
    )
    print(
        "cpu_envelope={cpu_envelope_ms:.3f} ms "
        "gpu_span={gpu_main_span_ms:.3f} ms "
        "gpu_busy={gpu_main_busy_ms:.3f} ms "
        "gpu_gap={gpu_main_gap_ms:.3f} ms".format(**totals)
    )
    print(
        "gap_breakdown: host_unsubmitted={host_unsubmitted_gap_ms:.3f} ms "
        "queued={queued_gap_ms:.3f} ms "
        "ambiguous={ambiguous_gap_ms:.3f} ms".format(**totals)
    )
    print("top host-unsubmitted gaps:")
    for row in analysis["top_host_gaps"][:5]:
        overlap = row["top_cpu_overlaps"][0] if row["top_cpu_overlaps"] else None
        overlap_text = (
            f"{overlap['category']}:{overlap['name']} "
            f"({overlap['overlap_ms']:.3f} ms overlap)"
            if overlap is not None
            else "no recorded CPU op"
        )
        print(
            f"  iter={row['iteration']} host_gap={row['host_unsubmitted_ms']:.3f} ms "
            f"{row['previous_gpu_op']} -> {row['next_gpu_op']}; {overlap_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    payload = {"baseline": analyze_trace(args.baseline)}
    _print_summary("baseline", payload["baseline"])
    if args.candidate is not None:
        payload["candidate"] = analyze_trace(args.candidate)
        _print_summary("candidate", payload["candidate"])
        baseline = payload["baseline"]["totals"]
        candidate = payload["candidate"]["totals"]
        delta = {
            key: float(candidate[key]) - float(baseline[key])
            for key in baseline
            if key != "main_stream_gpu_ops"
        }
        payload["candidate_minus_baseline"] = delta
        print("[candidate - baseline]")
        print(
            "cpu_envelope={cpu_envelope_ms:+.3f} ms "
            "gpu_span={gpu_main_span_ms:+.3f} ms "
            "gpu_busy={gpu_main_busy_ms:+.3f} ms "
            "gpu_gap={gpu_main_gap_ms:+.3f} ms".format(**delta)
        )
        print(
            "gap_breakdown: host_unsubmitted={host_unsubmitted_gap_ms:+.3f} ms "
            "queued={queued_gap_ms:+.3f} ms "
            "ambiguous={ambiguous_gap_ms:+.3f} ms".format(**delta)
        )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
