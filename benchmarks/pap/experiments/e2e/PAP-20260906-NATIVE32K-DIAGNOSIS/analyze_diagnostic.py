# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect diagnostic metrics and request-correlated control completion."""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import analyze

DIR = Path(__file__).resolve().parent
RUN = DIR / "diagnostic-run"
analyze.PAP = RUN


def summary(values):
    values = sorted(values)
    return (
        {
            "count": len(values),
            "mean": mean(values),
            "p50": median(values),
            "max": max(values),
        }
        if values
        else {"count": 0}
    )


base = analyze.analyze()
requests = {r["id"]: r for r in base["requests"]}
latest = {}
for line in (RUN / "prefill_metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row.get("lines"):
        latest[row["pa"]] = row
metric_rows = []
totals = defaultdict(float)
for pa, row in sorted(latest.items()):
    vals = {
        line.split("{")[0]: float(line.rsplit(" ", 1)[1])
        for line in row["lines"]
        if "_bucket" not in line
    }
    for key, val in vals.items():
        totals[key] += val
    metric_rows.append({"pa": pa, "time": row["time"], "values": vals})
n = totals["vllm:request_queue_time_seconds_count"]
stage_metrics = {
    key: totals[f"vllm:request_{key}_time_seconds_sum"] / n
    for key in ("queue", "prefill", "inference")
}
ipc = defaultdict(lambda: defaultdict(list))
for line in (RUN / "service_logs/proxy.log").read_text().splitlines():
    if "PAP proxy prefill IPC profile" not in line:
        continue
    rid = re.search(r"request_id=(\S+)", line)[1]
    for key, val in re.findall(r"(\w+_ms)=([\d.]+)", line):
        ipc[str(requests[rid]["turn"])][key].append(float(val))
releases, pins, commits = [], [], []
uuid = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
for pa in range(7):
    path = RUN / f"service_logs/prefill_{pa}.log"
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not any(
            s in line
            for s in (
                "PAP KV lease release lease_id=",
                "PAP KV lease pin request_id=",
                "unknown_request",
            )
        ):
            continue
        stamp = re.search(r"09-06 (\d\d:\d\d:\d\d)", line)
        rid_match = re.search(uuid, line)
        if not stamp or not rid_match or rid_match[0] not in requests:
            continue
        rid = rid_match[0]
        t = datetime.fromisoformat("2026-09-06T" + stamp[1]).timestamp()
        req = requests[rid]
        event = {
            "pa": pa,
            "id": rid,
            "time": t,
            "source": str(path),
            "line": number,
            "client_end": req["end"],
            "lag_after_client_end_s": t - req["end"],
        }
        if "PAP KV lease release lease_id=" in line:
            event["blocks"] = int(re.search(r"blocks=(\d+)", line)[1])
            releases.append(event)
        elif "PAP KV lease pin request_id=" in line:
            pins.append(event)
        else:
            commits.append(event)
attention_retires = {}
for pa in range(7):
    path = RUN / f"service_logs/attention_{pa}_0.log"
    for line in path.read_text().splitlines():
        match = re.search(r"release session lease request_id=(\S+)", line)
        if match:
            attention_retires[match[1]] = analyze.epoch(line)
for row in releases:
    if row["id"] in attention_retires:
        row["attention_retire_time"] = attention_retires[row["id"]]
        row["attention_to_prefill_retire_s"] = (
            row["time"] - row["attention_retire_time"]
        )
result = {
    "run": str(RUN),
    "metrics_completed_count": n,
    "mean_native_stage_seconds": stage_metrics,
    "by_turn": base["by_turn"],
    "ipc_by_turn_ms": {
        turn: {k: mean(v) for k, v in values.items()} for turn, values in ipc.items()
    },
    "release_lag_seconds_coarse_1s": summary(
        [r["lag_after_client_end_s"] for r in releases]
    ),
    "failed_commit_lag_seconds_coarse_1s": summary(
        [r["lag_after_client_end_s"] for r in commits]
    ),
    "attention_to_prefill_retire_s_coarse_1s": summary(
        [
            r["attention_to_prefill_retire_s"]
            for r in releases
            if "attention_to_prefill_retire_s" in r
        ]
    ),
    "lease_releases": releases,
    "lease_pins": pins,
    "failed_commits": commits,
    "metric_snapshots": metric_rows,
    "attention": base["attention"],
    "cross_turn_pa_changes": base["cross_turn_pa_changes"],
}
(DIR / "diagnostic_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in result.items()
            if k
            not in {
                "lease_releases",
                "lease_pins",
                "failed_commits",
                "metric_snapshots",
                "attention",
            }
        },
        indent=2,
    )
)
