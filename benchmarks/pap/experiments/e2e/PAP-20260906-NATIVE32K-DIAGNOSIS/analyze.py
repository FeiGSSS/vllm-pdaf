# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Join saved client timing and gateway events without changing serving."""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1] / "_runs"
PAP = ROOT / "pap/20260906_native32k_7pa1p_conc60_32k_attempt3"
DP = ROOT / "dynamo/20260906_native32k_dp8_conc60"


def records(root):
    return [
        json.loads(s) for s in (root / "aiperf/profile.jsonl").read_text().splitlines()
    ]


def epoch(line):
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()


def metric(row, name):
    return row["metrics"][name]["value"]


def analyze():
    routes, ready = {}, {}
    proxy = PAP / "service_logs/proxy.log"
    for number, line in enumerate(proxy.read_text().splitlines(), 1):
        m = re.search(r"placement request_id=(\S+) selected_pa=(\d+)", line)
        if m:
            routes[m[1]] = {"time": epoch(line), "pa": int(m[2]), "line": number}
        m = re.search(r"request_id=(\S+).*prefill_ms=(\d+)", line)
        if m:
            ready[m[1]] = {"time": epoch(line), "prefill_ms": int(m[2]), "line": number}
    joined = []
    for row in records(PAP):
        meta = row["metadata"]
        rid = meta["x_request_id"]
        start = meta["request_start_ns"] / 1e9
        first = start + metric(row, "time_to_first_token") / 1000
        end = meta["request_end_ns"] / 1e9
        route, handoff = routes[rid], ready[rid]
        joined.append(
            {
                "id": rid,
                "conversation": meta["conversation_id"],
                "turn": meta["turn_index"],
                "pa": route["pa"],
                "start": start,
                "route": route["time"],
                "ready": handoff["time"],
                "first": first,
                "end": end,
                "ttft_ms": metric(row, "time_to_first_token"),
                "prefill_ms": handoff["prefill_ms"],
                "client_to_route_ms": 1000 * (route["time"] - start),
                "route_to_ready_other_ms": 1000 * (handoff["time"] - route["time"])
                - handoff["prefill_ms"],
                "ready_to_first_ms": 1000 * (first - handoff["time"]),
                "input": metric(row, "input_sequence_length"),
                "output": metric(row, "output_sequence_length"),
                "cached": metric(row, "usage_prompt_cache_read_tokens"),
                "route_line": route["line"],
                "ready_line": handoff["line"],
            }
        )
    groups = {}
    for turn in range(3):
        rs = [r for r in joined if r["turn"] == turn]
        keys = [
            "ttft_ms",
            "prefill_ms",
            "client_to_route_ms",
            "route_to_ready_other_ms",
            "ready_to_first_ms",
            "input",
            "cached",
        ]
        groups[str(turn)] = {k: mean(r[k] for r in rs) for k in keys}
    samples, warnings = [], Counter()
    for pa in range(7):
        path = PAP / f"service_logs/prefill_{pa}.log"
        lines = path.read_text().splitlines()
        cap = int(
            re.search(r"GPU KV cache size: ([\d,]+) tokens", path.read_text())[
                1
            ].replace(",", "")
        )
        for number, line in enumerate(lines, 1):
            if "unknown_request" in line:
                warnings[f"pa{pa}_unknown_commit"] += 1
            m = re.search(
                r"09-06 (\d\d:\d\d:\d\d).*Running: (\d+) reqs, "
                r"Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%",
                line,
            )
            if not m:
                continue
            t = datetime.fromisoformat("2026-09-06T" + m[1]).timestamp()
            # Log timestamps are coarse and scheduler snapshots may be older.
            # This is client-observed decode activity, not authoritative ownership.
            active = [r for r in joined if r["pa"] == pa and r["ready"] <= t < r["end"]]
            routed = [
                r for r in joined if r["pa"] == pa and r["route"] <= t < r["ready"]
            ]
            samples.append(
                {
                    "pa": pa,
                    "time": t,
                    "line": number,
                    "running": int(m[2]),
                    "waiting": int(m[3]),
                    "usage_pct": float(m[4]),
                    "capacity": cap,
                    "client_decode_count": len(active),
                    "client_decode_logical_context_upper": sum(
                        r["input"] + r["output"] + 256 for r in active
                    ),
                    "routed_not_ready_count": len(routed),
                }
            )
    migrations = 0
    conv = defaultdict(list)
    for r in joined:
        conv[r["conversation"]].append(r)
    for rows in conv.values():
        rows.sort(key=lambda r: r["turn"])
        migrations += sum(a["pa"] != b["pa"] for a, b in zip(rows, rows[1:]))
    telemetry = defaultdict(list)
    for s in (PAP / "aiperf/profile_gpu_telemetry.jsonl").read_text().splitlines():
        row = json.loads(s)
        telemetry[row["gpu_index"]].append(row["telemetry_data"])
    gpu = {
        str(i): {
            k: mean(r[k] for r in rows if k in r)
            for k in ("gpu_utilization", "tensor_active", "dram_active")
            if any(k in r for r in rows)
        }
        for i, rows in telemetry.items()
    }
    stats = json.loads((PAP / "attention_fast_path_stats.json").read_text())
    capacities = [
        {
            "pa": r["attention_index"],
            **{k: v for k, v in r["stats"].items() if k.startswith("decode_capacity")},
            "mean_batch": r["stats"]["offload_exec_peer_rows"]
            / r["stats"]["offload_exec_peer_batches"],
        }
        for r in stats["instances"]
    ]
    return {
        "pap_run": str(PAP),
        "dp_run": str(DP),
        "joined_requests": len(joined),
        "by_turn": groups,
        "cross_turn_pa_changes": migrations,
        "cross_turn_transitions": 120,
        "warnings": dict(warnings),
        "gpu_means": gpu,
        "attention": capacities,
        "samples": samples,
        "requests": joined,
    }


if __name__ == "__main__":
    result = analyze()
    output = Path(__file__).with_name("analysis.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"samples", "requests", "attention"}
            },
            indent=2,
        )
    )
