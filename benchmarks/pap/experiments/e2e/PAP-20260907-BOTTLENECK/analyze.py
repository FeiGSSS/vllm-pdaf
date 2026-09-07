# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute client SLOs and boundary timings from retained PAP evidence."""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import regex as re


def summary(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"count": 0}

    def quantile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": mean(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "min": values[0],
        "max": values[-1],
    }


def client(run):
    run_year = datetime.fromtimestamp(
        json.loads((run / "exit.json").read_text())["start"]
    ).year
    rows = [
        json.loads(line)
        for line in (run / "aiperf/profile.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 180 and all(not r["metadata"]["was_cancelled"] for r in rows)
    profile = json.loads((run / "aiperf/profile.json").read_text())
    duration = profile["benchmark_duration"]["avg"]
    result = {
        "run": run.name,
        "duration_s": duration,
        "throughput_rps": len(rows) / duration,
    }
    for metric in ("time_to_first_token", "inter_token_latency", "request_latency"):
        result[metric] = summary(r["metrics"][metric]["value"] for r in rows)
    result["cache_hit_fraction"] = sum(
        r["metrics"]["usage_prompt_cache_read_tokens"]["value"] for r in rows
    ) / sum(r["metrics"]["input_sequence_length"]["value"] for r in rows)
    result["slo"] = {}
    for name, limits in {"D2": (10000, 75, 20000), "D3": (15000, 50, 20000)}.items():
        good = sum(
            all(
                r["metrics"][k]["value"] <= v
                for k, v in zip(
                    ("time_to_first_token", "inter_token_latency", "request_latency"),
                    limits,
                )
            )
            for r in rows
        )
        result["slo"][name] = {
            "good": good,
            "total": len(rows),
            "attainment": good / len(rows),
            "goodput_rps": good / duration,
            "feasible_95pct": good / len(rows) >= 0.95,
        }
    requests = {r["metadata"]["x_request_id"]: r for r in rows}
    reclaimed = []
    captures = 0
    for path in (run / "service_logs").glob("*.log"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            captures += "PAP Projection whole-step CUDA Graph capture begin" in line
            if "PAP allocator reclaimed released requests:" not in line:
                continue
            stamp = re.search(r"\b(\d\d-\d\d) (\d\d:\d\d:\d\d)", line)
            assert stamp
            timestamp = datetime.fromisoformat(
                f"{run_year}-{stamp[1]}T{stamp[2]}"
            ).timestamp()
            for rid in re.findall(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", line):
                assert rid in requests
                end = requests[rid]["metadata"]["request_end_ns"] / 1e9
                reclaimed.append(
                    {
                        "request_id": rid,
                        "source": str(path.relative_to(run)),
                        "line": number,
                        "client_end_s": end,
                        "reclaim_log_s": timestamp,
                        "lag_lower_s": timestamp - end,
                        "lag_upper_s": timestamp + 1 - end,
                    }
                )
    assert len(reclaimed) == 180 and len({r["request_id"] for r in reclaimed}) == 180
    result["reclaim_lag_upper_s"] = summary(r["lag_upper_s"] for r in reclaimed)
    result["reclaim_lag_lower_s"] = summary(r["lag_lower_s"] for r in reclaimed)
    result["projection_graph_captures"] = captures
    (run / "reclaim_join.json").write_text(json.dumps(reclaimed, indent=2) + "\n")
    stats = [
        json.loads(p.read_text())
        for p in sorted(run.glob("attention_fast_path_stats_[0-6].json"))
    ]
    assert len(stats) == 7
    result["decode_capacity"] = {
        k: sum(s[k] for s in stats)
        for k in (
            "decode_capacity_requests",
            "decode_capacity_waits",
            "decode_capacity_failures",
        )
    }
    lo = min(r["metadata"]["request_start_ns"] for r in rows)
    hi = max(r["metadata"]["request_end_ns"] for r in rows)
    gpu = defaultdict(lambda: defaultdict(list))
    for line in (run / "aiperf/profile_gpu_telemetry.jsonl").read_text().splitlines():
        row = json.loads(line)
        if lo <= row["timestamp_ns"] <= hi:
            for key in ("gpu_utilization", "gpu_power_usage", "mem_utilization"):
                gpu[row["gpu_index"]][key].append(row["telemetry_data"][key])
    assert set(gpu) == set(range(8)), "Incomplete GPU telemetry"
    result["gpu_request_window"] = {
        index: {key: summary(values) for key, values in data.items()}
        for index, data in gpu.items()
    }
    return result, rows


def trace(run):
    import torch

    torch.set_num_threads(1)
    p = torch.load(run / "pa_trace.pt", map_location="cpu", weights_only=True)
    start = p["start_ns"].double().min(2).values
    gather = p["projection_gather_done_ns"].double()
    assert (gather > start).all()
    between = start[:, 1:] - gather[:, :-1]
    assert (between > 0).all()
    consecutive = p["step_id"][1:] == p["step_id"][:-1] + 1
    cycle = (start[1:, 0] - start[:-1, 0])[consecutive] / 1e6
    boundary = (start[1:, 0] - gather[:-1, -1])[consecutive] / 1e6
    assert torch.allclose(
        cycle,
        (gather - start).sum(1)[:-1][consecutive] / 1e6
        + between.sum(1)[:-1][consecutive] / 1e6
        + boundary,
    )
    result = {
        "samples": len(start),
        "scope": (
            "Only steps with all 7 PA routes active and a consecutive valid "
            "next step; not the full run."
        ),
        "pa_path_ms_per_step": summary((gather - start).sum(1) / 1e6),
        "projection_35_interlayer_intervals_ms": summary(between.sum(1) / 1e6),
        "step_boundary_interval_ms": summary(
            p["projection_latency_ns"][:, -1].double() / 1e6
        ),
        "exact_adjacent_step_cadence_ms": summary(cycle),
        "exact_adjacent_step_boundary_ms": summary(boundary),
        "pa_path_ms_by_layer": [float(v) for v in (gather - start).mean(0) / 1e6],
    }
    kernel = torch.empty_like(p["latency_ns"], dtype=torch.float64)
    pa_rows = []
    for pa in range(7):
        a = torch.load(
            run / f"attention_pa_{pa}_kernel_trace.pt",
            map_location="cpu",
            weights_only=True,
        )
        single = a["request_count"] == 1
        assert torch.equal(
            a["request_block_counts"][single].sum(1), a["unique_block_count"][single]
        ), "Active KV alias in a single-request PA step"
        assert torch.equal(
            a["request_leased_block_counts"][single].sum(1),
            a["unique_leased_block_count"][single],
        ), "Lease KV alias in a single-request PA step"
        epochs = {int(v): i for i, v in enumerate(a["local_epoch"])}
        assert len(epochs) == len(a["local_epoch"])
        indices = [epochs[int(v)] for v in p["peer_epoch"][:, pa]]
        assert torch.equal(a["request_count"][indices], p["route_counts"][:, pa])
        kernel[:, :, pa] = a["latency_ns"][indices].double() / 1e6
        pa_rows.append(
            {
                "pa": pa,
                "single_request_alias_checks": int(single.sum()),
                "all_local_steps": len(epochs),
                "matched_steps": len(indices),
                "kernel_ms_per_step": summary(kernel[:, :, pa].sum(1)),
                "request_count": summary(a["request_count"][indices]),
                "logical_context_tokens": summary(a["logical_context_tokens"][indices]),
                "backend_counts": dict(
                    Counter(a["attention_backend"][i] for i in indices)
                ),
                "context_prepare_ms": summary(
                    a["context_prepare_ns"][indices].double() / 1e6
                ),
            }
        )
    result["matched_attention"] = pa_rows
    result["max_pa_kernel_sum_ms"] = summary(kernel.max(2).values.sum(1))
    result["max_minus_mean_kernel_sum_ms"] = summary(
        (kernel.max(2).values - kernel.mean(2)).sum(1)
    )
    result["unattributed_pa_path_ms"] = summary(
        (gather - start).sum(1) / 1e6 - kernel.max(2).values.sum(1)
    )
    result["warning"] = (
        "Subtraction is a localization residual, not transport latency or a "
        "predicted optimization gain. No absolute clocks joined across GPUs."
    )
    return result


def metrics(run):
    latest = {}
    peaks = defaultdict(dict)
    for line in (run / "prefill_metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("lines"):
            latest[row["pa"]] = row
            for metric in row["lines"]:
                if metric.startswith(
                    (
                        "vllm:num_requests_running{",
                        "vllm:num_requests_waiting{",
                        "vllm:kv_cache_usage_perc{",
                    )
                ):
                    key = metric.split("{")[0]
                    value = float(metric.rsplit(" ", 1)[1])
                    peaks[row["pa"]][key] = max(peaks[row["pa"]].get(key, 0), value)
    totals = defaultdict(float)
    for row in latest.values():
        for line in row["lines"]:
            if "_bucket" not in line:
                totals[line.split("{")[0]] += float(line.rsplit(" ", 1)[1])
    assert totals["vllm:request_queue_time_seconds_count"] == 180
    return {
        "request_count": 180,
        "queue_mean_s": totals["vllm:request_queue_time_seconds_sum"] / 180,
        "prefill_residence_mean_s": totals["vllm:request_prefill_time_seconds_sum"]
        / 180,
        "preemptions": totals["vllm:num_preemptions_total"],
        "sampled_peaks_by_pa": dict(peaks),
        "latest_samples": latest,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    reference = None
    results = []
    for run in args.runs:
        assert json.loads((run / "exit.json").read_text())["exit_code"] == 0
        result, rows = client(run)
        lengths = {
            (r["metadata"]["conversation_id"], r["metadata"]["turn_index"]): r[
                "metrics"
            ]["output_sequence_length"]["value"]
            for r in rows
        }
        if reference is None:
            reference = lengths
        assert len(lengths) == 180 and lengths == reference, (
            "Workload output lengths differ"
        )
        if (run / "pa_trace.pt").exists():
            result["trace"] = trace(run)
        if (run / "prefill_metrics.jsonl").exists():
            result["prefill_metrics"] = metrics(run)
        (run / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(
            run.name,
            json.dumps(
                {k: result[k] for k in ("throughput_rps", "slo", "reclaim_lag_upper_s")}
            ),
        )
    (args.runs[0].parent.parent / "comparison.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
