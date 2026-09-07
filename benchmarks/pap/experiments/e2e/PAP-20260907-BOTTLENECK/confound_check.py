# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check whether the SM comparison preserved the observed Decode workload."""

import json
from pathlib import Path

root = Path(__file__).resolve().parent
result = {}
for label, names in {
    "80/12": ("baseline_r1", "baseline_r2"),
    "72/20": ("attention20_r1", "attention20_r2"),
}.items():
    runs = []
    for name in names:
        run = root / "runs" / name
        profile = json.loads((run / "aiperf/profile.json").read_text())
        records = [
            json.loads(line)
            for line in (run / "aiperf/profile.jsonl").read_text().splitlines()
        ]
        assert len(records) == 180
        assert all(not row["metadata"]["was_cancelled"] for row in records)
        duration = profile["benchmark_duration"]["avg"]
        generation = sum(
            (
                row["metrics"]["request_latency"]["value"]
                - row["metrics"]["time_to_first_token"]["value"]
            )
            / 1000
            for row in records
        )
        stats = [
            json.loads((run / f"attention_fast_path_stats_{pa}.json").read_text())
            for pa in range(7)
        ]
        batches = sum(row["offload_exec_peer_batches"] for row in stats)
        rows = sum(row["offload_exec_peer_rows"] for row in stats)
        assert duration > 0 and generation > 0 and batches > 0 and rows > 0
        runs.append(
            {
                "run": name,
                "duration_s": duration,
                "generation_request_seconds": generation,
                "pa_step_count": batches,
                "pa_row_count": rows,
                "client_generation_population_proxy": generation / duration,
                "mean_rows_per_active_pa_step": rows / batches,
            }
        )
    result[label] = {
        "runs": runs,
        "pooled_population_proxy": (
            sum(row["generation_request_seconds"] for row in runs)
            / sum(row["duration_s"] for row in runs)
        ),
        "pooled_rows_per_active_pa_step": (
            sum(row["pa_row_count"] for row in runs)
            / sum(row["pa_step_count"] for row in runs)
        ),
    }
(root / "decode_load_confound.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
