# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calculate effective KV bandwidth from current PA-local kernel markers."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--source",
    type=Path,
    default=root.parents[1] / "e2e/PAP-20260907-BOTTLENECK/runs/diagnostic_80_12_v2",
)
parser.add_argument("--label", choices=("original", "fixed"), default="original")
args = parser.parse_args()
source = args.source.resolve()
model = json.loads(Path("/data/ssd1/llm-models/Qwen3-8B/config.json").read_text())
heads = model["num_key_value_heads"]
dim = model.get("head_dim", model["hidden_size"] // model["num_attention_heads"])
layers = model["num_hidden_layers"]
bytes_per_token = 2 * heads * dim * 2
assert (heads, dim, layers, bytes_per_token) == (8, 128, 36, 4096)
groups = defaultdict(list)
manifest = []
for pa in range(7):
    path = source / f"attention_pa_{pa}_kernel_trace.pt"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest.append({"path": str(path), "sha256": digest})
    data = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(data["seq_lens"].sum(1), data["logical_context_tokens"])
    assert data["latency_ns"].shape == (len(data["local_epoch"]), layers)
    assert (data["latency_ns"] > 0).all()
    single = data["request_count"] == 1
    assert torch.equal(
        data["request_block_counts"][single].sum(1), data["unique_block_count"][single]
    )
    for i, backend in enumerate(data["attention_backend"]):
        tokens = int(data["logical_context_tokens"][i])
        unique = int(data["unique_context_tokens"][i])
        ns = float(data["latency_ns"][i].double().sum())
        row = {
            "pa": pa,
            "epoch": int(data["local_epoch"][i]),
            "requests": int(data["request_count"][i]),
            "logical_tokens": tokens,
            "unique_tokens": unique,
            "layer_mean_us": ns / layers / 1000,
            "kernel_sum_ns": ns,
            "logical_bytes": tokens * bytes_per_token * layers,
            "unique_bytes": unique * bytes_per_token * layers,
        }
        groups[backend].append(row)
        groups[f"{backend}/B{row['requests']}"].append(row)
        for limit in (16384, 32768, 65536, 131072, 262144):
            if tokens <= limit:
                groups[f"{backend}/total_tokens_le_{limit}"].append(row)
                break


def summarize(rows):
    time = sum(row["kernel_sum_ns"] for row in rows)
    per_step = torch.tensor([row["layer_mean_us"] for row in rows]).double()
    return {
        "pa_steps": len(rows),
        "layer_invocations": len(rows) * layers,
        "mean_layer_us": time / len(rows) / layers / 1000,
        "p95_layer_mean_us": float(torch.quantile(per_step, 0.95)),
        "mean_logical_tokens": sum(r["logical_tokens"] for r in rows) / len(rows),
        "logical_effective_GBs": sum(r["logical_bytes"] for r in rows) / time,
        "unique_payload_GBs": sum(r["unique_bytes"] for r in rows) / time,
    }


result = {
    "bytes_per_token_per_layer": bytes_per_token,
    "layers": layers,
    "units": "decimal GB/s; bytes/ns. Ratio of summed bytes to summed time.",
    "scope": "PA-local kernel intervals, excludes KV append and transport; "
    "PA times pooled as service samples, not global critical path.",
    "not_hardware_bandwidth": True,
    "source_manifest": manifest,
    "groups": {key: summarize(rows) for key, rows in groups.items()},
}
(root / f"{args.label}_trace_bandwidth.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
with (root / f"{args.label}_pa_step_samples.jsonl").open("w") as stream:
    for backend in ("triton", "pat"):
        for row in groups[backend]:
            stream.write(json.dumps({"backend": backend, **row}) + "\n")
print(json.dumps({key: result["groups"][key] for key in ("triton", "pat")}, indent=2))
