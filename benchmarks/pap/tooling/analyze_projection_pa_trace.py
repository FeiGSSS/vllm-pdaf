# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and summarize a Projection-side PAP layer trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    flattened = values.flatten().to(torch.float64)
    return {
        "mean": float(flattened.mean()),
        "p50": float(torch.quantile(flattened, 0.50)),
        "p90": float(torch.quantile(flattened, 0.90)),
        "p99": float(torch.quantile(flattened, 0.99)),
        "max": float(flattened.max()),
    }


def analyze(path: Path) -> dict[str, object]:
    """Load and validate one trace payload."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    latency_us = payload["latency_ns"].to(torch.float64) / 1000
    step_ids = payload["step_id"]
    route_counts = payload["route_counts"]
    if latency_us.ndim != 3:
        raise ValueError("latency_ns must have [step, layer, PA] shape")
    if latency_us.shape[0] < 2 or not torch.equal(
        step_ids[1:] - step_ids[:-1],
        torch.ones_like(step_ids[1:]),
    ):
        raise ValueError("trace step IDs are not contiguous")
    if not bool(latency_us.gt(0).all()):
        raise ValueError("trace contains non-positive latency")

    barrier_delta_us = latency_us.max(dim=2).values - latency_us.mean(dim=2)
    estimated_tbt_loss_ms = barrier_delta_us.sum(dim=1) / 1000
    stable_layers = latency_us[:, 1:, :]
    stable_layer_cv = stable_layers.std(dim=1) / stable_layers.mean(dim=1)
    stable_layer_range_us = (
        stable_layers.max(dim=1).values - stable_layers.min(dim=1).values
    )
    slowest_pa = latency_us.argmax(dim=2)
    return {
        "shape": list(latency_us.shape),
        "step_id": {
            "first": int(step_ids[0]),
            "last": int(step_ids[-1]),
            "contiguous": True,
        },
        "route_counts": {
            "min_per_pa": route_counts.min(dim=0).values.tolist(),
            "max_per_pa": route_counts.max(dim=0).values.tolist(),
            "mean_per_pa": route_counts.to(torch.float64).mean(dim=0).tolist(),
        },
        "latency_us": _quantiles(latency_us),
        "per_pa_mean_us": latency_us.mean(dim=(0, 1)).tolist(),
        "per_layer_mean_us": latency_us.mean(dim=(0, 2)).tolist(),
        "layers_1_to_end_invariance": {
            "cv": _quantiles(stable_layer_cv),
            "range_us": _quantiles(stable_layer_range_us),
        },
        "slowest_pa_fraction": (
            torch.bincount(slowest_pa.flatten(), minlength=latency_us.shape[2])
            .to(torch.float64)
            .div(slowest_pa.numel())
            .tolist()
        ),
        "barrier_imbalance": {
            "layer_max_minus_mean_us": _quantiles(barrier_delta_us),
            "estimated_tbt_loss_ms": _quantiles(estimated_tbt_loss_ms),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.trace.with_suffix(".analysis.json")
    output.write_text(json.dumps(analyze(args.trace), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
