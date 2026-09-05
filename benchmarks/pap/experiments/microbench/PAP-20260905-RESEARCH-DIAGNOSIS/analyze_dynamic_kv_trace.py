# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlate PAP Decode lease growth with exact Projection step cadence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _summary(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.to(torch.float64).flatten()
    if values.numel() == 0:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p99_ms": None}
    return {
        "count": values.numel(),
        "mean_ms": values.mean().item() / 1e6,
        "p50_ms": values.quantile(0.50).item() / 1e6,
        "p99_ms": values.quantile(0.99).item() / 1e6,
        "max_ms": values.max().item() / 1e6,
    }


def analyze(path: Path, *, block_size: int) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step_ids = payload["step_id"].to(torch.int64)
    if not step_ids.diff().eq(1).all():
        raise ValueError("exact cadence analysis requires consecutive steps")

    next_dispatch = payload["projection_next_dispatch_done_ns"].to(torch.int64)
    dispatch = torch.cat((next_dispatch[:-1, -1:], next_dispatch[1:, :-1]), dim=1)
    gather = payload["projection_gather_done_ns"][1:].to(torch.int64)
    wait_ns = gather - dispatch
    projection_ns = next_dispatch[1:] - gather
    cycle_ns = next_dispatch[1:, -1] - dispatch[:, 0]
    if not wait_ns.gt(0).all() or not projection_ns.gt(0).all():
        raise ValueError("trace contains a nonpositive adjacent interval")
    if not torch.equal((wait_ns + projection_ns).sum(dim=1), cycle_ns):
        raise ValueError("adjacent intervals do not telescope to step cadence")

    request_ids = payload["request_ids"]
    leased_counts = payload["request_leased_block_counts"].to(torch.int64)
    prefix_lens = payload["prefix_lens"].to(torch.int64)
    growth = torch.zeros(len(step_ids), dtype=torch.bool)
    growth_blocks = torch.zeros(len(step_ids), dtype=torch.int64)
    growth_pas: list[list[int]] = [[] for _ in step_ids]
    first_observation_growths = 0
    previous: dict[str, int] = {}
    for step_index, peers in enumerate(request_ids):
        current: dict[str, int] = {}
        for pa_index, requests in enumerate(peers):
            for row_index, request_id in enumerate(requests):
                count = int(leased_counts[step_index, pa_index, row_index])
                old = previous.get(request_id)
                if old is None and step_index > 0:
                    prefix = int(prefix_lens[step_index, pa_index, row_index])
                    initial_blocks = (prefix + block_size - 1) // block_size
                    if count > initial_blocks:
                        old = initial_blocks
                        first_observation_growths += 1
                if old is not None and count > old:
                    growth[step_index] = True
                    growth_blocks[step_index] += count - old
                    if pa_index not in growth_pas[step_index]:
                        growth_pas[step_index].append(pa_index)
                current[request_id] = count
        previous.update(current)

    cycle_growth = growth[1:]
    layer0_pa_ns = payload["latency_ns"][1:, 0].max(dim=1).values
    attention_sum_ns = (
        payload["attention_kernel_latency_ns"][1:].max(dim=2).values.sum(dim=1)
    )
    single_request = payload["request_count"].eq(1)
    aliased = single_request & payload["unique_block_count"].lt(
        payload["request_block_counts"][:, :, 0]
    )
    top_indices = cycle_ns.argsort(descending=True)[:20]
    return {
        "source": str(path),
        "step_range": [int(step_ids[0]), int(step_ids[-1])],
        "exact_cycles": cycle_ns.numel(),
        "growth_detection": (
            "A growth is a request whose leased-block vector is longer than its "
            "previous observation in this consecutive window. A request first "
            "appearing after row zero is also a growth when it already exceeds "
            "the prompt block count."
        ),
        "block_size_tokens": block_size,
        "growth_steps": int(cycle_growth.sum()),
        "growth_blocks": int(growth_blocks[1:].sum()),
        "first_observation_growths": first_observation_growths,
        "single_request_cells": int(single_request.sum()),
        "single_request_alias_cells": int(aliased.sum()),
        "all_steps": {
            "exact_cycle": _summary(cycle_ns),
            "dispatch_to_gather_sum": _summary(wait_ns.sum(dim=1)),
            "gather_to_next_dispatch_sum": _summary(projection_ns.sum(dim=1)),
            "layer0_slowest_pa": _summary(layer0_pa_ns),
            "attention_kernel_max_sum": _summary(attention_sum_ns),
        },
        "growth_steps_only": {
            "exact_cycle": _summary(cycle_ns[cycle_growth]),
            "dispatch_to_gather_sum": _summary(wait_ns.sum(dim=1)[cycle_growth]),
            "gather_to_next_dispatch_sum": _summary(
                projection_ns.sum(dim=1)[cycle_growth]
            ),
            "layer0_slowest_pa": _summary(layer0_pa_ns[cycle_growth]),
            "attention_kernel_max_sum": _summary(attention_sum_ns[cycle_growth]),
        },
        "ordinary_steps_only": {
            "exact_cycle": _summary(cycle_ns[~cycle_growth]),
            "dispatch_to_gather_sum": _summary(wait_ns.sum(dim=1)[~cycle_growth]),
            "gather_to_next_dispatch_sum": _summary(
                projection_ns.sum(dim=1)[~cycle_growth]
            ),
            "layer0_slowest_pa": _summary(layer0_pa_ns[~cycle_growth]),
            "attention_kernel_max_sum": _summary(attention_sum_ns[~cycle_growth]),
        },
        "top_cycles": [
            {
                "step_id": int(step_ids[index + 1]),
                "cycle_ms": cycle_ns[index].item() / 1e6,
                "growth": bool(cycle_growth[index]),
                "growth_blocks": int(growth_blocks[index + 1]),
                "growth_pas": growth_pas[index + 1],
                "layer0_slowest_pa_ms": layer0_pa_ns[index].item() / 1e6,
            }
            for index in top_indices
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(analyze(args.trace, block_size=args.block_size), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
