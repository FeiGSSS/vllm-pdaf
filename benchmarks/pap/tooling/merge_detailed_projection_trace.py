# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Join Projection and PA-local Attention traces by each PA's epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


class IncompleteTraceWindow(ValueError):
    """The rolling trace files do not yet share a long enough complete window."""


_HOST_PHASE_NAMES = (
    "control_wait_ns",
    "control_decode_ns",
    "context_prepare_ns",
    "graph_lookup_ns",
    "graph_replay_submit_ns",
)


def _summary_us(values_ns: torch.Tensor) -> dict[str, float]:
    values = values_ns.to(torch.float64).flatten() / 1000
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _summary_ms(values_ns: torch.Tensor) -> dict[str, float]:
    values = values_ns.to(torch.float64).flatten() / 1_000_000
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p99": float(torch.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.to(torch.float64).flatten()
    return {
        "mean": float(values.mean()),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "p99": float(torch.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def merge(projection_path: Path) -> dict[str, object]:
    """Return one payload containing all three aligned latency tensors."""
    projection = torch.load(
        projection_path,
        map_location="cpu",
        weights_only=False,
    )
    pa_latency_ns = projection["latency_ns"].to(torch.int64)
    projection_latency_ns = projection["projection_latency_ns"].to(torch.int64)
    peer_epochs = projection["peer_epoch"].to(torch.int64)
    step_count, layer_count, pa_count = pa_latency_ns.shape
    if tuple(projection_latency_ns.shape) != (step_count, layer_count):
        raise ValueError("Projection latency shape does not match the PA trace")
    if tuple(peer_epochs.shape) != (step_count, pa_count):
        raise ValueError("peer_epoch shape does not match the PA trace")

    attention_paths: list[str] = []
    attention_payloads: list[dict[str, object]] = []
    common_steps = torch.ones(step_count, dtype=torch.bool)
    for pa_index in range(pa_count):
        attention_path = projection_path.with_name(
            f"attention_pa_{pa_index}_kernel_trace.pt"
        )
        attention_paths.append(str(attention_path))
        attention = torch.load(
            attention_path,
            map_location="cpu",
            weights_only=False,
        )
        attention_payloads.append(attention)
        local_epochs = attention["local_epoch"].to(torch.int64)
        target_epochs = peer_epochs[:, pa_index].contiguous()
        rows = torch.searchsorted(local_epochs, target_epochs)
        available = rows.lt(local_epochs.numel())
        matched = torch.zeros_like(available)
        matched[available] = local_epochs[rows[available]].eq(target_epochs[available])
        common_steps &= matched

    requested_steps = int(projection["metadata"].get("requested_samples", 512))
    candidates = common_steps.nonzero(as_tuple=False).flatten()
    if candidates.numel() < requested_steps:
        raise IncompleteTraceWindow(
            f"only {candidates.numel()} globally aligned steps are available; "
            f"need {requested_steps}"
        )
    candidate_step_ids = projection["step_id"][candidates]
    breaks = candidate_step_ids[1:].sub(candidate_step_ids[:-1]).ne(1)
    run_starts = torch.cat(
        (torch.zeros(1, dtype=torch.int64), breaks.nonzero().flatten().add(1))
    )
    run_ends = torch.cat(
        (breaks.nonzero().flatten().add(1), torch.tensor([candidates.numel()]))
    )
    selected = None
    for run_start, run_end in zip(run_starts.tolist(), run_ends.tolist()):
        if run_end - run_start >= requested_steps:
            selected = candidates[run_end - requested_steps : run_end]
    if selected is None:
        longest = int((run_ends - run_starts).max())
        raise IncompleteTraceWindow(
            f"longest contiguous aligned window has {longest} steps; "
            f"need {requested_steps}"
        )
    payload = {
        key: (
            value[selected]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == step_count
            else value
        )
        for key, value in projection.items()
    }
    pa_latency_ns = payload["latency_ns"].to(torch.int64)
    projection_latency_ns = payload["projection_latency_ns"].to(torch.int64)
    peer_epochs = payload["peer_epoch"].to(torch.int64)
    attention_latency_ns = torch.empty_like(pa_latency_ns)
    attention_kernel_start_ns = torch.empty_like(pa_latency_ns)
    attention_graph_start_ns = torch.empty(
        (pa_latency_ns.shape[0], pa_count),
        dtype=torch.int64,
    )
    attention_replay_start_ns = torch.empty_like(attention_graph_start_ns)
    aligned_rows: list[torch.Tensor] = []
    max_requests = 0
    for pa_index, attention in enumerate(attention_payloads):
        local_epochs = attention["local_epoch"].to(torch.int64)
        target_epochs = peer_epochs[:, pa_index].contiguous()
        rows = torch.searchsorted(local_epochs, target_epochs)
        aligned_rows.append(rows)
        attention_latency_ns[:, :, pa_index] = attention["latency_ns"][rows]
        attention_kernel_start_ns[:, :, pa_index] = attention["start_ns"][rows]
        attention_graph_start_ns[:, pa_index] = attention["graph_start_ns"][rows]
        attention_replay_start_ns[:, pa_index] = attention["replay_start_ns"][rows]
        max_requests = max(
            max_requests,
            int(attention["request_count"][rows].max()),
        )

    if not bool(attention_latency_ns.gt(0).all()):
        raise ValueError("Attention trace contains non-positive latency")
    if not bool(attention_graph_start_ns.gt(0).all()):
        raise ValueError("Attention trace contains a non-positive Graph start")
    if not bool(projection_latency_ns.gt(0).all()):
        raise ValueError("Projection trace contains non-positive latency")

    step_count = pa_latency_ns.shape[0]
    context_shape = (step_count, pa_count)
    request_shape = (step_count, pa_count, max_requests)
    request_count = torch.zeros(context_shape, dtype=torch.int32)
    seq_lens = torch.zeros(request_shape, dtype=torch.int32)
    prefix_lens = torch.zeros_like(seq_lens)
    request_block_counts = torch.zeros_like(seq_lens)
    request_leased_block_counts = torch.zeros_like(seq_lens)
    logical_context_tokens = torch.zeros(context_shape, dtype=torch.int64)
    unique_context_tokens = torch.zeros_like(logical_context_tokens)
    block_reference_count = torch.zeros(context_shape, dtype=torch.int32)
    unique_block_count = torch.zeros_like(block_reference_count)
    unique_leased_block_count = torch.zeros_like(block_reference_count)
    common_prefix_blocks = torch.zeros_like(block_reference_count)
    common_prefix_tokens = torch.zeros(context_shape, dtype=torch.int64)
    common_prefix_savings_tokens = torch.zeros_like(common_prefix_tokens)
    attention_is_pat = torch.zeros(context_shape, dtype=torch.bool)
    attention_reused_kv_tokens = torch.zeros(context_shape, dtype=torch.int64)
    host_phases = {
        name: torch.zeros(context_shape, dtype=torch.int64)
        for name in _HOST_PHASE_NAMES
    }
    request_ids = [[[] for _ in range(pa_count)] for _ in range(step_count)]
    for pa_index, (attention, rows) in enumerate(zip(attention_payloads, aligned_rows)):
        width = min(max_requests, int(attention["seq_lens"].shape[1]))
        request_count[:, pa_index] = attention["request_count"][rows]
        seq_lens[:, pa_index, :width] = attention["seq_lens"][rows, :width]
        prefix_lens[:, pa_index, :width] = attention["prefix_lens"][rows, :width]
        request_block_counts[:, pa_index, :width] = attention["request_block_counts"][
            rows, :width
        ]
        request_leased_block_counts[:, pa_index, :width] = attention[
            "request_leased_block_counts"
        ][rows, :width]
        logical_context_tokens[:, pa_index] = attention["logical_context_tokens"][rows]
        unique_context_tokens[:, pa_index] = attention["unique_context_tokens"][rows]
        block_reference_count[:, pa_index] = attention["block_reference_count"][rows]
        unique_block_count[:, pa_index] = attention["unique_block_count"][rows]
        unique_leased_block_count[:, pa_index] = attention["unique_leased_block_count"][
            rows
        ]
        common_prefix_blocks[:, pa_index] = attention["common_prefix_blocks"][rows]
        common_prefix_tokens[:, pa_index] = attention["common_prefix_tokens"][rows]
        common_prefix_savings_tokens[:, pa_index] = attention[
            "common_prefix_savings_tokens"
        ][rows]
        attention_reused_kv_tokens[:, pa_index] = attention[
            "attention_reused_kv_tokens"
        ][rows]
        for name in _HOST_PHASE_NAMES:
            host_phases[name][:, pa_index] = attention[name][rows]
        source_backends = attention["attention_backend"]
        attention_is_pat[:, pa_index] = torch.tensor(
            [source_backends[row] == "pat" for row in rows.tolist()],
            dtype=torch.bool,
        )
        source_request_ids = attention["request_ids"]
        for step_index, row in enumerate(rows.tolist()):
            request_ids[step_index][pa_index] = list(source_request_ids[row])

    if not torch.equal(request_count, payload["route_counts"].to(torch.int32)):
        raise ValueError("Attention request counts do not match Projection routes")
    request_mask = torch.arange(max_requests).view(1, 1, -1) < request_count.unsqueeze(
        2
    )
    if not bool(seq_lens[request_mask].gt(0).all()):
        raise ValueError("Attention request context contains non-positive lengths")
    if not bool(seq_lens[~request_mask].eq(0).all()):
        raise ValueError("Attention request context padding is nonzero")
    if not torch.equal((seq_lens * request_mask).sum(dim=2), logical_context_tokens):
        raise ValueError("Attention logical context total does not match seq_lens")
    if not bool(unique_context_tokens.le(logical_context_tokens).all()):
        raise ValueError("Attention unique context exceeds logical context")
    if not torch.equal(
        ((seq_lens + 15) // 16 * request_mask).sum(dim=2),
        block_reference_count,
    ):
        raise ValueError("Attention block references do not match seq_lens")
    if not torch.equal(common_prefix_tokens, common_prefix_blocks.to(torch.int64) * 16):
        raise ValueError("Attention common prefix is not block aligned")
    shared_context_tokens = logical_context_tokens - unique_context_tokens
    if not bool(common_prefix_savings_tokens.le(shared_context_tokens).all()):
        raise ValueError("Attention common prefix savings exceed total sharing")

    payload["attention_kernel_latency_ns"] = attention_latency_ns
    payload["attention_kernel_start_ns"] = attention_kernel_start_ns
    payload["attention_graph_start_ns"] = attention_graph_start_ns
    payload["attention_replay_start_ns"] = attention_replay_start_ns
    payload.update(host_phases)
    payload["request_ids"] = request_ids
    payload["request_count"] = request_count
    payload["seq_lens"] = seq_lens
    payload["prefix_lens"] = prefix_lens
    payload["request_block_counts"] = request_block_counts
    payload["request_leased_block_counts"] = request_leased_block_counts
    payload["logical_context_tokens"] = logical_context_tokens
    payload["unique_context_tokens"] = unique_context_tokens
    payload["shared_context_tokens"] = shared_context_tokens
    payload["block_reference_count"] = block_reference_count
    payload["unique_block_count"] = unique_block_count
    payload["unique_leased_block_count"] = unique_leased_block_count
    payload["common_prefix_blocks"] = common_prefix_blocks
    payload["common_prefix_tokens"] = common_prefix_tokens
    payload["common_prefix_savings_tokens"] = common_prefix_savings_tokens
    payload["attention_is_pat"] = attention_is_pat
    payload["attention_reused_kv_tokens"] = attention_reused_kv_tokens
    payload["metadata"] = dict(projection["metadata"])
    payload["metadata"].update(
        {
            "attention_kernel_shape": list(attention_latency_ns.shape),
            "attention_kernel_start_shape": list(attention_kernel_start_ns.shape),
            "attention_graph_start_shape": list(attention_graph_start_ns.shape),
            "attention_replay_start_shape": list(attention_replay_start_ns.shape),
            "projection_latency_shape": list(projection_latency_ns.shape),
            "seq_lens_shape": list(seq_lens.shape),
            "context_shape": list(logical_context_tokens.shape),
            "attention_sources": attention_paths,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("projection_trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.projection_trace.with_name("detailed_trace.pt")
    payload = merge(args.projection_trace)
    torch.save(payload, output)
    pa_latency_ns = payload["latency_ns"].to(torch.float64)
    attention_latency_ns = payload["attention_kernel_latency_ns"].to(torch.float64)
    attention_kernel_start_ns = payload["attention_kernel_start_ns"].to(torch.float64)
    attention_graph_start_ns = payload["attention_graph_start_ns"].to(torch.float64)
    attention_replay_start_ns = payload["attention_replay_start_ns"].to(torch.float64)
    projection_latency_ns = payload["projection_latency_ns"].to(torch.float64)
    logical_context_tokens = payload["logical_context_tokens"].to(torch.float64)
    unique_context_tokens = payload["unique_context_tokens"].to(torch.float64)
    request_count = payload["request_count"].to(torch.float64)
    common_prefix_tokens = payload["common_prefix_tokens"].to(torch.float64)
    common_prefix_savings_tokens = payload["common_prefix_savings_tokens"].to(
        torch.float64
    )
    attention_is_pat = payload["attention_is_pat"]
    attention_reused_kv_tokens = payload["attention_reused_kv_tokens"].to(torch.float64)
    shared_context_tokens = logical_context_tokens - unique_context_tokens
    common_prefix_coverage = torch.where(
        shared_context_tokens.gt(0),
        common_prefix_savings_tokens / shared_context_tokens,
        torch.zeros_like(shared_context_tokens),
    )
    logical_kv_gbps = logical_context_tokens[:, None, :] * 4096 / attention_latency_ns
    unique_kv_gbps = unique_context_tokens[:, None, :] * 4096 / attention_latency_ns
    sharing_amplification = logical_context_tokens / unique_context_tokens
    layer_mean_attention_ns = attention_latency_ns.mean(dim=1)
    context_latency_correlation = torch.corrcoef(
        torch.stack(
            (
                logical_context_tokens.flatten(),
                layer_mean_attention_ns.flatten(),
            )
        )
    )[0, 1]
    pa_barrier_ns = (pa_latency_ns.max(dim=2).values - pa_latency_ns.mean(dim=2)).sum(
        dim=1
    )
    attention_barrier_ns = (
        attention_latency_ns.max(dim=2).values - attention_latency_ns.mean(dim=2)
    ).sum(dim=1)
    pa_layer_averaged_ns = pa_latency_ns.mean(dim=1)
    layer_averaged_barrier_ns = pa_layer_averaged_ns.max(
        dim=1
    ).values - pa_layer_averaged_ns.mean(dim=1)
    approximate_cycle_ns = pa_latency_ns.max(dim=2).values.sum(
        dim=1
    ) + projection_latency_ns.sum(dim=1)
    layer0_kernel_start_ns = attention_kernel_start_ns[:, 0, :]
    layer0_replay_to_graph_ns = attention_graph_start_ns - attention_replay_start_ns
    layer0_graph_to_kernel_ns = layer0_kernel_start_ns - attention_graph_start_ns
    layer0_kernel_ns = attention_latency_ns[:, 0, :]
    layer0_outside_graph_and_kernel_ns = (
        pa_latency_ns[:, 0, :] - layer0_graph_to_kernel_ns - layer0_kernel_ns
    )
    layer0_slowest_pa = pa_latency_ns[:, 0, :].argmax(dim=1)
    row_indices = torch.arange(pa_latency_ns.shape[0])

    def slowest(values: torch.Tensor) -> torch.Tensor:
        return values[row_indices, layer0_slowest_pa]

    summary = {
        "output": str(output),
        "step_id": {
            "first": int(payload["step_id"][0]),
            "last": int(payload["step_id"][-1]),
        },
        "pa_end_to_end_shape": list(payload["latency_ns"].shape),
        "attention_kernel_shape": list(payload["attention_kernel_latency_ns"].shape),
        "projection_shape": list(payload["projection_latency_ns"].shape),
        "pa_end_to_end_latency_us": _summary_us(payload["latency_ns"]),
        "attention_kernel_latency_us": _summary_us(attention_latency_ns),
        "projection_latency_us": _summary_us(projection_latency_ns),
        "request_count": _summary(request_count),
        "logical_context_tokens": _summary(logical_context_tokens),
        "unique_context_tokens": _summary(unique_context_tokens),
        "shared_context_tokens": _summary(
            logical_context_tokens - unique_context_tokens
        ),
        "common_prefix_tokens": _summary(common_prefix_tokens),
        "common_prefix_shared_savings_coverage": _summary(common_prefix_coverage),
        "pat_step_fraction": float(attention_is_pat.to(torch.float64).mean()),
        "attention_reused_kv_tokens": _summary(attention_reused_kv_tokens),
        "logical_over_unique_context": _summary(sharing_amplification),
        "logical_kv_bandwidth_gbps": _summary(logical_kv_gbps),
        "unique_kv_bandwidth_gbps": _summary(unique_kv_gbps),
        "logical_context_vs_layer_mean_attention_correlation": float(
            context_latency_correlation
        ),
        "non_attention_pa_path_latency_us": _summary_us(
            pa_latency_ns - attention_latency_ns
        ),
        "pa_barrier_imbalance_tbt_ms": _summary_ms(pa_barrier_ns),
        "attention_kernel_imbalance_tbt_ms": _summary_ms(attention_barrier_ns),
        "layer_averaged_pa_barrier_us": _summary_us(layer_averaged_barrier_ns),
        "layer_averaged_pa_barrier_times_layers_ms": _summary_ms(
            layer_averaged_barrier_ns * pa_latency_ns.shape[1]
        ),
        "projection_sum_per_step_ms": _summary_ms(projection_latency_ns.sum(dim=1)),
        "approximate_traced_cycle_ms": _summary_ms(approximate_cycle_ns),
        "layer0_boundary_all_pa_us": {
            "gpu_replay_marker_to_graph_start": _summary_us(layer0_replay_to_graph_ns),
            "attention_graph_start_to_kernel_start": _summary_us(
                layer0_graph_to_kernel_ns
            ),
            "attention_kernel": _summary_us(layer0_kernel_ns),
            "outside_attention_graph_wait_and_kernel": _summary_us(
                layer0_outside_graph_and_kernel_ns
            ),
        },
        "layer0_boundary_slowest_pa_us": {
            "gpu_replay_marker_to_graph_start": _summary_us(
                slowest(layer0_replay_to_graph_ns)
            ),
            "attention_graph_start_to_kernel_start": _summary_us(
                slowest(layer0_graph_to_kernel_ns)
            ),
            "attention_kernel": _summary_us(slowest(layer0_kernel_ns)),
            "outside_attention_graph_wait_and_kernel": _summary_us(
                slowest(layer0_outside_graph_and_kernel_ns)
            ),
            "projection_dispatch_to_return": _summary_us(
                slowest(pa_latency_ns[:, 0, :])
            ),
        },
        "attention_host_phase_all_pa_us": {
            name.removesuffix("_ns"): _summary_us(payload[name])
            for name in _HOST_PHASE_NAMES
        },
        "attention_host_phase_slowest_pa_us": {
            name.removesuffix("_ns"): _summary_us(slowest(payload[name]))
            for name in _HOST_PHASE_NAMES
        },
        "per_layer_mean_us": {
            "pa_end_to_end": (pa_latency_ns.mean(dim=(0, 2)).div(1000).tolist()),
            "attention_kernel": (
                attention_latency_ns.mean(dim=(0, 2)).div(1000).tolist()
            ),
            "projection": projection_latency_ns.mean(dim=0).div(1000).tolist(),
        },
        "per_pa_mean": {
            "request_count": request_count.mean(dim=0).tolist(),
            "logical_context_tokens": logical_context_tokens.mean(dim=0).tolist(),
            "unique_context_tokens": unique_context_tokens.mean(dim=0).tolist(),
            "logical_over_unique_context": sharing_amplification.mean(dim=0).tolist(),
            "attention_kernel_us": layer_mean_attention_ns.mean(dim=0)
            .div(1000)
            .tolist(),
            "logical_kv_bandwidth_gbps": logical_kv_gbps.mean(dim=(0, 1)).tolist(),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
