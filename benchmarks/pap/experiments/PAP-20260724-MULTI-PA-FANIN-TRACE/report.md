# PAP multi-PA fan-in trace

## Question

For one Projection decode batch routed to up to three PA nodes, how much of
the layer latency is normal remote Attention work, and how much is an
avoidable straggler bubble while Projection waits for the slowest PA?

## Setup

- Model/hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs.
- Topology: 3PA1P, local-fast transport, static 72/20-SM PA split.
- Workload: AIPerf C12, 16 conversations, four turns, 64 requests.
- Input: randomized 8,192-token initial and 512-token append means.
- Output: randomized 32-token mean, 16-64-token bounds.
- Delay: 3-second think time and a 1-second tool delay every third turn.
- GPU trace: deferred CUDA events with no hot-path synchronization.
- Correlation trace: batch-key CPU timestamps with per-PA rows and total KV
  sequence lengths. This run is used for attribution, not absolute latency.

Both AIPerf runs completed 64/64 requests and 16/16 conversations with zero
client errors. The CPU trace passed all correctness, routing, session-drain,
decode-token-join, scheduler, and static-MPS audits. The GPU trace captured
25,056 Projection layer calls and 44,280 PA peer-batches with zero pending,
dropped, or errored records.

## GPU critical path

Values are milliseconds per layer. Startup maxima and means are not used
because service warmup produced second-scale outliers.

| Span | Count | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Projection remote stage | 25,056 | 0.177 | 0.372 | 0.826 |
| Multi-PA Projection join wait | 12,600 | 0.255 | 0.376 | 0.680 |
| PA output path ordinal 0 | 12,600 | 0.217 | 0.344 | 0.528 |
| PA output path ordinal 1 | 12,600 | 0.188 | 0.308 | 0.438 |
| PA output path ordinal 2 | 6,624 | 0.280 | 0.362 | 0.604 |
| Output scatter | 31,824 | 0.004 | 0.006 | 0.009 |
| QKV P2P copy | 44,280 | 0.005 | 0.008 | 0.013 |

`projection_join_wait_gpu_ms` is the time for which the Projection stream is
blocked after reaching the fan-in point. It includes useful remote Attention
work that has not completed yet; it is not itself equal to straggler waste.

## PA completion skew

The synchronized CPU correlation trace separates the three PA completion
times for the same layer:

| Metric | p50 | p90 | p99 |
| --- | ---: | ---: | ---: |
| Earliest-to-slowest PA completion skew | 0.048 ms | 0.222 ms | 2.102 ms |
| Mean PA idle time until slowest PA | 0.021 ms | 0.127 ms | 1.307 ms |
| Routed row-count range | 0 | 2 | 3 |
| Routed KV-token range | 5,652 | 15,538 | 23,736 |
| Maximum KV load / mean KV load | 1.105x | 1.454x | 1.532x |

The PA with the largest total routed sequence length was the slowest PA in
73.1% of matched multi-PA layers. Conversation-affinity round robin therefore
balances session counts, but not the actual bandwidth-dominant KV work.

The median 0.255-ms Projection join stall is much larger than the median
0.048-ms completion skew. Most of the join is normal dependency time; the
avoidable fan-in imbalance is smaller, although its p99 tail is material.

## Other bubbles

The Attention-side deferred trace found a larger resource-idle interval:

| PA | QKV-ready wait p50 | paged Attention p50 | step-prepare wait p50 |
| ---: | ---: | ---: | ---: |
| 0 | 0.643 ms | 0.119 ms | 0.002 ms |
| 1 | 0.675 ms | 0.135 ms | 0.004 ms |
| 2 | 0.658 ms | 0.113 ms | 0.002 ms |

After a PA finishes one layer, it waits roughly 0.65 ms for Projection to
finish that layer's output projection and MLP and produce the next layer's
QKV. This is idle PA capacity, but not extra Projection critical-path latency:
the layer-wise dependency prevents the next Attention operation from starting.
Using the idle capacity would require another independent batch or Projection
producer, which conflicts with the current single-Projection-batch policy.

Step preparation is already effectively hidden. The residual compute-stream
wait for the prepared metadata is only 0.002-0.004 ms, so further work there
has negligible upside.

## Decision

Do not pursue another local-fast copy or metadata-overlap optimization:
QKV copy, output scatter, and step-prepare residual waits are all single-digit
microseconds per layer.

The next scheduling-level experiment should preserve conversation affinity
but assign a new conversation using projected KV load and active decode load,
instead of pure round robin. The success criterion is a reduction in PA
completion-skew p90/p99 and multi-PA join p90/p99 without changing the
single-batch execution model.

Persistent kernels would reduce launch/control overhead, but they do not
remove the cross-layer QKV dependency or the bandwidth work measured here.
They remain a separate high-risk kernel project rather than the next PAP
scheduler optimization.

## Validity

This is diagnostic evidence, not a release performance baseline. Deferred
CUDA events and verbose CPU tracing were run separately; the latter
synchronizes kernel timing and intentionally perturbs scheduling. Absolute
AIPerf latency from either trace-on run must not replace the trace-off C12
milestone.
