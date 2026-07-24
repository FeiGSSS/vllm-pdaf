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

## C32 full-testbed extension

The same four-GPU topology was subsequently tested at C32 with the canonical
32-conversation, ten-turn AIPerf workload. The trace-off reference and the
deferred CUDA trace reused the exact same 320-request input file. Both
completed 320/320 requests and 32/32 conversations with zero errors, and both
passed correctness, routing, session-drain, decode-token-join, Projection
scheduling, and static-MPS audits.

| Run | Mean ITL | ITL p50 | ITL p90 | Request throughput |
| --- | ---: | ---: | ---: | ---: |
| Trace off | 52.436 ms | 50.682 ms | 72.565 ms | 4.880 req/s |
| Deferred trace | 56.680 ms | 54.703 ms | 78.964 ms | 4.904 req/s |

The deferred instrumentation adds 8.09% mean-ITL overhead, so only the
trace-off run is a performance reference. The diagnostic run captured 38,088
Projection layer calls and 94,032 PA peer-batches without pending, dropped, or
errored trace records.

Median GPU spans at C32:

| Location | Span | p50 |
| --- | --- | ---: |
| Projection | Entire remote stage | 0.408 ms/layer |
| Projection | Fan-in join after receive streams are armed | 0.459 ms/layer |
| Projection | QKV projection, norm, and RoPE | 0.085 ms/layer |
| Projection | QKV P2P copy | 0.006 ms/PA-layer |
| Projection | Output scatter | 0.005 ms/PA-layer |
| Attention | Wait for next-layer QKV, mean of three PA p50s | 0.714 ms/layer |
| Attention | Paged Attention, mean of three PA p50s | 0.286 ms/layer |
| Attention | Residual step-prepare wait, mean of three PA p50s | 0.005 ms/step |

The main Attention-side idle interval is therefore the wait for the next
layer's QKV. During this interval Projection is executing the current layer's
output projection, residual/norm/MLP path, and the next layer's QKV path. It
is a real PA idle bubble, but it is imposed by the layer dependency rather
than local-fast copying or metadata preparation.

Compared directionally with the shorter C12 trace, mean PA QKV-ready p50 rises
only from 0.659 ms to 0.714 ms (+0.055 ms), while mean paged-Attention p50
rises from 0.122 ms to 0.286 ms (+0.164 ms). The C12 and C32 traces have
different conversation counts and turn depth, so this is not an isolated
concurrency A/B; nevertheless it shows that the C32-specific increase is
dominated by real KV-bandwidth Attention work, not by a new control-plane or
P2P idle gap.

This trace is stage-aggregate evidence rather than a globally aligned GPU
timeline. A future Nsight or bilateral timestamp trace is only necessary if
the fixed 0.714-ms cross-layer dependency must be split further into output
projection, MLP, and next-layer QKV production.

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

The C32 extension also rules out transport and step preparation as the cause
of high-concurrency ITL growth. At C32, paged Attention itself grows much more
than the PA's QKV-ready wait. Removing the remaining PA idle would require
cross-batch work or a deeper model/kernel pipeline; neither is a low-risk
single-batch scheduling optimization.

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
