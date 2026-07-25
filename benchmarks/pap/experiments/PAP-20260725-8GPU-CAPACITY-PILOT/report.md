# Eight-GPU AIPerf capacity pilot

> Development evidence only. This report records one repetition per point and
> a targeted PAP trace. It validates the eight-GPU execution paths and narrows
> the next scan, but it is not a three-repetition release claim.

The compact scan requested by this pilot is complete. Use the
[full capacity report](../PAP-20260725-8GPU-CAPACITY-SCAN/report.md) for
PAP/PD/DP comparisons; retain this document for the matched 7PA1P/6PA2P
root-cause trace.

Date: 2026-07-25

## Scope

Every completed point serves the same 128 conversations and five turns per
conversation, for 640 requests. The Qwen3-8B FP16 workload uses eight NVIDIA
L20 GPUs, conversation concurrency 32, eager execution, randomized long input,
randomized 16-64-token output, and deterministic think/tool delays.

The seed-42 dataset has an 8,010.7-token mean initial document and a
1,394.2-token mean append. PAP PA, PD, and fused replicas use
`gpu_memory_utilization=0.90`; Projection uses the automatic checkpoint-weight
budget. The three request-level SLO tiers are:

| Tier | TTFT | ITL | Required good requests |
| --- | ---: | ---: | ---: |
| Strict | 5 s | 50 ms | 95% |
| Standard | 10 s | 75 ms | 95% |
| Relaxed | 20 s | 100 ms | 95% |

PAP and fused-DP results below use clean commit `152f64445`. PD 4P4D was also
clean. The PD 6P2D point is retained as observational development evidence
because its tracked worktree contained only matrix-default and documentation
edits that did not affect the running PD path.

## C32 results

All five points completed 640/640 requests correctly. A failed SLO cell means
that fewer than 95% of requests met both latency limits; it does not mean that
the service failed to finish.

| Architecture | TTFT avg / p95 | ITL avg / p95 | Requests/s | Strict | Standard | Relaxed |
| --- | ---: | ---: | ---: | --- | --- | --- |
| PAP 6PA2P | 3,209 / 9,909 ms | 33.53 / 39.79 ms | 4.388 | fail | **pass** | **pass** |
| PAP 7PA1P | 1,750 / 6,532 ms | 51.37 / 106.38 ms | 5.007 | fail | fail | fail |
| PD 4P4D | 4,870 / 18,364 ms | 27.86 / 33.23 ms | 3.829 | fail | fail | **pass** |
| PD 6P2D | 8,205 / 21,187 ms | 43.05 / 117.51 ms | 2.537 | fail | fail | fail |
| Fused replica pool ×8 | 1,002 / 3,492 ms | 47.77 / 116.58 ms | 6.381 | fail | fail | fail |

The fused baseline is eight independent dense-model vLLM replicas. AIPerf
routes conversations with sticky session affinity; it is not vLLM external
data parallelism, which this dense Qwen model does not support.

Good-request fractions and numerical goodput are:

| Architecture | Strict | Standard | Relaxed |
| --- | ---: | ---: | ---: |
| PAP 6PA2P | 79.69% / 3.497 rps | **96.25% / 4.224 rps** | **99.84% / 4.381 rps** |
| PAP 7PA1P | 72.03% / 3.607 rps | 84.53% / 4.232 rps | 93.59% / 4.686 rps |
| PD 4P4D | 66.88% / 2.560 rps | 87.50% / 3.350 rps | **95.31% / 3.649 rps** |
| PD 6P2D | 49.53% / 1.257 rps | 59.69% / 1.514 rps | 87.03% / 2.208 rps |
| Fused replica pool ×8 | 68.12% / 4.347 rps | 85.62% / 5.464 rps | 92.03% / 5.873 rps |

At C32, 6PA2P is the only tested point that passes Standard and Relaxed.
Against the best passing PD point, 4P4D, it has 14.6% higher request
throughput, 34.1% lower mean TTFT, and 20.1% higher relaxed compliant goodput.
The fused pool has the highest raw throughput, but its C32 ITL tail makes it
ineligible for all three SLO tiers. Lower-concurrency points are required
before comparing each architecture's maximum SLO-compliant goodput.

## Why 7PA1P misses the C32 SLO

The seventh PA is useful. Relative to 6PA2P, 7PA1P increases raw request
throughput by 14.1% and reduces mean TTFT by 45.5%. Its failure is specifically
an ITL-tail failure: 599/640 requests meet Relaxed, nine short of the 608
required by the 95% gate.

A matched trace at clean commit `4c8a96f26` completed 640/640 requests for
both PAP topologies. It isolates the Projection-side remote-Attention chain:

| Projection trace span | 7PA1P | 6PA2P Projection 0 | 6PA2P Projection 1 |
| --- | ---: | ---: | ---: |
| QKV norm/rope mean | 0.088 ms | 0.083 ms | 0.083 ms |
| Q/K repack mean | 0.008 ms | 0.007 ms | 0.007 ms |
| P2P QKV copy mean | 0.006 ms | 0.006 ms | 0.005 ms |
| Join wait p50 | 0.348 ms | 0.267 ms | 0.256 ms |
| Join wait p90 | 0.517 ms | 0.410 ms | 0.412 ms |
| Join wait p99 | **3.654 ms** | 0.625 ms | 0.663 ms |

Projection must wait at every layer until every active PA shard has returned
before the full scheduler batch can continue. In 7PA1P, one Projection creates
one seven-PA synchronization domain. The seventh PA reduces each PA's
Attention work, but the layer still completes at the slowest PA; occasional
slow returns therefore affect the whole batch. In 6PA2P, affinity divides the
same conversations into two independent, smaller Projection synchronization
domains. The second Projection is valuable here because it splits the fan-in
barrier, not because Projection stores KV.

This also explains why the change is not a simple Projection-compute
bottleneck. QKV preparation, repack, and P2P-copy costs remain near-identical,
while the join tail changes materially. Removing repeated Projection info
logging changed 7PA1P throughput only from roughly 4.95 to 5.01 requests/s and
did not remove the ITL tail.

Three independent 7PA1P C32 observations produced relaxed good-request
fractions of 94.22%, 96.25%, and 93.59%. The topology is therefore a
throughput-oriented point close to the current Relaxed boundary, not a broken
execution path. It must not be selected as the latency-stable C32 baseline
without further scheduling work or repeated evidence.

## Decisions and next gate

1. Keep both 7PA1P and 6PA2P supported. Use 6PA2P as the current
   latency-stable C32 development configuration.
2. Do not report 7PA1P or the fused pool as zero-goodput systems. Their
   numerical goodput is high, but the points are ineligible under the 95%
   admission gate.
3. Replace the original C32/64/96/128 default with C16/24/32/48. Existing C64
   diagnostics already show overload for PAP 6PA2P and PD 4P4D; the lower
   points are necessary to find Strict and Standard capacity.
4. Run three repetitions only for the final per-SLO boundary points selected
   by the compact scan.

Raw results and traces remain machine-local under
`benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_*`. The executable
testbed and workload contract remain
`benchmarks/pap/aiperf/run_capacity_matrix.sh`.
