# PAP versus PD: audited-capacity AIPerf baseline

## Scope

This four-GPU eager-mode scan is the capacity baseline after removing
experiment-side scheduler limits that could mask the actual compute, KV, or
SLO boundary.

- vLLM/PAP commit: `79b31742f`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Work per point: 32 conversations, ten turns, 320 requests
- Timing: conversation concurrency with delays
  `0,3,3,1,3,3,1,3,3,1` seconds
- Dataset seed: 42
- Dataset SHA-256:
  `a2d77fb2748a9f1bb02495abc89ed9d2c8da7db947cfbe8afcfd7a6b4ee43969`
- Execution mode: eager for both PAP and PD

The randomized distributions are unchanged from the preceding O32 testbed:

| Dimension | Mean / median | Bounds | Sample mean / median | Sample range |
| --- | ---: | ---: | ---: | ---: |
| Initial user content | 8,192 / 8,000 | 4,096-11,264 | 8,146.406 / 8,349 | 4,999-11,264 |
| Later-turn user content | 512 / 500 | 256-768 | 508.014 / 499 | 289-768 |
| Output | 32 / 30 | 16-64 | 32.594 / 31 | 16-64 |

The longest estimated request, including its output budget, is 16,224 tokens.
This leaves 3,776 tokens below `max_model_len=20000`.

## Audited runtime limits

The active settings were derived from the scheduler and connector source:

| Role | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| PAP PA / PD Prefill | 64 | 16,384 |
| PAP Projection / PD Decode | 64 | 64 |

- At most 32 conversations can be live, so `max_num_seqs=64` cannot reject a
  request in this testbed.
- Prefill uses the documented throughput-oriented 16K iteration budget.
- Decode and Projection execute at most one local model token per live request.
  NIXL asynchronous KV loads use zero scheduled model tokens and therefore do
  not consume this 64-token scheduler budget.
- `max_num_partial_prefills` remains at its default value 1. No CLI override is
  set, and `long_prefill_token_threshold` remains 0.
- PAP reserves writable unified KV from each request's sampled output limit
  (16-64 here), with 64 only as a compatibility fallback. It no longer reserves
  512 decode tokens for every request.
- PAP keeps `gpu_memory_utilization=0.76` because PA GPUs also host Attention;
  PD keeps 0.90. The scheduler retains full-input admission and zero watermark.

These values and their rationale are also recorded in
[`benchmarks/pap/aiperf/README.md`](../../aiperf/README.md).

## Validity

Eleven matrix points ran. Every point completed all 320 requests and passed
output-length, routing, conversation-affinity, KV handoff, and session-drain
audits. The matrix therefore contains 3,520 valid request records and no
partial or failed point.

This is one clean repetition per point. It is the development baseline for the
following CUDA Graph comparison, not a three-repetition release claim.

A request is good only when both its TTFT and request-level mean ITL meet the
tier. A point passes when at least 95% of its requests are good:

| Tier | TTFT | ITL |
| --- | ---: | ---: |
| Strict | 5,000 ms | 50 ms |
| Standard | 10,000 ms | 75 ms |
| Relaxed | 20,000 ms | 100 ms |

## Results

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 8 | 2,024.84 | 37.21 | 1.994 | pass | pass | pass |
| PAP | 3PA1P | 12 | 3,552.38 | 41.32 | 2.492 | fail | pass | pass |
| PAP | 3PA1P | 20 | 10,068.56 | 49.22 | 2.715 | fail | fail | pass |
| PAP | 3PA1P | 28 | 39,325.89 | 53.31 | 1.316 | fail | fail | fail |
| PD | 1P3D | 8 | 48,108.72 | 28.98 | 0.386 | fail | fail | fail |
| PD | 2P2D | 10 | 7,431.79 | 31.87 | 1.801 | fail | pass | pass |
| PD | 2P2D | 16 | 11,570.79 | 33.67 | 2.813 | fail | fail | pass |
| PD | 2P2D | 20 | 29,167.72 | 36.54 | 2.193 | fail | fail | fail |
| PD | 3P1D | 8 | 4,565.87 | 34.70 | 1.835 | pass | pass | pass |
| PD | 3P1D | 14 | 9,598.16 | 40.49 | 2.341 | fail | fail | pass |
| PD | 3P1D | 20 | 28,392.52 | 41.67 | 1.407 | fail | fail | fail |

The tested concurrency envelope is:

| SLO | PAP 3PA1P | Best PD | PAP difference |
| --- | ---: | ---: | ---: |
| Strict | C8 | C8, 3P1D | 0 |
| Standard | C12 | C10, 2P2D | +2 / +20% |
| Relaxed | C20 | C16, 2P2D | +4 / +25% |

## Compliant goodput

Only complete and correct configurations with at least 95% good requests are
eligible:

| SLO | PAP best | PD best | PAP versus PD |
| --- | ---: | ---: | ---: |
| Strict | 1.944 req/s, C8 | 1.761 req/s, 3P1D C8 | +10.4% |
| Standard | 2.461 req/s, C12 | 1.818 req/s, 3P1D C8 | +35.3% |
| Relaxed | 2.656 req/s, C20 | 2.690 req/s, 2P2D C16 | -1.3% |

## Observations and conclusion

PAP has a measured advantage for strict and standard goodput and sustains a
larger standard and relaxed concurrency envelope. Under the relaxed tier its
best goodput is effectively near PD but is 1.3% lower in this single run, so no
relaxed-goodput advantage is claimed.

PD 1P3D C8 is a real transfer-overload result, not a KV-capacity rejection. One
Decode worker reported NIXL transfers as long as 221 seconds and throughput
below 1 MB/s while GPU KV usage was only about 33-40%. Its ITL stayed low once
generation began, but TTFT p95 reached 48.1 seconds. This topology therefore
fails because one Prefill source and the transfer path cannot reliably feed
three Decode workers under the fixed workload.

The next experiment changes only execution mode: add PAP CUDA Graph support,
keep this dataset and capacity configuration fixed, and compare against PD
CUDA Graph at the informative strict, standard, and relaxed points.

The complete 80 MiB evidence bundle is preserved locally under
[`20260721_79b31742f_aiperf_audited_eager_o32_s32`](runs/20260721_79b31742f_aiperf_audited_eager_o32_s32/raw/capacity_results.md).
