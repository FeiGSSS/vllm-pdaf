# PAP low-SM decode Attention specialization

Date: 2026-07-30

## Question

Can PAP reduce the Attention allocation from 20 to 12 visible SMs, give the
remaining SMs to Prefill, and preserve Decode latency by recovering HBM
memory-level parallelism inside the Attention kernel?

Static MPS allocation on this L20 host is quantized in four-SM chunks. The
controlled end-to-end comparison is therefore:

```text
baseline:  Prefill 72 SM / Attention 20 SM
treatment: Prefill 80 SM / Attention 12 SM
```

## Correction to the initial diagnosis

The first version of this report used a paged-FlashAttention (FA2) probe to
explain the end-to-end 12-SM slowdown. That attribution was not valid:
the current PAP main path has used vLLM's Triton grouped-query decode kernel
since `PAP-20260716-TRITON-72-20-BASELINE`. The FA2 result remains a useful
counterexample showing why a bandwidth-bound kernel may still need enough
request generators, but it does not diagnose the production PAP kernel.

The current investigation therefore measures and changes the exact Triton
kernel called by `vllm/pap/attention/kernels.py`.

## Controlled workload

The exact-shape kernel probe uses Qwen3-8B's decode layout:

- batch 3, 32 query heads, 8 KV heads, head dimension 128;
- sequence lengths 17,344, 17,334, and 17,324 tokens;
- page size 16 and the production cross-layer KV-cache layout;
- identical tensors for every launch configuration;
- FA2 output only as an independent numerical reference.

The end-to-end point uses:

- Qwen3-8B, 8x L20, PAP 7PA1P, eager execution;
- per-PA FIFO Prefill admission of one;
- 60 sessions, 3 turns, 180 requests, concurrency 20;
- initial context about 9.5K tokens; later contexts about 18.9K and 28.6K;
- mean output 101.98 tokens;
- dataset SHA256
  `4faa9f1cf3423f11f83cbf38bad19f2c73865e290b99bfc7e716f84cf9e8ea7b`;
- project UCX 1.22.0 NIXL runtime with protocol emulation disabled.

All reported end-to-end runs completed 180/180 requests and 60/60 sessions
and passed correctness, routing, MPS-allocation, NIXL-runtime, and
session-drain audits.

## Original end-to-end observation

Before the kernel change, reducing Attention from 20 to 12 SMs improved
Prefill TTFT but worsened Decode ITL:

| Metric | 72/20 old kernel | 80/12 old kernel | 80/12 delta |
| --- | ---: | ---: | ---: |
| TTFT average | 4.279 s | **3.583 s** | **-16.27%** |
| TTFT p95 | 7.287 s | **6.010 s** | **-17.52%** |
| ITL average | **44.71 ms** | 48.76 ms | **+9.06%** |
| ITL p95 | **53.05 ms** | 59.31 ms | **+11.79%** |
| Raw throughput | 1.894 req/s | 1.897 req/s | +0.17% |
| Strict goodput | **1.073 req/s** | 1.033 req/s | -3.76% |
| Standard goodput | 1.883 req/s | 1.886 req/s | +0.17% |

This established the optimization target: retain the TTFT benefit while
removing most of the 12-SM ITL penalty.

## Triton specialization search

The old PAP launch used:

```text
KV splits=4, num_warps=4, BLOCK_H=16, num_stages=2
```

The probe exhaustively tested 24 combinations at both 12 and 20 SMs:

```text
KV splits:   4, 8
num_warps:   4, 8
BLOCK_H:     4, 8, 16
num_stages:  1, 2
```

All 24 configurations were numerically close to FA2 at both allocations.
For the selected configuration, maximum absolute error was
`1.907e-6`.

The robust low-resource winner is:

```text
KV splits=8, num_warps=4, BLOCK_H=4, num_stages=1
```

| Visible SMs | Old kernel | Selected kernel | Improvement |
| ---: | ---: | ---: | ---: |
| 12 | 0.4680 ms | **0.4105 ms** | **12.30%** |
| 20 | 0.4560 ms | **0.3855 ms** | **15.46%** |

The selected 12-SM kernel is about 10.0% faster than the old 20-SM kernel.
Its remaining 12-versus-20 latency gap is 6.5%, rather than the old
12-SM end-to-end ITL penalty of about 9%--12%.

The result also holds across the sampled batch/context shapes:

| Shape | 12-SM old | 12-SM selected | Improvement |
| --- | ---: | ---: | ---: |
| B3, 4K context | 0.1057 ms | **0.0922 ms** | **12.8%** |
| B3, 10K context | 0.2698 ms | **0.2368 ms** | **12.2%** |
| B3, 17K context | 0.4678 ms | **0.4107 ms** | **12.2%** |
| B1, 17K context | 0.1975 ms | **0.1489 ms** | **24.6%** |
| B6, 17K context | 0.9297 ms | **0.8087 ms** | **13.0%** |

## Why it works

For the exact B3/H32 shape, the old launch creates only:

```text
3 batches * ceil(32 heads / BLOCK_H 16) * 4 splits = 24 CTAs
```

The selected launch creates:

```text
3 batches * ceil(32 heads / BLOCK_H 4) * 8 splits = 192 CTAs
```

Compiled Triton metadata reports:

| Kernel | Shared memory / CTA | Registers / thread | Threads / CTA |
| --- | ---: | ---: | ---: |
| Old | 21,504 B | 64 | 128 |
| Selected | 9,472 B | 64 | 128 |

The old grid has only two CTAs per visible SM at 12 SMs, regardless of the
higher theoretical occupancy limit. It therefore exposes too few warps and
independent memory operations. The selected kernel both expands the grid
from 24 to 192 CTAs and reduces shared-memory use. Register pressure then
permits up to eight resident CTAs, or 32 resident warps, per SM.

This does not reduce the required KV bytes. It provides enough independent
CTAs and warps for a small number of SMs to keep more HBM reads in flight.
Simply changing to eight warps per CTA was consistently slower; more warps
inside one CTA are not a substitute for enough independent CTAs.

## End-to-end validation

One complete AIPerf run of the selected kernel produced:

| Metric | 72/20 old kernel | 80/12 old kernel | 80/12 selected |
| --- | ---: | ---: | ---: |
| TTFT average | 4.279 s | **3.583 s** | 4.003 s |
| TTFT p95 | **7.287 s** | 6.010 s | 7.646 s |
| TTFT p99 | **8.531 s** | 7.353 s | 11.585 s |
| ITL average | **44.71 ms** | 48.76 ms | **45.40 ms** |
| ITL p95 | **53.05 ms** | 59.31 ms | **54.67 ms** |
| ITL p99 | **54.50 ms** | 59.68 ms | **55.90 ms** |
| Raw throughput | 1.894 req/s | 1.897 req/s | **1.902 req/s** |
| Strict goodput | 1.073 req/s | 1.033 req/s | **1.173 req/s** |
| Standard goodput | **1.883 req/s** | 1.886 req/s | 1.849 req/s |

Relative to the old 80/12 path, the selected kernel improves mean ITL by
6.89%, ITL p95 by 7.82%, ITL p99 by 6.33%, and Strict goodput by 13.59%.
Relative to the old 72/20 path, mean ITL is now only 1.54% worse while mean
TTFT remains 6.47% better.

The five Standard-SLO failures in this first selected-kernel run were all
third-turn TTFT failures; their ITLs remained between 36.9 and 55.9 ms. A
second run reproduced the phase boundary: Projection first output accounted
for only about 0.15 seconds, while the slow requests accumulated several
seconds before Projection.

## TTFT-tail diagnosis and fix

Stage-level tracing found two coupled sources:

1. several third-turn Prefills took 5--11 seconds instead of the normal
   3--4 seconds because their reusable prefix had been displaced; and
2. per-PA single-slot admission amplified one slow Prefill into a FIFO convoy,
   with later requests waiting up to 7.27 seconds.

The capacity pressure was caused by duplicate lease ownership. PAP Attention
already pins each live request through its own exact KV lease, but the generic
NIXL producer also retained the same completed Prefill request for its
30-second PD handoff window. Projection is KV-unaware and never performs that
PD read. Logs consequently showed every request expiring after being retrieved
by zero remote workers, while per-PA KV usage climbed above 90%.

PAP now keeps its 300-second, pressure-evictable ownership lease and uses a
one-second NIXL producer bookkeeping lease. This does not change NIXL transfer
metadata or the later migration data plane. Once the generic connector
finishes bookkeeping, the vLLM free path observes the still-active PAP lease
and transfers block ownership to it rather than returning live Attention
blocks to the allocator.

On the same dataset and C20 point:

| Tail metric | Before | One-second producer lease |
| --- | ---: | ---: |
| Prefill executions above 5 s | 3 | **0** |
| Admission waits above 5 s | 3 | **0** |
| Third-turn Prefill maximum | 7.29 s | **3.85 s** |
| Third-turn admission maximum | 7.27 s | **4.86 s** |
| Request TTFT maximum | 10.90 s | **8.85 s** |
| Standard-SLO good requests | 179/180 | **180/180** |
| Mean ITL | 45.62 ms | **43.36 ms** |
| Raw throughput | 1.920 req/s | 1.920 req/s |

Mean TTFT in the treatment run was 4.127 seconds versus 3.783 seconds in the
immediately preceding run because all three turns had 10%--13% slower normal
Prefill execution. The pathological multi-second cache-miss and convoy tail
nevertheless disappeared, throughput was unchanged, and every Standard-SLO
request passed. The following concurrency scan is the stability check for
normal Prefill variance.

## Full concurrency comparison

The committed 80/12 configuration and one-second connector lease were then
compared with one-way PD 6P2D at C16/C20/C24/C28/C32. Every point restarted
all services and replayed the same 60-session, three-turn dataset. All ten
points completed 180/180 requests and 60/60 sessions with correctness,
routing, NIXL-runtime, and architecture-specific runtime audits passing.

| Architecture | C | Raw req/s | TTFT avg / p99 | ITL avg / p99 | Standard goodput | Relaxed goodput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PD 6P2D | 16 | 1.518 | 3.38 / 7.50 s | 53.37 / 74.50 ms | **1.501 pass** | 1.518 pass |
| PD 6P2D | 20 | 1.513 | 4.42 / 10.94 s | 61.89 / 106.56 ms | 1.337 fail | 1.488 pass |
| PD 6P2D | 24 | 1.588 | 5.53 / 12.09 s | 63.94 / 86.02 ms | 1.358 fail | 1.588 pass |
| PD 6P2D | 28 | 1.446 | 8.07 / 18.67 s | 72.29 / 134.21 ms | 0.578 fail | 1.382 pass |
| PD 6P2D | 32 | **1.715** | 8.40 / 17.91 s | 71.76 / 97.24 ms | 0.648 fail | **1.696 pass** |
| PAP 7PA1P | 16 | 1.722 | 3.30 / 6.05 s | 42.88 / 52.40 ms | 1.722 pass | 1.722 pass |
| PAP 7PA1P | 20 | 1.975 | 3.50 / 6.58 s | 46.05 / 57.29 ms | 1.975 pass | 1.975 pass |
| PAP 7PA1P | 24 | 2.062 | 4.04 / 8.06 s | 48.63 / 60.28 ms | 2.062 pass | 2.062 pass |
| PAP 7PA1P | 28 | 2.126 | 4.75 / 8.57 s | 50.45 / 62.12 ms | 2.126 pass | 2.126 pass |
| PAP 7PA1P | 32 | **2.343** | 5.71 / 10.51 s | 51.39 / 64.74 ms | **2.252 pass** | **2.343 pass** |

Neither architecture has a point at which 95% of requests meet the Strict
5-second-TTFT/50-ms-ITL tier, so the observed Strict goodput values are not
reported as capacity. For the two passing tiers:

| SLO / metric | Best PD | Best PAP | PAP delta |
| --- | ---: | ---: | ---: |
| Standard goodput | 1.501 req/s, C16 | **2.252 req/s, C32** | **+50.0%** |
| Relaxed goodput | 1.696 req/s, C32 | **2.343 req/s, C32** | **+38.1%** |
| Raw throughput | 1.715 req/s, C32 | **2.343 req/s, C32** | **+36.6%** |

At the matched C32 point, PAP lowers average TTFT by 32.0%, TTFT p99 by
41.3%, average ITL by 28.4%, and ITL p99 by 33.4% relative to PD. PAP passes
Standard with 173/180 good requests, just above the predeclared 95% gate;
PD passes Standard only at C16.

The long-tail diagnosis also remains stable across the scan. PAP's maximum
individual Prefill execution is 3.50 seconds and no Prefill execution exceeds
five seconds at any point. The maximum single-slot admission wait grows from
4.00 seconds at C16 to 9.08 seconds at C32. Consequently, the C32 TTFT maximum
of 11.51 seconds is capacity queueing, not a recurrence of the old
30-second-lease cache displacement. C16--C28 have TTFT maxima of 6.95, 7.48,
8.28, and 9.17 seconds respectively.

## Implementation decision

Accept the specialization for PAP processes that expose at most 20 SMs:

```text
visible SMs <= 20: split8 / BLOCK_H4 / 4 warps / 1 stage
visible SMs > 20:  preserve split4 / BLOCK_H16 / 4 warps / 2 stages
```

The threshold keeps full-GPU and unverified xPAyP paths on the old launch.
The generic vLLM Triton defaults are unchanged; only PAP passes the new
optional launch parameters.

Together with the corrected lease ownership, this establishes 80/12 as the
PAP AIPerf baseline. The runner defaults are 20 Prefill MPS chunks and 3
Attention chunks, audited as 80 and 12 visible L20 SMs. This does not establish
that HBM bandwidth is fully saturated; the decision is limited to the measured
Qwen3-8B shapes and keeps the generic vLLM kernel defaults unchanged.

## Evidence

- Triton 12/20-SM sweep:
  `benchmarks/pap/experiments/_staging/runs/20260730_191240_attention_sm_sweep`
- independent focused repeat:
  `benchmarks/pap/experiments/_staging/runs/20260730_1920_attention_sm_sweep_repeat`
- cross-shape probes:
  `benchmarks/pap/experiments/_staging/runs/20260730_attention_sm_sweep_b3_ctx4k`,
  `20260730_attention_sm_sweep_b3_ctx10k`,
  `20260730_attention_sm_sweep_b1_ctx17k`, and
  `20260730_attention_sm_sweep_b6_ctx17k`
- old 72/20 C20 control:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pap7pa1p_c20_mps72_20_admit1_fifo_control_r1`
- old 80/12 C20:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pap7pa1p_c20_mps80_12_admit1_r1`
- selected-kernel 80/12 C20:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pap7pa1p_c20_mps80_12_low_resource_kernel_r1`
- independent selected-kernel TTFT trace:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pap7pa1p_c20_mps80_12_low_resource_kernel_ttft_trace_r2`
- one-second producer-lease treatment:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pap7pa1p_c20_mps80_12_nixl_lease1_ttft_fix_r2`
- full 80/12 PAP-versus-PD concurrency scan:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pap80_12_nixllease1_pd_full_concurrency_r1`

## Scope

Kernel evidence has two exact-shape repetitions plus four cross-shape checks.
The selected kernel has three complete C20 end-to-end observations: the
initial result, an independent stage trace, and the one-second producer-lease
treatment. All completed 180/180 requests with strict correctness audits.
The subsequent comparison contains one complete repetition at each of ten
architecture/concurrency points. It is controlled development evidence; a
paper or release claim still requires repeated boundary points.
