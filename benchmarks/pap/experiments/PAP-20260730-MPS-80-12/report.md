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

The TTFT tail is not yet a stable positive claim. The five Standard-SLO
failures in the selected run are all third-turn TTFT failures; their ITLs
remain between 36.9 and 55.9 ms. This separates the observed tail from the
decode-kernel latency gain, but one repetition cannot determine whether it
is run variance or increased HBM contention with concurrent Prefill.

## Implementation decision

Accept the specialization for PAP processes that expose at most 20 SMs:

```text
visible SMs <= 20: split8 / BLOCK_H4 / 4 warps / 1 stage
visible SMs > 20:  preserve split4 / BLOCK_H16 / 4 warps / 2 stages
```

The threshold keeps full-GPU and unverified xPAyP paths on the old launch.
The generic vLLM Triton defaults are unchanged; only PAP passes the new
optional launch parameters.

This establishes that the current kernel can preserve most of Attention's
latency under a small SM allocation. It does not establish that HBM
bandwidth is fully saturated or that 80/12 should replace 72/20 as the
system default. A repeated interleaved end-to-end A/B is still required for
the TTFT-tail and Standard-goodput decision.

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

## Scope

Kernel evidence has two exact-shape repetitions plus four cross-shape
checks. End-to-end evidence has one complete repetition. A second
end-to-end attempt could not start because the host GPU device nodes became
unavailable before service launch; it produced no benchmark profile and is
not counted as evidence.
