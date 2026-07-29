# PAP research loop L02: batch aggregation and observer effect

> The trace-off run is the performance reference. Deferred CUDA spans are
> diagnostic evidence. The blocking L01 trace is excluded from absolute
> performance claims.

Date: 2026-07-29

## Question and decision

L02 tested whether a non-intrusive service model could explain at least 75%
of the 7PA1P versus 6PA2P ITL gap from their actual Decode batch shapes.

The registered threshold is not met, so C02 is falsified as written. The
broader load-aggregation mechanism is nevertheless the largest observed
component: 7PA1P executes much larger Projection batches, and the resulting
remote-Attention stage explains most of the complete model-forward median
difference. The remaining scheduler/cadence term prevents a 75% attribution
of total ITL.

The next comparison must use matched achieved throughput rather than matched
client concurrency.

## Identity

- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Topologies: 7PA1P and 6PA2P.
- Load: 128 conversations, five turns, 640 requests, concurrency 32.
- Dataset SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Routing: conversation affinity; zero migrations.
- Trace-off and first deferred run code: clean commit `f7cc7d185`.
- Model-span diagnostic code: clean commit `f90dfc084`.

Machine-local artifacts:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l02_current_notrace_s128_c32/
  20260729_l02_current_deferred_s128_c32/
  20260729_l02_modelspan_s128_c32/
```

## Trace-off end-to-end result

Both runs completed 640/640 requests and passed correctness, routing,
session-drain, scheduling, and static-MPS audits.

| Metric | 7PA1P | 6PA2P | 7PA1P relative |
| --- | ---: | ---: | ---: |
| Request throughput | 11.935 req/s | 9.304 req/s | +28.3% |
| Output throughput | 196.16 tok/s | 152.91 tok/s | +28.3% |
| TTFT average | 0.916 s | 1.670 s | -45.1% |
| ITL average | 36.54 ms | 28.34 ms | +28.9% |
| ITL p50 | 33.07 ms | 27.49 ms | +20.3% |
| ITL p95 | 63.55 ms | 32.69 ms | +94.4% |

7PA1P is not globally slower. At C32 it chooses a higher-throughput,
lower-TTFT operating point and pays higher per-user token latency.

| SLO | 7PA1P good fraction / goodput | 6PA2P good fraction / goodput |
| --- | ---: | ---: |
| Strict | 88.28% / 10.536 req/s, fail | 98.59% / 9.173 req/s, pass |
| Standard | 97.50% / 11.636 req/s, pass | 99.69% / 9.275 req/s, pass |
| Relaxed | 99.69% / 11.897 req/s, pass | 99.84% / 9.289 req/s, pass |

Goodput is only admissible for a tier when its required 95% good-request
fraction passes. The strict 7PA1P number is therefore not a valid capacity
point despite its larger numerator.

## Observer effect

| Instrumentation | 7PA1P ITL avg | 6PA2P ITL avg | Gap |
| --- | ---: | ---: | ---: |
| Trace off | 36.54 ms | 28.34 ms | 8.20 ms |
| Deferred CUDA events | 38.37 ms | 30.08 ms | 8.30 ms |
| Blocking L01 trace | 58.53 ms | 39.25 ms | 19.28 ms |

Deferred events perturb mean ITL by 5.0% and 6.1%, respectively, while
preserving the gap within 0.10 ms. The blocking trace perturbs 7PA1P much
more strongly and overstates the topology gap by 2.35x. It remains useful
only for matched-shape correlation.

All accepted deferred traces have zero pending, dropped, and errored records,
and their layer counts are divisible by 36.

## Batch formation

Both topologies produced the same 10,519 output tokens. In the L01
matched-shape trace:

| Metric | 7PA1P | 6PA2P |
| --- | ---: | ---: |
| Projection forwards | 947 | 3,201 |
| Rows per forward, mean | 11.11 | 3.29 |
| Rows per forward, median | 11 | 3 |

The independent model-span run reproduces the mechanism with 1,193 versus
3,585 total forwards, or 8.82 versus 2.93 rows/forward. Exact batch counts
change with instrumentation and timing, but the roughly 3x aggregation ratio
does not.

Two Projection processes do not merely divide one synchronized batch. They
are independent vLLM scheduling domains. The gateway additionally prevents a
PA from changing Projection owner in the middle of an active request wave.
Consequently, 6PA2P forms more, smaller Decode forwards, trading batching
efficiency for shorter token cycles.

## Non-blocking phase attribution

The complete model-forward span was added at `f90dfc084`. Its 7PA1P run
experienced a simultaneous NVIDIA driver fault and second-scale tail events,
so only medians are admitted from this run.

| Median component | 7PA1P | 6PA2P | Step-level difference |
| --- | ---: | ---: | ---: |
| Complete model forward | 30.886 ms | 27.396 ms | 3.490 ms |
| Remote-Attention stage | 0.2478 ms/layer | 0.1623 ms/layer | 3.078 ms / 36 layers |
| Paged Attention, mean PA p50 | 0.1106 ms/layer | 0.0853 ms/layer | 0.909 ms / 36 layers |
| QKV batched fan-out | 0.0176 vs 0.0100 ms/layer | — | about 0.28 ms / 36 layers |
| Step prepare wait | 0.0015 ms | 0.0015 ms | negligible |

Subtracting the remote stage from the complete forward leaves 21.965 versus
21.553 ms. Dense Projection/MLP therefore differs by only about 0.41 ms,
consistent with these small GEMM shapes remaining underutilized rather than
scaling linearly with rows.

The remote stage explains 88.2% of the complete-forward median difference,
but the complete-forward difference explains only 62.6% of the trace-off ITL
p50 gap. A residual scheduler/cadence term remains, so the pre-registered 75%
total-ITL criterion fails.

## Decision and next loop

Do not optimize dense Projection compute or local-fast copies next. Their
measured contribution is too small.

L03 will compare 7PA1P C20 against 6PA2P C32, because historical same-dataset
pilots place both near 9.3 req/s. Those pilots are not formal evidence: they
used a dirty benchmark worktree and produced unstable strict-SLO outcomes.
The clean test will use two repetitions and decide whether 7PA1P offers a
better latency/goodput Pareto point at matched achieved throughput.
