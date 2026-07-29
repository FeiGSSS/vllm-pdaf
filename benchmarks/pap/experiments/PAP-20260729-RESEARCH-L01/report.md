# PAP research loop L01: fan-in dominance test

> Diagnostic evidence only. Blocking CUDA-event synchronization makes the
> trace unsuitable as a formal performance baseline.

Date: 2026-07-29

## Question and decision

L01 tested whether multi-PA completion spread, amplified by the Projection
fan-in join, dominates the 7PA1P ITL loss relative to 6PA2P.

The pre-registered claim is falsified. The median spread difference accounts
for 3.20 ms over 36 layers, only 16.6% of the 19.28 ms trace-mode mean ITL
gap. Fan-in imbalance is real and affects the tail, but it is not the dominant
typical-latency explanation in this run.

## Identity

- Code: clean commit `ee6b307c7`.
- Model: Qwen3-8B, FP16, eager.
- Hardware: eight NVIDIA L20 GPUs.
- Topologies: 7PA1P and 6PA2P.
- Load: 128 conversations, five turns, 640 requests, concurrency 32.
- Dataset SHA-256:
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- Initial input mean: 4,015.6 actual tokens.
- Later-turn append mean: 711.3 actual tokens.
- Output mean/range: 16.44 / 8--32 tokens.
- Think/tool schedule: 0/1/1/0.3/1 seconds.
- Routing: conversation affinity; zero migrations.

Raw artifacts remain machine-local at:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260729_l01_current_trace_s128_c32/
```

## End-to-end diagnostic result

Both runs completed 640/640 requests and passed correctness and routing audits.

| Topology | Req/s | TTFT avg / p95 | ITL avg / p95 |
| --- | ---: | ---: | ---: |
| 7PA1P | 10.525 | 0.920 / 3.069 s | 58.53 / 96.29 ms |
| 6PA2P | 8.825 | 1.823 / 4.190 s | 39.25 / 47.05 ms |

The topology trade-off is explicit: 7PA1P has 19.3% higher request throughput
and 49.5% lower mean TTFT, but 49.1% higher mean ITL.

## Fan-in attribution

| Per-layer metric | 7PA1P median | 6PA2P median | Difference |
| --- | ---: | ---: | ---: |
| Participating PAs | 6 | 3 | 3 |
| First ready | 0.156 ms | 0.154 ms | 0.002 ms |
| Last ready | 0.327 ms | 0.230 ms | 0.097 ms |
| First-to-last spread | 0.163 ms | 0.074 ms | 0.089 ms |

Multiplying the median spread difference by 36 layers yields 3.20 ms. This is
well below the pre-registered threshold of half the observed ITL gap.

## Matched-shape result

The aggregate shapes differ substantially:

| Shape | 7PA1P median | 6PA2P median |
| --- | ---: | ---: |
| Projection rows per model forward | 11 | 3 |
| Rows processed by one PA call | 2 | 1 |

Conditioning removes most topology dependence:

- For Projection rows 1--10, matched median forward time is generally within
  6% across topologies.
- For PA rows 1--5, matched paged-Attention kernel medians are within 5.7%.
- At one PA row, for example, kernel medians are 0.073 and 0.075 ms.
- At three PA rows, kernel medians are 0.194 and 0.191 ms.

This supports a new hypothesis: one Projection domain in 7PA1P aggregates a
larger active Decode batch, while two Projection domains in 6PA2P form smaller
independent batches. Barrier spread adds cost but does not by itself explain
the aggregate ITL ordering.

## Instrumentation caveat

`PAP_OFFLOAD_EXEC_TRACE=1` synchronizes the per-layer Attention timing event
and all Projection ready events. At matched PA rows, the CUDA kernel time is
nearly unchanged, but the synchronous `paged_flash_ms` wall time is about
0.21--0.26 ms larger in 7PA1P for rows 1--5. This is observer overhead and
must not be charged to the normal runtime.

Two trace-off retries failed before service startup because `nvidia-smi`
returned driver error 9 while automatic Projection memory sizing queried GPU
6 or 7. They produced no profiles and are not evidence:

```text
20260729_l01_current_notrace_s128_c32/
20260729_l01_current_notrace_retry_s128_c32/
```

## Next loop

L02 will rerun the byte-identical C32 point without blocking trace and use
deferred CUDA spans to fit a per-step service model over:

1. Projection batch rows;
2. per-PA rows and resident KV tokens;
3. communication;
4. slowest PA completion; and
5. scheduler/host residual.

No scheduling implementation is justified until that model explains the
trace-free gap or exposes a specific residual.
