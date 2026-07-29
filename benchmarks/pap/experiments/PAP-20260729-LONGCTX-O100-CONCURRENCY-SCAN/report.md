# Long-context O100 concurrency scan

Date: 2026-07-29

## Question and decision

This scan asks whether PAP's larger distributed KV pool becomes useful when a
long-context, multi-turn workload is driven by a fixed concurrency cap rather
than a request-rate limit.

The completed C16 and C20 points use AIPerf concurrency mode with no `-r`
limit. All six points complete 90 of 90 requests and pass correctness.
At C20, PAP 7PA1P exceeds PD 6P2D by 5.9% in Standard goodput and 3.5% in raw
request throughput. PAP 6PA2P does not show the same result: it fails the
Standard SLO at C20 because its TTFT tail grows substantially.

This is one-repetition development evidence. It identifies a promising
7PA1P operating point but is not yet a paper-ready statistical claim.

## Workload

- Qwen3-8B FP16 eager on eight L20 GPUs
- 30 sessions, three turns, 90 total requests
- about 10K new input tokens per turn
- estimated input sequence mean 18,941 tokens and maximum 30,023 tokens
- randomized output length: mean 105.5, median 100.5, range 50--187 tokens
- delay schedule: 0/1/1 seconds
- `max_model_len=32768`
- dataset SHA-256:
  `ff6206989c25b69cfc4ad0b3d8e299e7ae9f888c531cb8c11c2956ef12b9da7d`
- code commit: `c2497fc4ff`

SLO tiers are:

- Strict: TTFT <= 5 seconds and ITL <= 50 ms
- Standard: TTFT <= 10 seconds and ITL <= 75 ms
- Relaxed: TTFT <= 20 seconds and ITL <= 100 ms
- A tier passes only when at least 95% of requests meet both limits.

## Fixed-concurrency results

These are the primary results. AIPerf ran in concurrency mode without a
request-rate limit. Every point reached its configured maximum concurrency.

| System | C | Raw req/s | Mean / p95 TTFT | Mean / p95 ITL | Strict goodput / fraction | Standard goodput / fraction | Relaxed goodput / fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PD 6P2D | 16 | 1.434 | 3629 / 6074 ms | 50.67 / 65.70 ms | 0.542 / 37.8% fail | 1.434 / 100% pass | 1.434 / 100% pass |
| PAP 6PA2P | 16 | 1.316 | 5326 / 9776 ms | 37.30 / 43.68 ms | 0.643 / 48.9% fail | 1.257 / 95.6% pass | 1.316 / 100% pass |
| PAP 7PA1P | 16 | 1.442 | 4101 / 7146 ms | 42.27 / 64.28 ms | 0.945 / 65.6% fail | 1.410 / 97.8% pass | 1.442 / 100% pass |
| PD 6P2D | 20 | 1.380 | 4953 / 7909 ms | 56.72 / 72.51 ms | 0.322 / 23.3% fail | 1.334 / 96.7% pass | 1.380 / 100% pass |
| PAP 6PA2P | 20 | 1.163 | 6754 / 14548 ms | 39.08 / 47.65 ms | 0.401 / 34.4% fail | 0.943 / 81.1% fail | 1.163 / 100% pass |
| PAP 7PA1P | 20 | 1.429 | 4612 / 6893 ms | 44.05 / 68.77 ms | 0.587 / 41.1% fail | 1.413 / 98.9% pass | 1.429 / 100% pass |

At C20, 7PA1P relative to 6P2D has:

- 5.9% higher Standard goodput
- 3.5% higher raw request throughput
- 6.9% lower mean TTFT
- 22.3% lower mean ITL

At C16, 7PA1P has 0.6% higher raw throughput but 1.6% lower Standard
goodput than PD. The crossover therefore appears between C16 and C20 for this
workload and single repetition.

The topology distinction is material. At C20, 6PA2P has 29.3% lower Standard
goodput than PD and fails the Standard tier, while 7PA1P passes it. The extra
PA improves long-Prefill capacity enough to dominate the loss of one
Projection replica at this point.

No PD Decode log at C16 or C20 contains an explicit preemption, recompute,
eviction, KV-full, or OOM marker. The data therefore establishes a
goodput crossover, not yet a proven KV-capacity-wall mechanism.

All PAP points use the fail-closed same-node NIXL runtime:

- NIXL/UCX 1.22
- `UCX_PROTO_EMULATION_ENABLE=n`
- `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=y`

## Earlier R=4 context

The earlier C8/C12 points used constant request rate R=4 with a concurrency
cap. They reached their configured maximum concurrency, but the arrival mode
is not identical to the primary fixed-concurrency C16/C20 scan. They are
retained as contextual evidence rather than silently merged into one
homogeneous curve.

| System | C | Raw req/s | Mean / p95 TTFT | Mean / p95 ITL | Standard goodput / fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| PD 6P2D | 8 | 1.018 | 2469 / 3313 ms | 37.29 / 44.73 ms | 1.018 / 100% pass |
| PAP 6PA2P | 8 | 0.891 | 3680 / 6572 ms | 34.55 / 39.47 ms | 0.871 / 97.8% pass |
| PAP 7PA1P | 8 | 0.975 | 2909 / 3831 ms | 35.27 / 39.18 ms | 0.975 / 100% pass |
| PD 6P2D | 12 | 1.239 | 2530 / 3362 ms | 43.15 / 55.86 ms | 1.239 / 100% pass |
| PAP 6PA2P | 12 | 1.103 | 4306 / 8024 ms | 35.70 / 41.25 ms | 1.066 / 96.7% pass |
| PAP 7PA1P | 12 | 1.161 | 3600 / 6517 ms | 37.80 / 46.31 ms | 1.161 / 100% pass |

## Invalidated timing controls

An R=1 C16 attempt was stopped after showing that the rate limiter prevented
the requested concurrency:

- PD 6P2D reached maximum total concurrency 9 and Decode concurrency 7.
- PAP 6PA2P reached maximum total concurrency 10 and Decode concurrency 7.
- PAP 7PA1P was interrupted and is ineligible.

The completed R=1 points are retained as low-arrival-load diagnostics but must
not be used as C16 capacity evidence. A later R=4 C16 attempt was also
interrupted when the protocol was corrected to pure concurrency mode; it is
ineligible.

## Provenance

- R=4 C8:
  `benchmarks/pap/experiments/_staging/capacity/20260729_longctx_o100_s30_t3_r4_c8_r1/`
- R=4 C12:
  `benchmarks/pap/experiments/_staging/capacity/20260729_longctx_o100_s30_t3_r4_c12_r1/`
- invalid R=1 C16 diagnostic:
  `benchmarks/pap/experiments/_staging/capacity/20260729_longctx_o100_s30_t3_r1_c16_r1/`
- interrupted R=4 C16 attempt:
  `benchmarks/pap/experiments/_staging/capacity/20260729_longctx_o100_s30_t3_r4_c16_r1/`
- primary fixed-concurrency C16:
  `benchmarks/pap/experiments/_staging/capacity/20260729_longctx_o100_s30_t3_concurrency_c16_r1/`
- primary fixed-concurrency C20:
  `benchmarks/pap/experiments/_staging/capacity/20260729_longctx_o100_s30_t3_concurrency_c20_r1/`
