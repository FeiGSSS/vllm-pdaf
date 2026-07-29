# PAP research loop L12: near-limit KV capacity

Date: 2026-07-29

## Question and decision

L12 tests whether PAP's 2.63x larger aggregate KV-token pool becomes at least
10% more Standard and Relaxed goodput than 6P2D when four multi-turn contexts
approach the 40,960-token model limit.

The claim is falsified at the registered discovery boundary. In the valid
48-session scan, both systems pass Standard at C12 and fail Standard at C16.
At C12, PAP Standard goodput is 1.502 req/s versus 2.290 req/s for PD, a
34.4% deficit. PAP therefore reaches its TTFT SLO boundary before its larger
KV pool becomes usable goodput.

## Workload

- Qwen3-8B FP16 eager on eight L20 GPUs
- PAP 6PA2P versus PD 6P2D
- 48 sessions, four turns, 192 requests per discovery point
- about 10K new input tokens per turn; fourth-turn input mean 37,818 and
  maximum 39,732 tokens
- O16 output with sampled range 8--32
- delay schedule 0/1/1/0.3 seconds
- dataset SHA-256:
  `8ef6c8017930b8549ba077f14c1592d683fbd69d9de3795931657ba9f9dd1e73`

Forty-eight sessions give two complete waves at the highest planned C24. The
short workload is a boundary locator, not a replacement for long-sample tail
confirmation.

## Valid discovery results

All four completed points use clean commit `8b9228871`, complete every
request, and pass client, routing, decode-token join, and session-drain
correctness.

| System | C | Raw req/s | Mean TTFT | TTFT p95 | Mean ITL | ITL p95 | Standard goodput / fraction | Relaxed goodput / fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PD 6P2D | 12 | 2.314 | 3819 ms | 6022 ms | 35.38 ms | 51.20 ms | 2.290 / 98.96% pass | 2.314 / 100% pass |
| PD 6P2D | 16 | 2.140 | 5555 ms | 10839 ms | 44.43 ms | 71.22 ms | 1.884 / 88.02% fail | 2.107 / 98.44% pass |
| PAP 6PA2P | 12 | 1.576 | 5666 ms | 9361 ms | 34.52 ms | 46.66 ms | 1.502 / 95.31% pass | 1.559 / 98.96% pass |
| PAP 6PA2P | 16 | 1.347 | 7857 ms | 19789 ms | 38.63 ms | 42.29 ms | 1.059 / 78.65% fail | 1.235 / 91.67% fail |

At C16, PAP has 13.1% lower mean ITL than PD but 41.4% higher mean TTFT.
The near-limit failure is therefore Prefill/TTFT dominated rather than an
Attention ITL or KV-capacity failure.

## Short-scan validity

The completed 128-session PD controls independently bracket the same Standard
boundary:

| PD C | Raw req/s | Standard goodput / fraction | Relaxed goodput / fraction |
| ---: | ---: | ---: | ---: |
| 8 | 1.859 | 1.859 / 100% pass | 1.859 / 100% pass |
| 10 | 2.139 | 2.139 / 100% pass | 2.139 / 100% pass |
| 12 | 2.268 | 2.233 / 98.44% pass | 2.264 / 99.80% pass |
| 16 | 1.993 | 1.491 / 74.80% fail | 1.748 / 87.70% fail |

At PD C12, reducing 512 to 192 requests changes raw throughput by +2.0% and
Standard goodput by +2.5%, without changing the pass decision. At C16,
Standard still fails, but Relaxed changes from fail in 512 requests to pass
in 192. Short scans are therefore adequate for locating the Standard
boundary, but not for final Relaxed/P95/P99 evidence.

The 128-session PAP C8 and C12 request streams completed, but their launchers
failed the release-count routing audit despite zero active sessions after
drain. They are diagnostic only and are excluded from the comparison.

## Early stop

PAP C20 and C24 were stopped before request execution because valid C12/C16
already bracket the PAP boundary and satisfy the registered falsification
condition. Their incomplete startup directories contain no performance
evidence.

## Provenance

- Full-size staging bundle:
  `benchmarks/pap/experiments/_staging/capacity/20260729_l12_nearlimit_kv_capacity_r1/`
- Discovery staging bundle:
  `benchmarks/pap/experiments/_staging/capacity/20260729_l12_nearlimit_kv_capacity_s48_r1/`
- Discovery protocol commit: `8b9228871`
- Full-size dataset SHA-256:
  `c9c7b6e36d8a45b2d87d8af308ecdc66f9006b429502fbd7820a0ec85555f78b`

## Successor

L13 tests the causal interpretation: PAP reserves 20 of 92 visible SMs for
Attention on every PA, leaving only 72 for the much larger near-limit Prefill.
A Prefill-heavier static partition is a bounded intervention before deciding
whether PAP needs dynamic temporal resource allocation.
