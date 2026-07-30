# PAP final-commit and Prefill-admission experiment

Date: 2026-07-30

> Update: the PD rows below use Prefill `max_num_seqs=256`. A controlled
> rerun with PD Prefill `max_num_seqs=1` materially improves the PD baseline
> and supersedes the cross-architecture effect sizes in this report. See
> `PAP-20260730-PD-PREFILL-MAX-SEQS1/report.md`.

## Question

Can PAP return `[DONE]` after submitting final KV lifecycle updates, without
the Prefill/Attention overload that appeared after removing synchronous
per-token commit acknowledgements?

## Controlled workload

- Qwen3-8B, 8x L20, PAP 7PA1P, static MPS 80/20
- 60 sessions, 3 turns, concurrency 20--32
- 180 requests; mean output 101.98 tokens
- initial context about 9.5K tokens; later turns about 18.9K and 28.6K
- dataset SHA256:
  `4faa9f1cf3423f11f83cbf38bad19f2c73865e290b99bfc7e716f84cf9e8ea7b`
- Standard SLO: TTFT <= 10 s and average request ITL <= 75 ms

All eligible rows completed 180/180 requests with no correctness error.

## C20 root-cause controls

| Variant | TTFT avg | ITL avg | Throughput | Standard goodput | Standard pass |
| --- | ---: | ---: | ---: | ---: | --- |
| PD 6P2D baseline | 4.255 s | 62.51 ms | 1.513 req/s | 1.337 req/s | no |
| PAP old execution-ack baseline | 4.535 s | 45.35 ms | 1.629 req/s | 1.611 req/s | yes |
| PAP final-only, submit-only | 6.744 s | 52.92 ms | 1.463 req/s | 1.235 req/s | no |
| PAP final-only, Prefill token budget 16K | 7.000 s | 54.39 ms | 1.462 req/s | 1.105 req/s | no |
| PAP final-only, load placement, no migration | 6.520 s | 59.42 ms | 1.391 req/s | 1.051 req/s | no |
| PAP final-only, execution-ack | 4.536 s | 46.19 ms | 1.584 req/s | 1.487 req/s | no |
| PAP final-only, submit-only, Prefill admission=1 | **4.388 s** | **43.56 ms** | **1.872 req/s** | **1.820 req/s** | **yes** |

The C20 admission result improves over PD 6P2D by 23.7% in raw request
throughput and 36.2% in Standard goodput. Its ITL is 30.3% lower, while
average TTFT is 3.1% higher.

## Capacity extension

The same dataset and SLO were used at every concurrency. C23, C24, and C32
use the final per-PA FIFO admission queue. C20 predates FIFO ordering; C28 is
retained only as supporting pre-FIFO evidence.

| C | Architecture | TTFT avg / p95 | ITL avg | Throughput | Standard goodput / good fraction | Pass |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 20 | PD 6P2D | 4.255 / 8.584 s | 62.51 ms | 1.513 | 1.337 / 88.3% | no |
| 20 | PAP 7PA1P admission=1 | 4.388 / 8.480 s | 43.56 ms | 1.872 | 1.820 / 97.2% | yes |
| 23 | PAP 7PA1P FIFO admission=1 | 5.138 / 9.728 s | 44.96 ms | 1.896 | 1.801 / 95.0% | yes |
| 24 | PD 6P2D | 6.261 / 11.388 s | 62.69 ms | 1.554 | 1.019 / 65.6% | no |
| 24 | PAP 7PA1P FIFO admission=1 | 5.363 / 10.175 s | 45.95 ms | 1.902 | 1.796 / 94.4% | no |
| 28 | PD 6P2D | 7.852 / 15.147 s | 68.61 ms | 1.504 | 0.535 / 35.6% | no |
| 28 | PAP 7PA1P admission=1, pre-FIFO | 7.321 / 16.421 s | 46.58 ms | 1.781 | 1.464 / 82.2% | no |
| 32 | PD 6P2D | 9.152 / 14.953 s | 70.33 ms | 1.670 | 0.584 / 35.0% | no |
| 32 | PAP 7PA1P FIFO admission=1 | 8.533 / 20.676 s | 46.36 ms | 1.862 | 1.438 / 77.2% | no |

The largest Standard-SLO concurrency verified for PAP is C23. C24 misses the
95% threshold by one request: 170/180 requests are good. PD already fails
Standard at C20. At C32, PAP still exceeds PD by 11.5% in raw throughput and
146.0% in Standard goodput, with 34.1% lower average ITL. PAP's
high-concurrency failure is therefore a TTFT queueing limit, not a Decode
throughput or correctness collapse.

## Mechanism

Submit-only control changed the closed-loop arrival pattern. The mean
time-weighted number of simultaneous Prefill requests per PA rose from about
1.11 to 1.35. Second- and third-turn Prefill service time increased from
4.08/5.03 seconds to 6.13/7.32 seconds.

Waiting for final control execution restored performance, but added about
1.19 seconds after the last token of each HTTP request. This confirmed that
the old cleanup path acted as implicit admission rather than making
per-token KV updates useful.

The explicit policy admits at most one in-flight Prefill request per PA. It
keeps final commit and lease updates submit-only, so the prior request can
return `[DONE]` without waiting for Prefill scheduler execution. Queueing is
visible at the gateway and is audited per PA.

The initial condition-variable implementation woke every waiter without
preserving arrival order. At C32, first-turn admission wait p95 reached
21.49 seconds while third-turn p95 was 9.10 seconds, showing that later
requests could overtake older requests. Per-PA FIFO ordering reduced C32
TTFT p95 from 21.78 to 20.68 seconds and raised Standard goodput from 1.396
to 1.438 requests/s. At C24 it raised Standard goodput from 1.695 to 1.796
requests/s.

## Rejected controls

- Admission=2 at C24 reduced average Gateway wait from 2.27 to 1.04 seconds,
  but raised Prefill service from 2.99 to 4.61 seconds and ITL from 45.89 to
  54.53 ms. Standard goodput fell from 1.695 to 1.141 requests/s.
- A 21K estimated-context token budget at C32 allowed two first-turn
  Prefills. It reduced TTFT p95 from 21.78 to 18.17 seconds, but raised ITL
  to 49.81 ms and reduced Standard goodput to 1.276 requests/s. The
  experimental code was removed.
- Work-conserving first-turn placement caused retained conversation owners
  to become imbalanced: routed requests ranged from 21 to 33 per PA. The
  routing audit rejected the run, and the experimental code was removed.

## Evidence

- submit-only baseline:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c20_r1`
- 16K token-budget control:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c20_prefill16k_r1`
- load-placement control:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c20_loadplace_nomigrate_r1`
- execution-ack control:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c20_execack_r1`
- explicit admission C20:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c20_prefill_admit1_r1`
- FIFO C23 boundary:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c23_prefill_admit1_fifo_r1`
- FIFO C24:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c24_prefill_admit1_fifo_r1`
- FIFO C32:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c32_prefill_admit1_fifo_r1`
- rejected admission=2:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c24_prefill_admit2_r1`
- rejected 21K token budget:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c32_prefill_tokens21k_r1`
- rejected work-conserving placement:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c32_prefill_admit1_workconserving_r1`

## Scope

Each point has one controlled repetition. The result supports retaining the
configurable admission mechanism and FIFO ordering, but not making
admission=1 a universal default before checking shorter-context workloads
and repeating the C23/C24 boundary.
