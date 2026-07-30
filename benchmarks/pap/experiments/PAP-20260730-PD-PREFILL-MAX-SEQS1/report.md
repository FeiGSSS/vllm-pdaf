# PD long-Prefill `max_num_seqs=1` control

Date: 2026-07-30

## Question

Does the long-Prefill serialization benefit observed in PAP also improve the
PD 6P2D baseline?

## Controlled change

The experiment changes only the Prefill scheduler sequence limit:

- PD Prefill `max_num_seqs`: 256 -> 1
- PD Decode `max_num_seqs`: unchanged at 256
- Prefill `max_num_batched_tokens`: unchanged at 32768
- Decode `max_num_batched_tokens`: unchanged at 256
- execution mode: eager
- same-node NIXL: UCX 1.22.0 with protocol emulation disabled

The AIPerf dataset is byte-identical to the earlier baseline:

- 60 sessions, 3 turns, 180 requests
- initial context about 9.5K tokens
- later-turn contexts about 18.9K and 28.6K tokens
- mean output 101.98 tokens
- Standard SLO: TTFT <= 10 s and average request ITL <= 75 ms
- SHA256:
  `4faa9f1cf3423f11f83cbf38bad19f2c73865e290b99bfc7e716f84cf9e8ea7b`

Each point restarted all services. All points completed 180/180 requests and
60/60 sessions without a correctness error.

## PD result

| C | PD variant | TTFT avg / p95 | ITL avg / p95 | Raw throughput | Standard goodput / good fraction | Pass |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 20 | Prefill max seqs 256 | 4.255 / 8.584 s | 62.51 / 78.10 ms | 1.513 req/s | 1.337 / 88.3% | no |
| 20 | Prefill max seqs 1 | **3.669 / 6.655 s** | **58.29 / 74.51 ms** | **1.637 req/s** | **1.583 / 96.7%** | **yes** |
| 24 | Prefill max seqs 256 | 6.261 / 11.388 s | 62.69 / 83.87 ms | 1.554 req/s | 1.019 / 65.6% | no |
| 24 | Prefill max seqs 1 | **4.511 / 8.874 s** | **62.26 / 76.27 ms** | **1.750 req/s** | **1.497 / 85.6%** | no |
| 32 | Prefill max seqs 256 | 9.152 / 14.953 s | 70.33 / 83.55 ms | 1.670 req/s | 0.584 / 35.0% | no |
| 32 | Prefill max seqs 1 | **6.924 / 14.662 s** | **67.20 / 77.86 ms** | **1.852 req/s** | **1.296 / 70.0%** | no |

Relative to the old PD baseline, Prefill `max_num_seqs=1`:

- raises raw throughput by 8.2%, 12.6%, and 10.9% at C20/C24/C32;
- lowers average TTFT by 13.8%, 28.0%, and 24.4%;
- raises Standard goodput by 18.4%, 47.0%, and 121.8%;
- changes C20 from Standard failure to Standard pass.

The benefit is therefore not PAP-specific. For this long-context workload,
admitting multiple long Prefills to one full GPU is an avoidable scheduler
confound in the old PD baseline.

## Fairer PAP comparison

The following table compares the optimized PD control with PAP 7PA1P using
per-PA FIFO Prefill admission of one:

| C | Architecture | TTFT avg / p95 | ITL avg | Raw throughput | Standard goodput / good fraction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | PD 6P2D, Prefill max seqs 1 | **3.669 / 6.655 s** | 58.29 ms | 1.637 req/s | 1.583 / 96.7% |
| 20 | PAP 7PA1P, admission 1 | 4.388 / 8.480 s | **43.56 ms** | **1.872 req/s** | **1.820 / 97.2%** |
| 24 | PD 6P2D, Prefill max seqs 1 | **4.511 / 8.874 s** | 62.26 ms | 1.750 req/s | 1.497 / 85.6% |
| 24 | PAP 7PA1P, admission 1 | 5.363 / 10.175 s | **45.95 ms** | **1.902 req/s** | **1.796 / 94.4%** |
| 32 | PD 6P2D, Prefill max seqs 1 | **6.924 / 14.662 s** | 67.20 ms | 1.852 req/s | 1.296 / 70.0% |
| 32 | PAP 7PA1P, admission 1 | 8.533 / 20.676 s | **46.36 ms** | **1.862 req/s** | **1.438 / 77.2%** |

The complete latency distributions are:

| C | Architecture | TTFT avg / p95 / p99 | ITL avg / p95 / p99 | Output throughput | Duration |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20 | PD 6P2D | 3.669 / 6.655 / 8.115 s | 58.29 / 74.51 / 76.06 ms | 166.96 tok/s | 109.94 s |
| 20 | PAP 7PA1P | 4.388 / 8.480 / 12.315 s | 43.56 / 51.74 / 52.12 ms | 190.92 tok/s | 96.08 s |
| 24 | PD 6P2D | 4.511 / 8.874 / 10.559 s | 62.26 / 76.27 / 78.60 ms | 178.45 tok/s | 102.86 s |
| 24 | PAP 7PA1P | 5.363 / 10.175 / 13.206 s | 45.95 / 53.10 / 53.65 ms | 193.93 tok/s | 94.59 s |
| 32 | PD 6P2D | 6.924 / 14.662 / 17.560 s | 67.20 / 77.86 / 80.33 ms | 188.86 tok/s | 97.19 s |
| 32 | PAP 7PA1P | 8.533 / 20.676 / 24.760 s | 46.36 / 55.23 / 57.76 ms | 189.91 tok/s | 96.59 s |

The three SLO tiers require at least 95% of requests to satisfy both limits:
Strict is TTFT <= 5 s and ITL <= 50 ms, Standard is TTFT <= 10 s and
ITL <= 75 ms, and Relaxed is TTFT <= 20 s and ITL <= 100 ms.

| C | Architecture | Strict goodput / fraction | Standard goodput / fraction | Relaxed goodput / fraction |
| ---: | --- | ---: | ---: | ---: |
| 20 | PD 6P2D | 0.327 / 20.0% (fail) | 1.583 / 96.7% (pass) | 1.637 / 100% (pass) |
| 20 | PAP 7PA1P | 1.165 / 62.2% (fail) | 1.820 / 97.2% (pass) | 1.872 / 100% (pass) |
| 24 | PD 6P2D | 0.301 / 17.2% (fail) | 1.497 / 85.6% (fail) | 1.750 / 100% (pass) |
| 24 | PAP 7PA1P | 0.792 / 41.7% (fail) | 1.796 / 94.4% (fail) | 1.902 / 100% (pass) |
| 32 | PD 6P2D | 0.144 / 7.8% (fail) | 1.296 / 70.0% (fail) | 1.852 / 100% (pass) |
| 32 | PAP 7PA1P | 0.341 / 18.3% (fail) | 1.438 / 77.2% (fail) | 1.748 / 93.9% (fail) |

Against the fairer PD baseline, PAP's raw-throughput advantage is 14.4% at
C20, 8.7% at C24, and 0.6% at C32. Its Standard-goodput advantage is 15.0%,
20.0%, and 10.9%, respectively. PAP retains a 25.3%--31.0% lower average
ITL, but optimized PD has lower average TTFT at all three points.

| C | PAP raw-throughput delta | PAP Standard-goodput delta | PAP average-ITL delta | PAP average-TTFT delta | PAP p95-TTFT delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | +14.4% | +15.0% | -25.3% | +19.6% | +27.4% |
| 24 | +8.7% | +20.0% | -26.2% | +18.9% | +14.7% |
| 32 | +0.6% | +10.9% | -31.0% | +23.2% | +41.0% |

Therefore, the earlier claim of a 23.7% raw-throughput and 36.2%
Standard-goodput PAP advantage at C20 was inflated by the PD scheduling
configuration. The controlled result still supports a PAP Decode-latency
and Standard-goodput advantage, but it no longer supports the original
effect size.

## TTFT compute-share interpretation

The optimized PD/PAP average-TTFT ratios are stable:

| C | PD TTFT | PAP TTFT | PD / PAP |
| ---: | ---: | ---: | ---: |
| 20 | 3.669 s | 4.388 s | 0.836 |
| 24 | 4.511 s | 5.363 s | 0.841 |
| 32 | 6.924 s | 8.533 s | 0.811 |

PAP's static MPS partition assigns 72 of the L20's 92 visible SMs to
Prefill and 20 SMs to Decode Attention. The single-PA Prefill compute share
is therefore:

```text
alpha = 72 / 92 = 0.783
```

Using 90 SMs as a rough denominator gives the simpler approximation
`alpha = 0.8`. Two different ratios must not be conflated:

```text
single-node Prefill ratio = 72 / 92 = 0.783
aggregate PAP/PD Prefill ratio = (7 * 72) / (6 * 92) = 0.913
```

The aggregate ratio would be 0.933 under the 72/90 approximation. Thus the
equation `PD_TTFT * 6 = PAP_TTFT * 5.6` predicts
`PD_TTFT / PAP_TTFT = 0.933`, not 0.8.

With admission one, however, a request cannot combine the aggregate compute
of all PA nodes. One long Prefill runs on one full PD Prefill GPU or one
72-SM PAP PA. A useful latency model is:

```text
PD_TTFT(C) =
    S
  + W_pd(C, 6)

PAP_TTFT(C, N_pa) =
    S / alpha
  + W_pap(C, N_pa)
  + H_pap
```

Here `S` is full-GPU single-request Prefill service time, `W` is admission
and scheduler queueing, and `H_pap` is PAP's remaining fixed overhead. The
observed ratios of 0.811--0.841 lie between the single-node ratio 0.783 and
the aggregate ratio 0.913, but are much closer to the single-node ratio.
This supports the interpretation that long-request Prefill service time is
the dominant TTFT component, while PAP's seventh PA partially compensates
by reducing queueing.

Increasing the number of PA nodes reduces `W_pap` and should narrow the TTFT
gap while PAP has meaningful queueing. It does not change `S / alpha`, so it
cannot remove the intrinsic service-time cost of running one request on
about 80% of a GPU. In the no-queue limit, adding PA nodes has almost no
TTFT benefit. Under higher load, additional PA capacity can make PAP equal
or outperform PD once the PD Prefill queue grows sufficiently.

This yields a falsifiable scaling prediction: with the workload fixed,
increasing `N_pa` should reduce PAP admission wait while leaving measured
single-request Prefill service time nearly unchanged. The PAP/PD TTFT gap
should shrink first through queueing, rather than through faster execution
of an individual Prefill.

## Evidence

- new PD control:
  `benchmarks/pap/experiments/_staging/capacity/20260730_pd6p2d_prefill_maxseq1_c20_c24_c32_r1`
- old PD baselines:
  `benchmarks/pap/experiments/_staging/capacity/20260730_longctx_o100_s60_t3_concurrency_c{20,24,32}_r1`
- PAP admission controls:
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c20_prefill_admit1_r1`
  and
  `benchmarks/pap/experiments/_staging/capacity/20260730_final_commit_pap7pa1p_c{24,32}_prefill_admit1_fifo_r1`

## Scope

Each point has one repetition. This is controlled development evidence, not
yet a paper-ready confidence interval. The next fair comparison should use
the optimized PD Prefill setting by default and repeat only the selected
boundary points.
