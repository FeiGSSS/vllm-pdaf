# PAP versus PD: randomized-length AIPerf capacity scan

## Scope

This controlled four-GPU scan is the first PAP/PD capacity comparison using
randomized input and output lengths. It supersedes the fixed-length scans for
new development claims.

- vLLM/PAP commit: `84d19035eb2f1fb9b00f2fcb30eecbe80213895f`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Work per point: 32 conversations, ten turns, 320 requests
- Timing: pure conversation concurrency with the delay schedule
  `0,3,3,1,3,3,1,3,3,1` seconds
- Dataset seed: 42
- Dataset SHA-256:
  `2880b1e61f7bad0541d2a5cf1ddc54dcb2a180b7125eca60706a8c361638b880`

Lengths use AIPerf's lognormal mean/median parameterization, followed by the
recorded bounds:

| Dimension | Configured mean / median | Bounds | Sample mean / median | Sample range |
| --- | ---: | ---: | ---: | ---: |
| Initial user content | 8,192 / 8,000 | 4,096-11,264 | 8,146.406 / 8,349 | 4,999-11,264 |
| Later-turn user content | 512 / 500 | 256-768 | 508.014 / 499 | 289-768 |
| Output | 32 / 30 | 16-64 | 32.594 / 31 | 16-64 |

The output sample contains 46 distinct lengths and has p95 54 tokens. After
chat templating and conversation accumulation, the online request input length
has mean 10,740.038, median 10,697.5, and range 5,017-16,195 tokens. This tiny
difference from the offline estimate of 10,740.125 is expected tokenizer and
chat-template variation. The largest estimated request remains 3,776 tokens
below `max_model_len=20000`.

PAP uses 3PA1P, `gpu_memory_utilization=0.76`, and the accepted static
72/20-SM path. PD uses one-way P-to-D transfer and
`gpu_memory_utilization=0.90`. Both use `max_num_batched_tokens=8192` and
`max_num_seqs=32`.

## Validity

Ten matrix points were executed. Every point completed all 320 requests, so
the result contains 3,200 complete request records. All per-request output
targets matched, all correctness and routing audits passed, request errors
were zero, and conversation migration was zero.

The runner stopped each topology after its first valid relaxed-SLO failure.
PAP C32 and PD 2P2D C24 were therefore skipped rather than launched. This is
one clean repetition per point, so the result is controlled development
evidence, not a formal three-repetition release claim.

A request meets a tier only when both TTFT and request-level mean ITL meet its
limits. A point passes only when it is complete and correct and at least 95%
of all 320 requests meet the tier:

| Tier | TTFT | ITL |
| --- | ---: | ---: |
| Strict | 5,000 ms | 50 ms |
| Standard | 10,000 ms | 75 ms |
| Relaxed | 20,000 ms | 100 ms |

## Results

| Architecture | Topology | C | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 12 | 4,058.20 | 40.89 | 2.563 | pass | pass | pass |
| PAP | 3PA1P | 20 | 9,928.83 | 42.36 | 2.805 | fail | pass | pass |
| PAP | 3PA1P | 28 | 29,383.06 | 52.66 | 1.528 | fail | fail | fail |
| PD | 1P3D | 8 | 15,398.70 | 29.25 | 1.141 | fail | fail | pass |
| PD | 2P2D | 10 | 9,112.51 | 31.57 | 1.841 | fail | pass | pass |
| PD | 2P2D | 16 | 14,873.39 | 35.89 | 1.985 | fail | fail | pass |
| PD | 2P2D | 20 | 27,154.79 | 35.86 | 1.991 | fail | fail | fail |
| PD | 3P1D | 8 | 4,364.73 | 34.29 | 1.838 | pass | pass | pass |
| PD | 3P1D | 14 | 13,926.55 | 39.02 | 2.096 | fail | fail | pass |
| PD | 3P1D | 20 | 22,111.26 | 203.53 | 1.358 | fail | fail | fail |

The tested capacity envelope is:

| SLO | PAP 3PA1P | Best PD | PAP advantage |
| --- | ---: | ---: | ---: |
| Strict | C12 | C8, 3P1D | +4 conversations / +50% |
| Standard | C20 | C10, 2P2D | +10 conversations / +100% |
| Relaxed | C20 | C16, 2P2D | +4 conversations / +25% |

## Compliant goodput

Goodput counts complete and correct requests per wall-clock second that meet
both latency limits. Only configurations with at least 95% compliant requests
are eligible for the comparison.

| SLO | PAP best compliant | PD best compliant | PAP versus PD |
| --- | ---: | ---: | ---: |
| Strict | 2.467 req/s, C12 | 1.769 req/s, 3P1D C8 | +39.4% |
| Standard | 2.674 req/s, C20 | 1.810 req/s, 3P1D C8 | +47.8% |
| Relaxed | 2.744 req/s, C20 | 2.096 req/s, 3P1D C14 | +30.9% |

## Conclusion

PAP has a clear advantage on this randomized short-output workload: it has a
higher tested concurrency envelope and higher compliant request goodput in all
three SLO tiers. The result also shows why all three PD ratios must be tested:
3P1D gives PD its best strict and goodput points, while 2P2D gives its largest
standard and relaxed tested concurrency.

The grid is deliberately coarse and does not claim the exact boundary between
passing and failing points. If a formal confirmation is needed, repeat only
PAP C12/C20, PD 3P1D C8/C14, and PD 2P2D C10 three times. There is no reason to
reopen the full matrix first.

The complete 69 MiB raw result is preserved locally under
[`20260721_aiperf_capacity_random_o32_s32`](runs/20260721_aiperf_capacity_random_o32_s32/raw/capacity_results.md).
