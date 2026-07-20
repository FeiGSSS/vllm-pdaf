# PAP versus PD: AIPerf think/tool capacity scan

## Scope

This clean controlled scan adds deterministic inter-turn waiting to the fixed
four-GPU AIPerf workload. It retains burst admission for the first request of
every session and changes only continuation timing.

This historical scan used `total sessions = C`, so each point contained one
cohort. It is superseded as the long-running goodput methodology by the fixed
96-session scan; the measurements remain useful pilot evidence.

- vLLM/PAP commit: `62dc45439dc066c74532a2a4a5fde0f04fc26b5a`
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Workload: ten turns, 8K initial user text, +512 user tokens per turn,
  exactly 256 output tokens per turn
- Delay schedule: `0,3,3,1,3,3,1,3,3,1` seconds
- Delay meaning: six human-think gaps and three tool-execution gaps per session
- Dataset SHA-256:
  `927a9dc37d12a2531fbf144dd0b3870a336724a0033a2df28ba4f15848039359`

The prior capacity-boundary runs spent 11.5-15.8 seconds per request on
average. Think time is therefore fixed at 3 seconds and tool time at 1 second:
large enough to desynchronize continuations, but small enough that serving time
still dominates each session.

PAP remains fixed at 3PA1P with 0.76 GPU-memory utilization and the static
72/20-SM path. PD remains one-way with 0.90 GPU-memory utilization and tests
1P3D, 2P2D, and 3P1D. The concurrency points and three SLO tiers are unchanged.

## Validity

All 16 executed points passed correctness and routing audits. All 188 sessions
completed all ten turns: 1,880/1,880 requests completed with exactly 256 output
tokens and no conversation migration.

The observed continuation gaps confirm that AIPerf applied the dataset timing:

| Gap | Samples | Minimum | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Think | 1,128 | 3,001.0 ms | 3,002.1 ms | 3,002.7 ms |
| Tool | 564 | 1,001.2 ms | 1,002.0 ms | 1,002.5 ms |

The first-turn start spread remains 0.11-18.5 ms across points. This is an
inter-turn think/tool experiment, not a staggered session-arrival experiment.

## SLO goodput

Goodput is the number of correct requests per second that meet both the tier's
TTFT and ITL limits. The primary comparison below uses the highest observed
goodput, whether or not 95% of all requests pass the tier.

| SLO | PAP best observed | PD best observed | PAP versus PD |
| --- | ---: | ---: | ---: |
| Strict | 0.953 req/s (C16) | 0.952 req/s (2P2D C16) | approximately equal |
| Standard | 1.272 req/s (C24) | 0.952 req/s (2P2D C16) | +33.7% |
| Relaxed | 1.339 req/s (C24) | 0.972 req/s (2P2D C16) | +37.8% |

The strict maxima both have 90% SLO attainment. The best-observed PD points for
standard and relaxed have 90.0% and 91.9% attainment, while PAP C24 has 95% and
100%, respectively.

Restricting the comparison to configurations where at least 95% of requests
meet the SLO gives:

| SLO | PAP compliant | PD compliant | PAP versus PD |
| --- | ---: | ---: | ---: |
| Strict | 0.350 req/s (C4) | 0.340 req/s (3P1D C4) | +3.2% |
| Standard | 1.272 req/s (C24) | 0.572 req/s (3P1D C8) | +122% |
| Relaxed | 1.339 req/s (C24) | 0.880 req/s (2P2D C12) | +52.2% |

Every request produces 256 tokens, so multiplying request goodput by 256 gives
output-token goodput without changing the ordering.

## Capacity and conclusion

The maximum passing tested concurrency is unchanged from the no-delay scan:

| SLO | PAP 3PA1P | Best PD | Capacity ratio |
| --- | ---: | ---: | ---: |
| Strict | 4 | 4 (3P1D) | 1.0x |
| Standard | 24 | 8 (3P1D) | 3.0x |
| Relaxed | 24 | 12 (2P2D or 3P1D) | 2.0x |

PAP has no meaningful strict-goodput advantage, but it has a clear standard and
relaxed SLO-goodput advantage. The think/tool schedule lowers raw request rate
for every architecture because each session intentionally waits 21 seconds;
raw throughput should therefore not be compared directly with the previous
zero-delay workload.

The complete generated table is under
[`20260720_62dc45439_aiperf_think_tool`](../../test/baseline/pap/results/capacity/20260720_62dc45439_aiperf_think_tool/capacity_results.md).
This is a one-repetition development result, not a formal release claim.
