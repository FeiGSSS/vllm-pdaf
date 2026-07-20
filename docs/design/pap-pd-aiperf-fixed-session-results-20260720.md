# PAP versus PD: fixed-session AIPerf capacity scan

## Scope

This clean controlled scan fixes the amount of work at every matrix point:
96 conversations, ten turns per conversation, and 960 requests. Concurrency
only controls how many conversations AIPerf keeps active. When one conversation
finishes all ten turns, another conversation takes its slot.

- vLLM/PAP commit: `c6c29817cb27c9cc466fca61749a147cae00abf8`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0 at `854ff91a4a221f899b806e7660a89b41b80d5689`
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Workload: 8K initial user text, +512 user tokens on turns 2-10,
  exactly 256 output tokens per turn
- Delay schedule: `0,3,3,1,3,3,1,3,3,1` seconds
- Dataset SHA-256:
  `9c10aab8d3a0bcedc8289415685984799ae49dc8e09e5b67b8e5b18f3db8d6bc`

PAP uses 3PA1P, `gpu_memory_utilization=0.76`, and the accepted static
72/20-SM path. PD uses one-way transfer and `gpu_memory_utilization=0.90`.
Both use `max_model_len=20000`, `max_num_batched_tokens=8192`, and
`max_num_seqs=32`.

The scan deliberately retains only the useful region identified by the pilot
runs:

| Topology | Tested concurrency |
| --- | --- |
| PAP 3PA1P | 16, 24, 32 |
| PD 1P3D | 8 |
| PD 2P2D | 12, 16 |
| PD 3P1D | 4, 8, 12, 16 |

This replaces the earlier `total sessions = C` pilot methodology for ongoing
goodput comparisons. It retains burst admission for the first cohort of `C`
conversations; the remaining conversations enter as slots become available.

## Validity

Eight of ten points completed normally. Across those points, all 768
conversations and all 7,680 requests completed, every request produced exactly
256 output tokens, all routing audits passed, and conversation migration was
zero. PAP assigned 32 conversations to each PA node. Each PD topology also
distributed conversations evenly across equivalent Prefill and Decode nodes.

The measured continuation gaps confirm that the dataset timing was applied:

| Gap | Samples | Minimum | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Think | 4,608 | 3,000.4 ms | 3,002.0 ms | 3,002.7 ms |
| Tool | 2,304 | 1,000.5 ms | 1,002.0 ms | 1,002.7 ms |

The first cohort start spread was 0.14-20.44 ms across the eight complete
points. This remains a concurrency test with an initial cohort burst, not an
arrival-rate or staggered-admission test.

PD 2P2D C16 and PD 3P1D C16 entered severe overload. They were stopped once a
95% relaxed-SLO result was mathematically impossible: a 960-request point can
contain at most 48 bad requests, while the partial records already contained
147 and 49 relaxed-SLO violations, respectively. Their missing aggregate
throughput and failed correctness/routing fields are therefore intentional
fail-closed results, not successful partial measurements.

## Results

A request meets an SLO only when both TTFT and request-level mean ITL are below
the tier limits. A point passes only when at least 95% of all 960 expected
requests meet the SLO and the complete correctness and routing gate passes.

| Architecture | Topology | C | Completed | TTFT p95 ms | ITL p95 ms | Req/s | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 3PA1P | 16 | 960/960 | 2,005.49 | 49.45 | 1.087 | fail | pass | pass |
| PAP | 3PA1P | 24 | 960/960 | 17,280.20 | 60.76 | 1.309 | fail | fail | pass |
| PAP | 3PA1P | 32 | 960/960 | 32,026.78 | 80.50 | 0.986 | fail | fail | fail |
| PD | 1P3D | 8 | 960/960 | 129,626.26 | 32.77 | 0.275 | fail | fail | fail |
| PD | 2P2D | 12 | 960/960 | 4,912.91 | 38.76 | 0.904 | pass | pass | pass |
| PD | 2P2D | 16 | 628/960 | 315,940.39 | 43.02 | - | fail | fail | fail |
| PD | 3P1D | 4 | 960/960 | 3,883.02 | 33.80 | 0.369 | pass | pass | pass |
| PD | 3P1D | 8 | 960/960 | 4,050.72 | 43.23 | 0.604 | pass | pass | pass |
| PD | 3P1D | 12 | 960/960 | 4,918.19 | 52.29 | 0.768 | fail | pass | pass |
| PD | 3P1D | 16 | 339/960 | 28,843.76 | 53.36 | - | fail | fail | fail |

## Compliant goodput

Goodput is the number of correct requests per wall-clock second that meet both
latency limits. A configuration is eligible below only if the entire point is
correct and at least 95% of its 960 requests meet the tier.

| SLO | PAP best compliant | PD best compliant | PAP versus PD |
| --- | ---: | ---: | ---: |
| Strict | no passing tested point | 0.862 req/s (2P2D C12) | inconclusive |
| Standard | 1.084 req/s (C16) | 0.885 req/s (2P2D C12) | +22.5% |
| Relaxed | 1.280 req/s (C24) | 0.899 req/s (2P2D C12) | +42.5% |

Every request has 256 output tokens. The standard result is therefore 277.6
versus 226.7 compliant output tokens/s, and the relaxed result is 327.7 versus
230.0 compliant output tokens/s.

The strict row is not a PD win. PAP C16 delivered 1.029 strict-good req/s, but
only 909/960 requests met the strict limits. It missed the 912-request passing
threshold by three requests and is ineligible. Because this lean scan did not
test PAP below C16, it does not establish PAP's best strict-compliant point.

Similarly, PAP C24's 1.198 standard-good req/s is not used in the standard
comparison because only 879/960 requests met the tier. This distinction keeps
high non-compliant throughput from being reported as SLO capacity.

## Conclusion and next gate

With identical total work at every point, PAP has a clear advantage under the
standard and relaxed SLOs. Its largest passing tested concurrency is C16 versus
PD C12 for standard, and C24 versus PD C12 for relaxed. More importantly, its
best compliant goodput is 22.5% and 42.5% higher, respectively.

This is one clean repetition per point, so it is development evidence rather
than a formal release claim. A minimal confirmation should repeat only PAP C16,
PAP C24, and PD 2P2D C12. If strict SLO performance becomes important, add only
PAP C12 rather than reopening the full concurrency sweep.

The complete 523 MiB raw result remains local under
[`20260720_172558_aiperf_capacity`](../../test/baseline/pap/results/capacity/20260720_172558_aiperf_capacity/capacity_results.md).
