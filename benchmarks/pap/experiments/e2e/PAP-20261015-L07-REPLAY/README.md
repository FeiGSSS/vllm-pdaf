# E2b: July L07 replay on the current runtime

Request: `paper/handoff/requests/R-002-phase1-e2e-e2a-e2b.md` (E2b), with the
dataset decision in `paper/handoff/requests/X-002-l07-replay-session-isolation.md`.

## Question

July's L07 (`origin/feature/pap:benchmarks/pap/experiments/PAP-20260729-RESEARCH-L07`)
found PAP at 0.63× PD 6P2D's Standard-SLO goodput on a short-output (O16)
multi-turn workload. Does that gap remain on the current PAP runtime?

## Protocol

- Dataset `multiturn/l07-s128-t5-o16-seed42-session-isolated`: July's
  generator parameters and seed (sampled append mean 697.338, as in July);
  128 sessions × 5 turns = 640 requests; a per-session first line replaces
  July's `cache_salt` so sessions share no KV block (X-002 option 1).
- Architectures, all on source `9b814d13` with the fixed PAT build:
  PAP 7PA1P and 6PA2P (`pap.env`, KV migration on, `auto` kernels); Dynamo PD
  6P2D and DP8 (`dynamo.env` = the E2a-confirmed baseline,
  `PAP-20260907-ALIGNED-C30/results/common.env`). Closed-loop concurrency,
  one service start per run; `run.sh` waits for idle GPUs before starting.
- Points: the request's grid, two rounds interleaved across architectures
  (`run_round.sh`). Because both PAP topologies were still feasible at their
  top point, PAP was extended to C56, C64, C72 and then C96 (a coarse step
  instead of +8, to bound the search). Boundary points and the next point up
  were taken to three runs.
- July's rule (`analyze.py`): a point is feasible for a tier if every run has
  ≥95% of requests within both the TTFT and ITL limits (Strict 5 s/50 ms,
  Standard 10 s/75 ms, Relaxed 20 s/100 ms); its goodput is the lowest run's
  good requests per second; capacity is the best feasible goodput. D2 (with
  E2E ≤ 20 s) is also computed.

## Results (2026-10-15)

71 runs, all valid (640/640 requests, all audits passed). Per point
(`results/points.csv`; worst run for tails, lowest run for throughput):

| Arch | C | Runs | req/s (min) | TTFT P95 (ms) | ITL P95 (ms) | Feasible (Strict/Std/Relaxed) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PAP 7PA1P | 24 | 2 | 12.83 | 1500 | 31.9 | S S R |
| | 30 | 2 | 14.51 | 1852 | 34.9 | S S R |
| | 34 | 2 | 15.65 | 2317 | 40.3 | S S R |
| | 40 | 2 | 15.64 | 2707 | 42.7 | S S R |
| | 48 | 3 | 17.15 | 3555 | 47.6 | S S R |
| | 56 | 3 | 16.50 | 4942 | 56.1 | – S R |
| | 64 | 3 | 17.43 | 6922 | 54.3 | – S R |
| | 72 | 3 | 18.45 | 6416 | 56.5 | – S R |
| | 96 | 2 | 17.87 | 8728 | 67.8 | – S R |
| PAP 6PA2P | 24 | 2 | 12.72 | 1695 | 29.7 | S S R |
| | 32 | 2 | 14.92 | 2291 | 31.5 | S S R |
| | 40 | 3 | 15.57 | 3535 | 34.8 | S S R |
| | 48 | 3 | 16.22 | 4187 | 40.7 | – S R |
| | 56 | 3 | 16.03 | 4983 | 41.9 | – S R |
| | 64 | 3 | 16.54 | 6514 | 44.7 | – S R |
| | 72 | 3 | 17.15 | 8277 | 45.9 | – S R |
| | 96 | 2 | 16.31 | 10395 | 49.3 | – – R |
| PD 6P2D | 31 | 3 | 13.64 | 1978 | 46.4 | S S R |
| | 38 | 3 | 13.65 | 2852 | 73.9 | – S R |
| | 44 | 3 | 14.71 | 3536 | 98.6 | – – R |
| | 48 | 3 | 15.06 | 3806 | 91.2 | – – R |
| | 56 | 3 | 13.71 | 5208 | 143.8 | – – – |
| DP8 | 8 | 3 | 5.70 | 879 | 32.0 | S S R |
| | 18 | 3 | 9.78 | 1190 | 81.5 | – – R |
| | 28 | 3 | 11.98 | 1356 | 129.8 | – – – |
| | 36 | 2 | 12.59 | 1565 | 156.3 | – – – |
| | 44 | 2 | 13.31 | 1872 | 191.1 | – – – |

Capacity (good req/s) under July's rule, with July's L07 values:

| SLO | Best PAP | PD 6P2D | DP8 | PAP/PD | July PAP/PD | PAP/DP8 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Strict | 7PA1P C48, 16.61 | C31, 13.39 | C8, 5.64 | **1.24** | 0.68 | 2.94 |
| Standard | 7PA1P C72, 18.45 | C31, 13.55 | C8, 5.69 | **1.36** | 0.63 | 3.24 |
| Relaxed | 7PA1P C72, 18.45 | C48, 14.52 | C18, 9.62 | **1.27** | 0.76 | 1.92 |

July (PAP / PD / DP): Strict 9.445 / 13.932 / 5.985; Standard 9.659 / 15.305
/ 10.994; Relaxed 11.654 / 15.405 / 14.083.

### Fairness checks (R-002)

1. Turn-0 cross-session cache hit rate: 0.000 in every run (prompt cache-read
   tokens over prompt tokens of first turns).
2. PD reproduction: current 6P2D Standard capacity 13.55 vs July's 15.305,
   ratio 0.885, inside the ±15% band (barely). Strict (0.96×) and Relaxed
   (0.94×) are closer. July's PD ran eager with
   `PREFILL_MAX_NUM_BATCHED_TOKENS=32768`; the current baseline uses CUDA
   Graphs and 2048.

## Conclusion

- **Verdict: "contrary result removed".** PAP/PD is 1.24–1.36 across all three
  tiers (July 0.63–0.76), and both fairness checks pass.
- PD's capacity is close to July's; the change comes from PAP, whose Standard
  capacity rose from 9.66 to 18.45 good req/s (1.9×) between the July runtime
  and the current one with Dynamo routing and KV migration.
- PD 6P2D is ITL-bound: its ITL P95 passes 75 ms by C44, while PAP 7PA1P stays
  at ≤57 ms through C72. PAP's Standard limit is TTFT; 7PA1P was still
  feasible at C96 but its goodput had already fallen from the C72 peak.
- DP8 is weaker than in July (Standard 5.69 vs 10.99): its ITL P95 already
  exceeds 75 ms at C18. July's DP config differed (eager, larger batches);
  this was not part of the request and was not investigated.
- 7PA1P beats 6PA2P in every tier; the second Projection GPU does not pay for
  the lost Prefill GPU on this workload.

## Correction (2026-10-16)

The DP8 grid above (C8, C18, …) skipped DP8's Standard knee.
`PAP-20261015-DP8-STRENGTH` (same dataset and rule, C8–C32 in steps of 4–8)
measures DP8 capacity Strict 7.45 (C12), Standard 8.91 (C16), Relaxed 10.12
(C20). PAP/DP8 is therefore 2.23 / 2.07 / 1.82; the DP8 bullet in the
conclusion overstates its weakness. PAP/PD is unaffected.
