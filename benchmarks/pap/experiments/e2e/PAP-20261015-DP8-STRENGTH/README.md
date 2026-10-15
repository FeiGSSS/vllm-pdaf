# R-005: is the DP8 baseline a strawman on L07?

Request: `paper/handoff/requests/R-005-dp8-strength-check.md`.

## Question

On the session-isolated L07 replay, current DP8 reaches only 5.69 good req/s
under the Standard SLO (July's DP: 10.99). Do July's batch settings recover
the gap?

## July's DP versus the current DP8

July (`origin/feature/pap:benchmarks/pap/scripts/run_dp_multiturn.sh`, called
by `run_capacity_matrix.sh`) and the current baseline
(`PAP-20260907-ALIGNED-C30/results/common.env` via Dynamo) differ in:

| Setting | July DP | Current DP8 |
| --- | --- | --- |
| Serving | 8 independent `vllm serve` instances | Dynamo frontend + 8 aggregated workers |
| Routing | AIPerf client, `sticky-user-sessions` (one session → one instance) | Dynamo KV-aware router |
| Execution | `--enforce-eager` | CUDA Graphs (capture up to 128), v2 model runner |
| `max_num_batched_tokens` | 32768 | 2048 |
| `max_num_seqs` | 256 | 256 |

The current Dynamo runner has no eager option and no client-side sticky
routing, so this experiment tests only the batched-token limit without a
runner change.

## Protocol

- Dataset `multiturn/l07-s128-t5-o16-seed42-session-isolated`, July's rule
  (`analyze.py`, as in `PAP-20261015-L07-REPLAY`).
- `dp8base`: current baseline. `dp8b32k`: baseline with
  `DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS=32768`.
- Concurrency C8, 12, 16, 20, 24, 32; two rounds interleaving both settings
  (`run_round.sh <k>`), 24 runs.

## Criterion (from R-005)

If a setting reaches ≥ 0.9× July's 10.99 good req/s (Standard), it replaces
DP8 in R-003 runs not yet started.

## Results (2026-10-16)

24 valid runs (640/640 requests, audits passed). Per point (lowest run for
throughput, worst run for tails):

| Setting | C | req/s (min) | TTFT P95 (ms) | ITL P95 (ms) | Feasible (Strict/Std/Relaxed) |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 8 | 5.75 | 855 | 28.5 | S S R |
| | 12 | 7.76 | 1055 | 47.0 | S S R |
| | 16 | 9.30 | 1170 | 69.4 | – S R |
| | 20 | 10.46 | 1219 | 84.9 | – – R |
| | 24 | 11.09 | 1316 | 109.1 | – – – |
| | 32 | 12.50 | 1524 | 137.3 | – – – |
| batch 32768 | 8 | 5.88 | 845 | 27.1 | S S R |
| | 12 | 7.94 | 1005 | 43.4 | S S R |
| | 16 | 9.40 | 1124 | 75.4 | – – R |
| | 20 | 10.53 | 1213 | 88.3 | – – R |
| | 24 | 11.21 | 1284 | 116.5 | – – – |
| | 32 | 11.71 | 1547 | 148.4 | – – – |

Capacity under July's rule (good req/s):

| SLO | Baseline | Batch 32768 | July DP |
| --- | --- | --- | ---: |
| Strict | C12, 7.45 | C12, 7.70 | 5.985 |
| Standard | C16, 8.91 | C12, 7.90 | 10.994 |
| Relaxed | C20, 10.12 | C20, 10.25 | 14.083 |

## Conclusion

- Neither setting reaches 0.9× July's Standard capacity (9.89): baseline
  0.81×, batch 32768 0.72×. The current DP8 stays the baseline for R-003.
- The batched-token limit is not the cause: 32768 changes ITL P95 by a few ms
  either way, and at C16 it crosses the 75 ms limit (75.4 ms) where the
  baseline does not (69.4 ms).
- **Correction to `PAP-20261015-L07-REPLAY`:** that grid (C8, C18, …)
  skipped DP8's Standard knee at C12–C16, so it reported 5.69 good req/s.
  With this denser grid DP8's Standard capacity is 8.91, Relaxed 10.12.
- The remaining gap to July (0.81× Standard, 0.72× Relaxed) can only come from
  the settings this runner cannot change: eager execution and client-side
  sticky session routing over eight independent instances. DP8's limit here is
  ITL: P95 rises from 47 ms at C12 to 85 ms at C20, while TTFT P95 stays
  ≤1.5 s.
