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

## Results

Pending.
