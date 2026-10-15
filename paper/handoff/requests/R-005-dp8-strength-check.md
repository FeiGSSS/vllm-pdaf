# R-005: DP8 baseline strength check

status: open
priority: P1
claims: C13
opened: 2026-10-15
from: research agent
common settings: `paper/handoff/requests/COMMON.md`

depends: none (may interleave with R-003 and R-004; do not delay R-003)

## Hypothesis

DP8's weak L07 result (Standard 5.69 vs July's fused DP 10.99 good req/s, ITL
P95 > 75 ms from C18) comes from its batch settings, not from data-parallel
serving itself; July-like batch settings recover most of the gap.

## Why it matters for the paper

R-002 Decision 2. A reviewer comparing to July or to plain vLLM DP will ask
whether DP8 is a strawman. C13 currently covers only single-knob changes at
C30 on `s60`.

## Runs

- L07 session-isolated dataset (X-002), July's capacity rule, Standard SLO.
- Two DP8 arms: current baseline and July's DP settings (eager vs CUDA Graphs
  and the batched-token limit as in
  `origin/feature/pap:benchmarks/pap/aiperf/run_capacity_matrix.sh`; list the
  exact diff in the report).
- Concurrency grid C8–C32 in steps of 4–8 around the knee; two repetitions
  per point (July's rule: the lower one counts, both must pass).

## Expected result and falsification

- If July-like DP8 reaches ≥ 0.9× July's 10.99 good req/s, DP8 is replaced by
  that setting in R-003 runs not yet started, and the report says which.
- If it stays near 5.7, record the cause (ITL source: batch size, KV pressure)
  and keep the current DP8.

## Deliverables

- `paper/handoff/reports/R-005.md` with the capacity table and the settings
  diff; update C13's Evidence column.
