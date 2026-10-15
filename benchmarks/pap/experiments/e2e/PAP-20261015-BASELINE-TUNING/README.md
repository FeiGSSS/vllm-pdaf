# E2a: single-knob tuning of the PD 6P2D and DP8 baselines

Request: `paper/handoff/requests/R-002-phase1-e2e-e2a-e2b.md` (E2a).

## Question

Are the paper's Dynamo PD 6P2D and DP8 baselines already tuned, or does one
knob change improve D2 goodput or throughput beyond its CI?

## Protocol

- `dynamo.env` is a byte copy of `PAP-20260907-ALIGNED-C30/results/common.env`
  (the baseline). `s60-t3-native32k-stratified-seed42`, closed-loop C30, one
  service start per run.
- One knob per variant:

| Variant | Architecture | Change |
| --- | --- | --- |
| `6p2d-base` | 6P2D | none |
| `6p2d-pbt4096`, `6p2d-pbt8192` | 6P2D | `DYNAMO_PREFILL_MAX_NUM_BATCHED_TOKENS` 4096, 8192 (base 2048) |
| `6p2d-ls1`, `6p2d-ls4` | 6P2D | `DYNAMO_ROUTER_PREFILL_LOAD_SCALE` 1.0, 4.0 (base 2.0) |
| `dp8-base` | DP8 | none |
| `dp8-abt8192` | DP8 | `DYNAMO_AGG_MAX_NUM_BATCHED_TOKENS` 8192 (base 2048) |

- `run_all.sh` runs all seven variants in one round, three rounds (21 runs),
  so every variant is interleaved with its baseline in time.
- Every 6P2D run is followed by `analyze_dynamo_ttft.py` (TTFT breakdown).

## Criterion (from R-002)

A setting whose D2 goodput or throughput beats its baseline with a CI that
excludes zero replaces the baseline in all later experiments.

## Results

Pending.
