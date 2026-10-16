# E2d-lite: when does PAP pay off? Input × output 2×2

Request: `paper/handoff/requests/R-003-phase2-e1-e2d.md` (E2d-lite), trimmed
plan in `X-003`.

## Question

How does PAP's capacity advantage over PD 6P2D and DP8 change with input
length (short L07 ≈4K + 0.7K appends vs native-32K) and output length (short
O16/O80 vs long O256)?

## Cells

| | Short output | Long output (this experiment) |
| --- | --- | --- |
| Short input (L07) | O16: `PAP-20261015-L07-REPLAY` + `PAP-20261015-DP8-STRENGTH` | `l07o256`: `multiturn/l07-s128-t5-o256-seed42-session-isolated` (same texts and delays as the O16 file, outputs mean 256) |
| Long input (native-32K) | O80: `PAP-20261015-LOAD-SWEEP` (s240) | `n32ko256`: `agentic-code/s120-t3-native32k-o256-stratified-seed42` (120 sessions, 30 per 4K stratum, outputs mean 251) |

## Protocol

- PAP 7PA1P (KV migration on, fixed PAT), Dynamo PD 6P2D and DP8 baselines;
  the dataset is selected per variant (`variants/<cell>-<arch>-c<C>.env`).
- Coarse scan: C8, 16, 32, 48, one run each (`run_round.sh 1`, 24 runs). Then
  four points around each architecture's D2 boundary in each cell, two runs
  each.
- Capacity uses the R-003 rule (feasible if mean D2 attainment ≥ 95%) and is
  also reported with July's tiers for the L07 cell.

## Results

Pending.
