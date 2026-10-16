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

## Results (2026-10-16)

59 valid runs (24 coarse, 35 fine). One run
(`20261016T050550Z_l07o256-pd6p2d-c24_r2`) started while a paper report was
being edited, so it recorded a dirty tracked worktree; it is excluded and was
rerun (`..._c24_r3`). Runtime code was identical.

Maximum feasible goodput (good req/s; feasible if mean D2 attainment ≥ 95%;
run-level 95% CI):

| Cell | SLO | PAP 7PA1P | PD 6P2D | DP8 |
| --- | --- | --- | --- | --- |
| L07 + O256 | D2 | C32, 2.29 [2.29, 2.29] | C32, 2.43 [2.43, 2.43] | ≥ C64, 4.61 [4.58, 4.65] |
| | D4 | C48, 2.71 | C48, 2.75 | C64, 4.72 |
| native-32K + O256 | D2 | C16, 1.15 [1.15, 1.15] | C8, 0.74 [0.74, 0.74] | C24, 1.67 [1.65, 1.68] |
| | D4 | C24, 1.44 | C16, 1.10 | C24, 1.74 |

D1 is infeasible for every architecture in both cells (256 output tokens do
not fit its 10 s E2E limit). DP8 was still feasible at C64, the top of the
L07 grid, so its L07 + O256 capacity is a lower bound.

Points near the boundaries (all points: `results/points.csv`):

| Cell | Arch | C: D2 attainment, ITL P95 |
| --- | --- | --- |
| L07 + O256 | PAP | 24: 99.1%, 40 ms · 32: 96.5%, 45 ms · 40: 92.9%, 51 ms · 48: 86.1%, 58 ms |
| | PD 6P2D | 24: 99.7%, 39 ms · 32: 97.6%, 44 ms · 40: 94.0%, 52 ms · 48: 87.3%, 60 ms |
| | DP8 | 40: 100%, 39 ms · 48: 99.8%, 43 ms · 56: 99.1%, 46 ms · 64: 97.7%, 50 ms |
| native-32K + O256 | PAP | 8: 100%, 37 ms · 16: 95.6%, 44 ms · 24: 85.0%, 53 ms · 32: 72.4%, 64 ms |
| | PD 6P2D | 8: 100%, 38 ms · 16: 92.1%, 51 ms · 24: 74.6%, 65 ms · 32: 41.9%, 68 ms |
| | DP8 | 16: 98.8%, 46 ms · 24: 95.1%, 55 ms · 32: 83.9%, 68 ms · 40: 65.1%, 79 ms |

### 2×2 with the short-output cells

| Input \ Output | Short | Long (O256, this experiment) |
| --- | --- | --- |
| Short (L07) | O16 (`L07-REPLAY` + `DP8-STRENGTH`, July rule, Standard): PAP 18.45, PD 13.55, DP8 8.91; **PAP/PD 1.36, PAP/DP8 2.07** | PAP 2.29, PD 2.43, DP8 ≥ 4.61; **PAP/PD 0.94, PAP/DP8 ≤ 0.50** |
| Long (native-32K) | O80 (`LOAD-SWEEP`, D2): PAP 3.63, PD 3.00, DP8 2.32; **PAP/PD 1.21, PAP/DP8 1.56** | PAP 1.15, PD 0.74, DP8 1.67; **PAP/PD 1.56, PAP/DP8 0.69** |

Bytes per request (PD: measured NIXL MB per transfer in the cell's 6P2D
runs; PAP: 737,280 B of activations per generated token):

| Cell | PD KV per transfer | PAP activations per request | PAP / PD |
| --- | ---: | ---: | ---: |
| L07 + O16 | 445 MB | 12 MB | 0.03 |
| native-32K + O80 | 1742 MB | 59 MB | 0.03 |
| L07 + O256 | 400 MB | 189 MB | 0.47 |
| native-32K + O256 | 1480 MB | 185 MB | 0.13 |

## Conclusion

- **Output length decides PAP versus DP8.** With short outputs PAP beats DP8
  by 1.6–2.1×; with 256-token outputs DP8 beats PAP in both cells (PAP/DP8
  0.69 with long inputs, ≤ 0.50 with short inputs).
- **Input length decides PAP versus PD.** With long inputs PAP beats PD for
  both output lengths (1.21 with O80, 1.56 with O256). With short inputs and
  long outputs PAP is slightly below PD (0.94; flagged per X-003).
- The bytes account matches the cell where PAP loses to PD: there PAP's
  per-request activation traffic reaches 47% of PD's one-time KV transfer,
  against 3–13% in the other cells.
- With long outputs, D2's 20 s E2E limit binds for PAP and PD at the
  boundary (mean E2E 11–17 s), not only TTFT and ITL.
- PAP's advantage region is therefore long input with short-to-moderate
  output — the multi-turn agentic shape — and it does not extend to
  short-prompt, long-generation traffic.
