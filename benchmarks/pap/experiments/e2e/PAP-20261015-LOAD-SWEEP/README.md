# E1: load sweep on native-32K multi-turn (main figure)

Request: `paper/handoff/requests/R-003-phase2-e1-e2d.md` (E1).

## Question

Across load, which architecture sustains the most goodput under the D1–D4
SLOs on native-32K multi-turn agentic-code traffic, and what limits each?

## Protocol

- Dataset `agentic-code/s240-t3-native32k-stratified-seed42` (240 sessions,
  3 turns, 720 requests, 60 final contexts per 4K stratum from 16K to 32K,
  mean output 80 tokens).
- Architectures: PAP 7PA1P (KV migration on, fixed PAT, `pap.env`), Dynamo
  PD 6P2D, PD 4P4D and DP8 (`dynamo.env`, the E2a/R-005-confirmed baseline).
- Concurrency C10–C60 in steps of 10, one service start per run, three rounds
  with C30 first in each round and all architectures interleaved (72 runs).
  Each architecture's D2 knee (last feasible point) and the next point were
  taken to five runs (16 more runs).
- `analyze.py`: a point is feasible if its mean attainment over runs is
  ≥ 95% (COMMON §2.4); goodput is the mean of good requests per second over
  runs; latency-percentile CIs bootstrap requests within runs, goodput CIs
  bootstrap runs. SLOs D1–D4 from `PAP-20260907-ALIGNED-C30/GOODPUT.md`;
  SLO scaling multiplies all three D2 limits by 0.5–2.0.
- Every PD run carries `ttft_breakdown.json` (NIXL transfer statistics).

## Consistency check (R-003)

C30 round 1: PAP/6P2D throughput 3.611/2.983 = 1.21 on s240 versus
3.159/2.703 = 1.17 on s60 (R-001 fixed PAT, E2a baseline). Same direction.

## Results (2026-10-16)

88 valid runs, none failed. Per point (`results/points.csv`; percentiles pool
all requests of the point's runs):

| Arch | C | Runs | req/s | TTFT P50 / P99 (ms) | ITL P50 / P95 / P99 (ms) | D2 attainment | KV transfer (ms) |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| PAP 7PA1P | 10 | 3 | 1.99 | 2001 / 4488 | 35.2 / 38.2 / 39.5 | 100% | |
| | 20 | 3 | 3.14 | 2585 / 7014 | 43.0 / 45.4 / 46.3 | 100% | |
| | 30 | 3 | 3.62 | 3673 / 9261 | 50.0 / 55.5 / 57.0 | 99.4% | |
| | 40 | 5 | 3.74 | 5613 / 12664 | 53.0 / 63.1 / 66.5 | 97.1% | |
| | 50 | 5 | 3.76 | 7944 / 15158 | 52.9 / 65.3 / 70.3 | 82.1% | |
| | 60 | 3 | 3.67 | 10935 / 18564 | 50.9 / 63.1 / 66.5 | 33.5% | |
| PD 6P2D | 10 | 3 | 1.96 | 2113 / 4810 | 35.3 / 40.5 / 43.0 | 100% | 116 |
| | 20 | 3 | 2.75 | 3265 / 7132 | 45.6 / 53.8 / 57.4 | 100% | 138 |
| | 30 | 5 | 3.04 | 4907 / 9409 | 55.0 / 69.0 / 73.6 | 98.6% | 182 |
| | 40 | 5 | 3.09 | 7424 / 13122 | 58.1 / 73.7 / 79.6 | 79.4% | 194 |
| | 50 | 3 | 3.01 | 11078 / 18362 | 55.1 / 72.5 / 77.3 | 31.0% | 194 |
| | 60 | 3 | 2.80 | 15361 / 29497 | 50.9 / 66.1 / 71.0 | 7.0% | 194 |
| PD 4P4D | 10 | 3 | 1.92 | 2757 / 6441 | 27.5 / 31.3 / 33.6 | 100% | 125 |
| | 20 | 5 | 2.21 | 6435 / 11008 | 28.5 / 32.9 / 35.4 | 96.7% | 133 |
| | 30 | 5 | 2.21 | 10865 / 15477 | 28.4 / 33.5 / 36.1 | 32.6% | 138 |
| | 40 | 3 | 2.00 | 16167 / 30688 | 27.7 / 32.6 / 35.6 | 3.6% | 134 |
| | 50 | 3 | 1.64 | 23365 / 52905 | 26.9 / 31.5 / 33.3 | 3.6% | 136 |
| | 60 | 3 | 1.41 | 35469 / 67277 | 26.3 / 30.0 / 31.8 | 4.0% | 139 |
| DP8 | 10 | 5 | 2.36 | 1553 / 3047 | 26.5 / 59.9 / 79.3 | 98.4% | |
| | 20 | 5 | 2.99 | 1741 / 4434 | 57.3 / 100.4 / 127.7 | 77.6% | |
| | 30 | 3 | 3.26 | 1880 / 5901 | 84.4 / 144.4 / 177.4 | 37.5% | |
| | 40 | 3 | 3.36 | 1999 / 8166 | 112.1 / 185.1 / 230.7 | 15.6% | |
| | 50 | 3 | 3.39 | 2144 / 10390 | 142.0 / 227.8 / 271.8 | 7.6% | |
| | 60 | 3 | 3.48 | 2320 / 12148 | 169.4 / 259.9 / 304.0 | 5.4% | |

Maximum feasible goodput (good req/s, run-level 95% CI):

| SLO | PAP 7PA1P | PD 6P2D | PD 4P4D | DP8 |
| --- | --- | --- | --- | --- |
| D1 (5 s / 50 ms / 10 s) | C10, 1.95 [1.94, 1.96] | C10, 1.91 [1.88, 1.92] | infeasible | infeasible |
| D2 (10 s / 75 ms / 20 s) | **C40, 3.63 [3.59, 3.67]** | C30, 3.00 [2.96, 3.03] | C20, 2.14 [2.12, 2.17] | C10, 2.32 [2.31, 2.33] |
| D3 (15 s / 50 ms / 20 s) | C20, 3.14 [3.12, 3.17] | C10, 1.96 [1.94, 1.97] | C20, 2.21 [2.20, 2.22] | infeasible |
| D4 (15 s / 75 ms / 25 s) | C40, 3.74 [3.70, 3.77] | C30, 3.03 [2.99, 3.06] | C20, 2.21 [2.21, 2.22] | C10, 2.32 [2.31, 2.33] |

D2 limits scaled by a common factor (maximum feasible goodput):

| Factor | PAP 7PA1P | PD 6P2D | PD 4P4D | DP8 |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | – | – | – | – |
| 0.75 | 3.12 | 2.66 | 1.91 | – |
| 1.0 | 3.63 | 3.00 | 2.14 | 2.32 |
| 1.25 | 3.70 | 3.04 | 2.21 | 2.36 |
| 1.5 | 3.74 | 3.09 | 2.21 | 2.92 |
| 2.0 | 3.76 | 3.09 | 2.21 | 3.15 |

Mean TTFT by turn (ms): PAP C30 4150 / 3639 / 3754, C40 6022 / 5626 / 5570;
6P2D C30 4498 / 5108 / 5411; 4P4D C20 5958 / 6609 / 6893; DP8 C10
1780 / 1382 / 1580.

## Conclusion

- **C15 holds.** PAP 7PA1P has the highest maximum D2 goodput, 3.63 vs 3.00
  for the best PD split (6P2D, +21%), 2.32 for DP8 (+56%) and 2.14 for 4P4D
  (+70%); CIs do not overlap. The order is the same under D4 and at every SLO
  scale from 0.75× to 2×. Under the tight-ITL D3 SLO PAP leads 4P4D by 42%;
  under D1 only C10 is feasible for either PAP (1.95) or 6P2D (1.91), so
  the gap there is just 2%.
- **What limits each architecture.** PAP is TTFT-limited (ITL P95 ≤ 65 ms at
  all loads; TTFT P99 passes 10 s at C40–C50). 6P2D is limited by both: ITL
  P95 reaches 69–74 ms at C30–C40, near the 75 ms limit, while TTFT grows.
  4P4D has the best ITL (≤ 36 ms P99) but saturates at about 2.2 req/s, so
  TTFT explodes from C30. DP8 has the best TTFT but its ITL P95 passes 75 ms
  already at C20 (100 ms).
- **C20 (native-32K part):** PD 6P2D becomes ITL-bound earlier than PAP
  (ITL P95 69 ms at C30 vs PAP 55.5 ms), and PAP stays below 66 ms P95 through
  C60.
- **Fan-in (R-002 follow-up):** 6P2D's NIXL time per transfer rises from 116
  ms at C10 to 182 ms at C30 and plateaus at 194 ms from C40 (+67%), less than
  the 3× microbenchmark fan-in penalty; 4P4D (one Decode per Prefill) stays at
  125–139 ms. Transfer stays a small part of TTFT (194 ms of 7.4–15 s).
- **Per-turn TTFT:** PAP and DP8 are faster after turn 0, PD splits are slower
  on later turns. PAP is not strictly monotone at C30 (turn 2 is 115 ms above
  turn 1), so "monotonically decreasing" overstates it; "later turns are
  faster than the first" holds.
- Throughput: DP8 reaches the highest raw request rate at high load
  (3.48 req/s at C60) but almost none of it meets D2.
