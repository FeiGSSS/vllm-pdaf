# R-003 figure data

Exported by the exp agent; see `paper/handoff/reports/R-003.md`.

| File | Source |
| --- | --- |
| `e1_load_sweep_points.csv` | E1: every architecture × concurrency point (throughput, TTFT/ITL percentiles, D1–D4 attainment and goodput) |
| `e1_max_goodput_d1_d4.csv` | E1: maximum feasible goodput per SLO with run-level 95% CI |
| `e1_slo_scaling.csv` | E1: maximum feasible goodput when all D2 limits are scaled by 0.5–2.0 |
| `e1_ttft_by_turn.csv` | E1: mean TTFT per conversation turn at every point |
| `e1_pd_transfer_vs_load.csv` | E1: PD NIXL time and size per transfer versus concurrency |
| `e2d_regime_points.csv` | E2d-lite: every cell × architecture × concurrency point in the two long-output cells |
| `e2d_regime_2x2.csv` | E2d-lite: maximum goodput and PAP ratios in all four input × output cells (rules per row) |
| `e2d_bytes_per_request.csv` | E2d-lite: PD KV bytes per transfer vs PAP activation bytes per request in each cell |
