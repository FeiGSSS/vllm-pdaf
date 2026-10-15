# R-002 figure data

Exported by the exp agent from the R-002 experiments; see
`paper/handoff/reports/R-002.md`.

| File | Source |
| --- | --- |
| `handoff_share_by_context.csv` | E2e(a): archived Sept PD runs, handoff bounds as a share of TTFT by input-length bucket (`microbench/PAP-20261015-PD-KV-TRANSFER/analyze_offline.py`) |
| `kv_transfer_microbench.csv` | E2e(b): NIXL READ into GPU 7, 1→1 same/cross NUMA and 3→1 fan-in, 4K–32K full-prompt KV |
| `l07_replay_points.csv` | E2b: every architecture × concurrency point (min throughput, worst tails, tier attainment and goodput) |
| `l07_replay_capacity.csv` | E2b: capacity per SLO tier under July's rule, with July's L07 values |
