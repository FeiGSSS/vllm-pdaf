# E2e: PD P→D KV handoff cost on PCIe L20

Request: `paper/handoff/requests/R-002-phase1-e2e-e2a-e2b.md` (E2e).

## Question

What share of TTFT does Dynamo PD spend moving KV from Prefill to Decode for
16–32K contexts, and does it grow when several Prefill workers feed one Decode
GPU?

## Protocol

**(a) Offline, archived runs.** `analyze_offline.py` reads
`analyze_dynamo_ttft.py` output for `PAP-20260907-ALIGNED-C30` 6P2D/4P4D/2P6D
(r1, r2) and `PAP-20260907-ALIGNED-C30-7P1D` (run_1, run_2): Qwen3-8B,
`s60-t3-native32k-stratified-seed42`, C30, September code. Dynamo logs NIXL
transfer statistics only per time window, not per request, so two per-request
bounds are reported by input-length bucket:

- `d2ft`: Decode receipt to first token (KV read + Decode queueing + first
  step), an upper bound on the handoff;
- `xfer_est`: full-prompt KV bytes (147,456 B/token) divided by the run's
  measured aggregate NIXL rate. Decode reads only the blocks it lacks
  (measured MB per transfer is 0.6–0.7 of full-prompt size), so this also
  overestimates.

**(b) Point-to-point microbenchmark.** `run.sh` runs
`tooling/nixl_read_write_probe.py` with the Dynamo baseline UCX settings
(`configure_same_node_nixl.sh`): READ initiated by Decode GPU 7, 72 regions,
32 KiB segments (one 16-token block per layer and K/V), full-prompt sizes for
4K–32K contexts, 5 timed repetitions after one warm-up. Cases: same NUMA
(6→7), cross NUMA (0→7), and 3→1 fan-in (0, 1, 2 → 7 concurrently). The
topology has no GPU pair under a common PCIe switch (`results/topo.txt`: GPUs
0–3 and 4–7 are NODE within a socket, SYS across), so "same switch" in the
request is the same-NUMA pair. The kernel log was checked for PCIe/AER/GHES
errors after every step: none.

## Results (2026-10-15)

### (a) Archived runs, share of TTFT by input length (mean / P99 over requests)

| Arch | Context | n | TTFT mean (ms) | `d2ft` share | `xfer_est` share |
| --- | --- | ---: | ---: | --- | --- |
| 6P2D | <16K | 122 | 4258 | 8.8% / 20.2% | 4.4% / 8.7% |
| 6P2D | 16–20K | 102 | 6439 | 7.6% / 30.7% | 4.6% / 18.2% |
| 6P2D | 20–24K | 70 | 6555 | 7.6% / 25.8% | 5.5% / 16.6% |
| 6P2D | 24–28K | 36 | 6020 | 10.1% / 21.5% | 6.6% / 15.1% |
| 6P2D | 28–32K | 30 | 6640 | 9.1% / 26.0% | 6.9% / 20.3% |
| 4P4D | 16–32K | 238 | 11.7–13.1 s | 1.8–2.8% | 1.8–2.6% |
| 2P6D | 16–32K | 238 | 33.6–54.9 s | 0.4–0.6% | 0.6–0.7% |
| 7P1D | 16–32K | 238 | 9.1–11.6 s | 59–79% (Decode queueing) | 3.0–3.8% |

Per run, the measured NIXL average is 113–188 ms per transfer, 1.6–1.9 GB per
transfer, 10.2–14.7 GB/s aggregate (`results/offline_runs.json`). With the
measured bytes (0.6–0.7× full prompt) instead of full-prompt bytes, the 6P2D
16–32K transfer share is about 3–4.5%.

### (b) Microbenchmark, one full-prompt KV READ into GPU 7

| Context | Same NUMA 1→1 | Cross NUMA 1→1 | 3→1 fan-in, per stream |
| ---: | ---: | ---: | ---: |
| 4K (576 MiB) | 63 ms, 9.5 GB/s | 66 ms, 9.2 GB/s | 204–207 ms, 2.9–3.0 GB/s |
| 8K (1152 MiB) | 131 ms, 9.3 GB/s | 144 ms, 8.4 GB/s | 394–423 ms |
| 16K (2304 MiB) | 262 ms, 9.2 GB/s | 300 ms, 8.1 GB/s | 827–863 ms |
| 24K (3456 MiB) | 424 ms, 8.6 GB/s | 418 ms, 8.7 GB/s | 1205–1261 ms |
| 32K (4608 MiB) | 576 ms, 8.4 GB/s | 608 ms, 7.9 GB/s | 1684–1714 ms |

Raw lines: `results/microbench_raw.txt`; table: `results/microbench.csv`.

## Conclusion

- **Mean handoff share at C30 is below 5%.** In the best PD split (6P2D),
  the transfer is about 3–4.5% of TTFT for 16–32K contexts with measured
  bytes, and at most 5–7% with full-prompt bytes. The `d2ft` upper bound
  (7.6–10.1%) includes Decode queueing. Larger-Decode splits (4P4D, 2P6D) are
  below 3%; in 7P1D, TTFT is dominated by queueing on the single Decode GPU,
  not by the transfer.
- **Fan-in shares the Decode GPU's ingress.** Three concurrent READs into one
  GPU each take 3.0–3.3× as long as a single READ; the sum stays at about
  8.6 GB/s, so the receiver's ingress, not the source or NUMA distance, is the
  limit. At C30 (about 2.7 req/s, two Decode GPUs) September's 6P2D runs show
  no such slowdown (151–163 ms per transfer), consistent with roughly 20%
  ingress occupancy per Decode GPU. Whether it appears at higher load can be
  read from the `ttft_breakdown.json` of R-003's PD runs.
- Tail effects are larger than means: per-request `xfer_est` share reaches
  15–20% at P99 in 6P2D.
