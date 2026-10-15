# PAP code-cleanup performance regression

## Question

Did the 2026-10-15 cleanup change single-host PAP performance? The cleanup
removed the session router, Gateway load tracker, multi-Attention-rank support,
GPU-resident dispatch, retired flags and dead fields. It was intended to leave
the default execution path unchanged.

## Protocol

- Workload and service settings: `requested.env`, byte-identical to the PAP
  settings of `PAP-20260907-ALIGNED-C30` (Qwen3-8B FP16, 7PA1P on eight L20,
  native 32K, 60 sessions x 3 turns, concurrency 30).
- `current`: this checkout, with the uncommitted cleanup diff recorded per run
  in `source.diff`.
- `base`: commit `c565ecb904` (before the cleanup) in `.worktrees/cleanup-base`,
  imported through `PYTHONPATH`, with its own version-10 NVSHMEM bridge built on
  the same day and compiler under `.local/nvshmem-3.3.24-cuda13-bridge-v10`.
- Order: `current base current base`, one fresh service start per run.

The September reference ran on an older driver (595.84, now 595.99) and an
older commit, so it is a secondary comparison only.

```bash
bash benchmarks/pap/experiments/e2e/PAP-20261015-CLEANUP-REGRESSION/run.sh current 1
bash benchmarks/pap/experiments/e2e/PAP-20261015-CLEANUP-REGRESSION/run.sh base 1
```

## Results

All four valid runs completed 180/180 requests and passed all 14 startup,
correctness and drain audits. Two earlier attempts failed in harness setup
before any request was measured (`rg` missing from `PATH`; `AIPERF_PYTHON` not
overridden for the worktree); their directories are retained.

| Run | TTFT mean / P95 / P99 (ms) | ITL mean / P95 / P99 (ms) | Requests/s |
| --- | --- | --- | ---: |
| current r1 | 5921 / 10322 / 11646 | 43.40 / 53.47 / 55.89 | 2.743 |
| base r1 | 5767 / 9564 / 12047 | 44.19 / 52.38 / 52.86 | 2.836 |
| current r2 | 5451 / 9254 / 11243 | 45.03 / 57.07 / 58.62 | 2.872 |
| base r2 | 5829 / 9654 / 12073 | 43.48 / 54.75 / 56.54 | 2.752 |
| **current mean** | **5686 / 9788 / 11444** | **44.22 / 55.27 / 57.26** | **2.807** |
| **base mean** | **5798 / 9609 / 12060** | **43.83 / 53.56 / 54.70** | **2.794** |

Pooling 360 request records per variant, the current/base ratio of mean TTFT is
-1.94% (bootstrap 95% CI -6.99% to +3.76%) and of mean request-level ITL is
+0.88% (95% CI -1.03% to +2.86%). Both intervals contain zero and every
variant-mean difference is smaller than the spread between that variant's own
repetitions.

## Records

`results/summary.json` holds per-run metrics and the pooled bootstrap
comparison; `results/requests.csv` holds all 720 request records.
`results/archives/` keeps each attempt's configuration, launch environment,
source diff, audits, runner log and AIPerf exports; checksums are in
`results/SHA256SUMS`. Full service logs stay in the ignored local `runs/`.

## Conclusion

The cleanup has no measurable single-host performance effect at this load.
Both variants are about 6-8% slower in mean TTFT than the September reference
(5.364 s); that shift is common to the pre- and post-cleanup code and is
therefore attributable to changes since September (later commits, driver
595.84 to 595.99), not to the cleanup. Only C30 on 7PA1P was measured.
