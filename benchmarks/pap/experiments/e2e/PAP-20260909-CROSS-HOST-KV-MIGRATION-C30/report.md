# Same-host versus cross-host PAP with KV migration

## Question

This experiment tests whether PAP still has roughly one second of TTFT overhead
after enabling PA-to-PA KV migration in the cross-host topology. The prediction
was that cross-host TTFT would remain near five seconds, compared with the
4319.83-ms same-host Snapshot result.

All seven PA groups remain on L20T, so a PA change copies KV through CUDA IPC
between two processes on that host. Only Projection is on the other host. The
Gateway transports migration metadata but never opens the CUDA IPC handles.

## Controlled comparison

Both sides use Qwen3-8B FP16, 7PA1P, the identical 60-session three-turn input,
closed-loop concurrency 30, block size 16, Prefill budget 2048, Projection
budget 256, Dynamo routing, Snapshot export, and no routing trace. The AIPerf
input artifact has SHA256
`ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5`.
Each side contains two independent service starts and 360 requests.

The same-host result reuses `snapshot_r12` and `snapshot_r14` from
`PAP-20260909-KV-SNAPSHOT-EXPORT-C30`. The cross-host runs are `cross_r11` and
`cross_r12`. All 75 cross-host PA changes produced one source export and one
target CUDA IPC Copy, with zero source miss, migration failure, residual export,
client error, output-length mismatch, or GPU process after cleanup.

| Measurement | Same host | Cross host | Cross minus same |
| --- | ---: | ---: | ---: |
| Mean TTFT | 4319.83 ms | 4327.23 ms | **+7.40 ms (+0.17%)** |
| P99 TTFT | 12263.06 ms | **11775.36 ms** | -487.71 ms |
| Mean Prefill residence | **3772.19 ms** | 3836.09 ms | +63.90 ms |
| Cache read rate | 64.88% | **65.02%** | +0.14 pp |
| Computed tokens per run | 1,184,072 | **1,179,424** | -4,649 |
| Mean ITL | 46.96 ms | **43.89 ms** | -3.07 ms |
| Mean E2E | 8131.21 ms | **7882.83 ms** | -248.39 ms |
| Request throughput | 3.214 req/s | **3.250 req/s** | +1.13% |

The prediction is rejected. With the same migration behavior and essentially
identical cache work, the measured mean TTFT difference is 7.4 ms. The 63.9-ms
increase inside the Prefill HTTP boundary is offset elsewhere before first-token
delivery; aggregate subtraction localizes that offset but does not identify a
mechanism. There is no remaining one-second interval to diagnose.

## Per-turn check

| Turn | Same-host TTFT | Cross-host TTFT | Difference |
| --- | ---: | ---: | ---: |
| 0 | 5737.81 ms | **5564.09 ms** | -173.72 ms |
| 1 | **3956.18 ms** | 4022.51 ms | +66.33 ms |
| 2 | **3265.52 ms** | 3395.10 ms | +129.58 ms |

The earlier five-second cross-host runs disabled migration. Their later turns
recomputed KV after a PA change and increased queueing for the C30 workload.
That configuration mismatch, rather than a fixed cross-host PAP penalty,
created the apparent gap.

## Migration path

The cross-host runs copied 590,704 destination tokens across 75 migrations.
Mean source Snapshot export was 19.13 ms and mean target Copy was 76.28 ms. The
aggregate Copy bandwidth was 15.22 GB/s, compared with 16.78 GB/s for the two
same-host runs. Migration shapes and counts differ because Dynamo placements
are not deterministic, but both results use the same CUDA IPC implementation;
the KV bytes never cross hosts.

## Evidence

`results/archives/` retains both cross-host runs. The verifier reuses the two
already retained same-host Snapshot archives, validates every archive member,
then reconstructs `results/summary.json` and `results/requests.csv`.

```bash
.venv/bin/python \
  benchmarks/pap/experiments/e2e/PAP-20260909-CROSS-HOST-KV-MIGRATION-C30/verify.py
```
