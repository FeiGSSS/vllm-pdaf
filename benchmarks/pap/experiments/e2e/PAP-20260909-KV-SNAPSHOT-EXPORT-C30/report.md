# Allocator-safe PAP KV Snapshot export

## Question

The first same-host PA KV migration implementation synchronously entered the
source EngineCore utility queue to find and pin the reusable prefix. That
export RPC averaged about 1.4 seconds and ran before target Prefill, so it was
fully inside TTFT. This experiment tests an allocator-safe Snapshot side
channel that does the same lookup and pin without waiting behind model steps.

## Implementation and safety contract

Each Prefill EngineCore starts a local Unix-socket Snapshot server and collects
its process-lifetime CUDA IPC catalog once during startup. Export requests read
the live prefix index through `BlockPool.acquire_cached_prefix`. An allocator
lock makes prefix lookup and reference acquisition atomic with cache eviction,
free-list removal, and block reuse. The source block cannot be recycled until
the target finishes Prefill and the asynchronous release decrements its lease.

The server socket is owner-only. Invalid requests, duplicate block IDs, copy
errors, release errors, and active leases at drain fail closed. The old
EngineCore path remains available only as
`PAP_KV_MIGRATION_EXPORT_MODE=engine_core` for the controlled comparison.

## Controlled C30 result

Both modes have two independent service starts and 360 requests. Every run
uses Qwen3-8B FP16, 7PA1P on eight NVIDIA L20 GPUs, native 32K context, the
same 60-session three-turn input, closed-loop concurrency 30, block size 16,
Prefill budget 2048, Projection budget 256, and no Hugging Face override. All
four performance runs have the same tracked source patch and identical AIPerf
input SHA256
`ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5`.
Each mode happened to contain 88 PA changes.

| Measurement | EngineCore export | Snapshot export | Change |
| --- | ---: | ---: | ---: |
| Export RPC mean | 1507.96 ms | **11.55 ms** | **-99.23%** |
| Mean TTFT | 4564.67 ms | **4319.83 ms** | **-5.36%** |
| Mean ITL | **45.51 ms** | 46.96 ms | +3.18% |
| Mean E2E | 8248.79 ms | **8131.21 ms** | **-1.43%** |
| Request throughput | 3.024 req/s | **3.214 req/s** | **+6.30%** |

The Snapshot service itself averaged below one millisecond in both performance
runs. The remaining 11.55-ms Gateway-visible RPC includes HTTP, Unix-socket
transport, JSON/base64 serialization, and process scheduling. Export occurs on
only 88/360 requests, so the 1496.41-ms per-migration reduction removes about
365.8 ms when amortized over all requests. Closed-loop routing and queueing
offset part of this direct saving, leaving a measured 244.84-ms mean TTFT gain.

## Migration-conditioned TTFT

Within the new Snapshot runs, migrated later turns averaged 3236.30 ms TTFT;
later turns that stayed on their PA averaged 3827.69 ms. Turn 1 was
3611.32 versus 4278.78 ms, and turn 2 was 2511.28 versus 3516.93 ms.

This is observational. Dynamo chooses migration using cache and load state, so
the groups are not randomized. Across paired runs, only 44 of 88 migrated
request identities were common to both modes. Those 44 requests improved by
1593.65 ms, consistent with removing the measured Export wait. The 228 request
identities that stayed in both modes regressed by 111.19 ms. The present trace
does not instrument the adjacent Projection enqueue/start boundaries, so this
small run-wide queueing shift is not assigned a mechanism.

## Direct data correctness

The separate `snapshot_verify_r3` run uses the same 60-session C30 workload
with `PAP_KV_MIGRATION_VERIFY=1`. It produced 40 PA changes. Every destination
range across all 36 layers was compared byte-for-byte with its source CUDA IPC
mapping: 40/40 migrations passed, with zero mismatch, source miss, release
failure, active export, client error, output-length mismatch, or residual GPU
process.

An earlier ten-session verification attempt produced no PA change and therefore
did not exercise Copy. A subsequent attempt collided with a still-live internal
port during startup and is invalid. The runner now checks every managed child
inside its HTTP readiness loop, so a failed child immediately triggers full
cleanup. Neither invalid run contributes to the result.

## Evidence

`results/archives/` retains the four performance runs and the C30 byte-check
run. `analyze.py` validates workload, topology, configuration, routing,
migration, Snapshot health, output, source-pin drain, and cleanup before
recomputing `results/summary.json`. Run:

```bash
.venv/bin/python \
  benchmarks/pap/experiments/e2e/PAP-20260909-KV-SNAPSHOT-EXPORT-C30/verify.py
```
