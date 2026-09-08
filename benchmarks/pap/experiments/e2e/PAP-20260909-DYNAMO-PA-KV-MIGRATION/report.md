# Same-host Dynamo PA KV migration

## Question

This experiment keeps Dynamo's per-turn PA selection and tests whether copying
the previous PA's reusable prefix KV to a newly selected PA is better than
recomputing that prefix.

## Unit of work and timeline

A request is one conversation turn. A transition is turn 0 to 1 or turn 1 to 2
inside one conversation. Each run contains 180 requests and 120 transitions.
A PA migration means Dynamo selected a different Prefill/Attention pair from
the previous turn.

For an enabled migration, the measured state transition is:

1. Dynamo selects and reserves the target PA.
2. The Gateway sends the new prompt's full block hashes to the previous PA.
3. Source EngineCore finds the longest contiguous cached prefix, increments the
   source blocks' allocator references, and returns its cached CUDA IPC catalog.
4. Target EngineCore accounts those tokens as external KV and allocates its own
   destination blocks.
5. Before forward, the target worker opens the source mappings and copies only
   the blocks beyond its existing local cache. Source and destination block-ID
   runs are coalesced into CUDA peer copies.
6. Target Prefill computes the remaining suffix and returns. The Gateway then
   schedules source unpin outside the TTFT path. Shutdown and benchmark drain
   wait for every release task.

Every boundary uses the same Gateway request ID. Source export and release are
end-to-end control RPC times. Copy time is measured with CUDA events around the
physical peer copies and read after the following forward completes.

## Failure contract

Missing conversation identity disables migration for that request because
there is no previous-PA relation. Concurrent turns of the same identified
conversation, invalid descriptors, control failures, copy failures, or source
release failures fail closed. A source cache miss is recorded and normal
Prefill continues because no source KV remains to copy. Conversation and source
export tables are bounded; bindings are never silently evicted.

This implementation is same-host TP1. CUDA IPC is not a cross-host transport.

## Controlled runs

The accepted order alternates the only experimental variable,
`PAP_ENABLE_PA_KV_MIGRATION`:

1. `control_r5`: disabled
2. `migration_r5`: enabled
3. `control_r6`: disabled
4. `migration_r6`: enabled

All runs use Qwen3-8B FP16, one host with eight L20 GPUs, 7PA1P, static 80/12-SM
Prefill/Attention MPS partitions, native 32K context, Prefill budget 2048,
Projection budget 256, 60 three-turn sessions, and AIPerf concurrency 30. Each
run served 180/180 requests with no client error, cancellation, or output-length
mismatch.

The analyzer verifies identical source fingerprints, dataset selection,
effective configuration after removing output paths and the migration flag,
run metadata, and process/device topology. Service, CUDA-context, MPS, Graph,
routing, lifecycle-drain, source-pin-drain, and GPU cleanup audits passed.

## Aggregate result

The table pools two runs per mode, for 360 requests and 240 transitions each.

| Measurement | Recompute control | KV migration | Change |
| --- | ---: | ---: | ---: |
| Dynamo PA changes | 68 | 84 | +6.67 pp |
| Migration exports / misses | 0 / 0 | 84 / 0 | 100% available |
| Cache-read rate | 56.65% | 64.95% | +8.29 pp |
| Computed Prefill tokens | 2,923,502 | 2,363,952 | -19.14% |
| Mean Prefill residence | 5.039 s | 3.763 s | -25.32% |
| Mean TTFT | 5.562 s | 4.592 s | -17.44% |
| P95 TTFT | 9.814 s | 9.433 s | -3.88% |
| P99 TTFT | 11.896 s | 11.614 s | -2.37% |
| Mean ITL | 43.91 ms | 45.72 ms | +4.11% |
| Mean E2E latency | 9.113 s | 8.292 s | -9.01% |
| P99 E2E latency | 16.450 s | 16.646 s | +1.20% |
| Request throughput | 2.803 req/s | 3.071 req/s | +9.57% |
| Output throughput | 228.83 tok/s | 250.72 tok/s | +9.57% |

The enabled runs happened to contain more Dynamo PA changes, so the benefit is
not caused by an easier routing sequence. Reduced Prefill work also changes
subsequent queue state and therefore Dynamo's later decisions; migration count
is an observed system outcome rather than a fixed input.

For migrated transitions specifically, the recompute controls read 28.58% of
prompt tokens from cache and had 5.700-s mean TTFT. Enabled runs read 71.40% and
had 4.537-s mean TTFT. These subsets contain different requests and are
observational, while the full alternating A/B above is the causal comparison.

Turn 0 has no conversation source and performs no migration. Mean TTFT changed
from 6.532 to 5.907 s. Turns 1 and 2 changed from 4.991 to 4.383 s and from
5.164 to 3.488 s. Some turn-0 requests overlap later turns at concurrency 30,
so lower migration work can also reduce their queueing; turn number is not an
isolated control interval.

## Transfer work and remaining bottleneck

The 84 moves exported 1,176,736 cache tokens. Targets already held 526,912 of
those tokens locally, so the connector copied only 649,824 tokens. The copied
volume is 95.82 GB using the measured Qwen3-8B layout: 36 layers, 8 KV heads,
256 combined K/V values per token and FP16. Mean CUDA copy time was 128.1 ms;
mean per-request effective bandwidth was 19.77 GB/s.

Mean source-export RPC time was 1.417 s. The source catalog is cached after its
first export, so this number mainly contains waiting to enter the busy source
EngineCore control boundary plus hash lookup and response delivery. Mean source
release RPC time was 835.8 ms, but release runs asynchronously after target
Prefill and is excluded from its request's TTFT critical path. All final source
export counts were zero.

The data copy is therefore not the current dominant migration cost. A follow-up
should reduce source-control admission delay, for example by maintaining a
Gateway-visible, ownership-safe export snapshot or adding a prioritized
scheduler control lane. That optimization must preserve block pinning and the
exact hash-to-block validation used here.

## Direct data correctness

`migration_verify_smoke_r1` enabled `PAP_KV_MIGRATION_VERIFY=1` for ten sessions.
It triggered seven PA moves. After every copy, all destination ranges across all
36 layers were compared byte-for-byte with their source IPC mappings. All seven
passed, AIPerf completed 30/30 requests without errors, and source-pin drain
passed. This diagnostic synchronization is not enabled in performance runs.

## Invalid and superseded runs

- `migration_smoke_r1` contained no PA changes and therefore did not exercise
  the copy path.
- `migration_smoke_r2` exposed that `cuMemcpyBatchAsync` rejects the mapped
  cross-GPU IPC source address with CUDA `INVALID_VALUE`. The data plane now
  coalesces matching block-ID runs and uses standard CUDA peer copies.
- `migration_smoke_r3` validated the corrected copy path; the byte-verification
  smoke then established physical data equality.
- Formal r1/r2 completed before adjacent copy timing was added. Formal r3/r4
  showed that synchronous source release unnecessarily extended TTFT. The
  final implementation caches source catalogs and tracks release asynchronously.
  Only r5/r6 contribute to the result tables.

All failed and superseded raw runs remain in `experiments/_staging/` and are not
used by the analyzer.

## Evidence and replay

`results/archives/` contains the four accepted performance runs and the direct
byte-verification smoke. Copied Dynamo binaries and unrelated dirty-worktree
patches are omitted. Effective configuration, source fingerprints, AIPerf
records, service logs, topology and routing audits, and cleanup evidence remain.

Reproduce the saved analysis with:

```bash
.venv/bin/python \
  benchmarks/pap/experiments/e2e/PAP-20260909-DYNAMO-PA-KV-MIGRATION/verify.py
```
