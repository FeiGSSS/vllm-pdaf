# PAP post-Prefill Decode late binding

> Controlled 7PA1P development validation. Raw AIPerf and trace artifacts
> remain machine-local under `experiments/_staging/`.

Date: 2026-07-27

## Decision

Accept post-Prefill Decode placement as the experimental `attention_load`
policy's migration boundary.

A later conversation turn now always runs Prefill on its retained-history PA.
Only after Prefill reports the exact full context length does the Gateway
choose the Decode PA. If the selected Decode PA differs, the complete newly
Prefilled KV snapshot is pulled into the target Prefill process through NIXL,
installed into the colocated Attention process, and acknowledged before the
source session is released. A failed transfer leaves the source session live
and atomically falls Decode placement back to the Prefill PA.

First-turn requests remain on their admission PA and do not migrate.

## Why the boundary changed

The previous pilot selected a PA before Prefill and could migrate the retained
historical prefix before computing the new suffix. That made the placement
decision from estimated, not exact, current-turn work and could pay a migration
before the request had become an immediate Decode load.

The new ordering is:

```text
history-owner Prefill
  -> exact full-context observation
  -> Decode placement
  -> optional full-KV NIXL pull
  -> target Attention ready acknowledgement
  -> Projection Decode
```

This preserves local history reuse during Prefill and makes migration directly
change the next Decode barrier.

## Out-of-band migration correction

The original late-binding implementation represented a migration as a
synthetic target Prefill request. That request waited in the target Prefill
queue, executed an unnecessary one-token forward, and tied NIXL completion to
the Prefill engine step. This path is removed.

The corrected path is:

```text
target scheduler allocates final KV blocks and establishes the target lease
  -> NIXL READ starts on its independent stream
  -> worker background progress observes DMA completion
  -> existing CUDA-IPC manifest is published directly to Attention
  -> Gateway observes Attention readiness
  -> source lease is released
  -> scheduler later completes block-ownership bookkeeping
```

There is no target tokenization, fake prompt, Prefill wait-queue entry, or
model forward. Migration uses the existing NIXL connector, external KV-slot
allocator, lease registry, and Attention manifest protocol. At most one
migration remains unresolved. The background progress fast path is enabled
for TP=1; other TP shapes retain the safe post-forward completion fallback.

The Decode placement objective still uses only active Decode KV plus incoming
migration reservations. Prefill work is not added to Attention load. A
separate execution-readiness gate suppresses migration when the selected
target still has unfinished Prefill work, because its EngineCore cannot accept
the migration control message until the current forward boundary.

### Controlled C32 comparison and prefix-identity correction

The comparison uses the same Qwen3-8B FP16 eager 7PA1P deployment, the same
randomized long-context dataset, and 32 sessions with five turns
(160 requests). All three runs completed 160/160 requests with zero AIPerf
errors and drained all seven Attention nodes.

The initial async-late-binding run exposed a correctness/performance bug:
blocks installed by migration were usable by the current Decode session but
were anonymous to the target PA's prefix cache. A following conversation turn
could therefore recompute the complete history. This produced the previously
observed fifth-turn TTFT maximum of 3.477 seconds.

The fix carries the full-block token IDs and block hashes with the migration,
binds them to the imported blocks, and registers the resulting prefix in the
target KV-cache coordinator. It also handles the scheduler-to-lease-publisher
ordering window and uses the stable NIXL request ID instead of the Gateway
request UUID for source export and release.

| Metric | Affinity | Buggy late binding | Fixed late binding |
| --- | ---: | ---: | ---: |
| Completed requests | 160/160 | 160/160 | 160/160 |
| Successful migrations | 0 | 10 | 17 |
| Migration misses | 0 | 0 | 0 |
| Mean TTFT | 2,429.33 ms | 2,349.48 ms | 2,440.66 ms |
| Mean ITL | 43.93 ms | 39.09 ms | 42.78 ms |
| Request throughput | 4.833 req/s | 4.607 req/s | 4.732 req/s |
| Output throughput | 159.93 tok/s | 152.46 tok/s | 156.60 tok/s |
| Benchmark duration | 33.07 s | 34.68 s | 33.75 s |
| Turn-4 mean TTFT | 658.5 ms | 917.8 ms | 739.7 ms |
| Turn-4 maximum TTFT | 1,866.7 ms | 3,477.2 ms | 2,423.4 ms |
| Turn-4 mean ITL | 41.05 ms | 34.03 ms | 36.05 ms |

Relative to the buggy run, the fix reduces turn-4 mean TTFT by 19.4% and its
maximum by 30.3%. The eleven migrations followed by another conversation turn
had next-turn Prefill times of 199-1,253 ms (604 ms mean), rather than the
roughly 3.3-second full-history recomputations that exposed the bug. Repeated
migrations of the same conversation also retained reusable prefix identity.

Relative to affinity, overall mean TTFT is effectively equal (+0.5%), request
throughput is 2.1% lower, and mean ITL is 2.6% lower. Turn-4 maximum TTFT
remains higher than affinity, so migration/scheduling tail latency is not
declared solved.

The final run performed 17 migration attempts and installed all 17 in
71-204 ms, with zero migration misses or fallbacks. Its lifecycle audit passed
with 322 releases:

```text
160 current-turn releases
  + 128 retained-history releases
  + 17 migration-attempt session releases
  + 17 successful source-lease releases
  = 322
```

### Rejected intermediate executions

The staging directory retains diagnostic runs that must not be used as
baselines:

- `oob_priority_s32_c32` forced no-forward migration steps. It removed NIXL
  completion tails but paused target Prefill scheduling.
- `oob_overlap_notify_s32_c32` overlapped NIXL with Prefill but published only
  from `post_forward`, producing transfers reported as long as 2.56 seconds.
- `oob_notify_s32_c32` validated deferred completion notification but allowed
  18 migrations and retained Prefill interference.
- `oob_worker_async_s32_c32` validated worker-side progress and exposed one
  2.32-second control-admission tail on a busy target Prefill engine.

The accepted controlled run is:

```text
benchmarks/pap/experiments/_staging/scheduling/
  20260727_out_of_band_migration/runs/
    affinity_baseline_s32_c32/
    prefix_identity_fix_s32_c32_idfix/
```

`oob_worker_async_idle_gate_s32_c32` is retained only as the diagnostic
execution that exposed missing prefix identity.

Focused validation for the corrected path:

```text
.venv/bin/python -m pytest \
  tests/pap/test_pap_migration.py \
  tests/pap/test_decode_commit.py \
  tests/pap/test_pap_gateway_app.py -q

93 passed
```

The focused Ruff check, `git diff --check`, and workload-script syntax check
also passed.

## Workload

- Qwen3-8B FP16, eager execution, 7PA1P on eight NVIDIA L20 GPUs.
- Formal run: 128 conversations, five turns, 640 requests, concurrency 32.
- Long-tail randomized multi-turn input and randomized 16-64-token output.
- `max_model_len=32768`, PA memory utilization 0.90.
- Prefill and Projection `max_num_seqs=256`.
- Sparse migration defaults: balance gain 0.30, interval 64, max in-flight 1.

## End-to-end result

The comparison uses the same AIPerf dataset and topology. The prior row is the
rolled-back deferred-submission result from
`PAP-20260727-COLLECTIVE-QKV-FANOUT`; it remains useful only as a nearby
measurement, not as the current code baseline.

| Metric | Prior placement boundary | Post-Prefill late binding | Change |
| --- | ---: | ---: | ---: |
| Completed requests | 640/640 | 640/640 | equal |
| AIPerf errors | 0 | 0 | equal |
| Request throughput | 5.220 req/s | 5.312 req/s | +1.8% |
| Output throughput | not reported | 172.12 tok/s | — |
| Mean TTFT | 1,651.09 ms | 1,566.33 ms | -5.1% |
| P90 TTFT | not reported | 2,976.26 ms | — |
| Mean request latency | not reported | 3,113.13 ms | — |
| Mean ITL | 48.47 ms | 49.03 ms | +1.2% |
| P90 ITL | not reported | 60.17 ms | — |
| Successful migrations | not reported | 5 | — |
| Migration misses/fallbacks | 0 | 0 | equal |

All seven Attention nodes drained to zero sessions. The structured routing,
correctness, static-MPS, and lifecycle audits passed.

This is no-regression evidence for the mechanism and a positive TTFT/throughput
result. It is not evidence that Decode late binding has solved 7PA1P ITL:
mean ITL is effectively unchanged in this single formal repetition.

## Migration behavior

An aggressive eight-conversation canary used zero balance threshold and
one-turn migration spacing. Its valid repetition completed 40/40 requests,
performed five migrations with zero misses, and drained every Attention
session. Large transfers were generally 1.2-1.3 GiB and reached roughly
10-22 GB/s.

The formal sparse run also selected five migrations. End-to-end migration
latencies were 155, 170, 170, 1,008, and 1,207 ms. The median was 170 ms.
The two slow outliers show that long-running KV-page fragmentation remains a
real migration-tail risk. Observed NIXL transfer samples ranged from about
1.0 GB/s to 23.3 GB/s and from 4 to 1,000 descriptors. Sparse migration keeps
this cost rare; descriptor compaction remains future work.

## Fan-in trace

The trace-on diagnostic used the first 32 conversations, five turns, and C32.
It completed 160/160 requests with zero errors, two successful migrations, and
zero misses. Trace mode perturbs runtime, so only fan-in measurements are
compared with the earlier sparse-policy trace.

| Per-layer metric | Earlier pre-Prefill placement | Post-Prefill placement | Change |
| --- | ---: | ---: | ---: |
| Participating PAs, median | 6 | 6 | equal |
| First-to-last spread, median | 0.312 ms | 0.286 ms | -8.5% |
| Spread, P90 | 0.816 ms | 0.868 ms | +6.4% |
| Spread, P99 | 8.893 ms | 12.380 ms | +39.2% |
| PA compute-completion skew, median | 0.311 ms | 0.282 ms | -9.5% |
| PA compute-completion skew, P90 | 0.789 ms | 0.790 ms | equal |
| Mean PA idle-until-slowest, median | 0.164 ms | 0.154 ms | -5.9% |

Late binding improves the typical barrier modestly but does not improve its
tail in this sample. With only two migrations, the trace is diagnostic rather
than a statistically stable scheduler verdict.

## Invalid result retained

The first aggressive canary completed 40/40 model requests with zero errors
and five successful migrations, but the runner exited nonzero. Its legacy
lifecycle audit allowed at most 72 lease releases and observed 77. Each
post-Prefill migration creates exactly one additional source-or-target session
release, so `72 + 5 = 77`; the audit, not the data path, was wrong.

The audit now includes the migration-attempt count in the upper bound. The
identical second canary passed with 77 releases, five attempts, zero historical
release misses, and zero migration misses. The first run is retained as
negative test-infrastructure evidence and is not used as a valid performance
result.

## Validation

```text
.venv/bin/python -m pytest \
  tests/pap/test_pap_gateway_app.py \
  tests/pap/test_decode_commit.py \
  tests/pap/test_pap_attention_runtime.py -q

147 passed
```

The focused gateway and runtime-config suite passed 66 tests. `bash -n`,
`py_compile`, and `git diff --check` passed. The repository virtual
environment does not contain `ruff`, so no ruff result is claimed.

Raw artifacts:

```text
benchmarks/pap/experiments/_staging/scheduling/
  20260727_post_prefill_late_binding/runs/
    canary_migrate_all_s8_c8/       # invalid legacy audit
    canary_migrate_all_s8_c8_rep2/  # valid
    formal_sparse_s128_c32/         # valid
    trace_sparse_s32_c32/           # valid trace-on diagnostic
```
