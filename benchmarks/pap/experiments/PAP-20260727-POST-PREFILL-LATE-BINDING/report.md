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

## Workload

- Qwen3-8B FP16, eager execution, 7PA1P on eight NVIDIA L20 GPUs.
- Formal run: 128 conversations, five turns, 640 requests, concurrency 32.
- Long-tail randomized multi-turn input and randomized 16-64-token output.
- `max_model_len=32768`, PA memory utilization 0.90.
- Prefill and Projection `max_num_seqs=256`.
- Sparse migration defaults: balance gain 0.30, interval 64, max in-flight 1.

## End-to-end result

The comparison uses the same AIPerf dataset and topology. The prior row is the
accepted independent-stream result from
`PAP-20260727-COLLECTIVE-QKV-FANOUT`.

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
