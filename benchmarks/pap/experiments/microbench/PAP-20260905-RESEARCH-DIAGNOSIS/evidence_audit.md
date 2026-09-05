# Retained PAP evidence audit — 2026-09-05

Scope: CPU-only forensic analysis of existing artifacts. No server, GPU job,
full-suite benchmark, or production-code change was performed. This is an
analysis record, not a newly measured microbenchmark. Numerical results below
are recomputed by `audit_saved_evidence.py` unless explicitly marked historical.

## Bottom line

**Stop performance interpretation: the retained trace proves active KV block
aliasing within individual requests.** Completion/output-length audits did not
establish numerical correctness. All timing/load numbers below remain raw
execution observations of a potentially corrupted decode path, not admissible
correct-inference performance evidence. Physical/unique/shared token statistics
can mistake aliased decode blocks for legitimate reuse.

The current-source cleanup run establishes completion of a lightly loaded
half-length coding point, not reference-equivalent outputs. A newer retained
post-router-fix trace exists than the initially
suggested 11:45 trace. Its raw markers support a 48.997 ms mean sampled decode
cycle, not the 50.591 ms historical additive proxy. Larger, execution-audited pre-cleanup
workloads already show substantial Prefill queueing and high KV occupancy; the
half-length trace cannot establish that capacity and queueing are unimportant.
No available current-source trace establishes a GPU-kernel critical-path split
or causality for the high-pressure regime.

## Blocking correctness finding: one request, duplicate active block IDs

The newest retained merged window contains 466 PA-step cells with exactly one
request. **373 of those cells reference more logical block positions than unique
physical block IDs.** With one request, cross-request prefix sharing cannot
explain the difference. A concrete fully aligned observation is:

| Field | Raw observation |
| --- | --- |
| Global Projection step / PA / local epoch | 1983 / PA6 / 1840 |
| Request ID | `07e4ff81-d46b-42b2-a0c8-ec7334bc515e` |
| Request count | 1 |
| Prefix length / active sequence length | 29,781 / 30,003 tokens |
| Active logical block positions | 1,876 |
| Unique active physical block IDs | 1,863 |
| Exported lease-vector length / unique leased IDs | 1,908 / 1,863 |

The seven retained full PA rings independently contain 1,921 aliased cells among
2,922 single-request cells. All seven PAs have examples. PA0 local epoch 713,
request `dadd542b-ad32-47e0-837f-62d0c92a1950`, has sequence/prefix lengths
26,411/26,043, 1,651 active positions but only 1,629 unique physical IDs.

This conclusion follows directly from the recorder's counting implementation:
`request_block_counts` counts traversed positions in `state.block_ids`, while
`unique_block_count` is the number of distinct keys in `unique_tokens_by_block`.
`request_leased_block_counts` is the vector length; `unique_leased_block_count`
is the set cardinality. Raw actual block-ID vectors are **not** exported, so
the trace alone does not identify ID zero or independently prove why allocation
failed. Source inspection by the parallel source audit identifies the suspected
decode-headroom allocation path; that mechanism must be verified separately.

These are hashed, reproducibly joined observations from the pre-cleanup trace.
They cannot certify the current untraced run's numerical behavior either way.
The active-block aliasing invalidates treating these tensors as normal physical
KV reuse/performance evidence. The analysis script prints the witness and exits
2 intentionally; it does not quietly emit a passing performance report.

## Units and expected timeline

- A request is one user turn; the coding fixture has 60 conversations and
  180 requests. Configured client concurrency 60 is not 60 active decode rows.
- A global Projection step is one decode forward pass through 36 layers for
  its currently scheduled request batch. A PA-step is that step's subset of
  request rows assigned to one Attention worker, identified by its local epoch.
- A layer exchange dispatches QKV to active PAs, waits for Attention outputs,
  gathers/scatters them, and continues Projection work. PA peers execute in
  parallel; summing all peers' durations is not a critical path.
- A queued batch has been submitted but need not be executing. An in-flight
  batch's completion future need not be done. Neither is a new user request.
  Current records do not expose those transitions with shared batch IDs.
- Request ITL is AIPerf's per-request mean token interval, averaged over
  requests. A same-GPU step cadence is a different quantity. Replay throughput
  includes drain and is not peak service capacity.

Expected marker order for one layer on the Projection GPU is:

```text
S(l,p): per-peer dispatch block starts, before pack/put
  -> D(l): whole dispatch-done marker
  -> E(l,p): output-ready signal observed, before scatter copy
  -> G(l): whole gather-done marker
  -> Projection path and the next QKV dispatch
  -> D(l+1), or D(next step,0) at the last layer
```

PA-local Attention start/end markers use that PA's GPU clock. Only local
durations, not cross-device absolute timestamps, are compared here.

## Provenance and reusable artifacts

All run paths below are relative to
`benchmarks/pap/experiments/e2e/PAP-20260905-REFACTOR-VALIDATION/runs/`.

| Evidence | Source identity and execution-audit status | Reuse limit |
| --- | --- | --- |
| `20260905_190715_3493447/coding-half` | Current cleanup implementation: all 83 archived `vllm/pap/` files match working-tree commit `306b75a894fe6db2ae8df355b531e0c08433b2e5` byte-for-byte. Raw request records, audits, and source/dependency snapshots retained. | No trace; selected one-case success, not a fresh full-suite result. |
| `20260905_135804_3149938/coding-half-trace/trace_capture` | Newest available post-explicit-owner-Dynamo trace with passed execution audits. All 8 raw file hashes verified; raw rejoin reproduces all 35 tensor fields and request IDs. Steps 1980–2491, 512 samples. | Aliasing witness above; pre-cleanup source. Only 36 of 99 archived PAP files still match current files. |
| `20260905_135804_3149938/coding`, `coding-full` | Corrected-router seven-case queue passed execution audits; raw success records and Prefill logs retained. | Numerical equivalence unproved; larger/high-pressure workloads, but no detailed GPU or per-request Prefill phase traces. |
| `20260905_114511_3015679/coding-half-trace` | Older retained 512-sample raw window, steps 1861–2372, before explicit-owner router fix. | Superseded for current-policy diagnosis; sibling `coding` invalidated by premature reservation expiry. |
| `PAP-20260904-GPU-RESIDENT-TRACE` | Historical report and first-token report retained. | Referenced `attempt_002` tensors and first-token A/B raw directories are absent locally. Cannot recompute those claims from this artifact bundle. |

Both 19:07 and 13:58 suites record Git base
`9aca8d96ef8242f8bcd0110422b5e20d76da6f89` plus different dirty-source snapshots.
The base commit alone therefore does not identify either tested implementation.
Source archive SHA-256:

- 19:07: `911a1020340b09e05a86d8b48b437ab17e79dc5086d9408e3b65279364ae28e6`
- 13:58: `ad04cad064e00102482b967a047cbb05f488c0b2e2757271c3349ac0b4c1445c`

The two suites' three Python package inventories, model-file manifest, GPU
topology record, and native router binary match byte-for-byte. Raw configuration
and audits establish 7 PA groups on GPUs 0–6 and Projection on GPU 7, with 80
Prefill/12 Attention SMs per PA GPU. Eight captured whole-step Graph processes
means seven Attention plus one Projection, not eight total OS processes; the
Prefill engines/API processes are additional. Current drain stats show 180
completed lifecycle scopes and zero failed/active scopes or native reservations.

Both the current half run and the newest retained trace set
`PAP_ATTENTION_GPU_RESIDENT_DISPATCH=0`. The older September 4 resident-dispatch
report is a different operating mode. Native CUDA marker definitions and the
merge tool in the 13:58 archive match the currently inspected files byte-for-byte.

The llm-pipeline-analysis skill was consulted. Its scripts require Chrome-trace
kernel events, which these native NVSHMEM `.pt` marker arrays do not contain.
Consequently no GEMM/MLP/kernel composition or layer-anchor analysis was invented;
the native epoch-join and same-device boundaries were used instead.

## Corrected sampled cycle accounting

For 511 consecutive cycles we reconstruct D(step,0) from the preceding row's
saved next-step marker. Every adjacent interval is positive and the following
identity is checked exactly in integer nanoseconds:

`sum_l[(G(l)-D(l)) + (D(l+1)-G(l))] = D(next step,0)-D(step,0)`.

| Measured quantity, newest retained trace | Mean (ms) | Interpretation |
| --- | ---: | --- |
| Exact step cadence | 48.997 | Projection GPU dispatch-0 done to next dispatch-0 done; P50 47.198, P99 91.223, max 102.230. |
| Dispatch-done to gather-done sum | 21.869 | PA return wait plus gather/scatter and marker overhead. |
| Gather-done to next dispatch-done sum | 27.128 | Projection-path interval, including next dispatch and last-layer host/scheduling boundary; **not measured Projection compute alone**. |
| Historical `sum(max_PA(E-S)) + sum(Projection)` proxy, same 511 cycles | 50.591 | Nonadjacent boundaries; exceeds exact cadence by 1.594 ms on average (3.25%). |

The historical proxy includes pack/put in `E-S` while the previous Projection
interval also extends through dispatch-done, and it excludes some post-signal
scatter. Its net mismatch is directly measurable; it is not an exact additive
critical-path decomposition. This analysis stops using that proxy as a cycle
measurement and provides an integer telescoping guard against recurrence.
Historical records were not edited or relabelled.

Additional measured statistics over all 512 saved steps:

- Sum of per-layer mean PA Attention duration: 13.841 ms; sum of per-layer
  maximum Attention duration: 17.876 ms. Their difference is 4.035 ms.
- Fan-out-duration max-minus-mean spread sums to 6.801 ms per step. These
  spreads are **descriptive imbalance metrics, not proven achievable gains or
  causal speedup bounds**; reassigning requests changes reuse and kernel cost.
- Layer-0 slowest-peer fan-out latency averages 3.316 ms versus 0.575 ms for
  layers 1–35. Layer-0 P99 is 42.957 ms. This localizes a boundary anomaly but
  does not identify its cause.
- Last Projection transition averages 2.911 ms versus 0.692 ms for ordinary
  transitions. This overlaps other boundary views and must not be added again.

No raw marker here separates current enqueue, submit, GPU execution-start,
future-done, queue pop, output processing and client delivery for the same batch.
The older first-token report states such instrumentation existed, but its raw
evidence is missing; its `3*TBT` mechanism cannot be newly verified from current
records or inferred from the ratio.

## PAT and load observations — suspended as performance evidence

The following recomputed statistics describe the defective trace and are
retained for forensics only. Within-request aliasing contaminates apparent
unique/shared-token counts; do not fit a legitimate-reuse cost model from them.

The retained window contains 48 distinct request IDs, 15–26 simultaneous decode
rows (mean 21.201), and 86.998% PAT-selected PA-step cells. Consecutive request
membership is unchanged in 459 of 511 transitions. The collector intentionally
selects an all-seven-PAs-active window; this is not an unbiased sample of the
entire replay or drain.

| PA | Mean active rows | Logical context tokens | Unique physical context tokens | Mean Attention/layer (ms) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.86 | 126,678 | 38,657 | 0.426 |
| 1 | 3.10 | 108,029 | 45,700 | 0.392 |
| 2 | 4.47 | 121,287 | 38,437 | 0.431 |
| 3 | 2.92 | 93,095 | 42,118 | 0.370 |
| 4 | 1.62 | 57,239 | 41,881 | 0.368 |
| 5 | 2.50 | 71,277 | 37,565 | 0.353 |
| 6 | 1.74 | 56,646 | 40,229 | 0.351 |

Mean PA-step logical/unique/shared context is 90,607/40,656/49,952 tokens.
Unique tokens describe active decoded contexts, not whole-device allocated KV
occupancy. Flat Pearson correlation between mean layer Attention duration and
logical tokens/request count/unique tokens is 0.720/0.681/0.444. These correlated,
serially dependent and alias-contaminated observations identify no independent
causal coefficients. The current untraced run's 266 PAT rebuilds and
66 Triton selections count topology decisions, not fractions of decode steps.

## E2E regimes and capacity evidence

| Run/case | Successful records | Mean TTFT | Mean request ITL | Max sampled KV | Prefill queue-positive samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current 19:07 `coding-half` | 180 | 1.608 s | 43.109 ms | 37.6% | 0/140 |
| Pre-cleanup 13:58 `coding-half` | 180 | 1.605 s | 43.746 ms | 37.4% | 0/131 |
| Pre-cleanup 13:58 `coding` | 180 | 58.337 s | 80.961 ms | 98.3% | 188/233 |
| Pre-cleanup 13:58 `coding-full` | 195 | 45.197 s | 90.566 ms | 99.9% | 212/258 |
| Pre-cleanup 13:58 `coding-half-trace` | 180 | 1.637 s | 46.575 ms | 35.7% | 0/139 |

In `coding`, maximum sampled Prefill waiting count is 5; in `coding-full`, 7.
Queueing and KV occupancy >=90% coincide in 52/233 and 92/258 samples respectively.
All samples have at most one running Prefill request per engine. The long coding
run logs roughly 167,136–171,040 token KV capacity per PA engine. Occupancy samples
are sparse (~10 seconds) and not time-weighted; they cannot exclude short stalls
in the half workload or prove allocation failure is the high-pressure cause.
In particular, a Prefill queue can also arise from compute service demand.
There are zero detailed Prefill IPC phase records in these five runs.

The `coding-full` phase log records 255 sent, 195 completed, 60 cancelled and
600.00 seconds of phase time. AIPerf's success-profile `benchmark_duration` is
597.030 seconds; its 315.731 tokens/s uses that success-profile window. It is not
188,501/600 and excludes cancelled requests' outputs. Do not equate the 2,092
available sessions / 16,049 dataset turns with executed requests.

Current half: 84,155 output tokens, 291.477 s complete replay, 288.719 tokens/s.
Earlier half: same outputs, 294.756 s and 285.507 tokens/s. Matched by conversation
and turn, **98/180 actual input lengths differ**, with deltas -593 to +1,110 tokens;
66/180 cache-read lengths differ. All output lengths match. Thus even though the
frozen requested fixture and environment match, this is not a controlled equal-
realized-workload A/B. The 1.12% throughput difference supports no speedup claim.
The source of realized prompt-length drift was not diagnosed in this audit;
before any causal A/B, freeze/verify actual request payloads and arrival events.

## Cheapest next probe and missing causal evidence

**The cheapest next probe is correctness, not a routing benchmark:** a minimal
CPU test of scheduler allocation and exported lease ownership for one prompt
plus decode headroom. Confirm that every writable logical block maps to a
distinct, actually allocated non-padding block held for the request lifetime.
Use the retained single-request witness above to validate the failure model.
Stop optimization/capacity conclusions until the root cause is permanently
fixed and one bounded reference-equivalence test validates the decode path.

After correctness is established, a CPU cost-model test may use a *new valid*
trace with held-out membership epochs, comparing physical-token-only against
logical/request-count/PAT features. The currently retained aliased trace is
not admissible training/evaluation data for the intended inference mechanism.

For a capacity-oriented idea, the necessary next *measurement*, only with user
approval, is one bounded high-pressure reproduction rather than another full
suite: correlate request ID, Prefill enqueue/schedule reason, free/reclaimable/
leased blocks, PA decode membership, lease release, and first-token delivery.
This must distinguish compute queueing from blocked KV allocation and prefix-
locality placement. If a root-cause claim is selected, follow with one-variable
A/B at identical effective configuration, actual request payloads, processes,
devices and client state; stop and invalidate on mismatch.

## Reproduction and checks performed

```bash
.venv/bin/python benchmarks/pap/experiments/microbench/PAP-20260905-RESEARCH-DIAGNOSIS/audit_saved_evidence.py
```

This prints full numeric JSON, checks 8 raw hashes, recreates 35 tensor fields,
checks request-ID equality, validates consecutive steps and positive adjacent
intervals, and verifies exact nanosecond additivity. It additionally tests the
single-request no-alias invariant and **exits 2 on the retained aliasing witness**.
Source archives and original client/log files remain unchanged. CPU execution
reproduced the failure. The cached project
Ruff formatter/checker passed on the analysis script. No model eval or GPU test
was appropriate or run for this analysis-only change.
