# Refactor validation checkpoints

> **2026-09-05 research audit: numerical/performance interpretation suspended.**
> Later reanalysis found within-request KV block aliasing in the retained
> `135804_3149938/coding-half-trace` raw data. The source allocation/export
> defect also exists at the cleanup checkpoint. Completion, output lengths,
> Graph captures and drain still describe what those audits checked, but they
> do not certify numerically correct inference. Do not use affected PAP timing,
> physical-prefix reuse or the cleanup comparison below as correct-inference
> regression/performance evidence until the ownership/lifecycle repair and
> numerical revalidation pass. Raw numbers and artifacts are preserved rather
> than rewritten. See
> [the independent evidence audit](../../microbench/PAP-20260905-RESEARCH-DIAGNOSIS/evidence_audit.md).

## coding-half: passed before protocol/tracing extraction

Run: `runs/20260905_111034_2965031/coding-half/`.
Its exact source, dependency, hardware and model snapshots are in the suite's
`provenance/`. Later refactors still require their own end-to-end validation;
this checkpoint is not evidence for code changed after this run.

Configuration: 7PA1P, Dynamo routing, 2K Prefill budget, Qwen3-8B FP16/YaRN 131K,
60 sessions / 180 sequential turns, concurrency 60, Poisson 0.9 req/s, no warmup
or measurement cutoff. Dataset SHA-256:
`258b72c85772c9d372f1b63ee0bf6d710f27cb00234027e2c750c82a5fa9563c`.

| Evidence | Result |
| --- | --- |
| Client records | 180, no errors |
| Output-length mismatches | 0 |
| Total output tokens | 84,155 |
| Measured duration including replay drain | 291.445 seconds |
| Output throughput | 288.751 token/s |
| Mean TTFT | 1,616.764 ms |
| Mean request-level ITL | 43.277 ms |
| Mean end-to-end request latency | 21,692.383 ms |
| Whole-step Graph instances | 8/8 captured, all 7 PA active |
| Routing / lifecycle / correctness audits | Passed |
| Final Attention sessions | All 7 PA drained |
| Dynamo hash mismatch / missing-parent errors | None observed |
| PAT rebuild / Triton selection counters (sum over PA) | 263 / 68 |

Sources: `aiperf/profile.json`, `aiperf/profile.jsonl`,
`pap_whole_step_graph_audit.env`, `routing_audit.json`,
`correctness_audit.env`, `attention_fast_path_stats_*.json`, and `launcher.log`.
This run demonstrates execution of both PAT and Triton, but is not a controlled
performance A/B against an earlier revision. The configured arrival rate and
observed completed-request throughput are different quantities.

## short-context: invalidated

Run: `runs/20260905_105242_2939264/short-context/`.
All 14 requests completed and drained, but salted prefixes caused Dynamo event
index errors that the original audit missed. The suite's `INVALIDATED.md` records
why this result is excluded. The dataset and routing were not altered to bypass
the issue. Salted-Dynamo support remains an unresolved compatibility boundary.

## coding-half-trace: execution passed, raw retention incomplete

Run: `runs/20260905_112723_2989779/coding-half-trace/`, after protocol/tracing
module extraction. All 180 requests completed with zero errors and zero
output-length mismatches; Graph/lifecycle audits and drain passed. A valid
intermediate join contains steps 3314–3825 and the expected `[512,36,7]` PA and
Attention tensors plus `[512,36]` Projection tensor.

Final raw-ring export became empty during drain, so the join cannot be reproduced
from those final raw files. This is a trace-retention failure, not an inference
success claim for the entire diagnostic case. See the suite's
`TRACE_CAPTURE_INCOMPLETE.md`. An automatic raw-window collector is now part of
the diagnostic driver and must be verified by rerunning the case.

## coding-half-trace with automatic capture: passed

Run: `runs/20260905_114511_3015679/coding-half-trace/`. All 180 requests completed
without errors or output-length mismatches; Graph/correctness/lifecycle audits
and drain passed. The automatic collector froze steps 1861–2372 with two
`[512,36,7]` latency tensors and one `[512,36]` Projection tensor. All eight raw
file hashes matched the capture manifest, and replaying the join reproduced
all 35 saved tensor fields exactly. Raw files and `capture.json` are retained
under `trace_capture/`, independently of later rolling exports.

## coding: invalidated by active-reservation expiry

Run: `runs/20260905_114511_3015679/coding/`. All 180 inference requests completed,
but three live requests lost their native Dynamo load reservations before
completion. This result is excluded from valid load-accounting/performance
comparisons. See [reservation_lifetime.md](reservation_lifetime.md).

The queued `coding-full` case was stopped with exit 130 and has no completed
measurement. This suite has no overall success marker.

## Explicit-owner Dynamo + shared-prefix short-context: passed

Run: `runs/20260905_134349_3127559/short-context/`.
Uses the new immutable `qwen3-8b-yarn131k-shared-prefix` fixture and PAP-only
source-built Dynamo selector. All 14 requests completed; every output was 16
tokens. Correctness, routing, topology, Graph and lifecycle audits passed.
Four PA nodes received requests; three were idle and correctly did not need an
Attention Graph. The required five active-process Graph captures were present.

At drain, the gateway reported 14 selections, zero failures, zero reservations,
zero pending selections/cleanup, and a healthy `explicit-owner-v1` contract.
All seven native worker load records had zero active requests, Prefill tokens
and Decode blocks. All seven Attention session counts were zero; all 14 leases
were released. Service shutdown completed normally.

Cross-session reuse is observed in client records: first turns of conversations
`004`, `005`, and `006` each report 8,208 cached tokens out of an 8,210-token
prompt. These are first turns, so the hits cannot come from those conversations'
own previous turns. Other first turns reported no cache hits. The new policy
allows sharing; it does not require every conversation to use the same PA.

Sources: `aiperf/profile.jsonl`, `topology_runtime_stats.json`,
`routing_audit.json`, `pap_whole_step_graph_audit.env`, `gateway_drain.env`,
`session_drain.env` and `service_logs/proxy.log`. Dependency binary/build record
and the source snapshot are preserved under the suite's `provenance/`.
This short correctness run is not a performance comparison or full-suite pass.

Post-fix regression checks: `tests/pap` passed 270 tests in the sandbox, with
one CUDA test skipped for GPU visibility. That exact skipped test,
`test_qwen3_gqa_paged_decode_matches_reference`, was then rerun outside the
sandbox and passed (271 distinct tests covered). The targeted lifecycle/replay/
vLLM integration subset passed 70 tests; Ruff, mypy and ShellCheck passed.
Dataset SHA-256 checks passed for all nine registered files; all three new
shared-prefix fixtures passed Dynamo replay preflight.

## Post-fix all-active-dataset queue: passed before final model cleanup

Suite: `runs/20260905_135804_3149938/`, launched with no case filters. This is
the fresh seven-case queue, not a continuation of an invalidated old run.
The `short-context` case passed and released all Gateway/Attention resources
under the strengthened native-reservation drain audit. `long-context-1` also
passed: two requests with input lengths 125,018 and 129,058 tokens, 16 output
tokens each, zero output-length mismatches and 125,008 cached tokens in the
second turn. Both cases passed correctness, routing, Graph, decode-token join
and drain audits; native/router reservations and all worker load counters were
zero at drain. Other cases remain pending until individually audited.

`long-context-7` subsequently passed all 14 requests at the same two input
lengths, with 16 output tokens per request and no length mismatches. All seven
PAs served two requests each; all eight required whole-step Graph captures were
present. Correctness, routing, decode-token join and drain audits passed; the
native/router load counters returned to zero. The queue then entered
`coding-half`; this is three passed cases, not a full-suite success.

`coding-half` subsequently passed 180/180 requests, producing 84,155 output
tokens with zero output-length mismatches. All eight required Graph captures
were present. Correctness, routing, decode-token join, Gateway drain and all
seven PA session-drain audits passed. The selector recorded 180 selections,
zero failures and no remaining native or wrapper reservations/worker load.
Client measurement duration was 294.756 seconds; mean TTFT 1,604.934 ms and
mean request ITL 43.746 ms. These are checkpoint observations, not a controlled
performance A/B. The queue moved on to `coding` (four passed cases so far).

`coding` subsequently passed 180/180 requests and produced 168,402 output
tokens, with no output-length mismatches. All eight Graph captures and all
correctness/routing/decode-token/drain audits passed. Native and wrapper load
counts returned to zero, and the router reported zero failures.

Joining each Gateway `PAP Dynamo placement` log to AIPerf
`metadata.request_end_ns` by `metadata.x_request_id` gives four requests with
selection-to-client-completion duration above the former 300-second threshold:

| Request | Duration (s) |
| --- | ---: |
| `00f5cdbf-a4da-4c24-9781-ccd95a03ac3e` | 306.099 |
| `43d43471-e0b4-40d8-8f57-de2d0c8a3f02` | 307.538 |
| `2363c370-0ab6-4b82-bc47-6ca7d86be996` | 323.878 |
| `2c726643-0848-4fb8-af86-3cc6875dbe1c` | 519.783 |

The Gateway's Python placement timestamps use Asia/Shanghai; the client end
timestamp is Unix nanoseconds. There were no premature-expiry warnings or failed
native reservation releases. The 519.783-second case extends the real-workload
evidence beyond the old periodic expiry sweep, in addition to the controlled
370-second CPU probe. These durations are not Attention kernel latency or TBT.

Client measurement duration: 752.048 seconds; mean TTFT 58,337.241 ms; mean
request ITL 80.961 ms; output throughput 223.924 tokens/s. This is a new valid
checkpoint, not a performance A/B with the previously invalidated run. The
queue moved on to `coding-full` (five passed cases so far).

`coding-full` passed its 600-second timed protocol. AIPerf's phase log reports
255 requests sent, 195 completed, 60 cancelled, zero errors, and elapsed
600.00 seconds. All 60 outstanding credits were returned after cancellation.
The success profile contains 195 records, all non-cancelled, with zero output
length mismatches and 188,501 completed output tokens. Cancelled requests are
not included in that success profile; 255 is **not** the inference-success count.
The configured dataset still contains 2,092 sessions / 16,049 available turns;
this timed test did not execute them all.

The PAP lifecycle coordinator recorded 255 terminated/cleaned request scopes,
zero active scopes and zero failures. The local load tracker had no remaining
Prefill or Decode requests. The selector recorded 255 selections, zero failures,
zero native/wrapper reservations and zero per-worker load. All seven Attention
sessions drained; all eight required Graph captures and the correctness,
decode-token and cancellation-aware routing audits passed. Lease-release counts
are not equated with request counts: requests cancelled before creating a lease
need not have a lease to release.

Sources: `coding-full/launcher.log` phase-completion lines,
`aiperf/profile.json[l]`, `topology_runtime_stats.json`, and per-case audits.
The queue advanced to `coding-half-trace`: all six active dataset files now have
passing cases in this queue, but tracing and final model-interface cleanup remain.

`coding-half-trace` subsequently passed 180/180 requests with no output-length
mismatches and with all correctness/routing/Graph/decode-token/drain audits
passed. The frozen trace retains steps 1980–2491. All eight raw file hashes
matched both before and after drain, and recomputing from those files reproduced
all 35 tensor fields exactly. Shapes are `[512,36,7]`, `[512,36,7]` and
`[512,36]` for Projection-observed PA, PA-kernel and Projection latency.
The suite's seven exit codes are zero and `COMPLETE` exists; its driver exited
normally and GPU process cleanup was verified.

After that queue completed, obsolete Projection direct-send/timeline interfaces,
the empty message-release return/loop and the redundant per-layer reset were
removed. Model-hook/debug booleans now use the shared strict parser. All 276 PAP
tests passed with GPU access, with no skips; Ruff and mypy passed. The validated
pre-cleanup queue remains preserved as-is. Post-cleanup validation is recorded
below with its separate source snapshot and the user's final scope decision.

## Post-cleanup validation: duplicate full queue stopped by user

Suite `runs/20260905_145710_3297423/` was launched with all seven cases after
the model-interface cleanup and the 276-test pass. It has a separate source,
configuration and dependency snapshot. Only its completed, audited cases are
counted; the earlier passing suite is not overwritten or relabelled.

The user objected to repeating the expensive full queue for this limited cleanup.
It was stopped during `coding-half` with exit 130; do not restart it automatically.
The three context cases had completed with exit zero. The interrupted coding
attempt is not a completed result. Use the earlier full-suite evidence and
proportionate post-cleanup validation with their distinct source snapshots.

The final-source `short-context` and `long-context-1` cases passed 14/14 and
2/2 requests respectively, with 16 output tokens per request and no length
mismatches. The long-context inputs were 125,018 and 129,058 tokens. Both passed
correctness, routing, Graph, decode-token join and resource-drain audits, with
zero remaining native/router reservations. A byte comparison confirmed that
all 98 PAP runtime/build source files in the suite archive match the current
working tree; the suite/case dependency snapshots match the verified native
router binary. The post-cleanup `long-context-7` case also completed 14/14
requests with matching output lengths, passing audits and zero remaining native
load. All three cases were rechecked from their existing records during closeout.
The remaining duplicate cases were stopped or never started; they are not
pending obligations under the user's instruction against full repetition.

Provenance checks confirmed that the source archive contains the native router
lockfile/patch and the new dataset manifest. The installed router library, suite
snapshot and first case snapshot have identical SHA-256
`63ce191ff8b52f525fad1577daf30d17e34df0742e6d88fc5f3145cc7ca1a639`.
No runtime implementation changes were made during either validation queue.

## Accepted closeout scope

The user confirmed using the completed 7/7 suite, 276 post-cleanup tests and
three post-cleanup E2E cases as the verification basis. No additional experiment
or test was run during closeout. The source snapshots remain distinct. The
interrupted duplicate attempt remains exit 130, not a failure of a completed
correctness run and not a pass. Any further test needs prior user approval.

## Recursive dead-code cleanup: single-case E2E regression check passed

The user requested this additional E2E check after recursive Vulture cleanup.
Only `coding-half` was selected; the other datasets were not replayed.

Command: `bash benchmarks/pap/experiments/e2e/PAP-20260905-REFACTOR-VALIDATION/run.sh coding-half`.
New run: `runs/20260905_190715_3493447/coding-half/`.
Reference: `runs/20260905_135804_3149938/coding-half/`.
Both use 7PA1P, 2K Prefill, concurrency 60, Poisson 0.9 req/s, seed 42,
60 conversations / 180 turns, no warmup, no tracing and full replay with drain.

| Metric | Earlier checkpoint | After cleanup | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT (ms) | 1,604.934 | 1,608.174 | +0.20% |
| Mean request-level ITL / TBT (ms) | 43.746 | 43.109 | -1.46% |
| Mean request end-to-end latency (ms) | 21,945.088 | 21,619.500 | -1.48% |
| Output throughput (tokens/s) | 285.507 | 288.719 | +1.12% |
| Replay duration including drain (s) | 294.756 | 291.477 | -1.11% |

No material performance regression was observed at this point. This is one
historical-checkpoint comparison, not repeated randomized A/B evidence: the
small differences do not establish a speedup caused by dead-code removal.
ITL here is AIPerf's average of per-request token intervals, not a GPU layer
or decode-step timing. Throughput includes the finite replay's drain.

Validation:

- 180/180 records, zero errors/cancellations/output-length mismatches;
  84,155 output tokens, matching the reference.
- Correctness, routing, decode-token join, Prefill Graph, whole-step Graph,
  Gateway drain and Attention session drain audits all passed.
- All seven PA nodes served requests; eight required whole-step Graph
  instances captured. Each PA's static MPS audit reports 80 Prefill SMs and
  12 Attention SMs on its assigned physical GPU; Projection uses GPU 7.
- All 180 KV leases released. Lifecycle failures, native/router reservations,
  pending cleanup, worker loads and Attention sessions were zero at drain.
- Driver exit code zero and suite `COMPLETE=passed` apply to this one selected
  case only. All eight GPUs returned to 14 MiB and zero utilization, with no
  compute processes remaining.

The effective configurations differ only in output paths and deletion of
three previously inactive trace/audit settings. Package inventories for the
PAP, AIPerf and Dynamo environments, model-file manifest and GPU topology
match the reference byte-for-byte. The current run's source archive matches
all 83 archived PAP files in the working tree. The dirty-source snapshot, not
the shared Git HEAD, identifies the tested implementation.

Both runs contain the same optional DeepGEMM import warning. This unquantized
FP16 L20 path does not use DeepGEMM; no dependency override was introduced for
the experiment. Raw client records, service logs, audits, effective settings
and source/dependency snapshots remain in the run directory. Earlier CPU/GPU
validation of this cleanup passed all 282 PAP tests with zero skips.

## Attention-owned Decode growth: bounded E2E passed

Run: `runs/20260905_231410_3633861/coding-half/`. This is the first bounded E2E
after replacing padded worker-table Decode capacity with allocator-owned,
Attention-initiated growth. Prefill initially owns only the blocks needed by
the computed prompt. At a Decode step boundary, Attention requests another
256-token allocation chunk from that request's Prefill engine only when the
next token would exceed its writable range.

The case completed 180/180 requests with no client errors, cancellations or
output-length mismatches and produced 84,155 output tokens. Routing,
topology, Prefill/Attention/Projection Graph, decode-token join and drain
audits passed. All eight GPUs returned to 14 MiB and zero utilization after
shutdown.

The final Attention counters record exactly 404 capacity requests, 404
installs, 5,254 newly owned blocks, and zero topology mismatches. Recomputing
the expected page crossings solely from each client's actual input/output
length also gives exactly 404. The final Prefill snapshot records zero
allocation failures. This workload did not trigger Prefill preemption, so the
revoke-before-free branch is covered by the real-scheduler pressure test rather
than claimed as E2E-observed behavior.

This run predates the final atomic exact-lease check in the publisher. That
follow-up closes only the revoke-versus-late-publication race and does not alter
the exercised no-preemption path; it is covered by the final full PAP suite and
the dedicated late-manifest test rather than attributed to this E2E snapshot.

| Metric | Dynamic allocator run |
| --- | ---: |
| Mean TTFT | 1,689.998 ms |
| Mean request-level ITL / TBT | 54.816 ms |
| Mean request end-to-end latency | 26,941.021 ms |
| Output throughput | 273.478 token/s |
| Replay duration including drain | 307.721 s |

These values are a new execution checkpoint, not a valid regression comparison
against `135804_3149938/coding-half`. That older implementation has proven
within-request block aliasing. The realized workloads also differ: matching by
conversation and turn, 100/180 input lengths and 79/180 prefix-cache-read
lengths differ, although all output lengths match. Current/old input totals are
5,233,066/5,228,520 tokens.

`dynamic_kv_e2e_analysis.json` classifies external client token intervals by
whether the corresponding token crosses a predicted allocation boundary. The
404 crossing intervals average 56.866 ms; the other 83,487 average 54.167 ms.
This request-local classification is not a system-wide cost estimate: one
crossing stalls the PA/Projection barrier for every request in that decode
batch. The earlier inference that weighting 0.482% of request intervals ruled
out allocation as the primary cause was incorrect and is superseded by the
step-level trace below. The JSON now records this limitation explicitly.

This run plus the allocator/registry/kernel tests establishes structural block
ownership, growth accounting and normal-path execution. It is not a direct
logit or generated-text equivalence test against an independent implementation;
completion and output-length equality alone are not labelled numerical proof.

## Dynamic-KV step trace: synchronous growth is the TBT regression

Run: `runs/20260906_002049_3682642/coding-half-trace/`. The source includes the
final exact-lease race fix. Effective configuration is 7PA1P, 2K Prefill,
Qwen3-8B YaRN 131K, 60 conversations / 180 turns, concurrency 60 and Poisson
0.9 req/s. The run completed all requests and 84,155 output tokens with no
errors, cancellations or output-length mismatches. Routing, topology,
Prefill/whole-step Graph, decode-token join and all drain audits passed. The
405 client-predicted Decode growth requests exactly match 405 Attention
requests and installs; 5,256 blocks were added and topology mismatches are zero.

The collector froze consecutive global steps 1156–1667. All eight raw hashes
verify after drain, and rejoining the raw files reproduces `[512,36,7]`
Projection-observed PA latency, `[512,36,7]` PA-local Attention kernel latency
and `[512,36]` Projection latency. Among 288 single-request PA-step cells, zero
has fewer unique active blocks than logical block positions; the prior alias
witness is absent in this window.

Exact step cadence is reconstructed only from adjacent Projection-GPU markers:
dispatch-done to gather-done plus gather-done to the next dispatch-done. The
integer nanosecond intervals telescope exactly; no nonadjacent timing buckets
are added.

| Exact step population | Count | Mean cycle | P50 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| No observed lease growth | 462 | 44.564 ms | 42.784 ms | 76.538 ms | 84.490 ms |
| At least one lease grew | 49 | 118.335 ms | 58.848 ms | 691.910 ms | 773.894 ms |
| All exact cycles | 511 | 51.638 ms | 43.164 ms | 249.296 ms | 773.894 ms |

Growth-step and ordinary-step compute are nearly unchanged:
gather-to-next-dispatch sums are 27.260/27.185 ms and max-PA Attention-kernel
sums are 14.929/14.701 ms. The difference is dispatch-to-gather wait:
91.075 ms on growth steps versus 17.379 ms otherwise. Layer-0 slowest-PA
latency is 75.491/2.048 ms respectively. The worst cycle is step 1416:
773.894 ms total, 728.957 ms in the layer-0 slowest PA path, while its full-step
Attention kernel work remains normal.

The 49 growth steps occupy 9.59% of this window. Relative to the ordinary-step
mean, their observed excess contributes 7.074 ms to the all-step mean. This
direct batch/step-level evidence explains most of the 8–11 ms raw TBT increase
that motivated the trace. AIPerf reports 54.117 ms mean request ITL, close to
the preceding untraced dynamic-allocation checkpoint's 54.816 ms; tracing is
not the source of the regression.

Measured root cause: synchronous capacity growth is inside the layer-0 PA
preflight and therefore inside Projection's global barrier. Source inspection
shows the request then traverses HTTP, the Prefill API control dispatcher and
an EngineCore utility call. EngineCore handles utility input only between model
steps, and the dispatcher serializes it with Decode commits. The trace does not
separate API-queue, dispatcher-queue and in-flight Prefill-step wait, so their
individual shares remain unverified. Do not label any one of those sub-buckets
the root cause without boundary instrumentation or a one-variable A/B.

Artifacts: `trace_capture/capture.json`, `merged.pt`,
`dynamic_kv_trace_analysis.json`, `dynamic_kv_e2e_analysis.json`, the raw
Projection/Attention files and the saved effective configuration/source
snapshot.

## Low-watermark asynchronous Decode growth: tracing A/B passed

Run: `runs/20260906_005320_3718392/coding-half-trace/`. This changes the
synchronous growth above to one in-flight prefetch per request. Prefill readiness
starts the first request; later requests start when writable headroom is below
256 tokens. Each request adds 256 tokens (16 blocks) unless capped by its output
limit. HTTP runs in a daemon thread, but returned ownership is installed only at
an Attention preflight boundary. Reaching the current writable end waits for the
existing request rather than issuing a duplicate.

The run completed 180/180 requests and the same 84,155 output tokens. All
correctness, routing, topology, Prefill/whole-step Graph, decode-token join and
drain audits passed. It made 405 allocation requests, all 405 as prefetches;
only four later reached a capacity boundary and waited. All 405 installed,
adding 5,256 blocks, with zero failures, pending requests or topology
mismatches at drain. Aggregate wait time over the four Attention processes was
1.532 seconds. All GPUs returned to 14 MiB and zero utilization.

| Metric | Synchronous growth | Async low watermark | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 1,745.255 ms | 1,605.288 ms | -8.02% |
| Mean request ITL / TBT | 54.117 ms | 48.540 ms | -10.31% |
| Mean end-to-end latency | 26,758.254 ms | 24,004.107 ms | -10.29% |
| Output throughput | 273.250 token/s | 278.213 token/s | +1.82% |
| Replay duration | 307.979 s | 302.484 s | -1.78% |

The trace confirms removal of the original growth long tail. Growth-step mean,
P99 and maximum cadence change from 118.335/691.910/773.894 ms to
56.700/107.500/146.985 ms. In the new window, growth and ordinary steps average
56.700 and 51.315 ms; the former 73.771 ms gap is 5.385 ms. The windows represent
different load phases (steps 1156–1667 versus 2049–2560), so their overall cycle
means are not compared as a direct speedup.

The initial AIPerf `inputs.json` hashes are identical. Multi-turn realized
prompts are not byte-identical after model generation: 93/180 input lengths and
75/180 cache-read lengths differ. Total input is 5,233,898 versus 5,231,436
tokens (-0.047%); total output is identical, while the async run has 34,432 more
cache-read tokens (+0.76%). Therefore the endpoint table is strong supporting
evidence, not a strict payload-identical randomized A/B. The within-trace
allocation-wait counters and disappearance of growth-step stalls provide the
direct mechanism evidence.
