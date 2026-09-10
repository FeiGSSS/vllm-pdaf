# Native-32K PAP TTFT and KV retirement diagnosis

## Evidence scope

2026-09-06. Workload: immutable `s60-t3-native32k-stratified-seed42`,
60 concurrent sessions, 180 turns, Qwen3-8B FP16, max length 32768, no YaRN,
7PA1P, Prefill budget 32768, Prefill 80 SM and Attention 12 SM per PA.
Baseline: `../_runs/pap/20260906_native32k_7pa1p_conc60_32k_attempt3`.
The diagnostic replay is `diagnostic-run/`; its runner, effective configuration,
service logs, API-cached Prometheus samples, and raw AIPerf records are preserved.
`run.sh` enables existing IPC and lease logging and samples cached `/metrics`
every 0.5 seconds. It does not query scheduler utility endpoints or change
production code. It is diagnostic evidence, not a speedup A/B.

Reproduce analysis with `.venv/bin/python <this directory>/analyze.py` and
`.venv/bin/python <this directory>/analyze_diagnostic.py`.
Request identifiers, source paths, and timestamps are saved in `analysis.json`
and `diagnostic_analysis.json`. Computed line indices use Python splitlines;
carriage-return graph progress can offset them from physical shell line numbers.

## Definitions and corrected interpretation

A request is one conversation turn. A Prefill scheduler iteration can execute
chunks from several requests. A PAP Decode step advances the currently active
Decode batch. A session handle, a Prefill engine request ID, and a lease ID
are different identities and must not be substituted for each other.

Request timeline: client send -> gateway processing/routing -> Prefill HTTP
dispatch -> EngineCore enqueue -> first scheduling -> Prefill output ->
Attention readiness -> Projection admission/execution -> first token -> last
token -> Attention retirement -> Prefill lease release -> allocator reuse.

The earlier claim that all high TTFT comes from active Decode filling KV was
not established. Running=0 in an API-server statistics snapshot does not prove
that the GPU is idle at that instant. This run explicitly uses Prefill async
scheduling OFF; do not explain it by inventing two in-flight Prefill batches.
Statistics and GPU work do not share an instantaneous observation boundary.

## Measured findings

Baseline means, seconds:

| Turn | TTFT | Prefill HTTP | Client to route selected | Other route-to-ready | Ready to first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 19.521 | 12.811 | 6.005 | 0.524 | 0.180 |
| 2 | 12.510 | 12.236 | 0.026 | 0.016 | 0.233 |
| 3 | 12.824 | 12.576 | 0.032 | 0.008 | 0.208 |

The diagnostic replay completed 180 turns in 76.980 seconds, mean TTFT 15.512 s
and TBT 51.916 ms. Its native Prefill histograms cover all 180 turns:

- First QUEUED to first SCHEDULED: mean 3.317 s.
- First SCHEDULED to first Prefill NEW_TOKEN: mean 6.694 s.
- Gateway Prefill HTTP: mean 13.174 s. The difference from the above two
  intervals is 3.164 s; this localizes overhead outside those native events,
  but cannot identify how much is input preprocessing, EngineCore ingress,
  output delivery, or event-loop scheduling without more boundaries.
- There were zero recorded Prefill preemptions.
- Prefill time is elapsed residence including chunk waits, not pure CUDA time.

All seven baseline Attention counters report zero Decode allocation requests,
waits, and failures. This short-output workload never exercises dynamic runway
growth. Reducing the 256-token allocation granularity cannot explain this run's
performance gap.

### Finished requests hold leases for seconds

Join each client request end to the same request's Prefill lease release:
180/180 matches, mean 12.391 s, median 12.507 s, maximum 25.521 s.
Prefill log resolution is one second; uncertainty is much smaller than this gap.
Join Attention's local retirement instead: mean 12.456 s, maximum 25.582 s.
Lease release is only removal from the registry; allocator reuse can occur later.

Concrete PA3 example, request `b605a9ee-30da-4e98-bb1a-d47654963243`:

- 18:27:13: Prefill pins lease `pap-lease-b781432a51954574`, 616 blocks.
- 18:27:20.178: Attention installs prompt KV, prefix length 9780.
- 18:27:23.418: Attention releases its local session/lease mapping.
- 18:27:23.479: client request ends.
- 18:27:49: Prefill removes the corresponding lease, 616 blocks = 9856 tokens.

Sources: `diagnostic-run/service_logs/attention_3_0.log`, physical lines 238/319;
`diagnostic-run/service_logs/prefill_3.log`, physical lines 265/411/412;
client record with the same request ID. These timestamps establish retirement
lag, not the exact fraction caused by each intervening queue.

Summing final lease block counts times post-client release lag gives 38.49
million token-reference-seconds across all requests. This is NOT unique physical
memory: shared-prefix references can overlap, and block sets were not logged.
Do not report this sum as available capacity or translate it directly to speedup.

### Final commit targets the wrong identity

Both baseline and diagnostic logs have 180 final commits rejected as
`unknown_request` (one per successful client request). Example:

- Actual Prefill owner: `chatcmpl-b605a9ee-30da-4e98-bb1a-d47654963243-88020cce`.
- Commit target: `b605a9ee-30da-4e98-bb1a-d47654963243@pap-session-5`.

`step_context_registry.py` uses the dispatch request ID as `commit_request_id`;
that is an Attention handle in this path. `PAPEngineAdapter.apply_decode_commit`
looks directly in `scheduler.requests` and then in the lease registry by that
string. Neither lookup resolves the Attention handle to the Prefill owner.
`submit_only=True` acknowledges queue acceptance; deferred failures are warnings,
not propagated to the original caller. Lease release uses the separate valid
lease ID and eventually succeeds. Thus these are not evidence of a permanent
memory leak, but final Decode cache commits are not applied.

Effects on multi-turn cache reuse must be measured after correction. With mean
output 82 tokens, losing only the final Decode suffix is small; this bug alone
does not establish that thousands of old prompt tokens were recomputed.
Client success and existing drain audits do not establish internal commit
correctness. Prior declarations of complete correctness need this qualification.

### Prefix locality and initial ingress

Baseline changes PA for 42 of 120 consecutive-turn transitions. Changed-PA turns
average 5240 cached tokens; same-PA turns average 15758, but input lengths differ
and this is observational, not a causal A/B. PAP overall hit ratio is 50.4%, DP8
59.0%. Matching DP8's hit ratio at fixed inputs would reduce PAP's uncached token
count by approximately 17%; this is a compute-work opportunity, not a guaranteed
17% TTFT or throughput improvement.

In the diagnostic first turn, instrumented gateway means are tokenization
1.535 s, routing 1.119 s, registration 0.625 s; they are wall waits including
contention. Client-to-handler delay is not included in these spans. In later
turns tokenization is 26-33 ms, routing 1-2 ms, registration 1.3-1.4 ms,
and Attention readiness 3-9 ms. Do not label the whole initial gap tokenizer
compute. Baseline cold-burst ingress alone contributes about 2.00 s to mean
TTFT when averaged over all three turns.

## Source-supported mechanism and repair order

`PAPControlDispatcher._run` awaits one EngineCore utility before submitting the
next. `EngineCore.step` calls model execution, waits for the result, and processes
input utilities at the outer loop boundary. UniProcExecutor's `non_block=True`
still invokes the worker method inline; it is not a general background executor.
Consequently long Prefill iterations can delay controls, and a serial dispatcher
can spread an accumulated retirement queue over multiple iterations. After a
registry release, PAPPrefillConnector.get_finished must return finished_sending
before the scheduler frees allocator references. Several progress dependencies
exist. The measured 12.4 s spans the complete path, not one RPC boundary.

1. Make final commit and retirement one validated transaction keyed by authoritative
   lease ID, generation, and Prefill owner; preserve final token IDs, hash/cache
   updates, and idempotency. A rejected commit must be observable and must prevent
   a successful retirement acknowledgment. A cached HTTP enqueue ACK is insufficient.
2. Drain a batch of eligible retirements before new Prefill admission at each
   allocator-safe boundary. Avoid one utility acknowledgment per request becoming
   one further Prefill-iteration wait. Return completed, no-longer-used references
   at that boundary instead of waiting for an unrelated later model iteration.
3. If boundary batching leaves multi-second waits, make the engine responsive to
   a narrow retirement/control mailbox while computation is outstanding. This
   requires explicit executor completion fences and single allocator ownership;
   simply toggling async scheduling or mutating the allocator from an HTTP thread
   is not a sufficient implementation.
4. Remove cold ingress bottlenecks with bounded concurrent tokenization/routing,
   proactive initialization, and event-loop-lag measurements. Confirm all stage
   boundaries before attributing wall waits to CPU compute.
5. Then evaluate cache-locality/load-aware turn placement and Prefill chunk budgets
   using a one-variable A/B. Smaller chunks may shorten control service latency,
   but can reduce compute efficiency; reducing the budget is not the root repair.

Safety contract: exact lease owner/generation; final accepted tokens reconciled;
all Attention layers quiescent; no Prefill/CUDA graph still accesses retired
blocks; shared block reference counts preserved; commit-before-release ordering;
allocator actual-free acknowledgment separated from command acceptance.

## Subsequent end-to-end correctness validation

Historical attempt: `../_runs/pap/20260906_native32k_commit_owner_fixed_v2/`.
This is a post-repair correctness observation, not a throughput or retirement
speedup experiment. Findings above describe the original source state.
The recorded workload is the same immutable native-32K dataset, 7PA1P, 60
concurrent sessions and 180 requests, with diagnostic profiling disabled.

The retained validation summary reports:

- 180 client successes and 180 unique applied final commits; zero unknown owners
  or deferred application failures.
- 14,516 committed tokens. Each request committed its output count minus one:
  the final sampled token has not yet been consumed to form KV.
- Correctness, Decode-token join, and session-drain audits passed.
- Mean TTFT 15.242 s and TBT 52.011 ms.

Evidence remains in that attempt's `effective_config.env`, `run_metadata.json`,
`tracked_worktree.patch`, `service_logs/`, `aiperf/`, `correctness_audit.env`,
`decode_token_join_audit.env`, and `session_drain.env`. These are historical
results, not checks re-executed during directory cleanup. The preceding invalid
attempt retains its original `validation_correction.md`; its initial successful
audit was invalid because the scan skipped ignored `_runs/` logs.

Further service validation uses the shared
`benchmarks/pap/scripts/run_pap_workload.sh` entry point with an explicit fresh
run destination. No repair-specific launcher is required. Source identity and
effective configuration must be compared before claiming historical reproduction.

## Research direction: reclaimable capacity as a scheduling resource

Working hypothesis: PAP's separation of compute is incomplete because resource
retirement still depends on progress of another compute stage. Treat completed
requests' pinned KV as a measurable capacity tax with a bounded retirement
latency, not as useful Decode demand. Build a per-lease retirement journal and
use allocator-confirmed reclamation credits for both admission and route scoring.
Do not count a future credit as presently free space.

Track disjoint physical block states: active Prefill, handoff, active Decode,
retirement pending, and evictable/free. Blocks referenced by any active request
must remain in the active union; shared prefixes must never be counted once per
request. Predict marginal newly pinned blocks after cache hits and the time
until useful work/retirement releases them. A route can then compare predicted
first-token finish time, including compute backlog and memory eligibility, rather
than comparing context sums alone.

Smallest falsification: fix identity/atomic retirement, then compare identical
workloads with immediate batched reclamation versus controlled delayed reclamation.
Measure retirement P50/P99, unique retired-pinned block-time, allocation failures,
native queue time, Prefill time, TTFT, TBT, and output throughput. The negative
control is low KV pressure: reclamation improvements should provide little TTFT
benefit there. If reduced retired block-time does not improve queue time/TTFT,
the capacity explanation is not dominant and prioritize Prefill compute/locality.

Novelty remains unvalidated. Prefix-aware routing already exists in
[Preble](https://arxiv.org/abs/2407.00023); chunked Prefill scheduling in
[Sarathi-Serve](https://arxiv.org/abs/2403.02310); segment-level KV pooling and
cache-independent compute scheduling in [TokenLake](https://arxiv.org/html/2508.17219v1).
Memory pooling, radix routing, or moving Q to KV alone should not be presented as
new contributions. A potential contribution is an end-to-end progress and capacity
contract for shared-KV compute disaggregation, supported by a validated resource
accounting model and a causal throughput/SLO evaluation. Bug repair alone is not
a conference contribution.

## Limits

No optimized production path or speedup has been implemented in this investigation.
The diagnostic run has logging/sampling overhead and is not a performance A/B.
No causal fraction of TTFT or guaranteed speedup is assigned to retirement lag.
Original PD audit changes based on selected bandwidth quantiles are not direct
transport traces and do not prove absence of TCP fallback on every transfer;
the present PAP diagnosis does not rely on that earlier claim.
