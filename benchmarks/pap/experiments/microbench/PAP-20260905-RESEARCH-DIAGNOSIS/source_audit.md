# PAP structural source audit

Audit base: `306b75a894`; 2026-09-05. Runtime source was not modified. No GPU
job, model forward, service restart, or workload replay was performed. Read
`AGENTS.md`, `vllm/pap/README.md`, idea-spark and model-pr-history runbooks; this
is source input to the parent research process, not a completed novelty audit.

## Units and expected timelines

- **Request:** one user turn, including Prefill, repeated Decode and cleanup.
- **Decode step:** one complete model forward for a scheduled set of request
  rows; this is not a complete request. A **PA route cohort** is the subset of
  those rows resident at one PA. A **layer operation** is one cohort's Attention
  for one layer. A **queued/submitted batch** is not necessarily executing.
- **TTFT:** request arrival to first externally delivered output. **TBT:**
  intervals between external token outputs for one request. **Step residence:**
  scheduling/submission through result processing. **Throughput:** completed
  work per interval. These are not interchangeable.

Request timeline, source-established:

```text
tokenize → Dynamo selects PA → fixed P selected → register A session
 → Prefill (possibly multiple chunks/manifests) → complete prompt readiness
 → P admission → repeated decode steps → output stream → request cleanup
```

Evidence: `gateway/request_pipeline.py:155`, `:169`, `:174`, `:215`, `:246`,
`:303`, `:321`, `:386`, `:400`; cleanup `:443`. Prefill uses `max_tokens=1`
(`gateway/payloads.py:42`); Projection starts from the last prompt token
(`integration/scheduler.py:147`), not from a newly transferred full KV cache.

For step s, when an A graph is already submitted, it may be waiting for P's
next layer QKV; this wait is not Attention kernel execution:

```text
P CPU: publish all PA step plans → prepare route tensors → replay P graph
P GPU layer ℓ: QKV/local ops → dispatch cohorts → gather ALL active PA results
              → output projection/MLP → next layer
A GPU layer ℓ: wait QKV(s,ℓ) → optional KV append → attention → signal result
A CPU: whole graph complete → commit layer lengths/KV readiness → next plan
```

Evidence: `model/step_graph.py:206-288`; `model/projection.py:199-227`;
`attention/step_graph.py:454-482`, `:511-543`, `:187-208`, `:552-572`;
`attention/step_graph.py:35-48`. A's `graph_replay_submit` host field includes
`stream.synchronize()` (`:201-204`), so its name does not establish pure submit
overhead. P's cached graph links its stream to the caller in both directions
(`model/step_graph.py:328-334`). The allowed two in-flight batches
(`integration/runner.py:103-106`) do not prove two independent GPU pipelines.

## Source-established structural constraints

1. **Per-layer all-PA completion coupling.** Dispatch and gather run on the
   same model stream (`model/projection.py:210-227`). Native gather has one
   block per PA, each waits for that PA's epoch/layer signal before copying
   (`transport/nvshmem/device_bridge.cu:202-245`). If `r_i(s,ℓ)` is each route's
   output-ready time, gather completion cannot precede `max_i r_i(s,ℓ)`.
   This is a dependency fact, not a measurement of wasted GPU time. Summing
   route readiness differences would count row waiting, not P idle time.

2. **Cache placement also determines compute ownership.** Dynamo selects PA
   every turn, then `gateway/topology.py:153-164` maps PA to fixed P. A explicitly
   rejects a second Projection peer (`attention/peers.py:118-123`). Thus a PA
   cache hit cannot independently choose another P under the present protocol.
   P retains only one structural KV scratch block
   (`integration/kv_cache.py:15-29`), which makes this binding structural rather
   than inherently required by Projection's own KV residency.

3. **KV is a sealed request layout, not an independently scheduled storage
   service.** Prefill publishes layer catalogs and a final-layer IPC-ready event
   (`model/prefill.py:268-298`). A maps all layer states from one manifest
   (`kv/session_registry.py:362-390`). Before Decode claims it, subsequent
   Prefill chunks may advance the manifest; after claim, changing prefix is
   rejected (`:328-345`). Attention appends into this Prefill-owned cache
   (`attention/step_graph.py:507-520`). Leases delay connector completion
   (`kv_connector.py:143-161`, `:217-232`). Accepted token and materialized KV
   readiness join by request/sequence (`kv/decode_commit.py:46-129`); the joined
   positions are coalesced on final flush (`:134-165`, `:187-214`). Do not call
   this a synchronous Prefill HTTP commit on every token.

4. **Resource split is static.** Config fixes Prefill 80 visible SMs and
   Attention 12 (`config.py:568-575`). No utilization or HBM-isolation claim
   follows from those SM counts. CPU graph/metadata improvements, PAT physical
   prefix reuse, and fixed microbatch overlap do not remove constraints 1–2.

## Falsifiable mechanism candidates (inferred, novelty unassessed)

### A. Readiness-driven layer cohorts with temporal re-batching

Replace the whole-batch Attention barrier with per-cohort continuations. Keep
request hidden states in indexed slots; collect ready rows at the same layer
into useful Projection GEMMs, then re-dispatch their next-layer QKV. Maintain
per-request generation/layer ordering, deadlines and credits; graph segments
capture compute, not the all-PA wait. This is materially different from dividing
one batch into two permanently fixed microbatches: ready cohorts can progress
and rejoin other cohorts without the same straggler set at every layer.

Test one variable: PA output-readiness skew, at fixed total Attention work,
Projection kernels, active requests and device placement. Compare the existing
whole-step graph, fixed-micro2, and continuation/re-batching. Prediction:
decreasing unnecessary cohort waiting improves SLO-constrained throughput only
if recovered overlap exceeds GEMM fragmentation, state gather/scatter and
scheduling cost. Negative control: equalize PA ready times; gain should mostly
disappear. Kill if total service throughput does not improve at matched TBT, or
if identical outputs require restoring the global barrier. Baseline must include
small independent cohort graphs: if those suffice, the proposed scheduler is
incremental. Need exact adjacent `(request,step,layer,PA)` timing evidence.

### B. Separate cache-home ownership from Projection execution ownership

Preserve KV at its selected PA but expose epoch-safe multi-Projection request
leases and per-request result mailboxes. Route Projection compute by available
compute/deadline independently of PA prefix locality. A arbitrates request
cohorts from multiple P workers, with per-request single-writer ownership; a
handover transfers hidden/token state and sequence ownership, not the long KV
prefix. This targets idle P capacity stranded by cache-hot PA ownership, not
faster kernel math. It requires replacing the single-peer and single-owner
contracts, not merely changing gateway selection.

Test one variable: skew of cache-home demand across PA→P ownership partitions,
holding total prefix reuse, total attention bytes, P compute and link placement
fixed. Compare fixed ownership, hash/random P, and queue-aware reassignment.
Prediction: matched-SLO throughput improves only when P skew is the bottleneck
and A/link load remains feasible. Balanced demand is the negative control.
Kill if A remains the bottleneck, communication/fairness costs erase gain, or
naive least-loaded P performs equally well. Novelty requires literature checking
against attention/FFN disaggregation, stateful scheduling and elastic pooling.

## Separate correctness concern: advertised decode capacity vs owned blocks

Do not treat the following as a performance mechanism or established cause of
any historical latency. Preserve it before making new capacity/accuracy claims.

**Measured CPU behavior:** `probe_decode_reservation.py` uses the existing
AsyncScheduler, PAPPrefillConnector and local Qwen3 config. Prompt64,
`max_tokens=1`, PAP decode capacity64, no speculation, block16: adapter reports64,
but final Prefill allocation is four blocks/64 token slots, not eight/128.
Chunked32 evolves two→four blocks; final state still lacks Decode capacity.
Raw output: `reservation_probe.log`. The scheduler's V2 branch was explicitly
selected in the existing test helper. The test uses CpuPlatform because no
accelerator is visible in the sandbox; configuration logs a model-runner
fallback. No model runner was constructed/executed. Therefore this is a CPU
scheduler-contract probe, not a replay of effective GPU service configuration.

**Source chain:** `integration/scheduler.py:166-173` returns reservation metadata
but no production call consumes it. Running/waiting scheduler allocation uses
only speculative lookahead (`vllm/v1/core/sched/scheduler.py:567-572`, `:927-965`);
KV allocation uses computed+new+lookahead
(`vllm/v1/core/kv_cache_manager.py:436-439`). In contrast, Prefill slices worker
metadata through prefix+advertised capacity (`model/prefill.py:319-324`) using a
helper that checks neither allocated count nor IDs (`:41-60`). V2's forward
table starts zero-filled (`vllm/v1/worker/gpu/block_table.py:69-70`), copies only
the allocated count (`:237-246`), and returns full-width rows (`:159`).

**Executed model of the boundary:** construct a fresh V2-style padded CPU row
from the actual scheduler IDs, then invoke the unchanged publication helper
and manifest class. It exports `[1,2,3,4,0,0,0,0]`; manifest validation accepts
it because its guard checks length (`protocol/descriptors.py:342-346`). This
does not execute the Triton gather kernel or CUDA IPC. The source makes a fresh
row's zero tail—and a reused row's possible stale tail—a concrete concern,
rather than evidence of allocated Decode headroom.

**Historical reconciliation:** the successful suite's archived scheduler at
`runs/20260905_135804_3149938/provenance/source.tar.gz` already has the same absent
reservation call; this is not automatically a new 306b75 regression. Its
short-context A log (`short-context/service_logs/attention_3_0.log:46-51`) shows
manifest growth prefix2048→8210, blocks129→515, before Decode. Such count-only
logs are compatible with either valid allocated headroom or exported padding.
Long successful output counts cannot establish KV-address ownership or output
correctness. Actual historical physical IDs/writes remain unverified here.
Do not invalidate specific historical runs without checking their captured
ownership/accuracy evidence; do not use their success to dismiss this contract
probe. Before request-level replay, reconcile actual scheduler block IDs with
published block IDs/leases at every chunk and add a fail-closed ownership guard
and permanent reservation/lifetime fix if confirmed. Runtime source remains
unchanged in this audit.

### Follow-up: independent trace confirmation and isolated repair

Subsequent independent audit established aliases inside the retained aligned
GPU trace, not merely the constructed CPU row: at global step1983, PA6
local_epoch1840, singleton request `07e4ff81-d46b-42b2-a0c8-ec7334bc515e` has
sequence30003, prefix29781, referenced1876 blocks but only1863 unique blocks;
its lease vector has1908 entries and1863 unique IDs. A singleton excludes
cross-request prefix sharing as the explanation. The evidence agent reports
373/466 aliased singleton cells within the aligned window. Refer to its saved
invariant-audit artifact for raw file paths and complete counting procedure.
This supersedes the earlier statement that actual historical mapping aliases
were unverified; the exact repeated ID (e.g. zero) remains unavailable.

A separate worktree at `/tmp/pap-research-reservation-fix-20260905` now contains
a proposed correctness repair: reserve effective Decode capacity, propagate
scheduler-owned blocks through connector metadata, validate worker publication
against that ownership, extend leases on chunk growth, reject aliased/unleased
manifests, and remove silent capacity truncation. This does not change the main
runtime or establish a performance gain. The first repaired revision passes124
PAP contract/lifecycle tests and pinned Ruff0.14. The complete130-test core
scheduler suite gives129 pass/1 pre-existing fixture failure on both unchanged
source and isolated revision under the same explicit offline CPU harness.

**The repair is not yet accepted as complete:** independent review/probe found
that `_preempt_request` can free blocks after an exported early Prefill chunk
while the lease still retains them. In the CPU probe, allocated blocks1–6
became free and request ownership became empty while the active lease remained
1–6. The minimal safe lifetime policy versus a distributed revoke/reset design
must be resolved before request-level validation or adoption of the repair.
