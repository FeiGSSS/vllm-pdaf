# PAP research diagnosis: evidence before novelty

Status: **research framing blocked, not a completed novel-method proposal**.
Source baseline `306b75a894`, one 8x48GB L20 PCIe node. The initial diagnosis
did not modify production source; the subsequently selected correctness design
is being implemented and validated in the main working tree.

## 1. What this investigation established

### A. Existing success audits did not prove correct KV ownership

Source inspection, a real-scheduler CPU probe and independent raw-trace audit
converged on a concrete failure chain:

1. Gateway requests a Decode tail reservation while Prefill computes one
   output token. The PAP adapter knows the requested reservation.
2. Running and waiting allocations do not use that adapter's reservation.
3. The publisher exports a slice of a padded worker block table through
   `prefix_length + declared_decode_capacity`, rather than authoritative
   scheduler-owned block IDs.
4. Tail padding/stale IDs can therefore become Decode write destinations.
   Counting vector length does not validate capacity or unique ownership.

Independent retained witness: global step1983, PA6, local epoch1840, one
request, context30003 tokens,1876 referenced active block positions but only
1863 unique physical IDs. A single request excludes cross-request shared-prefix
reuse as an explanation. In the joined512-step window,373/466 single-request
PA-step cells violate that invariant. The trace does not retain actual IDs,
so identifying a particular corrupted physical page comes from separate source
and probe evidence, not inference from these counts.

Two linked lifecycle defects also emerged: the lease initially exported for an
early Prefill chunk does not grow with later chunk manifests; and Prefill
preemption can free blocks without revoking those external mappings. Merely
adding Decode headroom allocation is therefore not a complete repair.

Consequences: old timing numbers remain observations of executions, not evidence
of correct-inference performance. In particular, aliased pages can masquerade
as reusable physical KV and contaminate PAT selection and reuse accounting.
Do not claim that all historical runs are proven numerically wrong: the raw
witness establishes aliasing in the inspected trace, and shared source exposes
the other runs to the same defect. A complete blast-radius audit remains open.

See [source audit](source_audit.md), [raw evidence audit](evidence_audit.md) and
`audit_saved_evidence.py` (intentionally exits2 on the reproduced violation).

### B. Projection fragmentation has a newly measured cost

The independent dense-backbone probe does not allocate request KV. It uses
36 distinct FP16 weight sets, existing GPU operators, Graph replay and eager
comparison, on an otherwise idle L20.

| Total request rows | Single batch ms/layer | Two halves' combined service ms/layer |
| --- | ---: | ---: |
| 8 | 0.571 | 1.138 |
| 16 | 0.575 | 1.143 |
| 32 | 0.625 | 1.151 |
| 64 | 0.655 | 1.250 |
| 128 | 0.705 | 1.311 |

The second column is not whole-layer latency: no Attention, communication,
QK norm/RoPE or vocabulary head is timed. The third column is combined GPU
service demand, not a predicted overlapped makespan. Within those boundaries,
the result confirms that reducing row count does not proportionally reduce
Projection work. For a fixed resident batch, microbatching must recover enough
overlap to pay for that extra work. It does not manufacture additional KV
capacity or automatically increase throughput.

See [complete probe definition and exclusions](projection_backbone.md), raw
`projection_backbone.json`, and replayable `probe_projection.py`.

### C. Attractive broad mechanisms are already occupied

| Proposed headline | Primary counterexample |
| --- | --- |
| KV-local Attention plus separate dense/FFN execution | Lamina; CrossPool |
| GPU-resident dynamic inference scheduling | MPK; CrossPool |
| Independent batch/layer cursors without a global batch barrier | CrossPool |
| Same-layer FFN starts on an efficient ready subset | Tarragon |
| Adaptive Prefill/Decode SM coexecution | POD-Attention; NanoFlow |
| Prefix-homogeneous batching versus batch-size efficiency | Feather |
| Distributed deduplicated prefix segments and remote partial Attention | TokenLake |
| Gather queries per prefix-tree node and exact tree reduction | CoDec |
| Joint nonadditive shared-prefix capacity and request selection | TOPAS |
| Cache demand/lifetime and memory-time-aware scheduling | PRISM; LAMPS |

These papers solve different scopes; their names are not interchangeable.
The method sections and limitations are in [the core collision audit](literature/mechanism_audit.md)
and [the targeted collision audit](literature/targeted_collision.md). Neither
the absence of a combination from one paper nor a renamed composition establishes
novelty. In particular, a distributed-prefix-tree proposal must beat a serious
TokenLake+CoDec baseline, not an invented unshared KV baseline.

## 2. Why the present evidence cannot support a paper claim

A valid architecture argument requires the chain:

```text
correct execution + audited resources
  -> repeatable loss under a specified workload
  -> a structural restriction causing that loss
  -> a mechanism not subsumed by the closest prior design
  -> matched-SLO goodput improvement with negative controls
```

The current chain breaks at correctness and at the specific unsolved mechanism.
The raw trace's timing subtraction also cannot turn its Projection interval
into pure computation: the exact sampled cycle is48.997ms, whereas the old
additive proxy overcounts1.594ms. These values are retained as a diagnostic
lesson, not promoted into a new performance model.

The idea-spark workflow completed real multi-source grounding and primary
full-text reading, then returned `do_not_generate` at bottleneck diagnosis.
This is **not** because the user's question is vague. It is because the inspected
mechanisms have close prior coverage and there is no validated residual
correct-inference loss to carry a new contribution. The terminal artifact is
[here](ideaspark_run/pap-inference-architecture/do_not_generate.md).

## 3. Chosen repair boundary

The isolated partial repair at `/tmp/pap-research-reservation-fix-20260905`
connects Decode reservation to allocation, propagates authoritative block IDs,
rejects within-request aliasing and extends chunk leases. It must **not** be
treated as complete while Prefill preemption can invalidate external mappings.

Two defensible lifetime designs have different semantics:

1. **Full reservation:** allocate the full prompt plus effective Decode capacity
   before first Prefill execution, and do not preempt it after external publication.
   This is simpler but changes memory admission and can increase queueing.
2. **Revocable publication:** retain chunked Prefill preemption, but revoke an
   old mapping generation, stop new readers, acknowledge outstanding consumers,
   then free blocks. Resumption publishes a new generation with new ownership.
   This preserves flexibility but requires a distributed reset/ack protocol.

The user selected a third, more precise ownership boundary: Prefill allocates
only prompt KV; Attention requests Decode growth from the Prefill-owned vLLM
allocator; and Prefill preemption revokes an unclaimed Attention generation
before recycling its blocks. The implementation is in the main working tree and
has passed its bounded structural E2E validation. The superseded isolated
worktree patch is deliberately excluded from the committed artifacts and must
not be applied on top of the selected implementation.

### Repair validation checkpoint

The real vLLM manager test proves that a 64-token prompt initially owns four
16-token blocks even when a 64-token Decode limit is declared. After Prefill is
retained for remote Decode, an Attention allocation request for token 65 grows
ownership to eight blocks (128 tokens) through the same manager, without
advancing `num_computed_tokens`. A real scheduler pressure test observes the
victim's blocks still owned when revocation is called; a failed revoke leaves
the request running and owned, while an acknowledged revoke permits preemption
and recycling. Unit contracts additionally reject aliases, non-monotonic block
changes, stale generations and late revoked publications.

The bounded `coding-half` E2E run at
`../../e2e/PAP-20260905-REFACTOR-VALIDATION/runs/20260905_231410_3633861/coding-half`
completed 180/180 requests. Client lengths predict 404 Decode growth crossings;
Attention reports exactly 404 requests and installs, adding 5,254 blocks with
zero topology mismatches and zero reported allocation failures. The workload
did not exercise Prefill preemption, so no stronger E2E claim is made for that
branch. The final publisher-side exact-lease atomicity fix followed this run;
it changes only the revoke race and is validated by the final PAP suite and a
late-manifest regression test.

The 404 external token intervals at predicted growth boundaries average
56.866 ms, versus 54.167 ms for 83,487 ordinary intervals. Since crossings are
only 0.482% of request-local intervals, the original audit incorrectly weighted
their association as a system-wide TBT contribution. One growth blocks the
whole decode batch at the Projection barrier, so that calculation cannot answer
the question. The new step-level trace below supersedes that inference. The
older run remains neither a correct-inference baseline nor an identical realized
workload: 100/180 paired input lengths and 79/180 cache-read lengths differ.

The remaining correctness evidence boundary is explicit: structural ownership,
the Attention kernel reference test, completion and output length have been
validated, but an independent end-to-end logit or generated-text equivalence
comparison has not. Under total non-evictable KV saturation, allocation also
currently fails closed before accepting QKV; production-grade admission and
backpressure for that condition remain future work.

### Step-level performance diagnosis after the repair

The final-source trace at
`../../e2e/PAP-20260905-REFACTOR-VALIDATION/runs/20260906_002049_3682642/coding-half-trace`
passed its 180-request execution and drain audits and retained steps 1156–1667.
All eight raw hashes verify. Client lengths predict 405 capacity requests and
Attention records exactly 405 installs. The 512-step window contains no alias
among its 288 single-request PA-step cells.

Within the 511 exact adjacent-boundary cycles, 49 steps grow at least one
request lease. Their mean cadence is 118.335 ms versus 44.564 ms for 462
ordinary steps. Dense-side time is 27.260/27.185 ms and max-PA Attention-kernel
work is 14.929/14.701 ms, but dispatch-to-gather wait is 91.075/17.379 ms.
The growth population contributes 7.074 ms over the ordinary-step mean. The
worst 773.894 ms cycle contains a 728.957 ms layer-0 PA path without an
Attention-kernel spike.

Therefore synchronous on-demand allocation, not additional Attention compute,
is the measured source of most of the repaired path's TBT increase. The capacity
check runs before Attention releases the layer-0 QKV control slot, placing its
HTTP/Prefill-control round trip inside the global barrier. Source inspection
shows three possible wait locations below that boundary: the API dispatcher
FIFO shared with Decode commits, waiting for Prefill EngineCore to reach an
input-processing boundary, and allocator execution. Their individual shares
are not present in this trace and remain unverified pending adjacent boundary
instrumentation.

### Low-watermark repair result

The selected repair asynchronously grows a request when its writable headroom
falls below 256 tokens. It permits one in-flight request, installs returned
ownership only at an Attention preflight boundary, and waits only if Decode
actually catches that request. Prefill readiness starts the initial prefetch.

The matched tracing replay at
`../../e2e/PAP-20260905-REFACTOR-VALIDATION/runs/20260906_005320_3718392/coding-half-trace`
completed all 180 turns. All 405 allocations were initiated as prefetches; only
four reached a boundary before completion. There were zero allocation failures,
pending requests and topology mismatches at drain. Client TBT improved from
54.117 to 48.540 ms, TTFT from 1.745 to 1.605 seconds, and mean end-to-end
latency from 26.758 to 24.004 seconds. Growth-step mean/P99/max cadence fell
from 118.335/691.910/773.894 ms to 56.700/107.500/146.985 ms.

The source dataset and AIPerf input artifacts match, but generated multi-turn
prompts drift slightly: total input differs by 0.047% and cache-read tokens by
0.76%. Endpoint means are therefore not labelled a strict payload-identical
A/B. The allocation wait counters and step-local removal of the growth long
tail establish the mechanism independently of that small workload drift.

## 4. Smallest useful continuation after that choice

1. Validate two requests with distinct generated KV, crossing several16-token
   pages; check actual allocator ownership, no intra-request aliases, expected
   cross-request sharing, lease extension, cancellation and preemption/resume.
   Compare actual Attention outputs/logits with an independent reference;
   output length alone is not sufficient.
2. Rebuild one untraced performance point plus one bounded valid trace. Do not
   rerun the old all-dataset matrix. Separate a low-pressure point from a
   capacity-pressure point; freeze request input tokens or pair identical
   requests so generated histories do not silently change compared work.
3. Capture adjacent `(request,step,layer)` boundaries, authoritative physical
   KV, Projection row count and prefill queue state. Measure release/byte-time,
   not just logical context sums or GPU utilization.
4. Only then select a residual and run an oracle upper-bound counterfactual.
   Keep total work, physical cache sharing, kernels, devices and SLO identical.
   Charge extra Projection invocations and communication. If even the charged
   oracle gives negligible goodput benefit, abandon that axis before developing
   an elaborate scheduler.
5. A full paper campaign must include fair DP, PD and tuned PAP baselines,
   relevant closest-mechanism baselines, no-prefix/shared-prefix/mixed lengths,
   bursts and capacity pressure, and a negative control that removes the
   proposed mechanism's load-bearing condition. Target correctness and
   SLO-qualified output throughput, not an arbitrary utilization percentage.

This is a concrete route to a defensible argument, not a promised accepted
paper, completed optimization or measured speedup of a new method.
