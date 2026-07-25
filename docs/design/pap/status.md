---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260725-8GPU-CAPACITY-SCAN
  - PAP-20260725-8GPU-CAPACITY-PILOT
  - PAP-20260724-STEP-OVERLAP
  - PAP-20260724-PROJECTION-SCHEDULER-OVERLAP
  - PAP-20260724-SINGLE-PROJECTION-BATCH
  - PAP-20260724-BATCH-SCALING-MICRO
  - PAP-20260722-AIPERF-PROJECTION-AUTO
  - PAP-20260722-AIPERF-PA090-EAGER
  - PAP-20260722-AIPERF-CONVERGENCE
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
last_validated_commit: 152f64445cd12ffda91ec1d46330c563a36ab475
---

# Current PAP development status

Snapshot date: 2026-07-25.

PAP has completed its runtime refactor and the first capacity-oriented
performance milestone. The source milestone at `cb6fe3500` has one accepted
runtime architecture, a source-audited long-context testbed, and an optional
piecewise CUDA Graph execution mode. Historical experimental algorithms are
not selectable branches in the current runtime.

## Current support boundary

| Capability | Source state | Current evidence |
| --- | --- | --- |
| Qwen3-8B, same-host PAP | Main path | Eight-GPU 7PA1P and 6PA2P completed 640/640 requests |
| Same-host xPAyP | Implemented | Multi-Projection 6PA2P C32 E2E complete |
| Cross-host xPAyP over NIXL | Preserved | Contract coverage only; no fresh E2E claim |
| Prefill-owned unified KV | Main path | AIPerf runtime and lifecycle audits |
| Triton split-4 paged decode | Main Attention kernel | AIPerf eager/Graph baselines |
| Piecewise CUDA Graph | Optional development mode | Nine valid PAP/PD four-GPU points |
| Full-model CUDA Graph | Unsupported | Host transport and KV publication cannot be replayed safely |

Eager execution remains the default. Piecewise mode captures graph-safe model
regions and leaves remote Attention, OFFLOAD_EXEC transport, and Prefill KV
publication outside the graph. Capture shapes select replay or eager fallback;
they do not cap admission, sequence count, KV capacity, or batch size.

## Runtime architecture

The current request path is:

1. The Gateway assigns the conversation to a PA owner and independently
   selects a Projection endpoint.
2. Prefill processes the prompt, owns all paged KV blocks, and publishes one
   sealed generation-bound manifest to its colocated Attention service.
3. Projection runs the KV-unaware decode path and sends current-step Q/K/V.
   PAP keeps the vLLM scheduler batch intact and may fan same-step route
   shards from that batch out to several PA groups. Before layer-0 QKV,
   Projection publishes one step descriptor so Attention can prepare its
   context, slot plan, metadata, and workspace asynchronously.
4. Attention appends K/V into the Prefill-owned blocks, executes one
   step-level Triton paged-decode plan across the model layers, and returns the
   Attention output.
5. Asynchronous sampled-token delivery joins KV completion before decode
   commit, ACK, lease release, and final session drain.

PAP-to-vLLM integration is owned by `vllm/pap/integration/`; model interception
is owned by `vllm/pap/model/`. Runtime packages do not import benchmark tooling
or historical experiment implementations.

## Validation lanes

The active runner is the **eight-GPU AIPerf testbed**: 128 conversations, five
turns, randomized 8K initial input, a broad append distribution sampled near
1.4K tokens, randomized 16-64-token output, think/tool delays, and conversation
concurrency. It compares PAP 7PA1P/6PA2P, one-way PD 4P4D/6P2D, and
an eight-replica fused vLLM pool. The initial C32 pilot is complete; the
compact C16/24/32/48 scan and Strict-boundary refinement are also complete.
The latest normalized performance milestone remains the four-GPU result until
the selected eight-GPU boundary points have three repetitions.

The former P17 1PA1P client, runner, and release gate are retired. Its profile
and results remain archived solely for historical manifest validation.

The capacity lane deliberately avoids artificial scheduler limits:

| Role | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| PAP PA / PD Prefill / DP | 256 | 32768 |
| PAP Projection / PD Decode | 256 | 256 |

`max_num_partial_prefills` stays at its vLLM default of 1 and
`max_model_len=32768`. PAP Prefill and every PD/DP executor use
`gpu_memory_utilization=0.90`. Projection is sized independently at launch:
the checkpoint weight bytes per TP rank are multiplied by 1.20 and divided by
the smallest selected Projection GPU's total memory, rounding utilization up
to four decimals. Qwen3-8B FP16 at TP1 on an L20 resolves to `0.4070`.

Projection owns no local request KV. Its vLLM integration therefore preserves
KV layer/group metadata and one required null block, but plans no physical KV
tensor and performs no ordinary max-context KV-capacity admission check. A
targeted 1PA1P E2E at this automatic budget completed 32 four-turn sessions
(128/128 requests), passed all PAP audits, and reported 0.0% Projection KV
usage throughout. PAP must still report physical PA-GPU headroom because
Attention is colocated outside the Prefill executor's budget. The complete
parameter rationale is in the
[AIPerf methodology](../../../benchmarks/pap/aiperf/README.md).

The automatic-memory eager and Graph baselines are now recorded. Every tested
PAP point started without OOM or Graph-capture failure. Explicit legacy
Projection budgets are not current configuration inputs; historical details
remain isolated in archived records and Git history.

## Current performance milestone

The current eager scan finds PAP best-goodput advantages of +46.0%, +85.5%,
and +95.2% under the strict, standard, and relaxed SLOs. Its concurrency
envelope is C12/C20/C32, versus best PD C8/C10/C16. Relative to the preceding
eager milestone, PAP goodput changes by -1.3%, +1.6%, and -1.4%; automatic
Projection sizing therefore has no measured eager regression beyond 2%.

With piecewise CUDA Graph enabled on both architectures, PAP leads by +13.3%,
+75.0%, and +118.8%. Its Graph envelope is C8/C20/C32 versus PD C8/C8/C16.
Graph is not a uniform speedup: PAP C12 narrowly loses Strict compliance while
higher-concurrency throughput is roughly flat or slightly improved. Eager
therefore remains the default and Graph remains an optional supported mode.

All included runs are complete and correct. PD continues to show NIXL
transfer variance: one 2P2D C10 Graph attempt had a persistent single-lane
slowdown and is retained only as a diagnostic; its targeted repeat and
independent PD topologies are used for comparison. This is controlled
single-repetition development evidence, not a release claim. See the
[current milestone](../../../benchmarks/pap/experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).

The subsequent C12 no-async diagnostic completed 640/640 requests, but tested
an overly strict interpretation of the single-batch requirement. Disabling
vLLM async scheduling increased ITL p95 by 10.16% and is rejected: that queue
only overlaps scheduler/metadata/output CPU work with a complete model
forward; it does not interleave PAP microbatches across model layers. The
component probe independently attributes 93.2% of the B4-to-B8 per-layer
growth to paged Attention.

The corrected async C12 regression at `29cc69029` completed 320/320 requests
with all audits passing. It reports 35.20 ms mean ITL and 2.529 requests/s:
within 1.25% and 0.22% of the earlier async C12 baseline, respectively, while
mean ITL is 7.96% lower than the rejected no-async mean.

The step/control-overlap regression at `31e0b3882` also completed 320/320
requests. Every prepared context was reused by all 36 layers. Relative to
`29cc69029`, mean ITL improves by 0.37%, ITL p95 by 5.19%, and request
throughput by 0.50%. The initial-burst TTFT p95 movement is retained as
single-run tail variance rather than an improvement claim. See the
[current control-overlap result](../../../benchmarks/pap/experiments/PAP-20260724-STEP-OVERLAP/report.md).

The initial eight-GPU C32 pilot completed all 640 requests for PAP 6PA2P,
PAP 7PA1P, PD 4P4D, PD 6P2D, and the fused eight-replica pool. PAP 6PA2P is
the only tested configuration that passes both Standard and Relaxed at C32.
It delivers 4.224 and 4.381 compliant requests/s, respectively. PD 4P4D
passes Relaxed at 3.649 requests/s; the other C32 points require a lower
concurrency to establish SLO capacity.

7PA1P is not a liveness failure: it completes 640/640 requests, improves raw
throughput by 14.1% and mean TTFT by 45.5% relative to 6PA2P, but only
599/640 requests meet Relaxed. A matched trace attributes the tail to its
single seven-PA fan-in barrier: Projection join-wait p99 is 3.654 ms per layer,
versus 0.625 and 0.663 ms for the two independent 6PA2P Projection domains.
QKV preparation and P2P-copy costs remain essentially unchanged. See the
[eight-GPU pilot](../../../benchmarks/pap/experiments/PAP-20260725-8GPU-CAPACITY-PILOT/report.md).

The completed compact scan finds the following best compliant request
goodputs:

| SLO | PAP | PD | Fused DP | PAP vs PD | PAP vs DP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | 3.347, 7PA1P C16 | 2.442, 6P2D C12 | 2.294, C8 | +37.0% | +45.9% |
| Standard | 4.339, 6PA2P C32 | 3.596, 6P2D C24 | 4.006, C16 | +20.7% | +8.3% |
| Relaxed | 4.894, 7PA1P C32 | 3.785, 6P2D C24 | 5.181, C24 | +29.3% | -5.5% |

The 7PA1P C32 Relaxed result is repeat-unstable. Four observations straddle
the 95% gate, so 6PA2P C32 remains the latency-stable PAP baseline. Its
Relaxed goodput is 4.443 requests/s: 17.4% above PD and 14.2% below fused DP.
See the
[full eight-GPU scan](../../../benchmarks/pap/experiments/PAP-20260725-8GPU-CAPACITY-SCAN/report.md).

## Remaining work

1. Run three repetitions only for the selected eight-GPU boundary points:
   PAP 7PA1P C16, PAP 6PA2P C32, PD 6P2D C12/C24, and fused DP C8/C16/C24.
2. Diagnose PD NIXL transfer variance, including the observed 2P2D single-lane
   stalls, before treating exact PD tail latency as stable.
3. Keep piecewise CUDA Graph optional until repeated evidence shows a
   consistent latency or goodput benefit worth changing the eager default.
4. Keep cross-host NIXL source-compatible without a fresh performance claim.
   Same-host 7PA1P and 6PA2P now have fresh C32 E2E evidence; other xPAyP
   shapes remain preserved-unverified.
5. Profile the independent multi-PA output streams before attributing a
   specific E2E gain to them, and continue only metadata/control overlap that
   preserves the complete scheduler-batch and layer order.
6. Continue owner-driven splits in unified-KV and transport internals only
   when they simplify a concrete feature; do not restore retired experiment
   selectors or per-layer scheduling paths.
7. Consider chunked asynchronous growth of sealed Prefill-owned KV
   reservations. The current path reserves and leases the request's complete
   decode capacity before handoff, using `max_completion_tokens` or
   `max_tokens` with an environment fallback, and fails closed beyond the
   published writable range. A future design may request another block chunk
   at a low-water mark and install a generation-versioned block-table and
   lease extension. It must remain off the per-token hot path. This is a
   deferred capacity-efficiency TODO, not current milestone scope.

Dated milestone documents and legacy experiment reports remain read-only
development evidence. Skill-generated execution plans were removed after
consolidation and remain available through Git history. This page, the
architecture/runtime documents, and the experiment index define the current
state.
