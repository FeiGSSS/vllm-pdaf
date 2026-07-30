---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260730-MPS-80-12
  - PAP-20260730-RESEARCH-L13
  - PAP-20260729-LONGCTX-O100-CONCURRENCY-SCAN
  - PAP-20260729-RESEARCH-L12
  - PAP-20260729-RESEARCH-L07
  - PAP-20260729-RESEARCH-L01
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
last_validated_commit: 9207a5538
---

# Current PAP development status

Snapshot date: 2026-07-30.

PAP has completed its runtime refactor and the first capacity-oriented
performance milestone. The source milestone at `9207a5538` has one accepted
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
| Low-SM Triton paged decode | Main Attention kernel | 12/20-SM microbench and 7PA1P scan |
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
5. The ModelRunner carries a GPU-frame-local sequence key to the Scheduler.
   Only Scheduler-accepted sampled tokens are delivered asynchronously and
   joined with KV completion before decode commit, ACK, lease release, and
   final session drain.

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

Paper research loops may use prospectively registered workload variants
without redefining the canonical regression testbed. The current long-context
O100 lane uses 60 sessions, three approximately 10K-token turns, randomized
50--200-token outputs, and pure AIPerf concurrency. Its evidence is scoped to
`PAP-20260729-LONGCTX-O100-CONCURRENCY-SCAN` and its accepted 80/12 successor,
`PAP-20260730-MPS-80-12`.

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

## Performance milestones and current research evidence

The July 22 and July 25 scans below remain valid runtime and benchmark
milestones for their recorded code and methodology. They are no longer the
current fair PAP-versus-PD paper comparison: corrected same-node PD transport,
clean repeated boundaries, and workload-controlled research loops L01--L13
supersede their directional performance conclusion.

The July 22 eager scan finds PAP best-goodput advantages of +46.0%, +85.5%,
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
599/640 requests meet Relaxed. The initial trace suggested its single
seven-PA fan-in barrier as the cause because Projection join-wait p99 is
3.654 ms per layer, versus 0.625 and 0.663 ms for the two independent 6PA2P
Projection domains. L01 later rejects fan-in completion skew as the dominant
explanation: the matched median spread delta accounts for only 16.6% of the
trace-mode ITL gap. See the
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

The clean corrected-PD O16 comparison reverses that earlier direction
(`PAP-20260729-RESEARCH-L07`). At its preselected repeated boundaries, PD
leads PAP goodput by 32.2%, 36.9%, and 24.4% under strict, standard, and
relaxed SLOs. PAP leads fused DP only under strict; it loses by 12.1% and
17.2% under standard and relaxed. These are the current short-output paper
baselines.

The earlier large PAP-over-PD margins are not architecture evidence. The old
PD KV-transfer path included TCP-emulated GET at approximately 0.42 GiB/s and
later severe lane instability, rather than a healthy 22--24.5 GiB/s
CUDA-IPC path. Since workload, topology, and capacity search also changed
before L07, the evidence supports “baseline-contaminated,” not the stronger
claim that transport alone explains every percentage point of the reversal.

The topology evidence is a trade-off rather than a universal winner. At an
achieved-throughput mismatch of only 0.59%, 7PA1P C21 lowers mean TTFT by
61.9% and raises mean ITL by 9.33% relative to 6PA2P C32, with
standard/relaxed goodput differences of -0.20%/+0.19%
(`PAP-20260729-RESEARCH-L04`). L01 also falsifies the earlier claim that
fan-in completion skew dominates the 7PA1P ITL loss: the median spread delta
explains only 16.6% of the trace-mode gap.

Near the model context limit, the larger PAP KV pool is not sufficient
(`PAP-20260729-RESEARCH-L12`). Both 6PA2P and 6P2D bracket their Standard
boundary at C12--C16, while PAP C12 Standard goodput is 34.4% below PD. PAP
C16 has lower mean ITL but much higher TTFT, motivating a Prefill-resource
test. The first 20/3 MPS treatment is invalid because lifecycle correctness
fails (`PAP-20260730-RESEARCH-L13`); its metrics are diagnostic only.

The long-context O100 development scan finds one narrower positive region
(`PAP-20260729-LONGCTX-O100-CONCURRENCY-SCAN`). On 60 sessions at C20,
7PA1P passes Standard at 1.611 good req/s while 6P2D fails at 1.337, a 20.5%
difference. At C24 both fail Standard, and PD retains the best passing Relaxed
goodput by 0.8%. This one-repetition result motivates clean confirmation and
causal attribution; it does not establish monotonic PAP scaling or a
KV-capacity-wall mechanism.

That confirmation is now complete on the accepted 80/12 baseline
(`PAP-20260730-MPS-80-12`). The production PAP Triton launch changes from
split4/BLOCK_H16 to split8/BLOCK_H4 when at most 20 SMs are visible, reducing
the measured 12-SM exact-shape latency by 12.3%. A redundant generic NIXL
producer lease is shortened from 30 seconds to one second; PAP's independent
300-second pressure-evictable Attention lease remains the safety owner. This
removes the observed third-turn cache-displacement tail: no Prefill execution
exceeds five seconds across the new C16--C32 scan.

All ten PAP 7PA1P and PD 6P2D points complete 180/180 requests. PAP's best
passing Standard goodput is 2.252 requests/s at C32 versus corrected PD's
1.586 at C20 (+42.0%); best Relaxed goodput is 2.343 versus 1.891 (+23.9%);
raw throughput is 2.343 versus 1.891 (+23.9%). At matched C32, PAP reduces
average TTFT by 18.8% and average ITL by 24.2%. These are controlled
one-repetition results. The initial PD curve with Prefill `max_num_seqs=256`
is configuration-confounded and no longer supplies the comparison claim.
The selected boundary points still require repetition before becoming a paper
or release-level claim.

## Remaining work

1. Repeat the selected long-context boundaries on clean committed code before
   using the +42.0%/+23.9% goodput result as a paper claim.
2. Retain 80/12 plus the low-SM Triton specialization as the PAP baseline;
   re-audit launch geometry before generalizing it beyond Qwen3-8B/L20.
3. Keep corrected NIXL settings and clean tracked provenance mandatory for
   every PAP/PD comparison; do not merge dirty submit-only diagnostics into
   committed baselines.
4. Keep piecewise CUDA Graph optional until repeated evidence shows a
   consistent latency or goodput benefit worth changing the eager default.
5. Keep cross-host NIXL source-compatible without a fresh performance claim.
   Same-host 7PA1P and 6PA2P now have fresh C32 E2E evidence; other xPAyP
   shapes remain preserved-unverified.
6. Profile the independent multi-PA output streams before attributing a
   specific E2E gain to them, and continue only metadata/control overlap that
   preserves the complete scheduler-batch and layer order.
7. Continue owner-driven splits in unified-KV and transport internals only
   when they simplify a concrete feature; do not restore retired experiment
   selectors or per-layer scheduling paths.
8. Consider chunked asynchronous growth of sealed Prefill-owned KV
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
