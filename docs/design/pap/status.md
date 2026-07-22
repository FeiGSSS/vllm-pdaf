---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260722-AIPERF-CONVERGENCE
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
last_validated_commit: aafcfb1ea1800b4dc0bcd1ea8299d9984e9624aa
---

# Current PAP development status

Snapshot date: 2026-07-22.

PAP has completed its runtime refactor and the first capacity-oriented
performance milestone. The source milestone at `aafcfb1ea` has one accepted
runtime architecture, a source-audited long-context testbed, and an optional
piecewise CUDA Graph execution mode. Historical experimental algorithms are
not selectable branches in the current runtime.

## Current support boundary

| Capability | Source state | Current evidence |
| --- | --- | --- |
| Qwen3-8B, same-host PAP | Main path | Four-GPU AIPerf matrix |
| Same-host xPAyP | Implemented | Controlled correctness smoke; not a performance gate |
| Cross-host xPAyP over NIXL | Preserved | Contract coverage only; no fresh E2E claim |
| Prefill-owned unified KV | Main path | AIPerf runtime and lifecycle audits |
| Triton split-4 paged decode | Main Attention kernel | AIPerf eager/Graph baselines |
| Piecewise CUDA Graph | Optional development mode | Six valid PAP/PD four-GPU points |
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
4. Attention appends K/V into the Prefill-owned blocks, executes one
   step-level Triton paged-decode plan across the model layers, and returns the
   Attention output.
5. Asynchronous sampled-token delivery joins KV completion before decode
   commit, ACK, lease release, and final session drain.

PAP-to-vLLM integration is owned by `vllm/pap/integration/`; model interception
is owned by `vllm/pap/model/`. Runtime packages do not import benchmark tooling
or historical experiment implementations.

## Validation lanes

The only active runtime lane is the **four-GPU AIPerf testbed**: 32
conversations, ten turns, randomized 8K initial input, roughly 512 appended
input tokens, randomized 16-64-token output, think/tool delays, and
conversation concurrency. PAP is 3PA1P; PD compares one-way 1P3D, 2P2D, and
3P1D.

The former P17 1PA1P client, runner, and release gate are retired. Its profile
and results remain archived solely for historical manifest validation.

The capacity lane deliberately avoids artificial scheduler limits:

| Role | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| PAP PA / PD Prefill | 64 | 16384 |
| PAP Projection / PD Decode | 64 | 64 |

`max_num_partial_prefills` stays at its vLLM default of 1 and
`max_model_len=20000`. PAP Prefill and PD use
`gpu_memory_utilization=0.90`, while PAP Projection remains at `0.76` because
it owns no prompt KV. PAP must additionally report physical PA-GPU headroom
because Attention is colocated outside the Prefill executor's budget. These
settings are justified from scheduler/model-runner source in the
[AIPerf methodology](../../../benchmarks/pap/aiperf/README.md).

Existing performance reports used PAP Prefill `0.76`. They are retained as
historical evidence; the `0.90` baseline is not a milestone until PAP startup,
PA-GPU headroom, and a matched PAP/PD AIPerf comparison are recorded.

## Current performance milestone

The source-audited eager scan found PAP best-goodput advantages of +10.4% and
+35.3% under the strict and standard SLOs, with a -1.3% relaxed-SLO gap. With
both PAP and PD using piecewise CUDA Graph, the measured PAP advantages were
+34.2% strict, +34.2% standard, and +7.2% relaxed.

These CUDA Graph results are one clean development repetition. They establish
correctness and a useful baseline, but mixed per-point latency changes mean
the exact percentages are not a release claim. See the
[eager capacity report](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md)
and the
[piecewise CUDA Graph report](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH/report.md).

## Remaining work

1. Run three AIPerf repetitions only when promoting a four-GPU result to a
   release-level performance claim.
2. Profile the mixed Graph deltas, especially PAP C20 and PD 2P2D C16, before
   expanding capture shapes or attempting broader graph coverage.
3. Keep same-host xPAyP and cross-host NIXL source-compatible, but do not claim
   performance or fresh E2E support until those lanes are explicitly rerun.
4. Continue owner-driven splits in unified-KV and transport internals only
   when they simplify a concrete feature; do not restore retired experiment
   selectors or per-layer scheduling paths.

Dated milestone documents and legacy experiment reports remain read-only
development evidence. Skill-generated execution plans were removed after
consolidation and remain available through Git history. This page, the
architecture/runtime documents, and the experiment index define the current
state.
