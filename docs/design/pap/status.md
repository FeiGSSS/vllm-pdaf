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

The matched `0.90` eager baseline is now recorded. Every tested PAP point
started without OOM; each PA obtained 167,264 KV tokens and peak observed KV
usage reached 89.0% at C32. The preceding `0.76` comparisons remain historical
evidence.

## Current performance milestone

The current eager scan found PAP best-goodput advantages of +38.7%, +80.4%,
and +101.5% under the strict, standard, and relaxed SLOs. Its concurrency
envelope is C12/C20/C32, versus the best observed PD C8/C10/C16. All runs were
complete and correct. A targeted PD 3P1D C8 repeat recovered from 0.432 to
1.833 req/s, exposing large NIXL transfer variance; the comparison uses the
better PD repeat rather than claiming advantage from the anomalous run.

This is controlled development evidence, not a three-repetition release
claim. The former piecewise CUDA Graph comparison used PAP Prefill `0.76` and
is archived until Graph is rerun on the `0.90` baseline. See the
[current eager report](../../../benchmarks/pap/experiments/PAP-20260722-AIPERF-PA090-EAGER/report.md).

## Remaining work

1. Run three AIPerf repetitions only when promoting a four-GPU result to a
   release-level performance claim.
2. Diagnose PD 3P1D NIXL transfer variance before treating that topology as a
   stable performance baseline.
3. Rerun piecewise CUDA Graph on the `0.90` memory baseline before making a
   current Graph-performance claim.
4. Keep same-host xPAyP and cross-host NIXL source-compatible, but do not claim
   performance or fresh E2E support until those lanes are explicitly rerun.
5. Continue owner-driven splits in unified-KV and transport internals only
   when they simplify a concrete feature; do not restore retired experiment
   selectors or per-layer scheduling paths.

Dated milestone documents and legacy experiment reports remain read-only
development evidence. Skill-generated execution plans were removed after
consolidation and remain available through Git history. This page, the
architecture/runtime documents, and the experiment index define the current
state.
