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
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
last_validated_commit: 152f64445cd12ffda91ec1d46330c563a36ab475
---

# PAP documentation

This directory is the canonical entry point for the current
Prefill–Attention–Projection implementation. Read these documents in order:

1. [Current status](status.md) — supported capabilities, active validation
   lanes, current performance milestone, and remaining work.
2. [Architecture](architecture.md) — roles, ownership, topology, transports,
   and support boundaries.
3. [Runtime](runtime.md) — the accepted request, KV, token, commit, lease, and
   drain paths.
4. [Benchmark methodology](benchmark-methodology.md) — the AIPerf testbeds,
   evidence grades, experiment records, and release criteria.
5. [Compatibility retirement](compatibility.md) — removed legacy façades and
   the current import-ownership rule.
6. [Historical runtime refactor milestone](milestones/2026-07-runtime-refactor.md)
   — frozen scope, decisions, and validation result from the 64/28-SM stage.

The detailed historical ledger remains at
[PAP development and experiment history](../../../benchmarks/pap/experiments/HISTORY.md).
Its normalized status index is generated at
[the PAP experiment index](../../../benchmarks/pap/experiments/INDEX.md).

## Document lifecycle

- This directory contains current design and validation contracts; `status.md`
  is the current snapshot.
- `milestones/` contains frozen stage records. Their original settings and
  conclusions are not rewritten when the runtime changes.
- `benchmarks/pap/experiments/PAP-*/` contains current or superseded experiment
  bundles; lifecycle is stated in each report and in `INDEX.md`.
- `benchmarks/pap/experiments/legacy/` and `test/baseline/pap/` contain
  historical evidence. Their dated “current”,
  “default”, and “TODO” language applies only to the recorded snapshot.

New architecture guidance belongs here, and new experimental evidence belongs
under `benchmarks/pap/experiments/`. Do not create another PAP documentation or
result root. Benchmark automation is likewise project-owned under
`benchmarks/pap/`; no repository skill defines PAP experiment behavior.

## Current boundary

The accepted runtime path is Qwen3-8B FP16, same-host `local_fast`, static MPS
72/20 on each PA GPU, asynchronous decode-token delivery,
asynchronous Prefill KV import, sealed manifest handoff, and Prefill-owned
unified KV. The active runner is the eight-GPU AIPerf matrix; the latest
normalized milestone remains the four-GPU result, while an initial eight-GPU
C32 development pilot and the compact capacity scan are complete.
PAP-to-vLLM glue is isolated behind owner-specific adapters in
`vllm/pap/integration/`; vLLM owners do not implement alternate PAP paths.
PAP keeps each vLLM scheduler batch intact and may split it only into
same-step PA route shards. vLLM async scheduling remains enabled for CPU-side
next-step preparation; PAP does not interleave microbatches across model
layers.

The current eight-GPU development lane compares PAP 7PA1P/6PA2P, one-way PD
4P4D/6P2D, and an eight-replica fused vLLM pool under a 128-conversation,
five-turn randomized long-context workload. The C32 pilot finds 6PA2P to be
the only topology passing both Standard and Relaxed. The completed
C16/24/32/48 scan finds PAP leading PD and fused DP under Strict and Standard
goodput; fused DP leads under Relaxed.
Eager remains the default execution mode. Optional piecewise CUDA Graph has a
completed development comparison, while host transport, remote Attention, and
KV publication remain outside captured regions. The former P17 lane is
archived and no longer defines current validation.

Arbitrary xPAyP and cross-host NIXL remain implemented and contract-covered,
but are `preserved-unverified`: this milestone does not claim fresh E2E
validation for them.

## Legacy documents

The [`benchmarks/pap/experiments/legacy/reports/`](../../../benchmarks/pap/experiments/legacy/reports/)
directory contains historical designs, experiments, root-cause reports, and
handoffs. They remain evidence, but they are not the source of current runtime
defaults. When a historical statement conflicts with this directory or the
experiment index, the current docs and experiment records win.

Skill-generated implementation plans were removed after their accepted
decisions and evidence had been consolidated here, in the runtime-refactor
milestone, and in the experiment history. Git history remains the source for
those execution plans. Current work is tracked by `status.md`, source changes,
and normalized experiments rather than a repository skill plan.

## Operational entry points

- AIPerf PAP xPAyP workload runner:
  `benchmarks/pap/scripts/run_pap_workload.sh`
- Canonical eight-GPU AIPerf matrix:
  `benchmarks/pap/aiperf/run_capacity_matrix.sh`
- Initial eight-GPU capacity and 7PA1P trace:
  [capacity pilot](../../../benchmarks/pap/experiments/PAP-20260725-8GPU-CAPACITY-PILOT/report.md)
- Completed PAP/PD/fused-DP capacity comparison:
  [capacity scan](../../../benchmarks/pap/experiments/PAP-20260725-8GPU-CAPACITY-SCAN/report.md)
- Current eager and Graph capacity report:
  [automatic Projection-memory milestone](../../../benchmarks/pap/experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md)
- Current Projection scheduling regression:
  [step/control-overlap milestone](../../../benchmarks/pap/experiments/PAP-20260724-STEP-OVERLAP/report.md)
- Superseded scheduler-overlap baseline:
  [scheduler-overlap milestone](../../../benchmarks/pap/experiments/PAP-20260724-PROJECTION-SCHEDULER-OVERLAP/report.md)
- Archived Projection no-async negative control:
  [scheduler-overlap diagnostic](../../../benchmarks/pap/experiments/PAP-20260724-SINGLE-PROJECTION-BATCH/report.md)
- Offline diagnostics: `benchmarks/pap/tooling/`, invoked through `tools/pap_*`
- Test policy: [tests/pap/README.md](../../../tests/pap/README.md)
- Runnable examples: [examples/pap/README.md](../../../examples/pap/README.md)
