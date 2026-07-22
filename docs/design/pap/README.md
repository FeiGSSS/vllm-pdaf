---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
last_validated_commit: e5190a84e37124c893cf66d5b1bb94f9e31dc408
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
4. [Benchmark methodology](benchmark-methodology.md) — the four-GPU AIPerf
   testbed, evidence grades, experiment records, and release criteria.
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
unified KV. The four-GPU AIPerf matrix is its only active runtime testbed.
PAP-to-vLLM glue is isolated behind owner-specific adapters in
`vllm/pap/integration/`; vLLM owners do not implement alternate PAP paths.

The current four-GPU development lane fixes PAP at 3PA1P and compares it with
one-way PD under a 32-conversation, ten-turn randomized long-context workload.
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

- Generic xPAyP workload runner: `benchmarks/pap/scripts/run_pap_workload.sh`
- Canonical four-GPU AIPerf matrix:
  `benchmarks/pap/aiperf/run_capacity_matrix.sh`
- Current capacity reports:
  [eager](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md)
  and
  [piecewise CUDA Graph](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH/report.md)
- Offline diagnostics: `benchmarks/pap/tooling/`, invoked through `tools/pap_*`
- Test policy: [tests/pap/README.md](../../../tests/pap/README.md)
- Runnable examples: [examples/pap/README.md](../../../examples/pap/README.md)
