---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260716-TRITON-72-20-BASELINE
  - PAP-20260715-VLLM-INTEGRATION-BOUNDARY
  - PAP-20260715-ARCHITECTURE-MILESTONE
  - PAP-20260715-RUNTIME-BOUNDARY-E2E
  - PAP-20260715-MODEL-ADAPTER-E2E
  - PAP-20260715-INTEGRATION-E2E
  - PAP-20260714-P17-PRE-REFACTOR
  - PAP-20260715-P17-POST-REFACTOR
  - PAP-20260714-SEAL-HANDOFF-KV
last_validated_commit: 3a6fe93d11245c1137d3ea6767cd5e27b3e88156
---

# PAP documentation

This directory is the canonical entry point for the current
Prefill–Attention–Projection implementation. Read these documents in order:

1. [Architecture](architecture.md) — roles, ownership, topology, transports,
   and support boundaries.
2. [Runtime](runtime.md) — the accepted request, KV, token, commit, lease, and
   drain paths.
3. [Benchmark methodology](benchmark-methodology.md) — P17, evidence grades,
   experiment records, and release gates.
4. [Compatibility retirement](compatibility.md) — removed legacy façades and
   the current import-ownership rule.
5. [Runtime refactor milestone](milestones/2026-07-runtime-refactor.md) — scope,
   decisions, and the frozen validation result.

The detailed historical ledger remains at
[PAP development and experiment history](../../../benchmarks/pap/experiments/HISTORY.md).
Its normalized status index is generated at
[the PAP experiment index](../../../benchmarks/pap/experiments/INDEX.md).

## Current boundary

The accepted runtime path is Qwen3-8B FP16, 1PA1P/TP1, same-host
`local_fast`, static MPS 72/20, asynchronous decode-token delivery,
asynchronous Prefill KV import, sealed manifest handoff, and Prefill-owned
unified KV. This is the only end-to-end release gate for the frozen milestone.
PAP-to-vLLM glue is isolated behind owner-specific adapters in
`vllm/pap/integration/`; vLLM owners do not implement alternate PAP paths.

Arbitrary xPAyP and cross-host NIXL remain implemented and contract-covered,
but are `preserved-unverified`: this milestone does not claim fresh E2E
validation for them.

## Legacy documents

The [`benchmarks/pap/experiments/legacy/reports/`](../../../benchmarks/pap/experiments/legacy/reports/)
directory contains historical designs, experiments, root-cause reports, and
handoffs. They remain evidence, but they are not the source of current runtime
defaults. When a historical statement conflicts with this directory or the
experiment index, the current docs and experiment records win.

`docs/superpowers/` contains historical implementation plans/specifications.
They are retained temporarily for traceability, are not an active workflow,
and must not be used as canonical PAP documentation. No new PAP plans belong
there; the active local implementation checklist lives outside `docs/`.

## Operational entry points

- P17 profile: `benchmarks/pap/profiles/p17_1pa1p.toml`
- P17 runner: `benchmarks/pap/scripts/run_p17_1pa1p.sh`
- Generic xPAyP workload runner: `benchmarks/pap/scripts/run_pap_workload.sh`
- Offline diagnostics: `benchmarks/pap/tooling/`, invoked through `tools/pap_*`
- Test policy: [tests/pap/README.md](../../../tests/pap/README.md)
- Runnable examples: [examples/pap/README.md](../../../examples/pap/README.md)
