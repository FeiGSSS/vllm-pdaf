# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L01`
- **Loop status:** `diagnosis`
- **Baseline commit:** `12294e9ea`
- **Last checkpoint:** `2026-07-29`

L01 determines whether multi-PA completion skew is the dominant cause of the
7PA1P latency gap relative to 6PA2P before selecting a scheduling or runtime
change.

## Current loop

- **Paper-level uncertainty:** Why does 7PA1P have worse ITL and tail latency
  than 6PA2P despite assigning one more GPU to PA work?
- **Hypothesis:** Layer-level completion skew across seven PA participants,
  amplified by the Projection fan-in join, dominates the 7PA1P latency loss;
  Projection compute is not the dominant cause.
- **Falsification condition:** Reject the hypothesis if fan-in skew explains
  less than half of the matched E2E ITL gap, or if an equal-load intervention
  does not materially narrow that gap.
- **Expected paper delta:** Establish or reject fan-in amplification as a PAP
  scaling challenge and decide whether barrier-aware placement is a core
  mechanism.
- **Minimal next evidence:** Audit existing matched 7PA1P/6PA2P reports and
  traces, reconstruct per-layer participant completion distributions, then
  run one controlled equal-load comparison only if existing evidence is
  insufficient.

## Evidence checkpoint

- **Completed evidence:** Existing experiment records are available but have
  not yet been admitted as L01 evidence.
- **Known contradictions:** Existing summaries report both better matched-shape
  7PA1P forward time and worse E2E 7PA1P tail latency; the causal relationship
  is unresolved.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical eight-GPU AIPerf
  workload for matched comparisons; record its exact identity after audit.

## Paper gap queue

- Core mechanism: active in L01
- Problem and motivation: depends on L01
- Novelty and closest related work: pending causal diagnosis
- Correct implementation
- Fair and tuned baselines
- Workload-region validity
- Model and hardware generality
- Causal ablations and sensitivity
- Reproducibility artifacts
- Complete English manuscript

## Next loop

- **Loop ID:** `L01`
- **Question:** What component causally explains the 7PA1P versus 6PA2P
  latency gap?
- **Next action:** Audit the authoritative experiment reports, raw trace
  identities, and current fan-in instrumentation before running new work.
- **Stop or pivot condition:** Pivot away from barrier-aware scheduling if
  completion skew is not a dominant, intervention-sensitive component of the
  matched latency gap.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
