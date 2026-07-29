# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L07`
- **Loop status:** `experiment`
- **Baseline commit:** `a0f0189d8`
- **Last checkpoint:** `2026-07-29`

L05 falsified static PA repartitioning as the remedy. L06 found that the
directional July 28 three-way result has correct workload/runtime settings but
was collected with a dirty tracked PAP runtime. L07 repeats only the selected
boundaries on clean current code.

## Current loop

- **Paper-level uncertainty:** Does the current clean implementation reproduce
  the directional result that PAP loses to corrected PD but beats fused DP
  under tighter SLOs?
- **Hypothesis:** PD leads PAP goodput by at least 15% in every tier; PAP leads
  DP by at least 30% strict and 5% standard. No relaxed PAP-over-DP benefit is
  hypothesized.
- **Falsification condition:** Reject if any margin is missed or any selected
  point fails correctness or two-repetition eligibility.
- **Expected paper delta:** Establish the honest current baseline that the
  next scheduling or workload-region mechanism must improve.
- **Minimal next evidence:** Two clean repetitions of the eight preselected
  boundary points, without a new concurrency search.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L06` verifies common dataset
  SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
  It also verifies corrected NIXL settings and repeated correctness for the
  directional selected points.
- **Known contradictions:** Every directional selected point has a dirty
  52,776-byte tracked patch including a PAP runtime file.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: no validated adaptive mechanism; pending L07 clean baseline
- Problem and motivation: L01/L02 narrow it to batching and synchronization
  domain size, not copy or dense Projection cost
- Novelty and closest related work: initial pass rejects basic barrier-aware
  placement and A/F ratio selection as novelty
- Correct implementation
- Fair and tuned baselines
- Workload-region validity
- Model and hardware generality
- Causal ablations and sensitivity
- Reproducibility artifacts
- Complete English manuscript

## Next loop

- **Loop ID:** `L07`
- **Question:** What are the clean current three-way goodput margins at the
  preselected boundaries?
- **Next action:** Commit this audit, copy the canonical dataset, and launch
  the exact eight-point, two-repetition matrix.
- **Stop or pivot condition:** After the clean comparison, move to controlled
  workload-region sensitivity regardless of whether C07 is supported.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
