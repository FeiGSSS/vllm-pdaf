# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L06`
- **Loop status:** `evidence-audit`
- **Baseline commit:** `57926aac2`
- **Last checkpoint:** `2026-07-29`

L04 found a matched standard/relaxed frontier point with a 9.33% ITL penalty.
L05 falsified static PA repartitioning as the remedy: 16/7 does not improve
ITL and reduces throughput by 4.98%. L06 re-audits the current three-way
capacity evidence before selecting another implementation direction.

## Current loop

- **Paper-level uncertainty:** Which PAP advantages remain after correcting
  the PD NIXL path and reducing the canonical workload?
- **Hypothesis:** Current tuned PAP loses goodput to PD under all three SLO
  tiers, but retains higher tested standard concurrency and strict/standard
  goodput over fused DP.
- **Falsification condition:** Reject the audit if a selected point lacks the
  same dataset digest, correctness, clean tracked runtime provenance,
  corrected NIXL settings where applicable, or two valid repetitions.
- **Expected paper delta:** Remove superseded universal PAP-win language and
  identify the precise workload or mechanism gap the next loop must address.
- **Minimal next evidence:** Provenance audit of the selected July 28 PAP, PD,
  and DP boundary summaries and their effective configurations.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L05` records two correct
  16/7 treatment repetitions and verifies 64/28 visible SMs on all PAs on
  dataset SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- **Known contradictions:** The treatment leaves ITL unchanged (+0.45%) and
  loses roughly 5% throughput and standard/relaxed goodput versus 18/5.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: no validated adaptive mechanism; pending L06 reorientation
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

- **Loop ID:** `L06`
- **Question:** Does the existing current-workload three-way scan qualify as
  evidence that PAP loses to corrected PD while retaining narrower benefits?
- **Next action:** Audit source matrices, commits, dataset hashes, runtime
  settings, repetitions, and correctness for every selected boundary.
- **Stop or pivot condition:** If the audit passes, use the negative result to
  preregister controlled workload-region sensitivity; if it fails, rerun only
  the missing boundary points.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
