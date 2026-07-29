# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L11`
- **Loop status:** `pre-registered`
- **Baseline commit:** `77548d1aa`
- **Last checkpoint:** `2026-07-29`

L10 falsifies the proposed long-input goodput advantage. In the coarse
bracket, only C20 is standard/relaxed eligible for each architecture; PAP
trails PD by 23.1%/21.8%. L11 refines the intermediate concurrency values
before treating that deficit as the tuned result.

## Current loop

- **Paper-level uncertainty:** Is L10's 21.8--23.1% PAP goodput deficit robust
  to a finer search of the actual long-input capacity boundary?
- **Hypothesis:** After refining the C20--28 interval, best tested PD standard
  and relaxed goodput remains at least 15% above best tested PAP.
- **Falsification condition:** Reject if the PD margin is below 15% in either
  tier, any run fails correctness, or no intermediate eligible PAP or PD point
  is measured when one exists in the tested sequence.
- **Expected paper delta:** Establish a credible tuned long-input negative
  result, or identify a narrow concurrency point where PAP converts latency
  headroom into goodput.
- **Minimal next evidence:** One clean repetition at PAP C21/C23/C25 and PD
  C22/C24/C26, combined with the L10 C20 controls.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L10` correctly completes all
  six bracket points. Best tested standard/relaxed goodput is PAP 5.704/5.844
  versus PD 7.418/7.477 req/s.
- **Known contradictions:** The scan jumps from C20 to C27/C28. An unmeasured
  intermediate boundary could narrow the apparent PAP deficit.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: no validated adaptive mechanism; pending workload-region
  diagnosis
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

- **Loop ID:** `L11`
- **Question:** What are the finer standard/relaxed boundaries between C20
  and the first failing points?
- **Next action:** Commit L10 and this refinement plan, then run the six
  intermediate points on the exact dataset.
- **Stop or pivot condition:** If no intermediate point passes, retain C20.
  If a point passes, select the highest-goodput eligible point. Decide C11
  before any repetitions or mechanism work.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
