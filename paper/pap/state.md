# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L08`
- **Loop status:** `pre-registered`
- **Baseline commit:** `163c2dfa1`
- **Last checkpoint:** `2026-07-29`

L07 cleanly confirms that PAP loses to corrected PD on the canonical O16
workload. PAP beats fused DP only under the strict SLO, not the registered
standard SLO. Because older O32 studies were PAP-favorable but changed several
workload dimensions together, L08 isolates output length before considering a
new mechanism.

## Current loop

- **Paper-level uncertainty:** Is the reversal between historical O32 and
  current O16 PAP/PD results caused materially by Decode output length?
- **Hypothesis:** Doubling only the output-length distribution from O16 to O32
  raises the fixed-point PAP 7PA1P C34 to PD 6P2D C48 mean raw
  request-throughput ratio by at least 10 percentage points, from the L07
  baseline ratio of 0.758.
- **Falsification condition:** Reject if input text, session order, or delays
  differ; either run fails correctness; or the O32 throughput ratio is below
  0.858.
- **Expected paper delta:** Bound the workload region in which PA-side
  Attention ownership can amortize PAP's synchronization and Projection
  costs. If falsified, output length is not the explanation and the next loop
  isolates prompt length or arrival timing.
- **Minimal next evidence:** Generate an input- and delay-identical O32
  dataset, verify the identity programmatically, then run one clean diagnostic
  repetition each at PAP 7PA1P C34 and PD 6P2D C48.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L07` provides 16 clean,
  correct repetitions on common dataset SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
  PD leads PAP goodput by 32.2%, 36.9%, and 24.4% under the three SLO tiers.
  PAP leads fused DP by 57.8% strict but loses by 12.1% standard and 17.2%
  relaxed.
- **Known contradictions:** Earlier O32 results favor PAP, but changed output
  length, prompt length, and delay schedule together. They do not identify
  which workload dimension causes the reversal.
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

- **Loop ID:** `L08`
- **Question:** Does output length alone explain a material part of the PAP/PD
  result reversal?
- **Next action:** Commit L07 and this preregistration, generate the O32
  derived dataset, prove non-output fields are unchanged, and run the two
  fixed diagnostic points.
- **Stop or pivot condition:** If the throughput-ratio improvement is below
  10 percentage points, falsify C08 and isolate prompt length or arrival
  timing. If supported, repeat and sweep output length before any mechanism
  change.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
