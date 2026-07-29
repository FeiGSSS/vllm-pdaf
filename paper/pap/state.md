# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L04`
- **Loop status:** `experiment`
- **Baseline commit:** `15b5b90af`
- **Last checkpoint:** `2026-07-29`

L01 falsified completion spread as the dominant cause. L02 found that
fixed-C32 7PA1P runs at a higher-throughput operating point. L03 controlled
throughput more closely but falsified its no-goodput-reduction criterion.
L04 localizes the 7PA1P operating point more accurately.

## Current loop

- **Paper-level uncertainty:** At accurately matched achieved throughput, does
  7PA1P provide a useful TTFT/ITL trade-off without losing SLO goodput?
- **Hypothesis:** 7PA1P C21 matches 6PA2P C32 throughput within 1.5%, retains
  at least a 40% mean TTFT advantage, stays within 12% mean ITL, and stays
  within 2% of standard and relaxed goodput.
- **Falsification condition:** Reject if any threshold is missed. If C21
  misses the throughput range, test at most one adjacent concurrency point
  selected from the direction of the mismatch.
- **Expected paper delta:** Establish a defensible topology-frontier point or
  retire 7PA1P as the preferred topology for this workload region.
- **Minimal next evidence:** Two clean trace-off 7PA1P C21 repetitions on the
  byte-identical L03 dataset, reusing its valid 6PA2P C32 control.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L03` records two correct
  repetitions per topology. Every repetition completed 640/640 requests and
  passed all three SLO tiers on dataset SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- **Known contradictions:** At C20 versus C32, 7PA1P has 60.5% lower mean
  TTFT and 6.64% higher mean ITL, but its 2.55% lower request throughput leads
  to 2.7--2.8% lower standard/relaxed goodput.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: pending L04 frontier decision
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

- **Loop ID:** `L04`
- **Question:** Can 7PA1P match the 6PA2P C32 achieved throughput closely
  enough to compare the latency/goodput frontier without a throughput
  confound?
- **Next action:** Run two clean 7PA1P C21 repetitions with tracing disabled.
- **Stop or pivot condition:** If C21, or at most one direction-selected
  adjacent point, misses C04, stop treating 7PA1P as the preferred topology
  for this workload and search a different workload region or mechanism.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
