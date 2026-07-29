# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L05`
- **Loop status:** `experiment`
- **Baseline commit:** `734223485`
- **Last checkpoint:** `2026-07-29`

L01/L02 eliminated fan-in spread, copies, and dense Projection compute as
dominant explanations. L03 exposed residual throughput mismatch. L04 found a
matched standard/relaxed frontier point but retained a 9.33% ITL penalty and
an unstable strict tail. L05 tests the fixed PA resource partition.

## Current loop

- **Paper-level uncertainty:** Is the remaining 7PA1P ITL penalty caused by
  the hard-coded 72/20-SM Prefill/Attention allocation rather than topology?
- **Hypothesis:** A 16/7-chunk (64/28-SM) split improves 7PA1P C21 mean ITL by
  at least 5% while preserving at least 40% lower mean TTFT and standard/
  relaxed goodput within 2% of the 6PA2P C32 control.
- **Falsification condition:** Reject if ITL improves by less than 5%, mean
  TTFT loses the 40% advantage, throughput differs from control by more than
  2%, or standard/relaxed goodput is more than 2% lower.
- **Expected paper delta:** Establish resource allocation as a controllable
  PAP mechanism or eliminate it before building a complex scheduler.
- **Minimal next evidence:** Two clean 7PA1P C21 repetitions at 16/7 on the
  byte-identical L04 dataset, with static-MPS audits proving 64/28 SMs.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L04` records two correct
  7PA1P C21 repetitions. It matches 6PA2P C32 throughput within 0.59%, lowers
  mean TTFT by 61.9%, and raises mean ITL by 9.33% on dataset SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- **Known contradictions:** One C21 repetition misses strict at 94.84%.
  Standard and relaxed pass, but claim maturity remains `observed` because
  the 6PA2P control was reused from L03.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: pending L05 resource-allocation decision
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

- **Loop ID:** `L05`
- **Question:** Can PA resource rebalancing turn the C04 trade-off into a
  better frontier point?
- **Next action:** Expose audited capacity-runner MPS chunk overrides, commit
  them, then run 7PA1P C21 at 16/7.
- **Stop or pivot condition:** If C05 fails, keep 18/5 as the workload
  baseline and move to controlled workload-region sensitivity rather than
  adding an adaptive partitioner.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
