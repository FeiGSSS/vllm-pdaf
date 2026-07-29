# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L10`
- **Loop status:** `pre-registered`
- **Baseline commit:** `5c19606e9`
- **Last checkpoint:** `2026-07-29`

L09 observes that doubling input and append lengths improves the fixed-point
PAP/PD raw throughput ratio by 11.17 percentage points, above its registered
threshold. PAP also has much lower mean TTFT and ITL at the fixed points, but
both points fail every SLO. L10 now brackets each architecture's long-input
capacity instead of treating overloaded raw throughput as goodput.

## Current loop

- **Paper-level uncertainty:** Does PAP's fixed-point long-context latency
  advantage translate into higher SLO goodput after both architectures are
  retuned for concurrency?
- **Hypothesis:** On the L09 dataset, the best tested 7PA1P standard and
  relaxed goodput exceeds the best tested 6P2D goodput by at least 10%.
- **Falsification condition:** Reject if PAP misses the 10% margin in either
  standard or relaxed, any selected run fails correctness, or the tested
  points do not include at least one eligible point and one higher-pressure
  boundary for each architecture.
- **Expected paper delta:** Convert the long-context observation into an SLO
  operating-region claim, or show that PAP's latency benefit still cannot
  overcome PD raw capacity.
- **Minimal next evidence:** One clean repetition at PAP C20/C27/C32 and PD
  C20/C28/C36 on the exact L09 dataset. Repeat only selected boundaries in the
  following loop if the bracket is valid.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L09` preserves O16 outputs
  and delays and completes both fixed points correctly. The PAP/PD raw
  throughput ratio improves from 0.758 to 0.870; PAP mean TTFT and ITL are
  33.3% and 46.0% lower at the overloaded points.
- **Known contradictions:** Both fixed points fail strict, standard, and
  relaxed SLO eligibility. No tuned long-input goodput advantage exists yet.
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

- **Loop ID:** `L10`
- **Question:** What are the PAP and PD standard/relaxed capacity boundaries
  on the long-input workload?
- **Next action:** Commit L09 and this bracket, then run the six preselected
  points with isolated service restarts.
- **Stop or pivot condition:** If the bracket lacks a passing and failing
  side, add at most one adjacent point per architecture. Otherwise decide C10
  and repeat only the selected boundary points.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
