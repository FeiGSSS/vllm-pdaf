# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L12`
- **Loop status:** `pre-registered`
- **Baseline commit:** `5299b5087`
- **Last checkpoint:** `2026-07-29`

L11 is invalidated by an asynchronous sampled-token correctness defect at PAP
C25. Commit `4a3e36820` fixes ownership by combining GPU-frame-local sequence
keys with Scheduler-side output acceptance; two 640-request diagnostics then
complete with zero mismatches. L12 now tests the mechanism that earlier
workloads did not isolate: PAP's 2.63x larger aggregate KV-token pool.

## Current loop

- **Paper-level uncertainty:** Does PA-side KV pooling turn into an SLO
  goodput advantage when four multi-turn contexts approach the model limit?
- **Hypothesis:** On the fixed four-round workload, the best tested PAP
  6PA2P Standard and Relaxed goodput is at least 10% above the best tested PD
  6P2D goodput because PD reaches its Decode KV wall first.
- **Falsification condition:** Reject if PAP leads by less than 10% in either
  tier, any selected run fails correctness, PD shows no capacity pressure by
  C16, PAP fails at or below C16, or the points do not bracket an eligible and
  a higher-pressure boundary for each architecture.
- **Expected paper delta:** Establish PAP's first mechanism-backed positive
  region, or show that remote execution overhead dominates even with a 2.63x
  aggregate KV pool.
- **Minimal next evidence:** One clean scan at PD C8/C10/C12/C16 and PAP
  C8/C12/C16/C20/C24, followed by repetitions only at selected boundaries.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L11` records the invalidated
  comparison and the accepted-token/frame-key correctness diagnosis.
- **Known contradictions:** L07's PAP-negative workload has roughly 4K
  initial input; L10 reaches only 13.8K mean final context. Neither isolates
  the near-capacity region predicted from actual KV-token budgets.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Generate 128 sessions and four turns with
  seed 42; every round targets 10K new input (range 8.5--9.9K), O16 output,
  and the existing short delay schedule. The validated dataset SHA-256 is
  `4e99a9ec2626899bb4625e36c135ee8609f4de473ab38f4a7097293145760876`.

## Paper gap queue

- Core mechanism: no validated adaptive mechanism; pending workload-region
  diagnosis
- Problem and motivation: L01/L02 narrow it to batching and synchronization
  domain size, not copy or dense Projection cost
- Novelty and closest related work: initial pass rejects basic barrier-aware
  placement and A/F ratio selection as novelty
- Correct implementation: accepted-token/frame-key fix validated at C25
- Fair and tuned baselines
- Workload-region validity
- Model and hardware generality
- Causal ablations and sensitivity
- Reproducibility artifacts
- Complete English manuscript

## Next loop

- **Loop ID:** `L12`
- **Question:** Does the measured 2.63x aggregate KV-token pool produce a
  usable-capacity advantage near 38K final contexts?
- **Next action:** Commit this pre-registration, run the fixed scan, and
  inspect Decode/PA KV usage plus waiting/deferred evidence before assigning
  causality.
- **Stop or pivot condition:** If both architectures remain below capacity,
  extend only the nearest registered edge. If PAP wins, repeat selected
  boundaries and add a context-length ablation; if not, retain the negative
  result and pivot away from KV pooling as the paper's primary benefit.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
