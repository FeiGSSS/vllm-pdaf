# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L13`
- **Loop status:** `pre-registered`
- **Baseline commit:** `8b9228871`
- **Last checkpoint:** `2026-07-29`

L12 falsifies KV pooling as a sufficient advantage. Both PAP and PD bracket
their Standard boundary at C12--C16 on the near-limit workload, while PAP
C12 Standard goodput is 34.4% below PD. PAP C16 still has lower mean ITL but
materially higher TTFT, pointing to PA Prefill resources rather than
Attention capacity as the next causal question.

## Current loop

- **Paper-level uncertainty:** Does PAP lose near-limit TTFT primarily because
  its fixed 18/5 PA MPS split withholds too much Prefill compute?
- **Hypothesis:** At PAP 6PA2P C12, a 20/3 split lowers mean TTFT by at least
  10% and raises raw throughput by at least 8%, while mean ITL remains at or
  below 55 ms and at least 95% of requests pass Standard.
- **Falsification condition:** Reject if either improvement threshold is
  missed, mean ITL exceeds 55 ms, Standard good fraction falls below 95%, or
  any correctness gate fails.
- **Expected paper delta:** Determine whether PAP needs workload-adaptive
  compute partitioning, rather than treating its larger KV pool as sufficient.
- **Minimal next evidence:** One clean PAP 6PA2P C12 treatment on the exact
  48-session L12 dataset; use the existing 18/5 C12 point as control.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L12` records the matched
  192-request boundary scan and the independent 512-request PD controls.
- **Known contradictions:** The full-size PAP C8/C12 request streams complete
  and drain to zero sessions, but fail the release-count routing audit. Their
  performance is diagnostic only; this scale-dependent audit defect must be
  resolved before any full-size PAP confirmation.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** L13 reuses the L12 discovery dataset: seed 42,
  four turns, about 10K new input per turn (configured range 8.5--9.9K), O16
  output, and the existing short delay schedule. It has 48 sessions, prefix
  `pap-pd-dp-s48-t4-seed42`, and SHA-256
  `8ef6c8017930b8549ba077f14c1592d683fbd69d9de3795931657ba9f9dd1e73`.
  Its final-turn input averages 37,818 tokens and peaks at 39,732. Final
  confirmation uses 128 sessions, prefix `pap-pd-dp-s128-t4-seed42`, and
  SHA-256
  `c9c7b6e36d8a45b2d87d8af308ecdc66f9006b429502fbd7820a0ec85555f78b`.
- **Completed L12 evidence:** In the valid discovery scan, PD/PAP Standard
  goodput at C12 is 2.290/1.502 req/s and both fail Standard at C16. Full-size
  PD independently confirms the C12--C16 boundary. The shorter scan preserves
  the Standard decision but under-samples Relaxed tails, so it remains a
  locator rather than final tail evidence.

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

- **Loop ID:** `L13`
- **Question:** Is the static PA Prefill/Attention SM split the primary cause
  of PAP's near-limit TTFT deficit?
- **Next action:** Commit the L12 result and L13 pre-registration, then run
  only PAP 6PA2P C12 with a 20/3 MPS split.
- **Stop or pivot condition:** If the registered TTFT and throughput gains
  both hold without violating ITL or correctness, test C16 and design dynamic
  temporal allocation. Otherwise profile the residual Prefill/control-plane
  gap and do not pursue MPS repartitioning as the paper mechanism.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
