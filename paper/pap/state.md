# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L13`
- **Loop status:** `correctness repair before clean rerun`
- **Baseline commit:** `8b9228871`
- **Last checkpoint:** `2026-07-30`

L12 falsifies KV pooling as a sufficient advantage. Both PAP and PD bracket
their Standard boundary at C12--C16 on the near-limit workload, while PAP
C12 Standard goodput is 34.4% below PD. PAP C16 still has lower mean ITL but
materially higher TTFT, pointing to PA Prefill resources rather than
Attention capacity as the next causal question. The first L13 treatment is
invalid because decode-commit and release-count correctness fail.

Separately, the long-context O100 development scan identifies a narrow positive
region: 7PA1P C20 has 20.5% higher Standard goodput than 6P2D C20 on the
60-session workload and passes Standard while PD fails. This is one-repetition
evidence and does not extend to the Relaxed frontier or establish KV pressure
as the cause.

The previous large PAP-over-PD headline is retired. Historical PD KV transfer
was degraded or unstable, including a pull GET that fell back to approximately
0.42 GiB/s TCP emulation instead of a 22--24.5 GiB/s CUDA-IPC path. Corrected
PD wins the clean O16 comparison. Because workload and topology also changed
between those studies, the evidence establishes baseline contamination rather
than attributing 100% of the reversal to transport alone. Old favorable
margins must not be used as architecture evidence.

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
  48-session L12 dataset. If the lifecycle/control path changes, collect a
  contemporaneous 18/5 control instead of reusing the old point.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L12` records the matched
  192-request boundary scan and the independent 512-request PD controls.
  `PAP-20260729-LONGCTX-O100-CONCURRENCY-SCAN` records the 30-session
  discovery and 60-session C20--C32 extension.
- **Known contradictions:** The full-size PAP C8/C12 request streams complete
  and drain to zero sessions, but fail the release-count routing audit. Their
  performance is diagnostic only; this scale-dependent audit defect must be
  resolved before any full-size PAP confirmation. The first L13 20/3 C12 run
  repeats this class of failure and also logs decode-commit token-count
  mismatches; see `PAP-20260730-RESEARCH-L13`.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
  The working tree contains an uncommitted batched-commit/submit-only control
  experiment. Results collected from it are dirty diagnostics and cannot be
  compared with the synchronous committed controls.
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
- **Completed O100 observation:** At C20 on the 60-session dataset, PAP/PD
  Standard goodput is 1.611/1.337 req/s and only PAP passes the 95% gate.
  At C24 both fail Standard; at C28/C32 PAP loses Relaxed while PD passes.
  This bounds the observation to a Standard-SLO region rather than a general
  scaling claim.
- **Baseline-validity decision:** The July 13 transfer diagnosis, July 21/22
  lane-instability evidence, and clean L07 reversal invalidate the old large
  PAP-over-PD margins as architecture evidence. Future comparisons require
  the fail-closed corrected PD transport path.

## Paper gap queue

- Core mechanism: no validated adaptive mechanism; pending workload-region
  diagnosis
- Problem and motivation: L01/L02 narrow it to batching and synchronization
  cadence/load aggregation, not fan-in skew dominance, copy, or dense
  Projection cost
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
- **Next action:** Finish the lifecycle correctness repair, run focused unit
  tests, and commit it. If that patch changes control semantics, rerun both
  PAP 6PA2P C12 18/5 and 20/3 on the same clean commit; otherwise rerun only
  the registered 20/3 treatment.
- **Stop or pivot condition:** If the registered TTFT and throughput gains
  both hold without violating ITL or correctness, test C16 and design dynamic
  temporal allocation. Otherwise profile the residual Prefill/control-plane
  gap and do not pursue MPS repartitioning as the paper mechanism. After L13,
  the next paper loop should repeat and causally attribute the O100 C20 region
  rather than assuming it is caused by KV capacity.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
