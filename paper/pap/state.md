# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L02`
- **Loop status:** `diagnosis`
- **Baseline commit:** `ee6b307c7`
- **Last checkpoint:** `2026-07-29`

L01 falsified fan-in completion spread as the dominant explanation of the
7PA1P latency gap. L02 separates intrinsic Decode load aggregation from
instrumentation and topology effects before selecting a mechanism.

## Current loop

- **Paper-level uncertainty:** Is 7PA1P's ITL cost an avoidable implementation
  penalty, or the expected service time of aggregating more Decode work behind
  one Projection domain?
- **Hypothesis:** One Projection domain forms larger Decode batches in 7PA1P;
  matched Projection and PA kernels are topology-neutral, and a phase model
  using batch rows, KV load, communication, and the slowest PA explains at
  least 75% of the trace-free ITL gap.
- **Falsification condition:** Reject the hypothesis if a non-intrusive phase
  model leaves more than 25% of the gap unexplained or matched-shape execution
  retains a material topology-specific penalty.
- **Expected paper delta:** Establish a measured PAP scaling model and identify
  whether the next mechanism should control placement, batch formation,
  provisioning, or runtime overhead.
- **Minimal next evidence:** Once the GPU driver is stable, rerun the same
  byte-identical C32 workload with trace disabled, then collect deferred CUDA
  spans without per-layer synchronization. Do not implement a scheduler before
  this decomposition.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L01` records the current
  trace-mode matched comparison. Both topologies completed 640/640 requests on
  dataset SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- **Known contradictions:** 7PA1P has higher throughput and lower TTFT but
  worse ITL. Blocking trace inflates its Attention wall time while matched
  CUDA kernel time remains similar, so trace E2E numbers are diagnostic only.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: active in L02
- Problem and motivation: L01 narrows it to aggregation versus state movement
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

- **Loop ID:** `L02`
- **Question:** Can a non-intrusive per-step service model explain the
  7PA1P/6PA2P ITL gap from their actual Decode batch shapes?
- **Next action:** Recover the trace-off C32 baseline and collect deferred CUDA
  spans after the recurring NVIDIA driver fault clears.
- **Stop or pivot condition:** If the service model succeeds, use it to choose
  a scheduling/provisioning mechanism; if it fails, isolate the residual with
  matched-shape microbenchmarks before changing runtime code.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
