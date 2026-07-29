# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L03`
- **Loop status:** `experiment`
- **Baseline commit:** `f90dfc084`
- **Last checkpoint:** `2026-07-29`

L01 falsified completion spread as the dominant cause. L02 found that
fixed-C32 7PA1P runs at a higher-throughput operating point and falsified the
registered 75% phase-attribution threshold. L03 tests the topology trade-off
at matched achieved throughput.

## Current loop

- **Paper-level uncertainty:** Does 7PA1P offer a better latency/goodput Pareto
  point than 6PA2P, or does its larger synchronization domain merely exchange
  ITL for throughput?
- **Hypothesis:** At about 9.3 req/s, 7PA1P C20 retains at least a 40% mean
  TTFT advantage, stays within 12% of 6PA2P C32 mean ITL, and does not reduce
  standard or relaxed goodput.
- **Falsification condition:** Reject if achieved throughput differs by more
  than 5%, TTFT improves by less than 40%, mean ITL is more than 12% worse, or
  standard/relaxed goodput is lower.
- **Expected paper delta:** Replace fixed-concurrency topology comparisons with
  a defensible Pareto/goodput methodology and decide whether PAP needs a
  tail-aware scheduling mechanism.
- **Minimal next evidence:** Two clean trace-off repetitions of 7PA1P C20 and
  6PA2P C32 on the byte-identical L02 dataset.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L02` records clean trace-off
  and valid deferred comparisons. Both topologies completed 640/640 requests
  on dataset SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.
- **Known contradictions:** At C32, 7PA1P has 28.3% higher throughput and
  45.1% lower TTFT but 28.9% higher mean ITL. Historical iso-throughput C20
  pilots have unstable strict-SLO outcomes and are not formal evidence.
- **Current implementation state:** Refer to `docs/design/pap/status.md`.
- **Current experiment state:** Refer to
  `benchmarks/pap/experiments/INDEX.md`.
- **Dataset/config identity:** Reuse the byte-identical 128-session,
  five-turn AIPerf dataset with SHA-256
  `b694ba148a0789e4056a6c3f21fe1f3cbaf3d2c3a2eff2d4d663553f1a2546ed`.

## Paper gap queue

- Core mechanism: pending L03 frontier decision
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

- **Loop ID:** `L03`
- **Question:** Which topology has the better Pareto point at approximately
  9.3 req/s?
- **Next action:** After the NVIDIA driver recovers, run two clean repetitions
  of 7PA1P C20 and 6PA2P C32 with tracing disabled.
- **Stop or pivot condition:** If 7PA1P meets C03, target its strict-ITL tail;
  otherwise stop treating 7PA1P as the preferred topology for this workload
  and search a different workload region or mechanism.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
