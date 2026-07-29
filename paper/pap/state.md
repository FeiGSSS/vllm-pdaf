# PAP Current Research State

- **Research lifecycle:** `active`
- **Execution gate:** `open`
- **Active loop:** `L09`
- **Loop status:** `pre-registered`
- **Baseline commit:** `21a4f705e`
- **Last checkpoint:** `2026-07-29`

L08 finds that doubling only output length improves the fixed-point PAP/PD
throughput ratio by 6.15 percentage points, below its registered 10-point
threshold. Output length is directionally relevant but does not explain the
historical reversal. L09 isolates input and append length, which increases
both Prefill work and Decode Attention bytes.

## Current loop

- **Paper-level uncertainty:** Does long-context Prefill and Decode Attention
  load, rather than output length, define PAP's favorable workload region?
- **Hypothesis:** Doubling only initial and append input distributions raises
  the fixed-point PAP 7PA1P C34 to PD 6P2D C48 mean raw request-throughput
  ratio by at least 10 percentage points from the L07 O16 baseline of 0.758.
- **Falsification condition:** Reject if output samples, session order, or
  delays differ; either run fails correctness; the context budget exceeds
  32,768; or the long-input throughput ratio is below 0.858.
- **Expected paper delta:** Establish whether PAP's seven-way Attention
  parallelism creates a defensible long-context region. If falsified, the old
  PAP-favorable result likely depended on arrival timing, stale transport, or
  combined effects rather than context length alone.
- **Minimal next evidence:** Generate a long-input/O16 dataset with doubled
  document and append distributions, verify all non-input fields, then run one
  clean diagnostic repetition each at PAP 7PA1P C34 and PD 6P2D C48.

## Evidence checkpoint

- **Completed evidence:** `PAP-20260729-RESEARCH-L08` proves exact non-output
  dataset identity and completes both fixed points correctly. O32 improves the
  PAP/PD raw throughput ratio from 0.758 to 0.820, while missing the registered
  0.858 threshold.
- **Known contradictions:** Longer output improves PAP's relative position,
  but not enough to reproduce the older PAP advantage. Prompt length and
  arrival timing remain confounded in the historical comparison.
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

- **Loop ID:** `L09`
- **Question:** Does doubling context input alone materially improve PAP's
  position relative to PD?
- **Next action:** Commit L08 and this preregistration, generate and validate
  the doubled-input O16 dataset, then run the two fixed diagnostic points.
- **Stop or pivot condition:** If the ratio improvement is below 10 percentage
  points, falsify C09 and isolate the delay schedule. If supported, repeat and
  map the input-length boundary before changing PAP.

## Pause and recovery

If work pauses after the gate opens, update this file with the authoritative
commit, experiment IDs, unresolved contradiction, and first recovery action.
Finishing one loop must produce another loop or a paper-level completion audit.
