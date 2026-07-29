# PAP Claim Ledger

This file records claim maturity separately from experiment evidence grades.

## Claim maturity

| Status | Meaning |
| --- | --- |
| `hypothesis` | Falsifiable statement with no supporting experiment yet |
| `observed` | Supported by preliminary or single-run evidence |
| `supported` | Repeated and causally tested within the stated conditions |
| `paper-ready` | Supported across the generality required by the manuscript |
| `falsified` | Contradicted by authoritative evidence |
| `superseded` | Replaced by a more precise claim |

## Active claims

| ID | Claim | Conditions | Status | Evidence | Counterevidence | Next falsification |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | Multi-PA completion skew amplified by the fan-in join dominates the current 7PA1P latency loss relative to 6PA2P. | Same eight-GPU model, request trace, concurrency, runtime mode, and valid participant-count accounting. | `falsified` | PAP-20260729-RESEARCH-L01: the median spread delta accounts for only 3.20 ms per 36-layer step, 16.6% of the trace-mode ITL gap. | Same-shape Projection and paged-Attention kernel medians are topology-neutral; trace synchronization also inflates the measured 7PA1P gap. | Reopen only if a non-intrusive matched run attributes at least half of the gap to completion skew. |
| C02 | The 7PA1P ITL penalty is primarily a load-aggregation effect: one Projection domain forms larger Decode batches, while fan-in imbalance adds a smaller tail term. | Qwen3-8B FP16 eager, eight L20 GPUs, current reduced multi-turn workload, matched concurrency and clean code. | `hypothesis` | PAP-20260729-RESEARCH-L01 conditionals show topology-neutral costs at matched Projection rows and matched PA rows. | Current trace uses blocking event synchronization; a trace-off/deferred comparison is still missing. | Reject if non-intrusive phase measurements leave more than 25% of the ITL gap unexplained or matched-shape topology effects remain material. |

### C01: Fan-in amplification limits 7PA1P scaling

- **Statement:** Multi-PA completion skew, amplified at the Projection fan-in
  join, is the dominant cause of the current 7PA1P latency loss relative to
  6PA2P.
- **Conditions:** Same eight-GPU model, request trace, concurrency, runtime
  mode, and valid participant-count accounting.
- **Status:** `falsified`
- **Paper section:** Background and Motivation; Scheduling and Placement.
- **Supporting evidence:** `PAP-20260729-RESEARCH-L01`.
- **Counterevidence:** The median fan-in spread delta explains only 16.6% of
  the trace-mode ITL gap. Matched-shape Projection and paged-Attention kernels
  are topology-neutral.
- **Falsification condition:** Measured fan-in skew explains less than half of
  the matched E2E ITL gap, or an equal-load intervention does not materially
  narrow the gap.
- **Next test:** Reopen only if non-intrusive matched evidence attributes at
  least half of the ITL gap to completion skew.

### C01 outcome

- **Status:** `falsified`
- **Evidence:** `PAP-20260729-RESEARCH-L01`.
- **Reason:** At C32, trace-mode mean ITL differs by 19.28 ms, while the
  median first-to-last fan-in spread differs by 0.089 ms/layer, or 3.20 ms
  across 36 layers. This is 16.6% of the observed gap and fails the
  pre-registered 50% threshold.
- **Instrumentation caveat:** The trace calls `CUDA Event.synchronize()` on
  both Attention kernel events and Projection ready events. It is diagnostic,
  not formal performance evidence.

### C02: Decode load aggregation, not topology labels, sets ITL

- **Statement:** The 7PA1P ITL penalty is primarily caused by one Projection
  domain forming larger Decode batches; wider fan-in contributes a smaller
  tail term but does not dominate typical latency.
- **Conditions:** Qwen3-8B FP16 eager on eight L20 GPUs, current reduced
  multi-turn workload, matched concurrency, and clean committed code.
- **Status:** `hypothesis`
- **Paper section:** Motivation; System Model; Evaluation.
- **Supporting evidence:** In `PAP-20260729-RESEARCH-L01`, matched Projection
  rows 1--10 are generally within 6% across topologies, and matched PA rows
  1--5 have paged-Attention kernel medians within 5.7%. Aggregate batch shapes
  differ sharply: Projection rows have medians 11 versus 3 and PA rows have
  medians 2 versus 1 for 7PA1P versus 6PA2P.
- **Counterevidence:** The current detailed trace synchronizes CUDA events and
  inflates 7PA1P Attention wall time even when kernel time is unchanged.
- **Falsification condition:** A non-intrusive phase model leaves more than
  25% of the matched ITL gap unexplained, or a material topology-specific
  penalty remains after matching Projection batch, PA rows, and KV load.
- **Next test:** Repeat the C32 point without blocking trace, then collect
  deferred CUDA spans and fit a per-step model over Projection rows, per-PA
  rows/KV, communication, and the slowest completion.

## Entry requirements

Each claim must:

1. be precise enough to falsify;
2. state the workload, model, hardware, topology, and SLO conditions that bound
   it;
3. link supporting and contradictory PAP experiment IDs;
4. distinguish observed correlation from causal evidence;
5. identify the manuscript section and the next test needed for promotion.

Use the following shape when adding an entry:

```markdown
### CXX: Short claim name

- **Statement:**
- **Conditions:**
- **Status:** `hypothesis`
- **Paper section:**
- **Supporting evidence:**
- **Counterevidence:**
- **Falsification condition:**
- **Next test:**
```

Do not promote a claim to `paper-ready` without completing the paper-level
evidence checks in `README.md`.
