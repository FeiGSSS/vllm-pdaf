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
| C02 | A non-intrusive phase model explains at least 75% of the 7PA1P versus 6PA2P ITL gap from Decode load aggregation. | Qwen3-8B FP16 eager, eight L20 GPUs, current reduced multi-turn workload, matched concurrency and clean code. | `falsified` | PAP-20260729-RESEARCH-L02: the remote stage explains 88.2% of the complete-forward median difference. | The complete-forward difference explains only 62.6% of the trace-off ITL p50 gap, leaving a scheduler/cadence residual above the registered 25% limit. | Reopen only with request-aligned, non-intrusive evidence that closes the residual. |
| C03 | Fixed-concurrency 7PA1P/6PA2P latency comparisons confound topology with achieved throughput; at about 9.3 req/s, 7PA1P retains materially lower TTFT with comparable mean ITL and SLO goodput. | Same dataset and runtime, 7PA1P C20 versus 6PA2P C32, achieved throughput within 5%, two clean repetitions. | `falsified` | PAP-20260729-RESEARCH-L03: 7PA1P lowers mean TTFT by 60.5% and raises mean ITL by only 6.64% at a 2.55% throughput mismatch. | Standard and relaxed goodput are 2.78% and 2.70% lower, violating the registered no-reduction criterion even though every SLO tier passes. | Reopen only with an operating point that matches throughput within 1.5% and meets a prospectively registered goodput bound. |
| C04 | At an accurately matched achieved-throughput point, 7PA1P preserves its TTFT advantage with a bounded ITL cost and comparable standard/relaxed goodput. | Same L03 dataset and runtime; 7PA1P C21 versus the valid 6PA2P C32 control; two clean repetitions; tracing disabled. | `observed` | PAP-20260729-RESEARCH-L04: throughput differs by 0.59%, TTFT is 61.9% lower, ITL is 9.33% higher, and standard/relaxed goodput differs by -0.20%/+0.19%. | One 7PA1P repetition fails strict at 94.84%; the control is reused rather than contemporaneous. | Repeat treatment and control contemporaneously; reject if the registered standard/relaxed bounds no longer hold. |
| C05 | The fixed 18/5 PA MPS partition is Pareto-suboptimal at the C04 operating point; shifting two chunks from Prefill to Attention reduces ITL while retaining TTFT and goodput advantages. | Same dataset and 7PA1P C21; compare 16/7 against the repeated 18/5 L04 baseline; two clean repetitions; topology and routing unchanged. | `hypothesis` | C04 has 61.9% TTFT headroom but 9.33% worse mean ITL than 6PA2P; L02 identifies remote Attention as the dominant complete-forward difference. | The lower Prefill share can increase queueing or reduce achieved throughput, and more Attention SMs need not remove request-level tails. | Reject if mean ITL improves by less than 5%, mean TTFT is not at least 40% below the 6PA2P control, throughput differs from that control by more than 2%, or standard/relaxed goodput is more than 2% lower. |

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

### C02: Non-intrusive load-aggregation model

- **Statement:** A non-intrusive phase model explains at least 75% of the
  7PA1P versus 6PA2P ITL gap from Decode load aggregation.
- **Conditions:** Qwen3-8B FP16 eager on eight L20 GPUs, current reduced
  multi-turn workload, matched concurrency, and clean committed code.
- **Status:** `falsified`
- **Paper section:** Motivation; System Model; Evaluation.
- **Supporting evidence:** `PAP-20260729-RESEARCH-L02` shows roughly 3x larger
  Projection forwards in 7PA1P. Its remote stage explains 88.2% of the
  complete-forward median difference.
- **Counterevidence:** The complete-forward median difference is 3.49 ms,
  only 62.6% of the trace-off ITL p50 gap. The unexplained 37.4% exceeds the
  registered residual limit.
- **Falsification condition:** A non-intrusive phase model leaves more than
  25% of the matched ITL gap unexplained, or a material topology-specific
  penalty remains after matching Projection batch, PA rows, and KV load.
- **Next test:** Reopen only with request-aligned non-intrusive evidence that
  closes the scheduler/cadence residual.

### C03: Compare PAP topologies on a throughput-latency frontier

- **Statement:** Fixed client concurrency compares different achieved
  throughput points. At about 9.3 req/s, 7PA1P retains materially lower TTFT
  with comparable mean ITL and SLO goodput relative to 6PA2P.
- **Conditions:** Same Qwen3-8B workload and runtime; 7PA1P C20 versus 6PA2P
  C32; achieved request throughput within 5%; two clean repetitions.
- **Paper section:** Evaluation Methodology; End-to-End Performance.
- **Status:** `falsified`
- **Supporting evidence:** `PAP-20260729-RESEARCH-L03` finds 7PA1P mean TTFT
  60.5% lower and mean ITL only 6.64% higher at a 2.55% request-throughput
  mismatch. All three SLO tiers pass in both repetitions.
- **Counterevidence:** 7PA1P standard and relaxed goodput remain 2.78% and
  2.70% lower, violating the registered no-reduction criterion. Good-request
  fractions differ by no more than 0.31 percentage points, so the deficit
  primarily follows the residual throughput mismatch.
- **Falsification condition:** Reject if achieved throughput differs by more
  than 5%, 7PA1P TTFT improves by less than 40%, mean ITL is more than 12%
  worse, or standard/relaxed goodput is lower.
- **Next test:** Reopen only if a prospectively registered, more accurately
  matched operating point satisfies its goodput bound.

### C04: Accurately matched topology operating point

- **Statement:** At an accurately matched achieved-throughput point, 7PA1P
  preserves its TTFT advantage with a bounded ITL cost and comparable
  standard/relaxed goodput relative to 6PA2P.
- **Conditions:** Same Qwen3-8B workload and runtime as L03; 7PA1P C21 versus
  the valid 6PA2P C32 control; two clean repetitions; tracing disabled.
- **Status:** `observed`
- **Paper section:** Evaluation Methodology; End-to-End Performance.
- **Supporting evidence:** `PAP-20260729-RESEARCH-L04` matches throughput
  within 0.59%. 7PA1P lowers mean TTFT by 61.9%, raises mean ITL by 9.33%,
  and changes standard/relaxed goodput by -0.20%/+0.19%.
- **Counterevidence:** One 7PA1P repetition fails strict at a 94.84%
  good-request fraction. The valid 6PA2P control is reused from L03 rather
  than measured contemporaneously.
- **Falsification condition:** Reject if mean achieved throughput differs by
  more than 1.5%, mean TTFT improves by less than 40%, mean ITL is more than
  12% worse, or standard/relaxed goodput is more than 2% lower. If C21 misses
  the throughput range, test at most one adjacent point chosen from the
  direction of the measured mismatch without changing any other threshold.
- **Next test:** Re-run treatment and control contemporaneously before
  promotion to `supported`; first use L05 to test whether the fixed PA
  resource split causes the remaining ITL penalty.

### C05: Rebalance PA Prefill and Attention resources

- **Statement:** The fixed 18/5-chunk PA MPS partition is Pareto-suboptimal at
  the C04 operating point. A 16/7 split reduces the remaining Attention
  latency while preserving PAP's TTFT and standard/relaxed goodput.
- **Conditions:** Same Qwen3-8B dataset and 7PA1P C21 as L04; two clean
  repetitions; topology, routing, execution mode, and Projection resources
  unchanged; 16/7 treatment versus 18/5 baseline.
- **Status:** `hypothesis`
- **Paper section:** Design; Resource Allocation; Sensitivity.
- **Supporting evidence:** C04 shows 61.9% mean TTFT headroom but 9.33% worse
  mean ITL. L02 attributes most of the complete-forward difference to the
  remote-Attention stage.
- **Counterevidence:** Moving SMs away from Prefill can increase queueing and
  reduce throughput. More visible Attention SMs may also leave request-level
  scheduler and tail effects unchanged.
- **Falsification condition:** Reject if mean ITL improves by less than 5%
  against L04, mean TTFT is not at least 40% below the 6PA2P control,
  throughput differs from that control by more than 2%, or standard/relaxed
  goodput is more than 2% lower.
- **Next test:** Add benchmark-level, audited MPS-chunk overrides and run two
  clean 7PA1P C21 repetitions at 16/7 on the byte-identical L04 dataset.

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
