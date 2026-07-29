# PAP: Prefill-Attention-Projection Disaggregation for LLM Serving

> Manuscript status: scaffold only. No paper claims have been approved.

## Abstract

To be written after the core claims and evidence are established.

## 1. Introduction

### 1.1 Problem

To be established.

### 1.2 Key insight

To be established.

### 1.3 Contributions

To be populated only from paper-ready entries in `claims.md`.

## 2. Background and Motivation

### 2.1 LLM serving and KV-cache ownership

To be established with primary-source citations.

### 2.2 Limits of existing deployment architectures

To be established with primary-source citations and controlled evidence.

### 2.3 Motivation for PAP

Prior attention-disaggregation systems already demonstrate partial Decode
Attention offload and load-aware admission, while recent analytical work
models the slowest-worker barrier in \(rA\)-to-\(1F\) deployments
[liang2025adrenaline; song2026analytical]. Barrier-aware KV-load placement is
also prior art when requests remain sticky on Decode workers
[chen2026universal].

PAP's working research question is therefore not whether Attention can be
disaggregated. It is whether full PA-side ownership of Prefill, Decode
Attention, and KV state can support a stateless, highly batched Projection
tier while using efficient KV migration at natural post-Prefill boundaries to
repair multi-round placement. This position remains a hypothesis: current
evidence has not established that migration benefit exceeds its cost or that
PAP has a repeat-stable advantage over tuned PD and fused deployment.

Current controlled evidence also shows why fixed client concurrency is an
insufficient comparison. On the same C32 workload, 7PA1P executes roughly
three times more rows per Projection forward than 6PA2P. It achieves 28.3%
higher request throughput and 45.1% lower mean TTFT, while mean ITL is 28.9%
higher. Non-blocking CUDA timing attributes most of the complete-forward
median difference to the remote-Attention stage, not dense Projection compute
or local copies (`PAP-20260729-RESEARCH-L02`). The evaluation must therefore
compare throughput-latency frontiers and SLO goodput, not declare a topology
winner at one shared concurrency.

A first iso-throughput localization reinforces that requirement but does not
yet establish a universal winner. The refined point matches request
throughput within 0.59%: 7PA1P C21 reduces mean TTFT by 61.9% and increases
mean ITL by 9.33% relative to 6PA2P C32, while standard and relaxed goodput
differ by only -0.20% and +0.19% (`PAP-20260729-RESEARCH-L04`). This is a
useful latency trade-off, not a strict-SLO result: one 7PA1P repetition has a
94.84% strict-good fraction. The large TTFT headroom and smaller ITL deficit
motivated testing PA Prefill/Attention resource allocation before introducing
a more complex scheduler. That treatment is negative: moving each PA from
72/20 to 64/28 visible Prefill/Attention SMs changes mean ITL by +0.45% while
reducing throughput by 4.98% and increasing TTFT by 18.2%
(`PAP-20260729-RESEARCH-L05`). Static resource repartitioning is therefore not
the missing mechanism for this workload.

The clean O16 confirmation establishes an unfavorable but necessary baseline
(`PAP-20260729-RESEARCH-L07`). Across eight preselected capacity boundaries
and two isolated repetitions per point, all 10,240 requests complete
correctly. Conservative goodput places PAP 32.2%, 36.9%, and 24.4% below
corrected PD under strict, standard, and relaxed SLOs. PAP exceeds fused DP by
57.8% under strict, but loses by 12.1% and 17.2% under standard and relaxed.
Thus PAP currently offers a tight-latency operating point, not a general
goodput advantage on this short-output workload.

This result also makes workload sensitivity a first-order paper question.
Earlier PAP-favorable O32 experiments simultaneously used longer prompts and
longer think/tool delays, so they cannot identify a cause. The next controlled
experiment changes output length alone before the design adopts another
scheduler or transport mechanism.

That output-only experiment moves PAP in the expected direction but does not
explain the reversal (`PAP-20260729-RESEARCH-L08`). At fixed 7PA1P C34 and
6P2D C48 points, doubling the sampled output mean from 16 to 32 reduces PAP
request throughput by 13.2% and PD by 19.7%. The PAP/PD ratio rises from 0.758
to 0.820, a 6.15-percentage-point improvement below the preregistered
10-point threshold. The next ablation isolates input/context length, which
simultaneously increases Prefill work and the memory traffic of each Decode
Attention step.

The input-only ablation provides the first positive workload-region signal
(`PAP-20260729-RESEARCH-L09`). Doubling document and append distributions
while preserving every O16 output and delay raises the fixed-point PAP/PD raw
throughput ratio from 0.758 to 0.870. At the overloaded C34/C48 points, PAP
has 33.3% lower mean TTFT and 46.0% lower mean ITL. Both points fail every SLO,
however, so this remains an observation rather than an end-to-end goodput
claim. A clean concurrency bracket is required to determine whether the
latency headroom survives fair retuning.

The first bracket rejects that extension (`PAP-20260729-RESEARCH-L10`). PAP
and PD both pass standard and relaxed SLOs at C20 and fail at the next tested
C27/C28 points. PAP delivers 5.704/5.844 good req/s under standard/relaxed,
versus 7.418/7.477 for PD, deficits of 23.1% and 21.8%. Thus lower tail
latency in an overloaded regime is not evidence of higher usable capacity.
Because the initial bracket is coarse, intermediate concurrency values must
be tested before this becomes the tuned long-context result.

That refinement cannot be used as performance evidence
(`PAP-20260729-RESEARCH-L11`). PAP C25 exposed an asynchronous output
ownership defect: a sampled token could be published before Scheduler
acceptance with a sequence key derived from already-advanced mutable state.
The corrected path carries a GPU-frame-local key to the Scheduler and
publishes only accepted tokens. Two full C25 diagnostics then complete with
zero sequence mismatches, but the registered comparison spans the defective
runtime and is invalidated rather than merged across commits.

More importantly, the existing input ablations do not isolate PAP's intended
KV-pooling mechanism. L07 uses roughly 4K initial input, while L10's fifth
turn averages only 13.8K input tokens. Startup logs expose 155,424 KV tokens
per PAP PA and 177,504 per PD Decode GPU; 6PA2P therefore pools 932,544 tokens
across six PA owners, versus 355,008 across two Decode owners in 6P2D, a
2.63x ratio. L12 prospectively tests whether this ratio becomes usable
goodput with four approximately 10K-token turns, reaching 37.6K mean final
context without exceeding the model's 40,960-token limit.

## 3. Design

### 3.1 Architecture

To be synthesized from the canonical PAP design documentation.

### 3.2 Scheduling and placement

To be established.

### 3.3 KV-cache lifecycle and migration

To be established.

### 3.4 Communication and synchronization

To be established.

## 4. Implementation

To be synthesized from the validated implementation boundary.

## 5. Evaluation

### 5.1 Methodology

To be linked to normalized experiment records.

### 5.2 End-to-end performance

No paper-ready result is registered.

### 5.3 Causal analysis and ablations

No paper-ready result is registered.

### 5.4 Scalability and generality

No paper-ready result is registered.

### 5.5 Sensitivity and limitations

No paper-ready result is registered.

## 6. Related Work

To be synthesized from `related-work.md` and `references.bib`.

## 7. Discussion and Limitations

To be updated whenever a claim's scope or counterevidence changes.

## 8. Conclusion

To be written after the paper-level completion audit.
