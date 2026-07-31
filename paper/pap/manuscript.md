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

The paper's working motivation considers a fixed GPU budget, long-context
multi-turn traffic, and a Prefill-heavy PD allocation with \(N_P > N_D\).
It is organized around four structural observations. These observations define
the mechanism to test; they are not yet all paper-ready empirical claims.

First, PD concentrates both Decode KV capacity and Decode Attention bandwidth
on the \(N_D\) Decode GPUs. Ignoring runtime reservations, the aggregate
resident-KV budget is approximately \(N_D C_{\mathrm{KV}}\), while the
aggregate HBM bandwidth available to long-context Decode Attention is
approximately \(N_D B_{\mathrm{HBM}}\). When the active KV working set exceeds
the former, requests must queue, evict or recompute state, or fail admission,
which can reduce goodput and increase TTFT. When long-context Attention
saturates the latter, its KV reads increase TPOT/ITL. This mechanism is
conditional: it does not imply that every workload reaches either limit, and
our evaluation must distinguish an actual KV-capacity wall from Prefill
compute, scheduler, and transport bottlenecks.

Second, Prefill and long-context Decode Attention stress complementary hardware
resources. Prefill is dominated by high-arithmetic-intensity dense operations,
whereas Decode Attention performs comparatively little computation per byte of
KV state read. A Prefill-only GPU therefore leaves memory-bandwidth headroom,
while an Attention-dominated Decode phase leaves compute headroom. Treating
entire P and D GPUs as phase-exclusive resource bundles can strand one resource
while the other is saturated. This statement applies to the relevant phases
and operators; Decode still contains compute-intensive projections and MLPs.

Third, conventional PD crosses the ownership boundary by moving prompt KV from
Prefill to Decode. Long-context multi-turn requests generate a large new KV
segment on every turn, so this handoff can become a material bandwidth,
latency, and software-progress cost. One-way implementations can avoid moving
Decode-generated KV back immediately, but then a later Prefill must recover
that state through retention, transfer, or recomputation. Systems such as
Mooncake and LMCache motivate treating KV movement and placement as first-class
serving concerns; final manuscript wording requires primary citations and a
mechanism-matched comparison.

Fourth, low-latency multi-turn reuse creates a retain-or-rebuild dilemma. After
Prefill KV is copied to Decode, keeping the Prefill-side common prefix makes
the next turn cheap but duplicates that prefix in two HBM pools while Decode
runs. Releasing it saves memory but makes the next turn pay for transfer,
reload, or recomputation. The defensible claim is therefore conditional
common-prefix duplication, not that every PD implementation always retains two
complete KV copies.

### 2.3 Motivation for PAP

PAP's proposed response is to replace the P/D KV-ownership boundary with
Prefill--Attention ownership. Each PA GPU retains one paged KV copy and uses it
for both future Prefill and Decode Attention, while a KV-stateless Projection
tier executes the dense Decode operators. Under a fixed GPU budget, this can
make the KV capacity and HBM bandwidth of all PA GPUs available to Decode
Attention instead of concentrating them on the smaller D subset. It also
removes the bulk Prefill-to-Decode KV handoff and the associated
retain-or-rebuild dilemma.

This reorganization does not make communication disappear. PAP replaces bulk
phase-boundary KV movement with QKV fan-out and Attention-output fan-in on
every model layer. Its viability therefore depends on low-overhead,
step-planned communication, efficient low-SM Attention, and scheduling that
does not amplify PA skew. Same-host `local_fast` addresses the first part by
preparing route ranges, peer addresses, byte counts, and layer generations
once per Decode step, then issuing one batched transfer per layer. Extending
the same contract across hosts is an explicit open implementation and
generality task, not a capability claimed by the current evaluation.

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

An earlier trace suggested that the wider 7PA1P fan-in synchronization domain
might itself dominate this ITL difference. A matched attribution rejects that
interpretation (`PAP-20260729-RESEARCH-L01`): the median PA-completion-spread
delta contributes only 3.20 ms per 36-layer step, or 16.6% of the trace-mode
ITL gap. Fan-in skew remains measurable, but synchronization-domain routing is
not currently a supported headline mechanism.

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
(`PAP-20260729-RESEARCH-L07`). It also retires the earlier large PAP-over-PD
headline. The historical PD path was not a healthy architecture baseline:
an old pull GET had fallen back to approximately 0.42 GiB/s TCP emulation
instead of the 22--24.5 GiB/s CUDA-IPC path, and later four-GPU scans still
recorded severe per-lane transfer instability. Those results remain useful
transport diagnostics but cannot support a PAP performance claim.

Across eight preselected L07 capacity boundaries and two isolated repetitions
per point, all 10,240 requests complete correctly. Conservative goodput places
PAP 32.2%, 36.9%, and 24.4% below corrected PD under strict, standard, and
relaxed SLOs. PAP exceeds fused DP by 57.8% under strict, but loses by 12.1%
and 17.2% under standard and relaxed. Thus PAP currently offers a
tight-latency operating point, not a general goodput advantage on this
short-output workload. Because the historical and L07 studies also differ in
workload, topology, and search envelope, the old headline is classified as
baseline-contaminated rather than assigning its entire numerical reversal to
transport alone.

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

That test rejects KV pooling as a sufficient mechanism
(`PAP-20260729-RESEARCH-L12`). On the valid 48-session discovery workload,
both 6P2D and 6PA2P pass Standard at C12 and fail at C16. At C12, PAP
Standard goodput is 1.502 req/s versus 2.290 req/s for PD, a 34.4% deficit.
At C16, PAP retains 13.1% lower mean ITL but has 41.4% higher mean TTFT.
Thus PAP does not exhaust its aggregate KV pool; it first loses the SLO to
near-limit Prefill work while every PA statically reserves 20 of 92 visible
SMs for Attention.

The first registered 20/3 Prefill/Attention treatment cannot answer that
question (`PAP-20260730-RESEARCH-L13`). It verifies 80/12 visible SMs and
completes all 192 requests, but the launcher fails correctness because three
lease-release acknowledgements are missing and several decode commits carry
one fewer token than their requested sequence-length advance. Its latency and
throughput numbers are diagnostic only. Because subsequent submit-only
control-path work changes more than the MPS split, any retained implementation
must be measured with a contemporaneous 18/5 control on the same clean commit.

A separate O100 scan identifies a more promising but bounded workload region
(`PAP-20260729-LONGCTX-O100-CONCURRENCY-SCAN`). With 60 sessions, three
approximately 10K-token turns, randomized mean-102-token outputs, and pure
concurrency, 7PA1P C20 passes Standard at 1.611 good req/s while 6P2D C20
fails at 1.337 good req/s, a 20.5% difference. PAP also has 27.5% lower mean
ITL and 16.7% lower TTFT p95. This is not monotonic scaling: both systems fail
Standard at C24, and PD retains the best passing Relaxed goodput by 0.8%.
PD logs show no explicit KV-pressure event, while 6PA2P fails Standard at the
same C20 point. The evidence therefore supports a narrow 7PA1P operating
region and motivates testing PA Prefill capacity, not a general KV-capacity
wall or universal PAP advantage.

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
