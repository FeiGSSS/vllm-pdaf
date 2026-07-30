---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260730-RESEARCH-L13
  - PAP-20260729-LONGCTX-O100-CONCURRENCY-SCAN
  - PAP-20260729-RESEARCH-L12
  - PAP-20260729-RESEARCH-L07
  - PAP-20260725-8GPU-CAPACITY-SCAN
  - PAP-20260725-8GPU-CAPACITY-PILOT
  - PAP-20260724-STEP-OVERLAP
  - PAP-20260724-PROJECTION-SCHEDULER-OVERLAP
  - PAP-20260724-SINGLE-PROJECTION-BATCH
  - PAP-20260722-AIPERF-PROJECTION-AUTO
  - PAP-20260722-AIPERF-PA090-EAGER
  - PAP-20260722-AIPERF-CONVERGENCE
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260701-PD-METHODOLOGY
last_validated_commit: 152f64445cd12ffda91ec1d46330c563a36ab475
---

# PAP benchmark methodology

## Canonical AIPerf testbed

`benchmarks/pap/aiperf/run_capacity_matrix.sh` is the single executable
runtime testbed. It fixes Qwen3-8B FP16 on eight L20 GPUs, PAP 7PA1P/6PA2P,
one-way PD 4P4D/6P2D, an eight-replica fused vLLM pool, 128 conversations,
five randomized long-context turns, think/tool delays, conversation
concurrency, three request-level SLOs, and role-specific scheduler limits.

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

Development runs select one topology and one concurrency point with the
documented environment overrides. The standard PAP runtime regression is
6PA2P C32 with one repetition and still completes all 128 conversations and
640 requests. It is preferred over 7PA1P because the latter is near the
Relaxed ITL-tail boundary despite higher raw throughput. A release performance
claim uses the same testbed with three repetitions; it does not introduce a
second client or load shape.

## Validity before performance

A current AIPerf run records only the audits that apply to that architecture
and experiment. Request completion, output validity, routing/ownership, and
session drain must pass; PAP runs additionally record the runtime-specific
joins, MPS, or Graph-capture audits exercised by that run. The archived P17
profile alone retains its historical fixed ten-audit contract. Dirty
worktrees, incomplete requests, failed applicable audits, or mixed profiles
cannot be labeled `formal-clean`.

Missing requests stay in the SLO denominator, and only complete, correct runs
contribute compliant goodput. Performance comparisons use request-level TTFT,
ITL, throughput, goodput, and the tested concurrency envelope.

The initial eight-GPU C32 comparison and the trace-based 7PA1P fan-in analysis
are recorded in the
[capacity pilot](../../../benchmarks/pap/experiments/PAP-20260725-8GPU-CAPACITY-PILOT/report.md).
The default compact scan uses C16/24/32/48, stopping a topology after its first
valid Relaxed failure.

The completed compact scan adds C8/C12/C20 only to resolve the Strict
boundary. Its PAP/PD/fused-DP goodput, raw-throughput, TTFT, and ITL comparison
is recorded in the
[eight-GPU capacity report](../../../benchmarks/pap/experiments/PAP-20260725-8GPU-CAPACITY-SCAN/report.md).

That July 25 scan remains a runtime milestone, not the current fair paper
comparison. The clean corrected-transport O16 confirmation in
[`PAP-20260729-RESEARCH-L07`](../../../benchmarks/pap/experiments/PAP-20260729-RESEARCH-L07/report.md)
is the current short-output comparison. Registered research variants may
change one workload or mechanism dimension while keeping AIPerf concurrency,
request-level SLO accounting, correctness, and provenance gates unchanged:
L12 tests four approximately 10K-token turns with O16 output, and the O100
scan tests three such turns with randomized mean-102-token output. Those
variants do not silently redefine the long-lived regression workload.

## Archived P17 evidence

The former P17 custom client and runner are retired. Its TOML profile remains
marked `archived` only because normalized historical run manifests reference
that profile ID. P17 metrics and conclusions remain unchanged in their dated
experiment records, but they do not define current defaults or release gates.

## Four-GPU capacity lane

The AIPerf lane compares PAP 3PA1P with one-way PD 1P3D, 2P2D, and 3P1D on
four L20 GPUs. Every point processes the same 32 conversations and 320 requests
under conversation concurrency. Each conversation has ten turns, a randomized 8K
initial input, roughly 512 new input tokens on later turns, randomized 16-64
token outputs, and deterministic think/tool delays.

The source-audited scheduler settings are `max_num_seqs=64`, Prefill
`max_num_batched_tokens=16384`, Decode/Projection
`max_num_batched_tokens=64`, default `max_num_partial_prefills=1`, and
`max_model_len=20000`. PAP Prefill and every PD executor use
`gpu_memory_utilization=0.90`. PAP Projection derives its utilization from
120% of checkpoint weight bytes per TP rank and the smallest selected
Projection GPU, rounded upward to four decimals. Projection retains only KV
metadata plus vLLM's null block; it allocates no local KV tensor. The Prefill
setting applies per vLLM executor, so PAP reports must also record physical
PA-GPU headroom for the colocated Attention runtime.
The full parameter rationale and reproducible commands live in the
[AIPerf testbed documentation](../../../benchmarks/pap/aiperf/README.md).

The July 22 four-GPU comparison validates the `0.90` PA/PD baseline together
with automatic Projection sizing at `0.4070`. It covers eager and piecewise
CUDA Graph execution with the same byte-identical dataset. PD 2P2D uses stable
conversation affinity over the complete Cartesian P/D pair set.

Best SLO-compliant goodput from that single-repetition development
scans is:

| Mode | SLO | PAP | Best PD | PAP versus PD |
| --- | --- | ---: | ---: | ---: |
| Eager | Strict | 2.423 req/s | 1.659 req/s | +46.0% |
| Eager | Standard | 3.330 req/s | 1.795 req/s | +85.5% |
| Eager | Relaxed | 4.929 req/s | 2.525 req/s | +95.2% |
| Piecewise Graph | Strict | 2.031 req/s | 1.792 req/s | +13.3% |
| Piecewise Graph | Standard | 3.236 req/s | 1.850 req/s | +75.0% |
| Piecewise Graph | Relaxed | 4.942 req/s | 2.259 req/s | +118.8% |

All included points completed every request and passed routing, output, and
runtime audits. One persistent single-lane PD Graph transport anomaly is
retained as a diagnostic and excluded; its targeted repeat and independent PD
topologies remain in the comparison. These are development results, not a
formal three-repetition claim or the current eight-GPU paper comparison. See
the
[dated runtime milestone report](../../../benchmarks/pap/experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).

## Evidence and decisions

Evidence grades are `formal-clean`, `controlled`, `diagnostic`, `smoke`,
`historical`, and `invalid`. Decisions are `accepted`, `optional`, `rejected`,
`rolled-back`, `superseded`, and `inconclusive`.

Normal performance and trace/diagnostic evidence are never mixed. xPAyP and
cross-host NIXL results remain `preserved-unverified` during this milestone and
must not be described as freshly validated.

## Experiment records

Current and future runs use a versioned run manifest plus experiment JSON under
`benchmarks/pap/experiments/`. Raw results are colocated with their experiment
when owned by this worktree, but remain ignored by Git.

The 44 reviewed legacy experiments and 16 negative results retain their
metrics, conclusions, and raw locations in the historical ledger. A compact
status overlay normalizes their evidence, decision, and successor without
duplicating 60 verbose JSON records. The experiment validator checks complete
ID coverage and the generated index combines both tiers.

Useful commands:

```bash
.venv/bin/python benchmarks/pap/validate_registry.py
.venv/bin/python benchmarks/pap/generate_experiment_index.py \
  --output benchmarks/pap/experiments/INDEX.md --check
```

Raw directories are never moved or rewritten by validation/import tools.
Unknown historical fields remain `missing`; they are not reconstructed from
guesswork.
