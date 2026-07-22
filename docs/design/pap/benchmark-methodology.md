---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260722-AIPERF-PA090-EAGER
  - PAP-20260722-AIPERF-CONVERGENCE
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260701-PD-METHODOLOGY
last_validated_commit: 6d8718aa83aa326960fc65a5ca3270f08fa4a528
---

# PAP benchmark methodology

## Canonical AIPerf testbed

`benchmarks/pap/aiperf/run_capacity_matrix.sh` is the single executable
runtime testbed. It fixes Qwen3-8B FP16 on four L20 GPUs, PAP 3PA1P, one-way PD
1P3D/2P2D/3P1D, 32 conversations, ten randomized long-context turns, think/tool
delays, conversation concurrency, three request-level SLOs, and role-specific
scheduler limits.

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

Development runs select one topology and one concurrency point with the
documented environment overrides. The standard PAP runtime regression is
3PA1P C12 with one repetition; it still completes all 32 conversations and 320
requests. A milestone performance claim uses the same testbed with three
repetitions; it does not introduce a second client or load shape.

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

The latest four-GPU eager comparison uses the `0.90` PA/PD baseline but
predates automatic Projection sizing. It remains controlled performance
evidence; the current memory policy has a separate 128-request 1PA1P E2E and
requires a fresh four-GPU run before it becomes a performance baseline. The
older piecewise CUDA Graph result is historical development evidence.

Best SLO-compliant goodput from the current single-repetition development
scans is:

| Mode | SLO | PAP | Best PD | PAP versus PD |
| --- | --- | ---: | ---: | ---: |
| Recorded eager | Strict | 2.455 req/s | 1.770 req/s | +38.7% |
| Recorded eager | Standard | 3.277 req/s | 1.816 req/s | +80.4% |
| Recorded eager | Relaxed | 5.000 req/s | 2.481 req/s | +101.5% |
| Archived Graph | Strict | 2.460 req/s | 1.833 req/s | +34.2% |
| Archived Graph | Standard | 2.556 req/s | 1.904 req/s | +34.2% |
| Archived Graph | Relaxed | 2.777 req/s | 2.591 req/s | +7.2% |

All included points completed every request and passed routing, output, and
runtime audits. The eager comparison uses the better of two valid PD 3P1D C8
runs because that topology showed extreme transfer variance; the report
preserves both observations. These are development results, not a formal
three-repetition claim. See the
[latest eager report](../../../benchmarks/pap/experiments/PAP-20260722-AIPERF-PA090-EAGER/report.md)
and the
[historical Graph report](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH/report.md).

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
