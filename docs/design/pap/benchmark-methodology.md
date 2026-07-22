---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260701-PD-METHODOLOGY
last_validated_commit: e5190a84e37124c893cf66d5b1bb94f9e31dc408
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

A run is usable only after request completion, output-length validity,
conversation ownership, Attention stats, correctness logs, asynchronous
token/KV join, routing, decode commit, lease, MPS visibility, and zero-session
drain all pass. Dirty worktrees,
missing fingerprints, incomplete requests, failed audits, or mixed profiles
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
`max_model_len=20000`. PAP uses `gpu_memory_utilization=0.76`; PD uses `0.90`.
The full parameter rationale and reproducible commands live in the
[AIPerf testbed documentation](../../../benchmarks/pap/aiperf/README.md).

Best SLO-compliant goodput from the current single-repetition development
scans is:

| Mode | SLO | PAP | Best PD | PAP versus PD |
| --- | --- | ---: | ---: | ---: |
| Eager | Strict | 1.944 req/s | 1.761 req/s | +10.4% |
| Eager | Standard | 2.461 req/s | 1.818 req/s | +35.3% |
| Eager | Relaxed | 2.656 req/s | 2.690 req/s | -1.3% |
| Piecewise Graph | Strict | 2.460 req/s | 1.833 req/s | +34.2% |
| Piecewise Graph | Standard | 2.556 req/s | 1.904 req/s | +34.2% |
| Piecewise Graph | Relaxed | 2.777 req/s | 2.591 req/s | +7.2% |

All included points completed every request and passed routing, output, and
runtime audits. The exact percentages are development evidence, not a formal
release claim: each point has one repetition and Graph produced mixed
per-configuration latency changes. See the
[eager report](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md)
and the
[piecewise Graph report](../../../benchmarks/pap/experiments/PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH/report.md).

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
