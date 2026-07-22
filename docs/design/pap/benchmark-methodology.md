---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260716-TRITON-72-20-BASELINE
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260715-VLLM-INTEGRATION-BOUNDARY
  - PAP-20260715-ARCHITECTURE-MILESTONE
  - PAP-20260715-RUNTIME-BOUNDARY-E2E
  - PAP-20260715-MODEL-ADAPTER-E2E
  - PAP-20260715-INTEGRATION-E2E
  - PAP-20260701-PD-METHODOLOGY
  - PAP-20260713-PD-THREE-LANE-C4
  - PAP-20260714-P17-PRE-REFACTOR
  - PAP-20260715-P17-POST-REFACTOR
last_validated_commit: e5190a84e37124c893cf66d5b1bb94f9e31dc408
---

# PAP benchmark methodology

## Canonical P17 profile

`benchmarks/pap/profiles/p17_1pa1p.toml` is the single release-gate profile. It
freezes Qwen3-8B FP16, 1PA1P/TP1, GPUs 1/2, same-host `local_fast`, static MPS
72/20, a 16K exact-token initial context, five rounds, C4, and 256 output tokens
per round. The runner reads these fields rather than maintaining a second set
of workload defaults.

```bash
bash benchmarks/pap/scripts/run_p17_1pa1p.sh quick c1
bash benchmarks/pap/scripts/run_p17_1pa1p.sh formal c4
```

Quick C1 is a smoke shape. Only three-repetition C4 is the performance release
gate. Model and corpus roots are local inputs (`PAP_MODEL_ROOT` and
`PAP_CORPUS_ROOT`); tracked profiles and records never store machine-specific
absolute artifact paths.

## Validity before performance

A run is usable only after request completion, token/cache validity, Attention
stats, correctness logs, asynchronous token/KV join, routing, decode commit,
lease, MPS visibility, and zero-session drain all pass. Dirty worktrees,
missing fingerprints, incomplete requests, failed audits, or mixed profiles
cannot be labeled `formal-clean`.

P17 reports R1 TTFT/TPOT separately from steady rounds 2–5. Aggregation uses
request-level samples across repetitions and nearest-rank p90 where applicable.
At each milestone freeze, R1 TTFT, R1 TPOT, steady TTFT, and steady TPOT are
compared with the immediately preceding accepted formal record. A metric over
5% worse is rerun once and, if repeated, blocks the freeze.

## Current formal evidence

`PAP-20260716-TRITON-72-20-BASELINE` validates clean commit
`3a6fe93d11245c1137d3ea6767cd5e27b3e88156`. Three C4 repetitions completed
60/60 requests and 48/48 multi-turn cache transitions with every strict gate
passing and no warnings. Relative to the preceding accepted formal record,
R1 TTFT changed by -10.74%, R1 TPOT by -10.75%, steady TTFT by -1.84%, and
steady TPOT by -18.02%. The raw run remains at
`benchmarks/pap/experiments/PAP-20260716-TRITON-72-20-BASELINE/runs/20260716_3a6fe93d1_p17_mps_72_20_formal/raw/`.

## Four-GPU capacity development lane

The AIPerf capacity lane complements P17; it does not replace the release
gate. It compares PAP 3PA1P with one-way PD 1P3D, 2P2D, and 3P1D on four L20
GPUs. Every point processes the same 32 conversations and 320 requests under
conversation concurrency. Each conversation has ten turns, a randomized 8K
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
