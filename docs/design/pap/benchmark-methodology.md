---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260716-TRITON-72-20-BASELINE
  - PAP-20260715-VLLM-INTEGRATION-BOUNDARY
  - PAP-20260715-ARCHITECTURE-MILESTONE
  - PAP-20260715-RUNTIME-BOUNDARY-E2E
  - PAP-20260715-MODEL-ADAPTER-E2E
  - PAP-20260715-INTEGRATION-E2E
  - PAP-20260701-PD-METHODOLOGY
  - PAP-20260713-PD-THREE-LANE-C4
  - PAP-20260714-P17-PRE-REFACTOR
  - PAP-20260715-P17-POST-REFACTOR
last_validated_commit: 3a6fe93d11245c1137d3ea6767cd5e27b3e88156
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
