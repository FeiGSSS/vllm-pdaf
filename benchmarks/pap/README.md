# PAP benchmark and experiment governance

This directory defines the canonical PAP benchmark profile and the tracked
metadata layer above immutable raw results. It does not move or rewrite raw run
directories.

## Layout

- `profiles/p17_1pa1p.toml` is the only runtime release-gate profile.
- `schemas/` defines versioned run-manifest and experiment-record contracts.
- `registry/runs/` records what happened in a run without copying raw data.
- `registry/experiments/` records hypotheses, evidence, conclusions, decisions,
  and supersede relationships.
- `registry/INDEX.md` is generated from experiment records.
- `import_legacy_run.py` converts a reviewed legacy formal directory into a new
  manifest without writing to the source directory.
- `validate_registry.py` applies schema and cross-record fail-closed checks.

The P17 profile freezes Qwen3-8B FP16, 1PA1P/TP1, same-host `local_fast`,
static MPS with 64 Prefill and 28 Attention SMs, async decode-token delivery,
async Prefill KV import, sealed handoff, Prefill-owned unified KV, and the 16K
five-turn C4 workload with 256 output tokens per turn. xPAyP and cross-host NIXL
remain `preserved-unverified`; they are not runtime release gates in this
milestone.

## Paths and missing metadata

Tracked records never store machine-specific absolute artifact paths. Every
path is represented as:

```json
{
  "root_id": "pap-worktree",
  "relative_path": "test/baseline/pap/results/runs/example/result.json"
}
```

Root IDs are resolved only by the command performing local verification. The
initial registry uses:

| Root ID | Local meaning |
| --- | --- |
| `pap-worktree` | This repository root |
| `model-store` | Local model storage supplied by `--root` |
| `reference-benchmarks` | External benchmark corpus root supplied by `--root` |

Unknown historical metadata is the literal string `missing`. It must not be
omitted or reconstructed by guesswork. `null` and `not-applicable` express a
known absence; they do not mean missing evidence.

## Evidence and decisions

Evidence grades are `formal-clean`, `controlled`, `diagnostic`, `smoke`,
`historical`, and `invalid`. Decisions are `accepted`, `optional`, `rejected`,
`rolled-back`, `superseded`, and `inconclusive`.

A `formal-clean` run must have a full commit, a clean tracked worktree, empty
tracked patches, at least three repetitions, zero failed requests, passing
validity, complete fingerprints, and passing client, cache, Attention stats,
correctness, decode-token join, routing, commit, lease, session-drain, and MPS
gates. The validator rejects a weaker record labeled `formal-clean`.

## Commands

Validate tracked structure and graph consistency:

```bash
.venv/bin/python benchmarks/pap/validate_registry.py
```

Resolve and hash every raw artifact in the initial record:

```bash
.venv/bin/python benchmarks/pap/validate_registry.py \
  --verify-artifacts \
  --root model-store=/data/ssd1/llm-models \
  --root reference-benchmarks=/home/fei/research/PD/refer_codes/vllm/benchmarks
```

Import a reviewed legacy run into a new output file:

```bash
.venv/bin/python benchmarks/pap/import_legacy_run.py RAW_RUN_DIRECTORY \
  --experiment-id PAP-YYYYMMDD-STABLE-ID \
  --run-id stable_run_id \
  --profile-id p17_1pa1p \
  --evidence historical \
  --root pap-worktree="$PWD" \
  --root model-store=/data/ssd1/llm-models \
  --root reference-benchmarks=/path/to/benchmarks \
  --output /tmp/stable_run_id.json
```

The importer reads the raw directory and writes only the requested output. A
human must review the resulting evidence grade, conclusion, metrics, missing
fields, and decision before adding it to the registry.

Regenerate or check the deterministic index:

```bash
.venv/bin/python benchmarks/pap/generate_experiment_index.py \
  --output benchmarks/pap/registry/INDEX.md
.venv/bin/python benchmarks/pap/generate_experiment_index.py \
  --output benchmarks/pap/registry/INDEX.md --check
```
