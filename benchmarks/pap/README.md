# PAP benchmark and experiment governance

This directory defines the canonical PAP benchmark profile and the tracked
metadata layer above immutable raw results. It does not move or rewrite raw run
directories.

## Layout

- `profiles/p17_1pa1p.toml` is the only runtime release-gate profile.
- `tooling/` contains offline trace summaries and remote-Attention diagnostics;
  runtime code under `vllm/pap/` does not import it.
- `schemas/` defines versioned run-manifest and experiment-record contracts.
- `registry/runs/` records what happened in a run without copying raw data.
- `registry/experiments/` records hypotheses, evidence, conclusions, decisions,
  and supersede relationships.
- `registry/history_status.toml` assigns normalized evidence, decision, and
  successor states to every reviewed row in the historical ledger without
  duplicating its metrics, conclusions, or raw paths into dozens of JSON files.
- `registry/INDEX.md` is generated from full records and the compact history
  status overlay.
- `import_legacy_run.py` converts a reviewed legacy formal directory into a new
  manifest without writing to the source directory.
- `validate_registry.py` applies schema and cross-record fail-closed checks.

The P17 profile freezes Qwen3-8B FP16, 1PA1P/TP1, same-host `local_fast`,
static MPS with 64 Prefill and 28 Attention SMs, async decode-token delivery,
async Prefill KV import, sealed handoff, Prefill-owned unified KV, and the 16K
five-turn C4 workload with 256 output tokens per turn. xPAyP and cross-host NIXL
remain `preserved-unverified`; they are not runtime release gates in this
milestone.

Run a one-conversation smoke check or the canonical three-repetition C4 gate:

```bash
bash benchmarks/pap/scripts/run_p17_1pa1p.sh quick c1
bash benchmarks/pap/scripts/run_p17_1pa1p.sh formal c4
```

The runner loads its model, workload, placement, transport, MPS, and audit
values from `profiles/p17_1pa1p.toml`. Machine-specific model and corpus roots
remain local overrides (`PAP_MODEL_ROOT` and `PAP_CORPUS_ROOT`).

### Standalone paged-FlashAttention SM probe

The paged-FA probe reproduces the P17 C4 Attention shape without launching PAP
services or importing code from `vllm/pap/`. It compares full 92-SM execution
with a static 28-SM MPS partition, and compares FA2 auto-split with fixed
single-split. MPS counters come from NSYS GPU-wide sampling; NCU is used only
after the MPS partition is removed because this installed NCU version does not
support reliable profiling under MPS.

```bash
PAP_FA_PROBE_GPU=3 \
PAP_FA_PROBE_RUN_TORCH_TRACE=1 \
PAP_FA_PROBE_OUTPUT_ROOT=/path/to/run \
  bash benchmarks/pap/scripts/run_paged_fa_sm_probe.sh

.venv/bin/python \
  benchmarks/pap/tooling/summarize_paged_fa_sm_probe.py \
  /path/to/run
```

The probe records CUDA-event timing, NVTX-scoped GPU metrics, main-kernel launch
geometry, and raw NSYS/NCU/torch traces. It is a diagnostic experiment, not a
P17 correctness or release gate.

#### Deferred optimization: low-smem FA2 decode specialization

This is a high-value, high-complexity follow-up and is not a current P17 release
blocker. The accepted diagnostic
[`PAP-20260715-PAGED-FA-SM-PROBE`](registry/experiments/PAP-20260715-PAGED-FA-SM-PROBE.json)
shows that the current FA2 kernel uses about 82 KiB of shared memory per CTA,
limiting residency to one CTA per SM. Revisit this work after lower-risk MPS
quota, overlap, and existing-backend optimizations are exhausted, or when FA2
remains on the measured PAP critical path.

The future specialization should be narrow: SM89, BF16, head dimension 128,
paged-KV decode, and the P17 GQA shape. Its acceptance gates are at most 50 KiB
shared memory per CTA, two resident CTAs per SM without a new register limit,
output agreement with the existing FA2 path, lower 28-SM matched-shape latency,
and no material full-92-SM or P17 E2E regression. Keep the existing FA2 kernel
as the default until those gates pass.

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

Full JSON run/experiment records are required for the current milestone and
all new experiments. The 43 pre-milestone experiments and 15 negative results
remain detailed in `docs/design/pap-experiment-history-index.md`; the compact
status overlay makes their coverage and lifecycle machine-checkable. A legacy
row is promoted to a full JSON record only when its raw artifacts are reviewed
and the additional metadata is useful, so migration does not fabricate fields
or duplicate prose.

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

Run offline diagnostics through the stable tool entry points:

```bash
.venv/bin/python tools/pap_trace_summary.py RUN_DIR/service_logs
.venv/bin/python tools/pap_remote_attention_diagnostics.py RUN_DIR
```
