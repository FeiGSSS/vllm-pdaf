# PAP benchmark and experiment governance

The current runtime and evidence snapshot is summarized in
[`docs/design/pap/status.md`](../../docs/design/pap/status.md). This directory
owns benchmark methodology, tracked conclusions, manifests, and colocated raw
artifacts; it does not define the runtime architecture. PAP experiments use
these project-owned entry points directly, without a repository skill layer.

The canonical [AIPerf lane](aiperf/README.md) provides standardized serving
load generation and metrics alongside project-owned runtime audits. Its
four-GPU testbed uses ten turns, randomized lognormal
lengths around 8K initial input and 512 new user tokens per later turn, plus an
output-length distribution with mean 32 tokens. It uses deterministic 3-second
think and 1-second tool delays, pure conversation concurrency, three
request-level SLO tiers, 32 conversations per point, and a lean
topology-specific boundary scan. The current eager and piecewise CUDA Graph
results are documented together in
[`PAP-20260722-AIPERF-PROJECTION-AUTO`](experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).
The current one-global-batch Projection regression is recorded in
[`PAP-20260724-SINGLE-PROJECTION-BATCH`](experiments/PAP-20260724-SINGLE-PROJECTION-BATCH/report.md).
The initial integration result is documented in
[`pap-pd-aiperf-four-gpu-results-20260716.md`](experiments/legacy/reports/pap-pd-aiperf-four-gpu-results-20260716.md).
The historical cohort-sized capacity scan is documented in
[`pap-pd-aiperf-capacity-results-20260720.md`](experiments/legacy/reports/pap-pd-aiperf-capacity-results-20260720.md).
The corresponding historical think/tool result is documented in
[`pap-pd-aiperf-think-tool-results-20260720.md`](experiments/legacy/reports/pap-pd-aiperf-think-tool-results-20260720.md).
The historical fixed-96-session result is documented in
[`PAP-20260720-AIPERF-FIXED96`](experiments/PAP-20260720-AIPERF-FIXED96/report.md).

The workload launchers default to eager execution. Their optional `piecewise`
mode uses vLLM's token-count CUDA Graph dispatch while keeping PAP transport
and KV-publication side effects outside captured regions. Role-specific graph
sizes cover the scheduled-token shapes expected from the 32-session testbed
and never act as admission limits; uncaptured shapes fall back to normal
execution. See the
[AIPerf lane](aiperf/README.md#piecewise-cuda-graph-lane) for the exact contract.

This directory defines the canonical PAP benchmark contract and experiment
storage. Repository-local raw results are colocated below `experiments/`;
machine-shared historical results stay external until their ownership is
resolved.

## Layout

- `aiperf/run_capacity_matrix.sh` is the executable current testbed contract.
- `profiles/archived/p17_1pa1p.toml` validates archived manifests only;
  it is not runnable and has no release-gate status.
- `tooling/` contains offline trace summaries, remote-Attention diagnostics,
  prefix-cache diagnostics, and deferred-trace validation; runtime code under
  `vllm/pap/` does not import it.
- `schemas/` defines versioned run-manifest and experiment-record contracts.
- `experiments/` colocates each experiment's metadata, conclusion, run
  manifests, and ignored raw artifacts.
- `experiments/history_status.toml` assigns normalized evidence, decision, and
  successor states to every reviewed row in the historical ledger without
  duplicating its metrics, conclusions, or raw paths into dozens of JSON files.
- `experiments/INDEX.md` is generated from full records and the compact history
  status overlay; `experiments/HISTORY.md` retains the detailed ledger.
- `experiments/legacy/README.md` classifies pre-schema reports and the few
  pre-migration raw bundles that remain in their original directories.
- `import_legacy_run.py` converts a reviewed legacy formal directory into a new
  manifest without writing to the source directory.
- `validate_registry.py` applies schema and cross-record fail-closed checks.

The canonical runtime policy uses `0.90` for PAP Prefill and every PD executor.
Projection is independent: `vllm/pap/model/memory.py` derives its budget from
120% of checkpoint weight bytes per TP rank, and the PAP-vLLM integration
plans no physical Projection KV tensors. The latest randomized
32-conversation [milestone](experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md)
validates that policy in eager and piecewise modes. It remains
single-repetition controlled evidence; a release-level claim requires three
repetitions of the same AIPerf testbed.
P17 records and earlier four-GPU reports remain historical evidence only.

Run the complete lean matrix, or select one topology and one concurrency point
through the environment overrides documented in the AIPerf README:

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

### Standalone paged-FlashAttention SM probe

The paged-FA probe reproduces an archived long-context Attention shape without
launching PAP services or importing code from `vllm/pap/`. It compares full
92-SM execution
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
current correctness or release gate.

The backend comparison tool reuses the same archived shape and exact
cross-layer KV stride to compare FA2 with the PAP-owned kernel integration:

```bash
.venv/bin/python \
  benchmarks/pap/tooling/paged_attention_backend_probe.py \
  --triton-splits 4 --expected-sms 92
```

The 2026-07-16 diagnostic used one dirty-worktree repetition, so it is decision
evidence rather than a formal release record:

| Condition | FA2 | PAP Triton split-4 | Max error vs FA2 |
| --- | ---: | ---: | ---: |
| full 92 SM | 0.3511 ms | 0.3313 ms | 1.91e-6 |
| static-MPS 28 SM | 0.5727 ms | 0.3383 ms | 1.91e-6 |

The matching historical C4 quick run reduced steady TPOT from 49.75 ms to 42.47 ms;
the PD control is 41.97 ms. All 20 requests, token digests, cache checks,
lifecycle audits, static-MPS checks, and session drain passed. C1/C4 raw
diagnostics remain machine-local legacy artifacts; the tracked accepted
experiment record is linked above.

#### Deferred alternative: low-smem FA2 decode specialization

This is an archived kernel diagnostic. The accepted diagnostic
[`PAP-20260715-PAGED-FA-SM-PROBE`](experiments/PAP-20260715-PAGED-FA-SM-PROBE/experiment.json)
shows that the current FA2 kernel uses about 82 KiB of shared memory per CTA,
limiting residency to one CTA per SM. PAP instead uses vLLM's existing Triton
paged-decode kernel with four KV splits; matched-shape and C4 E2E measurements
removed the measured FA2-related TPOT gap without adding a new CUDA kernel.
Revisit a low-smem FA2 specialization only if a future shape cannot use the
Triton path or new evidence puts FA2 back on the critical path.

The future specialization should be narrow: SM89, BF16, head dimension 128,
paged-KV decode, and the archived GQA shape. Its acceptance gates are at most 50 KiB
shared memory per CTA, two resident CTAs per SM without a new register limit,
output agreement with the current Triton path, lower 28-SM matched-shape
latency, and no material full-92-SM or matched AIPerf-point regression.

## Paths and missing metadata

Tracked records never store machine-specific absolute artifact paths. Every
path is represented as:

```json
{
  "root_id": "pap-worktree",
  "relative_path": "benchmarks/pap/experiments/PAP-ID/runs/RUN-ID/raw/result.json"
}
```

Root IDs are resolved only by the command performing local verification. The
experiment records use:

| Root ID | Local meaning |
| --- | --- |
| `pap-worktree` | This repository root |
| `model-store` | Local model storage supplied by `--root` |
| `reference-benchmarks` | External benchmark corpus root supplied by `--root` |

Unknown historical metadata is the literal string `missing`. It must not be
omitted or reconstructed by guesswork. `null` and `not-applicable` express a
known absence; they do not mean missing evidence.

An AIPerf run remains in `_staging/` until its matrix config, summaries, raw
artifacts, and reviewed conclusion are promoted into one experiment bundle.
Release-level claims also require normalized JSON run/experiment records. The
44 reviewed historical experiments and 16 negative results remain detailed in
`experiments/HISTORY.md`; the compact status overlay makes their coverage and
lifecycle machine-checkable. A legacy
row is promoted to a full JSON record only when its raw artifacts are reviewed
and the additional metadata is useful, so migration does not fabricate fields
or duplicate prose.

## Evidence and decisions

Evidence grades are `formal-clean`, `controlled`, `diagnostic`, `smoke`,
`historical`, and `invalid`. Decisions are `accepted`, `optional`, `rejected`,
`rolled-back`, `superseded`, and `inconclusive`.

A release-level AIPerf point must have a full commit, a clean tracked worktree,
at least three repetitions, all 320 expected requests, exact sampled output
lengths, stable conversation owners, zero client/runtime errors, and passing
Attention, decode-token, routing, commit, lease, session-drain, and MPS audits.
The archived registry retains the older P17 `formal-clean` contract only to
validate its historical records.

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
fields, and decision before adding it to an experiment bundle.

Regenerate or check the deterministic index:

```bash
.venv/bin/python benchmarks/pap/generate_experiment_index.py \
  --output benchmarks/pap/experiments/INDEX.md
.venv/bin/python benchmarks/pap/generate_experiment_index.py \
  --output benchmarks/pap/experiments/INDEX.md --check
```

Run offline diagnostics through the stable tool entry points:

```bash
.venv/bin/python tools/pap_trace_summary.py RUN_DIR/service_logs
.venv/bin/python tools/pap_remote_attention_diagnostics.py RUN_DIR
```
