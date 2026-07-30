# PAP benchmark and experiment governance

The current runtime and evidence snapshot is summarized in
[`docs/design/pap/status.md`](../../docs/design/pap/status.md). This directory
owns benchmark methodology, tracked conclusions, manifests, and colocated raw
artifacts; it does not define the runtime architecture. PAP experiments use
these project-owned entry points directly, without a repository skill layer.

Same-node PAP/PD NIXL runs are fail-closed on the validated data path:
NIXL 1.3.0, UCX 1.22.0 built with multi-threading, CUDA IPC, and protocol
emulation disabled. Install or verify the repository-local runtime with:

```bash
bash benchmarks/pap/scripts/setup_same_node_nixl.sh install
bash benchmarks/pap/scripts/setup_same_node_nixl.sh verify
```

The launchers also accept `PAP_UCX_PREFIX` and `PAP_NIXL_PLUGIN_DIR` for an
equivalent external build. They reject UCX 1.21, a plugin linked to another
UCX, or enabled software emulation instead of silently measuring TCP.

The canonical [AIPerf lane](aiperf/README.md) provides standardized serving
load generation and metrics alongside project-owned runtime audits. Its
default eight-GPU testbed uses 128 five-turn conversations, randomized input
lengths, and three SLO tiers. It compares PAP 7PA1P/6PA2P, one-way PD
4P4D/6P2D, and an eight-replica fused vLLM pool under the same workload. The
completed compact scan is recorded in
[`PAP-20260725-8GPU-CAPACITY-SCAN`](experiments/PAP-20260725-8GPU-CAPACITY-SCAN/report.md).
The initial C32 results and 7PA1P fan-in analysis are recorded in the
[`PAP-20260725-8GPU-CAPACITY-PILOT`](experiments/PAP-20260725-8GPU-CAPACITY-PILOT/report.md).
The attention-load placement and migration pilot, including the corrected
UCX 1.22/V2 cross-layer transfer status and the selected sparse-migration
policy, is recorded in
[`PAP-20260726-ATTENTION-LOAD-MIGRATION`](experiments/PAP-20260726-ATTENTION-LOAD-MIGRATION/report.md).
Its current history-local Prefill and post-Prefill Decode placement boundary
is validated in
[`PAP-20260727-POST-PREFILL-LATE-BINDING`](experiments/PAP-20260727-POST-PREFILL-LATE-BINDING/report.md).
The current Attention allocator and kernel cold-start tail fixes are recorded
in
[`PAP-20260727-ATTENTION-TAIL-LATENCY`](experiments/PAP-20260727-ATTENTION-TAIL-LATENCY/report.md).
The latest
completed four-GPU eager and piecewise CUDA Graph results remain documented in
[`PAP-20260722-AIPERF-PROJECTION-AUTO`](experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).
The current step/control-overlap regression is recorded in
[`PAP-20260724-STEP-OVERLAP`](experiments/PAP-20260724-STEP-OVERLAP/report.md).
The current 80/12 PAP baseline, low-SM Attention specialization,
TTFT-tail correction, and long-context O100 PAP/PD concurrency scan are
recorded in
[`PAP-20260730-MPS-80-12`](experiments/PAP-20260730-MPS-80-12/report.md).
Its scheduler-overlap predecessor is retained in
[`PAP-20260724-PROJECTION-SCHEDULER-OVERLAP`](experiments/PAP-20260724-PROJECTION-SCHEDULER-OVERLAP/report.md).
The rejected Projection no-async treatment is retained as a negative control in
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
cross-layer KV stride to compare FA2 with PAP Triton launch specializations.
The static-MPS sweep tests 12 and 20 visible SMs by default:

```bash
bash benchmarks/pap/scripts/run_paged_attention_backend_sm_sweep.sh
```

Its split, warp, grouped-head (`BLOCK_H`), stage, sequence-length, placement,
and output settings are exposed as `PAP_ATTENTION_SWEEP_*` environment
variables. Every candidate is checked against FA2 before timing.

The current PAP integration selects the measured low-resource specialization
only when the Attention process exposes at most 20 SMs:

```text
<= 20 SM: split8 / BLOCK_H4 / four warps / one stage
 > 20 SM: split4 / BLOCK_H16 / four warps / two stages
```

On the B3, 17K-context Qwen3-8B shape, this reduces 12-SM kernel latency from
0.4680 to 0.4105 ms and 80/12 end-to-end mean ITL from 48.76 to 45.40 ms.
See the [low-SM experiment report](experiments/PAP-20260730-MPS-80-12/report.md)
for the resource analysis, cross-shape results, and TTFT-tail correction.
The AIPerf runner now uses 80/12 as its default static-MPS allocation. PAP's
own KV lease retains Attention ownership, while the redundant generic NIXL
producer bookkeeping lease expires after one second so KV-unaware Projection
does not pin completed Prefill requests for 30 seconds.

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
at least three repetitions, every request expected by its profile, exact
sampled output lengths, stable conversation owners, zero client/runtime errors,
and passing Attention, decode-token, routing, commit, lease, session-drain,
and MPS audits.
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
