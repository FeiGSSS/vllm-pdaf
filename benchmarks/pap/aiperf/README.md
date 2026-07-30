# AIPerf benchmark lane

This directory defines the canonical PAP and PD serving testbed. All new
performance, capacity, and runtime E2E comparisons use AIPerf:

- AIPerf owns load scheduling, multi-turn session execution, per-request
  records, TTFT/ITL/latency/throughput metrics, time slices, and sweeps.
- Project-owned launchers wrap AIPerf with output-length, routing, KV handoff,
  correctness-log, decode-token, MPS, and lifecycle-drain audits.

The former P17/custom-client lane is archived evidence and is not a current
runner or release gate. Targeted pytest and one-request smoke checks remain for
source-level diagnosis, but they do not define benchmark performance.

PAP and the PD proxy accept AIPerf's default `X-Correlation-ID` as a
conversation identifier when the request body has no `conversation_id`. A body
value still takes priority. This lets AIPerf carry live assistant responses into
later turns without a custom AIPerf fork.

## Install

Keep AIPerf outside the vLLM environment:

```bash
git clone https://github.com/ai-dynamo/aiperf.git "$AIPERF_ROOT"
cd "$AIPERF_ROOT"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

The first local installation used AIPerf 0.11.0. Every run records the exact
AIPerf commit and version in its artifact directory.

## Randomized eight-GPU capacity testbed

The long-lived comparison workload is pure concurrency, not an arrival-rate
test. AIPerf keeps at most `C` sessions active; a session retains its slot from
its first request through its final turn. No separate Prefill concurrency cap
is applied.

The workload contract fixes the conversation structure while sampling token
lengths reproducibly:

| Field | Distribution |
| --- | --- |
| Turns per session | 5 |
| Total sessions per matrix point | 128 |
| Initial user document | log-normal: mean 4096, median 4000, range 2048-5632 |
| New user text on turns 2-5 | bounded log-normal: parameter mean 1100, median 400, range 4-2125 |
| Assistant output | log-normal: mean 16, median 15, range 8-32 |
| Normal think time | 1 second |
| Tool execution time | 0.3 second every third continuation |
| Maximum model length | 32768 tokens |

AIPerf treats `{mean, stddev}` as a truncated normal distribution and
`{mean, median}` as a right-skewed log-normal distribution. This testbed uses
the latter semantics and then bounds its long tail. Truncation means the
distribution's `mean=1100` parameter is not the resulting arithmetic mean:
the adjacent manifest is authoritative for the actual mean, median, percentiles,
and context headroom. This keeps per-turn append/output length variation
large enough to stress decode scheduling without creating a synthetic fixed-shape
workload.

The deterministic delay schedule is `0,1,1,0.3,1` seconds. The first turn has
no delay; ordinary continuations model user think time, while turn 4 models a
faster external-tool return. Each session waits for 3.3 seconds in total while
retaining its concurrency slot.

The capacity runner generates 128 sessions with this shape:

```bash
.venv/bin/python benchmarks/pap/aiperf/generate_multiturn_dataset.py \
  --model /data/ssd1/llm-models/Qwen3-8B \
  --corpus /path/to/sonnet_4x.txt \
  --sessions 128 --turns 5 \
  --document-tokens 4096 --document-tokens-median 4000 \
  --document-tokens-min 2048 --document-tokens-max 5632 \
  --append-tokens 1100 --append-tokens-median 400 \
  --append-tokens-min 4 --append-tokens-max 2125 \
  --output-tokens 16 --output-tokens-median 15 \
  --output-tokens-min 8 --output-tokens-max 32 \
  --random-seed 42 --sampled-mean-tolerance 0.40 \
  --max-model-len 32768 \
  --think-time-ms 1000 --tool-time-ms 300 --tool-every 3 \
  --output /tmp/pap-aiperf-8gpu-longtail-o16-t5.jsonl
```

The generator gives every session a stable, unique `cache_salt`, preventing
cross-session prefix sharing while preserving reuse across that session's
turns. The capacity runner derives the session prefix from the workload and
seed, never from the run or matrix ID. Each length dimension has an
independently derived random stream, so the same configuration produces a
byte-identical dataset across matrices. The adjacent manifest records the
session prefix, target and sampled statistics, tokenizer-measured text lengths,
cumulative input estimates, and context headroom. Small decode/re-tokenize
boundary differences are expected and recorded; they are not forced away.

For a workload whose cumulative per-session text exceeds the source corpus,
set `PAP_CAPACITY_REPEAT_CORPUS_TO_FIT=1` or pass
`--repeat-corpus-to-fit` to the generator. This opt-in repeats the tokenized
corpus only enough to satisfy the longest sampled session and records the
source length, required length, and repeat count in the dataset manifest.
The default remains fail-closed so an unintended short corpus cannot silently
change a workload.

## SLOs and correctness gate

Each completed request is evaluated against TTFT and its request-level mean
ITL. A tier passes only when at least 95% of all expected requests meet both
limits. Missing requests remain in the denominator. A run is eligible for an
SLO pass/fail result only when all sessions complete every configured turn, every
response exactly matches that request's sampled output length, AIPerf reports
no error/cancellation, runtime audits pass, and both Prefill/PA and
Decode/Projection routing retain one owner per conversation. Incomplete or
invalid runs are reported as `ineligible`, not as ordinary SLO failures.

| Tier | TTFT | ITL | Required good-request fraction |
| --- | ---: | ---: | ---: |
| Strict | <= 5 s | <= 50 ms | >= 95% |
| Standard | <= 10 s | <= 75 ms | >= 95% |
| Relaxed | <= 20 s | <= 100 ms | >= 95% |

AIPerf's native `--goodput` output uses the Standard tier by default. The
per-request records retain enough data to evaluate Strict and Relaxed
thresholds without rerunning inference; those two tiers remain report-level
views rather than separate load sweeps.

AIPerf owns the per-variation records, sweep aggregates, repeated-run
confidence data, and error counts. Project launchers add architecture-level
runtime audits and preserve the AIPerf directory. The
`summarize_capacity_{run,matrix}.py` readers build compact, traceable
cross-architecture tables from those native artifacts.

## Fixed-length historical shapes

The former O256 and partial O128 scans are preserved for diagnostics but are no
longer baselines because every input delta and output had one fixed length. See
the [archive notice](../experiments/legacy/reports/pap-pd-aiperf-fixed-length-preliminary-20260721.md).

## Run against an already-started gateway

```bash
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8gpu-longtail-o32-t5.jsonl \
AIPERF_TARGET_URL=http://127.0.0.1:9460 \
AIPERF_OUTPUT_DIR=/path/to/run/aiperf \
AIPERF_SESSIONS=128 \
AIPERF_CONCURRENCY=16,24,32,48 \
AIPERF_NUM_PROFILE_RUNS=1 \
AIPERF_PARAMETER_SWEEP_COOLDOWN_SECONDS=30 \
AIPERF_TIMING_MODE=concurrency \
AIPERF_REQUEST_RATE= \
  bash benchmarks/pap/aiperf/run_profile.sh
```

The project launchers can start the services and run the same AIPerf dataset:

```bash
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8gpu-longtail-o32-t5.jsonl \
PAP_TOPOLOGY=6pa2p \
PAP_AIPERF_TURNS=5 \
PAP_AIPERF_SESSIONS=128 \
PAP_AIPERF_CONCURRENCY=16,24,32,48 \
PAP_AIPERF_TIMING_MODE=concurrency \
  bash benchmarks/pap/scripts/run_pap_workload.sh

AIPERF_INPUT_FILE=/tmp/pap-aiperf-8gpu-longtail-o32-t5.jsonl \
PD_LOAD_TOPOLOGY=4p4d \
PD_LOAD_ROUNDS=5 \
PD_LOAD_CONVERSATIONS=128 \
PD_AIPERF_CONCURRENCY=16,24,32,48 \
PD_AIPERF_TIMING_MODE=concurrency \
  bash benchmarks/pap/scripts/run_pd_multiturn_topology.sh oneway
```

Supply the normal PAP GPU-list variables alongside the first command. Use the
identical generated file for every PAP, one-way PD, and fused-replica sweep.
These direct launcher examples start the architecture once, then AIPerf runs
every comma-separated concurrency variation and repeated trial.
`--parameter-sweep-same-seed`, sequential dataset sampling, and the fixed
JSONL make request contents, target lengths, delays, session order, and random
choices identical across variations.

The direct multi-point launcher is a steady-state sweep: AIPerf does not
restart vLLM between variations, so allocator state and prefix-cache contents
can carry over. It is useful for diagnostics, but it is not the canonical
cross-concurrency capacity comparison. `run_capacity_matrix.sh` defaults to
one service restart per concurrency point and reuses the byte-identical
dataset. This prevents an earlier point from warming the prefix cache for a
later point. Set `PAP_CAPACITY_RESTART_BETWEEN_POINTS=0` only when explicitly
studying steady-state variation ordering. The canonical capacity sweep uses
`conversation_affinity`.

The default `records` export retains aggregate and per-request metrics without
duplicating every long prompt and response. Set `AIPERF_EXPORT_LEVEL=raw` only
when wire-level debugging is required.

## Run the lean randomized matrix

For a focused eight-GPU DP/PD/PAP goodput campaign, use the checked-in scan
preset:

```bash
PAP_CAPACITY_MATRIX_ID=20260727_dp_pd_pap_goodput_scan \
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_6p2d,pap_6pa2p \
  bash benchmarks/pap/aiperf/run_goodput_scan.sh
```

It runs one-way PD 6P2D, fused DP, and one PAP topology across
topology-specific concurrency points around their known Strict, Standard, and
Relaxed boundaries.
It performs one discovery repetition by default. Override
`PAP_CAPACITY_REPETITIONS=3` only when confirming already selected points, and
override an architecture's point list to avoid repeating the whole discovery
matrix:

```bash
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_6p2d,pap_6pa2p \
PAP_CAPACITY_PAP_6PA2P_POINTS=24 \
PAP_CAPACITY_PD_6P2D_POINTS=24 \
PAP_CAPACITY_DP_8_POINTS=16,20,24 \
PAP_CAPACITY_REPETITIONS=3 \
  bash benchmarks/pap/aiperf/run_goodput_scan.sh

# Example including extra PD/PAP variants:
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_4p4d,pd_6p2d,pd_7p1d,pap_6pa2p,pap_7pa1p \
PAP_CAPACITY_PAP_7PA1P_POINTS=16,20,24 \
PAP_CAPACITY_PAP_6PA2P_POINTS=20,24,28 \
PAP_CAPACITY_PD_4P4D_POINTS=16,20,24 \
PAP_CAPACITY_PD_6P2D_POINTS=20,24 \
PAP_CAPACITY_PD_7P1D_POINTS=16,20,24 \
PAP_CAPACITY_DP_8_POINTS=16,20,24 \
PAP_CAPACITY_REPETITIONS=3 \
  bash benchmarks/pap/aiperf/run_goodput_scan.sh
```

Results are the native AIPerf artifacts below each architecture's `aiperf/`
directory. Single-trial sweeps use `concurrency_<C>/`; repeated sweeps use
`profile_runs/trial_<N>/concurrency_<C>/`. AIPerf writes its cross-variation
aggregate under `sweep_aggregate/` (or `aggregate/sweep_aggregate/` for
repeated runs).

After the sweep, generate the architecture-level capacity/goodput comparison:

```bash
MATRIX_ROOT=benchmarks/pap/experiments/_staging/capacity/${PAP_CAPACITY_MATRIX_ID}
.venv/bin/python benchmarks/pap/aiperf/summarize_capacity_matrix.py "${MATRIX_ROOT}"
```

To aggregate multiple matrix runs (for example, `primary` + `refine` + `final_edges`
when they were launched as separate IDs), pass each root positionally and set one
output root:

```bash
MATRIX_ROOTS=(
  benchmarks/pap/experiments/_staging/capacity/primary
  benchmarks/pap/experiments/_staging/capacity/refine
  benchmarks/pap/experiments/_staging/capacity/final_edges
)
MERGED_ROOT=benchmarks/pap/experiments/_staging/capacity/merged_primary_refine_final
.venv/bin/python benchmarks/pap/aiperf/summarize_capacity_matrix.py "${MATRIX_ROOTS[@]}" --output-root "${MERGED_ROOT}"
```

The default matrix compares the main DP/PD/PAP variants together (depending on the
topologies launched in this run):

- DP: `8dp`
- PD: `4p4d`, `6p2d` (and `7p1d` if included in this run)
- PAP: `6pa2p`, `7pa1p`

All runs use the same Qwen3-8B/8-GPU dataset and random seed. DP and all PD
components use `gpu_memory_utilization=0.90`. Scheduler limits, model, dataset,
and AIPerf settings remain identical across architectures and concurrency points.

The latest completed eager and piecewise four-GPU scans are recorded in
[`PAP-20260722-AIPERF-PROJECTION-AUTO`](../experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).
They validate automatic Projection sizing at `0.4070` for Qwen3-8B TP1 on an
L20 and remain single-repetition controlled development evidence.

To run with local uncommitted worktree changes, set
`PAP_BENCH_REQUIRE_CLEAN_TRACKED_WORKTREE=0` (default remains `1` for
reproducibility checks).

For a single command that runs matrix + summary (recommended for your recurring
SLO comparison), use:

```bash
bash benchmarks/pap/aiperf/run_three_way_slo_capacity.sh
```

If you want to constrain this sweep to a sub-set (for quick smoke), set
architecture and/or per-topology point sets explicitly.

```bash
PAP_CAPACITY_MATRIX_ID=20260728_dp_pd_pap_slo_fullscan \
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_4p4d,pd_6p2d,pd_7p1d,pap_6pa2p,pap_7pa1p \
PAP_CAPACITY_DP_8_POINTS=8,12,16,20,24,28,32 \
PAP_CAPACITY_PD_4P4D_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PD_6P2D_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PD_7P1D_POINTS=12,16,20,24,28,32 \
PAP_CAPACITY_PAP_6PA2P_POINTS=12,16,20,24,28,32,40,48 \
PAP_CAPACITY_PAP_7PA1P_POINTS=12,16,20,24,28,32,40,48 \
PAP_CAPACITY_REPETITIONS=1 \
PAP_CAPACITY_WAIT_FOR_GPUS=1 \
  bash benchmarks/pap/aiperf/run_three_way_slo_capacity.sh
```

To compare variants already in a completed matrix (skip re-running jobs):

```bash
PAP_CAPACITY_MATRIX_ROOT=benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_fullscan_c16_48 \
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_4p4d,pd_6p2d,pap_6pa2p,pap_7pa1p \
PAP_CAPACITY_SKIP_RUN=1 \
PAP_CAPACITY_WAIT_FOR_GPUS=0 \
PAP_CAPACITY_OUTPUT_TOKENS=16 \
PAP_CAPACITY_SKIP_MISMATCH=0 \
bash benchmarks/pap/aiperf/run_three_way_slo_capacity.sh

# If you intentionally want to reuse a matrix that was generated with different
# architecture/points definitions, set this override explicitly:
# PAP_CAPACITY_SKIP_MISMATCH=0
```

You can also directly regenerate a compact one-page three-way summary (contains
best points and full per-concurrency sweep rows):

```bash
.venv/bin/python benchmarks/pap/aiperf/compare_three_way_slo.py \
  benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_fullscan_c16_48
```

The generated file is:

```
benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_fullscan_c16_48/three_way_slo_summary.md
```

Set `PAP_CAPACITY_WAIT_FOR_GPUS=0` if you want to skip the GPU availability
gate and fail fast in non-GPU contexts.

### Capacity-parameter audit

The matrix uses role-specific scheduler limits. These values follow the
current scheduler and model-runner implementation rather than treating a
larger number as automatically safer:

| Role | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| PAP PA / PD Prefill / DP | 256 | 32768 |
| PAP Projection / PD Decode | 256 | 256 |

- AIPerf can have at most 128 live sessions, with one request per live session.
  Therefore 256 sequences cannot be the admission bottleneck.
- vLLM defines `max_num_batched_tokens` as the per-iteration compute-token
  budget and recommends values above 8192 for throughput in many small-model
  regimes. A 32768 Prefill budget here is consistent with the replay workload and
  leaves ample headroom for multi-turn growth.
- Decode and PAP Projection execute at most one new model token per live
  request. KV-connector prompt tokens are externally computed, and PAP
  Projection owns no local prompt slots. A budget of 256 covers every possible
  live session without an oversized Prefill-style dummy profile on decode-only
  workers.
- Concurrent partial Prefill is left at the vLLM default
  `max_num_partial_prefills=1`. On this source revision, setting it above one
  also changes `long_prefill_token_threshold` to four percent of
  `max_model_len`, fragmenting every 8K prompt. It is not a
  neutral capacity increase.
- `max_model_len=32768` retains more than 10K tokens of measured headroom over
  the seed-42 dataset's longest estimated request. Chunked Prefill remains
  enabled.
- The scheduler keeps its default full-input admission check and zero
  watermark. Requests wait when their complete prompt does not fit, without a
  separate reserved fraction or first-chunk over-admission.
- PAP reserves writable unified KV from each request's output limit. The
  environment value follows the 64-token output upper bound; the former
  fixed 512-token reservation no longer reduces PA capacity for this 16-64
  token output distribution.
- `gpu_memory_utilization` is a per-vLLM-executor budget, not an aggregate cap
  for every process on a physical GPU. The PAP Prefill executor now uses
  `0.90`, matching PD and avoiding an artificial KV-capacity disadvantage.
  Attention remains an additional colocated allocation, so every new hardware
  baseline must record startup success and physical-GPU headroom. Projection
  computes `ceil((checkpoint_bytes / TP) * 1.20 / gpu_total_bytes)` and rounds
  utilization upward to four decimals. It retains layer/group metadata and
  one null block but allocates no local KV tensor, so Projection KV capacity is
  not a PAP conversation limit.
- This revised baseline remains eager. CUDA Graph support and its memory budget
  are introduced and measured separately so graph effects are not conflated
  with scheduler-capacity changes.

### Piecewise CUDA Graph lane

Set `PAP_CAPACITY_EXECUTION_MODE=piecewise` to run the same matrix with
piecewise CUDA Graphs on both PAP and PD. The default remains `eager`, so an
existing result never changes execution mode implicitly. PAP keeps NIXL
OFFLOAD_EXEC and Prefill KV publication outside the captured regions; QKV,
MLP, normalization, and other graph-safe model work remain eligible for
capture. Full-model CUDA Graph is intentionally unsupported because replaying
host transport side effects would be incorrect.

Capture sizes are role-specific and describe scheduled tokens, not admitted
sessions. Shapes outside the configured lists execute normally without a
graph; they do not cap `max_num_seqs`, `max_num_batched_tokens`, KV capacity,
or request admission. The eight-GPU capacity baseline remains eager so Graph
coverage is not mixed into the topology comparison.

```bash
PAP_CAPACITY_EXECUTION_MODE=piecewise \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

Every variation replays the same 128 conversations and 640 requests from one
JSONL file. Concurrency only limits the number of live sessions; when one
five-turn session finishes, the next conversation takes its slot. The default
lean scan is:

| Topology | Concurrency points |
| --- | --- |
| PAP 7PA1P / 6PA2P | 16, 24, 32, 48 |
| PD 4P4D / 6P2D | 16, 24, 32, 48 |
| Fused vLLM replica pool ×8 | 16, 24, 32, 48 |

The project runner starts each architecture once. AIPerf owns the inner
concurrency grid, same-seed replay, cooldowns, repeated runs, and aggregate
artifacts. The lower points find Strict and Standard capacity, while C48
brackets the C32/C64 Relaxed boundary observed in the initial eight-GPU pilot.
Set `PAP_CAPACITY_REPETITIONS=3` only for a later confirmation run.

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

For a PAP runtime regression, run one complete canonical point instead of the
whole comparison matrix:

```bash
PAP_CAPACITY_ARCHITECTURES=pap_6pa2p \
PAP_CAPACITY_PAP_6PA2P_POINTS=32 \
PAP_CAPACITY_REPETITIONS=1 \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

This still serves all 128 conversations and 640 randomized requests. It changes
only the number of topology/concurrency points, not the workload definition.

The canonical PA partition is 20 Prefill chunks and 3 Attention chunks
(80/12 visible L20 SMs). The PAP low-resource Attention specialization uses
eight KV splits, `BLOCK_H=4`, four warps, and one stage at this allocation.
Resource-sensitivity experiments may override it
without changing the production default:

```bash
PAP_CAPACITY_PAP_PREFILL_CHUNKS=18 \
PAP_CAPACITY_PAP_ATTENTION_CHUNKS=5 \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

The capacity runner fails closed unless all 23 L20 chunks are assigned and
the live static-MPS audit observes four SMs per chunk.

PAP also sets the standard NIXL producer bookkeeping lease to one second.
Attention safety remains owned by PAP's separate KV lease (300-second TTL
with pressure eviction); the short connector lease prevents KV-unaware
Projection requests from retaining a redundant 30-second producer pin.

The completed compact scan and strict-boundary refinement are recorded in
[`PAP-20260725-8GPU-CAPACITY-SCAN`](../experiments/PAP-20260725-8GPU-CAPACITY-SCAN/report.md).
The initial C32 comparison and trace-based explanation of the 7PA1P ITL tail
remain in
[`PAP-20260725-8GPU-CAPACITY-PILOT`](../experiments/PAP-20260725-8GPU-CAPACITY-PILOT/report.md).
The accepted 80/12 baseline and its fixed-dataset long-context O100
C16--C36 fused-DP8, PAP 7PA1P, and PD 6P2D scan are recorded in
[`PAP-20260730-MPS-80-12`](../experiments/PAP-20260730-MPS-80-12/report.md).
That long-Prefill comparison sets PD Prefill `max_num_seqs=1`; leaving it at
the generic capacity default of 256 is a known scheduling confound for
approximately 10K-token Prefill requests. PD Decode remains at 256.

The output distribution defaults to mean 16, median 15, and range 8-32. The
mean is encoded in the default matrix ID and dataset filename; all four values,
the seed, and the actual sampled statistics are recorded in the matrix and
dataset manifests.

The runner waits in 60-second intervals when GPUs 0-7 are occupied, supports
resuming a matrix ID, and writes `matrix_config.env`, one run directory per
architecture, native AIPerf sweep artifacts, launcher logs, and architecture
runtime audits.
After GPUs first appear idle, it waits another 15 seconds and verifies them
again so a preceding job can finish releasing ports and processes. Invalid
startup or correctness failures are recorded but never treated as an SLO
capacity boundary.

The initial AIPerf integration comparison is recorded in
[`pap-pd-aiperf-four-gpu-results-20260716.md`](../experiments/legacy/reports/pap-pd-aiperf-four-gpu-results-20260716.md).
The historical cohort-sized capacity scan is recorded in
[`pap-pd-aiperf-capacity-results-20260720.md`](../experiments/legacy/reports/pap-pd-aiperf-capacity-results-20260720.md).
The historical think/tool scan is recorded in
[`pap-pd-aiperf-think-tool-results-20260720.md`](../experiments/legacy/reports/pap-pd-aiperf-think-tool-results-20260720.md).
The historical fixed-96-session scan is recorded in
[`PAP-20260720-AIPERF-FIXED96`](../experiments/PAP-20260720-AIPERF-FIXED96/report.md).
It is historical fixed-length evidence rather than the current baseline. The
2026-07-21 fixed-length runs and their replacement decision are recorded in the
[archive notice](../experiments/legacy/reports/pap-pd-aiperf-fixed-length-preliminary-20260721.md).
The current source-audited randomized O32 eager and Graph milestone is
recorded in
[`PAP-20260722-AIPERF-PROJECTION-AUTO`](../experiments/PAP-20260722-AIPERF-PROJECTION-AUTO/report.md).
The preceding eager comparison remains in
[`PAP-20260722-AIPERF-PA090-EAGER`](../experiments/PAP-20260722-AIPERF-PA090-EAGER/report.md).
The superseded eager baseline remains in
[`PAP-20260721-AIPERF-AUDITED-CAPACITY`](../experiments/PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md).
The preceding piecewise CUDA Graph comparison remains in
[`PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH`](../experiments/PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH/report.md).
The earlier superseded predecessor remains in
[`PAP-20260721-AIPERF-RANDOM-O32`](../experiments/PAP-20260721-AIPERF-RANDOM-O32/report.md).
