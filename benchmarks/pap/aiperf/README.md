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

PAP and the four-GPU PD proxy accept AIPerf's default `X-Correlation-ID` as a
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

## Randomized four-GPU capacity testbed

The long-lived comparison workload is pure concurrency, not an arrival-rate
test. AIPerf keeps at most `C` sessions active; a session retains its slot from
its first request through its final turn. No separate Prefill concurrency cap
is applied.

The workload contract fixes the conversation structure while sampling token
lengths reproducibly:

| Field | Distribution |
| --- | --- |
| Turns per session | 10 |
| Total sessions per matrix point | 32 |
| Initial user document | log-normal: mean 8192, median 8000, range 4096-11264 |
| New user text on turns 2-10 | log-normal: mean 512, median 500, range 256-768 |
| Assistant output | log-normal: mean 32, median 30, range 16-64 |
| Normal think time | 3 seconds |
| Tool execution time | 1 second every third continuation |
| Maximum model length | 20000 tokens |

AIPerf treats `{mean, stddev}` as a truncated normal distribution and
`{mean, median}` as a right-skewed log-normal distribution. This testbed uses
the latter semantics explicitly. The manifest reports both parameters and the
sampled arithmetic mean and median; the configured mean must not be confused
with a finite dataset's measured median. Bounds protect the 20K context limit.

The deterministic delay schedule is `0,3,3,1,3,3,1,3,3,1` seconds. The first
turn has no delay; ordinary continuations model user think time, while turns 4,
7, and 10 model a faster external-tool return. Each session waits for 21 seconds
in total while retaining its concurrency slot. Previous capacity-boundary runs
spent 11.5-15.8 seconds per request on average, so these delays desynchronize
sessions without making waiting time dominate serving time.

The capacity runner always generates 32 sessions with this shape:

```bash
.venv/bin/python benchmarks/pap/aiperf/generate_multiturn_dataset.py \
  --model /data/ssd1/llm-models/Qwen3-8B \
  --corpus /path/to/sonnet_4x.txt \
  --sessions 32 \
  --document-tokens 8192 --document-tokens-median 8000 \
  --document-tokens-min 4096 --document-tokens-max 11264 \
  --append-tokens 512 --append-tokens-median 500 \
  --append-tokens-min 256 --append-tokens-max 768 \
  --output-tokens 32 --output-tokens-median 30 \
  --output-tokens-min 16 --output-tokens-max 64 \
  --random-seed 42 --max-model-len 20000 \
  --think-time-ms 3000 --tool-time-ms 1000 --tool-every 3 \
  --output /tmp/pap-aiperf-8k-plus512-random-o32-t10.jsonl
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

## SLOs and correctness gate

Each completed request is evaluated against TTFT and its request-level mean
ITL. A tier passes only when at least 95% of all expected requests meet both
limits. Missing requests remain in the denominator. A run is eligible for an
SLO pass/fail result only when all sessions complete all ten turns, every
response exactly matches that request's sampled output length, AIPerf reports
no error/cancellation, runtime audits pass, and both Prefill/PA and
Decode/Projection routing retain one owner per conversation. Incomplete or
invalid runs are reported as `ineligible`, not as ordinary SLO failures.

| Tier | TTFT | ITL | Required good-request fraction |
| --- | ---: | ---: | ---: |
| Strict | <= 5 s | <= 50 ms | >= 95% |
| Standard | <= 10 s | <= 75 ms | >= 95% |
| Relaxed | <= 20 s | <= 100 ms | >= 95% |

The per-run summary also records an explicit execution state:

| State | Meaning |
| --- | --- |
| `completed` | All expected requests were observed; inspect SLO columns for pass/fail. |
| `early_stopped_slo_impossible` | The run received SIGINT/SIGTERM after the relaxed tier could no longer reach 95%. |
| `incomplete_slo_impossible` | The run ended incomplete after the relaxed tier could no longer reach 95%. |
| `request_timeout` | Request timeouts occurred, but the observed records had not yet made the relaxed tier mathematically impossible. |
| `service_failed` | The launcher failed before producing a complete result. |
| `incomplete` | The workload ended without enough evidence for a more specific state. |

Request error counts, launcher exit code, and the relaxed tier's observed and
maximum bad-request counts remain in `capacity_summary.json`. Matrix tables
show completion as `observed/expected`; partial TTFT and ITL values are
diagnostic only and never contribute eligible goodput.

`summarize_capacity_run.py` emits one compact JSON result per run.
`summarize_capacity_matrix.py` emits a TSV, a Markdown table, and the tested
PAP-versus-best-PD capacity envelope.

## Fixed-length historical shapes

The former O256 and partial O128 scans are preserved for diagnostics but are no
longer baselines because every input delta and output had one fixed length. See
the [archive notice](../experiments/legacy/reports/pap-pd-aiperf-fixed-length-preliminary-20260721.md).

## Run against an already-started gateway

```bash
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8k-plus512-random-o32-t10.jsonl \
AIPERF_TARGET_URL=http://127.0.0.1:9460 \
AIPERF_OUTPUT_DIR=/path/to/run/aiperf \
AIPERF_SESSIONS=32 \
AIPERF_CONCURRENCY=12 \
AIPERF_TIMING_MODE=concurrency \
AIPERF_REQUEST_RATE= \
  bash benchmarks/pap/aiperf/run_profile.sh
```

The project launchers can start the services and run the same AIPerf dataset:

```bash
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8k-plus512-random-o32-t10.jsonl \
PAP_AIPERF_TURNS=10 \
PAP_AIPERF_SESSIONS=32 \
PAP_AIPERF_CONCURRENCY=12 \
PAP_AIPERF_TIMING_MODE=concurrency \
  bash benchmarks/pap/scripts/run_pap_workload.sh

AIPERF_INPUT_FILE=/tmp/pap-aiperf-8k-plus512-random-o32-t10.jsonl \
PD_LOAD_TOPOLOGY=2p2d \
PD_LOAD_ROUNDS=10 \
PD_LOAD_CONVERSATIONS=32 \
PD_AIPERF_CONCURRENCY=10 \
PD_AIPERF_TIMING_MODE=concurrency \
  bash benchmarks/pap/scripts/run_pd_multiturn_topology.sh oneway
```

Supply the normal PAP topology variables alongside the first command. Use the
identical generated file for PAP 3PA1P and PD 1P3D/2P2D/3P1D one-way runs.
Restart the services for every matrix point so each point starts with cold
caches. A comma-separated AIPerf sweep intentionally runs points in one process
and is suitable only when warm-cache carryover is part of the experiment.

The default `records` export retains aggregate and per-request metrics without
duplicating every long prompt and response. Set `AIPERF_EXPORT_LEVEL=raw` only
when wire-level debugging is required.

## Run the lean randomized matrix

The matrix fixes PAP at 3PA1P and PD at one-way 1P3D, 2P2D, and 3P1D. PAP uses
the accepted static 72/20-SM path. Its Prefill executor and every PD executor
use `gpu_memory_utilization=0.90`; PAP Projection remains at `0.76` because it
does not own prompt KV. Scheduler limits, batching, model length, dtype, data,
and AIPerf settings remain unchanged across concurrency points.

This role-specific memory default postdates the published `0.76` PAP scans.
Those reports remain valid historical measurements, but a new PAP/PD baseline
must rerun both sides before making a performance claim.

### Capacity-parameter audit

The matrix uses role-specific scheduler limits. These values follow the
current scheduler and model-runner implementation rather than treating a
larger number as automatically safer:

| Role | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| PAP PA / PD Prefill | 64 | 16384 |
| PAP Projection / PD Decode | 64 | 64 |

- AIPerf can have at most 32 live sessions, with one request per live session.
  Therefore 64 sequences cannot be the admission bottleneck. It also avoids
  sizing persistent request buffers and future default graph-capture ranges
  for 256 requests that this testbed cannot generate.
- vLLM defines `max_num_batched_tokens` as the per-iteration compute-token
  budget and recommends values above 8192 for throughput on small models and
  large GPUs. A 16384 Prefill budget holds roughly two mean 8K prompts.
- Decode and PAP Projection execute at most one new model token per live
  request. KV-connector prompt tokens are externally computed, and PAP
  Projection owns no local prompt slots. A budget of 64 therefore covers the
  32-session testbed while avoiding an 8192-token dummy profile on decode-only
  workers.
- Concurrent partial Prefill is left at the vLLM default
  `max_num_partial_prefills=1`. On this source revision, setting it above one
  also changes `long_prefill_token_threshold` to four percent of
  `max_model_len` (800 tokens here), fragmenting every 8K prompt. It is not a
  neutral capacity increase.
- `max_model_len=20000` retains more than 3.7K tokens of measured headroom over
  this dataset's longest online request. Chunked Prefill remains enabled.
- The scheduler keeps its default full-input admission check and zero
  watermark. Requests wait when their complete prompt does not fit, without a
  separate reserved fraction or first-chunk over-admission.
- PAP reserves writable unified KV from each request's output limit. The
  environment value is only a 64-token compatibility fallback; the former
  fixed 512-token reservation no longer reduces PA capacity for this 16-64
  token output distribution.
- `gpu_memory_utilization` is a per-vLLM-executor budget, not an aggregate cap
  for every process on a physical GPU. The PAP Prefill executor now uses
  `0.90`, matching PD and avoiding an artificial KV-capacity disadvantage.
  Attention remains an additional colocated allocation, so every new hardware
  baseline must record startup success and physical-GPU headroom. Projection
  stays at `0.76`; it allocates no local request KV, so its nominal KV arena is
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
sessions. Prefill captures `1,2,4,8,16,32,64,128`; larger Prefill steps execute
normally without a graph. Projection and PD Decode capture
`1,2,4,8,12,16,20,24,28,32`, matching the maximum 32 live sessions. Falling
outside these sizes selects eager execution for that step; it does not cap
`max_num_seqs`, `max_num_batched_tokens`, KV capacity, or request admission.

```bash
PAP_CAPACITY_EXECUTION_MODE=piecewise \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

Every point processes the same 32 conversations and 320 requests. Concurrency
only limits the number of live sessions; when one ten-turn session finishes,
the next conversation takes its slot. The topology-specific scan retains only
the previously observed useful region:

| Topology | Concurrency points |
| --- | --- |
| PAP 3PA1P | 12, 20, 28, 32 |
| PD 1P3D | 8 |
| PD 2P2D | 10, 16, 20, 24 |
| PD 3P1D | 8, 14, 20 |

Every point restarts all services. Once a valid point fails the relaxed SLO,
higher points for that topology are skipped. This is deliberately a lean
boundary scan; set
`PAP_CAPACITY_REPETITIONS=3` only for a later confirmation run.

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

For a PAP runtime regression, run one complete canonical point instead of the
whole comparison matrix:

```bash
PAP_CAPACITY_ARCHITECTURES=pap_3pa1p \
PAP_CAPACITY_PAP_3PA1P_POINTS=12 \
PAP_CAPACITY_REPETITIONS=1 \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

This still serves all 32 conversations and 320 randomized requests. It changes
only the number of topology/concurrency points, not the workload definition.

The output distribution defaults to mean 32, median 30, and range 16-64. The
mean is encoded in the default matrix ID and dataset filename; all four values,
the seed, and the actual sampled statistics are recorded in the matrix and
dataset manifests.

The runner waits in 60-second intervals when GPUs 0-3 are occupied, supports
resuming a matrix ID, and writes `matrix_config.env`, per-run
`capacity_summary.json`, `capacity_results.tsv`, `capacity_results.md`, and
`capacity_envelope.json` below one matrix directory.
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
The current source-audited randomized O32 baseline is recorded in
[`PAP-20260721-AIPERF-AUDITED-CAPACITY`](../experiments/PAP-20260721-AIPERF-AUDITED-CAPACITY/report.md).
Its matched piecewise CUDA Graph comparison is recorded in
[`PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH`](../experiments/PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH/report.md).
Its superseded predecessor remains in
[`PAP-20260721-AIPERF-RANDOM-O32`](../experiments/PAP-20260721-AIPERF-RANDOM-O32/report.md).
