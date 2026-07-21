# AIPerf benchmark lane

This directory adds AIPerf as a standard serving-performance client. It does
not replace the project-owned PAP E2E client:

- AIPerf owns load scheduling, multi-turn session execution, per-request
  records, TTFT/ITL/latency/throughput metrics, time slices, and sweeps.
- The PAP E2E client remains the release gate for exact token continuity,
  cache-hit accounting, routing audits, lifecycle drain, and token correctness.

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

## Fixed four-GPU capacity testbed

The long-lived comparison workload is pure concurrency, not an arrival-rate
test. AIPerf keeps at most `C` sessions active; a session retains its slot from
its first request through its final turn. No separate Prefill concurrency cap
is applied.

The frozen request shape is:

| Field | Value |
| --- | ---: |
| Turns per session | 10 |
| Total sessions per matrix point | 96 |
| Initial user document | 8192 tokens |
| New user text on turns 2-10 | 512 tokens/turn |
| Assistant output | 256 tokens/turn |
| Normal think time | 3 seconds |
| Tool execution time | 1 second every third continuation |
| Maximum model length | 20000 tokens |

Chat-template text and prior 256-token assistant responses also enter the
later-turn context. The requested user-token shape reaches 12,800 tokens; the
complete final prompt remains below the fixed 20K model limit.

The deterministic delay schedule is `0,3,3,1,3,3,1,3,3,1` seconds. The first
turn has no delay; ordinary continuations model user think time, while turns 4,
7, and 10 model a faster external-tool return. Each session waits for 21 seconds
in total while retaining its concurrency slot. Previous capacity-boundary runs
spent 11.5-15.8 seconds per request on average, so these delays desynchronize
sessions without making waiting time dominate serving time.

The capacity runner generates 96 sessions with this shape:

```bash
.venv/bin/python benchmarks/pap/aiperf/generate_multiturn_dataset.py \
  --model /data/ssd1/llm-models/Qwen3-8B \
  --corpus /path/to/sonnet_4x.txt \
  --sessions 96 \
  --think-time-ms 3000 --tool-time-ms 1000 --tool-every 3 \
  --output /tmp/pap-aiperf-8k-plus512-o256-t10.jsonl
```

The generator gives every session a stable, unique `cache_salt`, preventing
cross-session prefix sharing while preserving reuse across that session's
turns. The adjacent manifest records requested and actual text-token counts.
Chat-template overhead and prior assistant outputs are intentionally additional
context and are visible in server-reported input token counts.

## SLOs and correctness gate

Each completed request is evaluated against TTFT and its request-level mean
ITL. A tier passes only when at least 95% of all expected requests meet both
limits. Missing requests remain in the denominator. A run is eligible for an
SLO pass/fail result only when all sessions complete all ten turns with exactly
256 output tokens, AIPerf reports no error/cancellation, runtime audits pass,
and both Prefill/PA and Decode/Projection routing retain one owner per
conversation. Incomplete or invalid runs are reported as `ineligible`, not as
ordinary SLO failures.

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

## Companion short-decode shape

The frozen 256-token output is a sustained-decode capacity workload, not an
unusually large generation by itself. It does, however, combine two effects:
Decode execution lasts long enough to maintain overlap, and prior assistant
outputs add about 2,304 tokens to the final-turn KV footprint.

Use a 128-token output as the first companion diagnostic rather than replacing
the frozen testbed. It halves Decode steps while reducing the final prompt by
only about 1,152 tokens. The 8K initial document and nine 512-token follow-ups
still create strong KV pressure: 3P1D C16 remains beyond one L20's 173,200-token
KV capacity, while C12 moves close to the boundary. Keep all other fields,
delays, SLOs, and the 96-session total unchanged.

The first short-decode comparison should contain only PAP C16/C24 and PD 2P2D
C12/C16. If PAP retains its advantage, the result supports a KV-capacity and
scheduling explanation. If the advantage appears only at 256 output tokens,
the existing result is partly a sustained-Decode throughput advantage. Report
request goodput and output-token goodput within each shape; do not compare raw
token throughput between the 128- and 256-token profiles.

## Run against an already-started gateway

```bash
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8k-plus512-o256-t10.jsonl \
AIPERF_TARGET_URL=http://127.0.0.1:9460 \
AIPERF_OUTPUT_DIR=/path/to/run/aiperf \
AIPERF_SESSIONS=12 \
AIPERF_CONCURRENCY=12 \
AIPERF_TIMING_MODE=concurrency \
AIPERF_REQUEST_RATE= \
  bash benchmarks/pap/aiperf/run_profile.sh
```

The project launchers can start the services and run the same AIPerf dataset:

```bash
PAP_BENCH_CLIENT_MODE=aiperf_multiturn \
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8k-plus512-o256-t10.jsonl \
PAP_MULTITURN_LOAD_CONVERSATIONS=12 \
PAP_AIPERF_TIMING_MODE=concurrency \
  bash benchmarks/pap/scripts/run_pap_workload.sh

PD_LOAD_CLIENT_MODE=aiperf_multiturn \
AIPERF_INPUT_FILE=/tmp/pap-aiperf-8k-plus512-o256-t10.jsonl \
PD_LOAD_TOPOLOGY=2p2d \
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

## Run the lean fixed matrix

The matrix fixes PAP at 3PA1P and PD at one-way 1P3D, 2P2D, and 3P1D. PAP uses
the accepted static 72/20-SM path and `gpu_memory_utilization=0.76`; PD uses
`0.90`. Scheduler limits, batching, model length, dtype, data, and AIPerf
settings remain unchanged across concurrency points. Projection does not own
prompt KV, so increasing only its memory reservation would not increase PAP
session capacity.

Every point processes the same 96 conversations and 960 requests. Concurrency
only limits the number of live sessions; when one ten-turn session finishes,
the next conversation takes its slot. The topology-specific scan retains only
the previously observed useful region:

| Topology | Concurrency points |
| --- | --- |
| PAP 3PA1P | 16, 24, 32 |
| PD 1P3D | 8 |
| PD 2P2D | 12, 16 |
| PD 3P1D | 4, 8, 12, 16 |

Every point restarts all services. This is deliberately a lean boundary scan;
set
`PAP_CAPACITY_REPETITIONS=3` only for a later confirmation run.

```bash
bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

The output length defaults to 256 tokens. Set
`PAP_CAPACITY_OUTPUT_TOKENS=128` for the companion short-decode testbed; the
value is encoded in the default matrix ID, dataset filename, and matrix
configuration.

The runner waits in 60-second intervals when GPUs 0-3 are occupied, supports
resuming a matrix ID, and writes `matrix_config.env`, per-run
`capacity_summary.json`, `capacity_results.tsv`, `capacity_results.md`, and
`capacity_envelope.json` below one matrix directory.

The initial AIPerf integration comparison is recorded in
[`pap-pd-aiperf-four-gpu-results-20260716.md`](../experiments/legacy/reports/pap-pd-aiperf-four-gpu-results-20260716.md).
The historical cohort-sized capacity scan is recorded in
[`pap-pd-aiperf-capacity-results-20260720.md`](../experiments/legacy/reports/pap-pd-aiperf-capacity-results-20260720.md).
The historical think/tool scan is recorded in
[`pap-pd-aiperf-think-tool-results-20260720.md`](../experiments/legacy/reports/pap-pd-aiperf-think-tool-results-20260720.md).
The current fixed-96-session scan is recorded in
[`PAP-20260720-AIPERF-FIXED96`](../experiments/PAP-20260720-AIPERF-FIXED96/report.md).
