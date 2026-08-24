# PAP AIPerf benchmark lane

AIPerf is the only current PAP serving-load client. This directory owns
workload generation, the client wrapper, capacity orchestration, and summary
generation.

## Shared client environment

All PAP, PD, DP, and Dynamo launchers use the project-local PyPI environment
at `.venv-aiperf` by default. Build it with `uv`; do not install AIPerf from a
local editable checkout:

```bash
UV_CACHE_DIR=/tmp/uv-cache-aiperf \
  uv venv --python 3.12 --seed .venv-aiperf
UV_CACHE_DIR=/tmp/uv-cache-aiperf \
  uv pip install --python .venv-aiperf/bin/python \
    -r benchmarks/pap/aiperf/requirements.txt
```

`AIPERF_ROOT` still identifies the optional source checkout used to record an
upstream Git commit. `AIPERF_BIN` selects the client executable and defaults
to `.venv-aiperf/bin/aiperf`. Every run records the PyPI package version and
executable in `aiperf_install.env`; a source-checkout commit, when available,
is supplemental provenance and does not describe the installed package.
`run_profile.sh` fails closed unless the executable reports AIPerf 0.11.0.

## Files

| File | Purpose |
| --- | --- |
| `run_profile.sh` | Invoke one AIPerf profile against an existing endpoint |
| `aiperf_compat_entry.py` | Preserve local tokenizer paths in AIPerf 0.11 worker processes |
| `run_agentic_code_profile.sh` | Replay the no-delay Agentic Coding workload at a Poisson request rate |
| `run_capacity_matrix.sh` | Restart and run PAP/Dynamo capacity points |
| `generate_multiturn_dataset.py` | Generate fixed-turn randomized workloads |
| `build_synthetic_longctx_dataset.py` | Generate long-context workloads |
| `summarize_capacity_run.py` | Validate and summarize one run |
| `summarize_capacity_matrix.py` | Aggregate architecture/concurrency points |

The current default matrix is `dynamo_dp8,dynamo_6p2d,pap_7pa1p`. Removed PAP
multi-Projection topologies and retired custom-client presets are not
runnable lanes.

## Default Dynamo baselines

Both non-PAP baselines use Dynamo 1.4.1, official vLLM 0.26.0, one shared
frontend configuration, and the KV-aware router. The only serving-architecture
choice is `dp8` (eight aggregated Prefill+Decode workers) versus `6p2d` (six
Prefill and two Decode workers with same-node NIXL KV transfer):

```bash
DYNAMO_ARCHITECTURE=dp8 \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
DYNAMO_ARCHITECTURE=6p2d \
  bash benchmarks/pap/scripts/run_dynamo_workload.sh
```

The default workload is the fixed 128-session Agentic Coding subset documented
below: request rate 2 turn/s, Poisson arrivals, concurrency 64, no authored
turn delay, Qwen3-8B FP16, 131K static YaRN, and piecewise CUDA Graphs.
The first complete fixed-protocol results are in
`../experiments/PAP-20260824-DYNAMO-ARCH-BASELINES/report.md`.

## Run one PAP workload

The PAP launcher starts all services and then calls `run_profile.sh`:

```bash
PAP_TOPOLOGY=7pa1p \
PAP_PREFILL_GPUS=0,1,2,3,4,5,6 \
PAP_PROJECTION_GPUS=7 \
PAP_ROUTING_POLICY=conversation_affinity \
PAP_AIPERF_INPUT_FILE=/path/to/multiturn.jsonl \
PAP_AIPERF_SESSIONS=128 \
PAP_AIPERF_CONCURRENCY=32 \
  bash benchmarks/pap/scripts/run_pap_workload.sh
```

For a variable-turn file, also set:

```bash
PAP_AIPERF_VARIABLE_TURNS=1
PAP_AIPERF_EXPECTED_REQUESTS=<sum-of-turn-counts>
```

Think/tool delays are encoded in the dataset. Concurrency limits live
sessions; it does not rewrite request lengths or turn order.

## Run the capacity matrix

```bash
PAP_CAPACITY_MATRIX_ID=<name> \
PAP_CAPACITY_ARCHITECTURES=dynamo_dp8,dynamo_6p2d,pap_7pa1p \
PAP_CAPACITY_POINTS=16,24,32,48 \
PAP_CAPACITY_REPETITIONS=1 \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

The matrix runner:

1. generates one byte-identical workload;
2. restarts services between points by default;
3. records commit, effective environment, dataset identity, and topology;
4. runs strict correctness and lifecycle audits;
5. invokes `summarize_capacity_run.py`;
6. leaves aggregate input for `summarize_capacity_matrix.py`.

Use three repetitions only after selecting boundary points. A single run is an
observation, not a paper-ready performance claim.

Aggregate one or more matrices with:

```bash
.venv/bin/python benchmarks/pap/aiperf/summarize_capacity_matrix.py \
  /path/to/matrix-a /path/to/matrix-b \
  --output-root /path/to/merged
```

## Current scheduler baseline

| Role | `max_num_seqs` | `max_num_batched_tokens` |
| --- | ---: | ---: |
| PAP PA Prefill | 256 | 2048 |
| PAP Projection | 256 | 256 |
| PD Prefill | 256 | 2048 |
| PD Decode | 256 | 2048 |
| Fused DP | 256 | 32768 |

The direct PAP runner defaults to Qwen3 static YaRN and
`max_model_len=131072`, static 80/12-SM MPS, and a Projection memory budget of
120% of checkpoint weight bytes plus 512 MiB runtime headroom per TP rank.
Frozen capacity matrices may explicitly retain their historical 32K model
limit. PAP's Projection vLLM process runs without a native outer Graph because
PAP owns the separate whole-step CUDA Graph.

## Required run artifacts

A valid PAP result contains at least:

- `effective_config.env` and `run_metadata.json`;
- AIPerf `profile.json` and `profile.jsonl`;
- `correctness_audit.env`;
- `routing_audit.json`;
- `decode_token_join_audit.env`;
- `projection_scheduling_audit.env`;
- `gateway_drain.env` and `session_drain.env`;
- per-service logs.

The routing audit requires every request's Prefill lease release and Decode
commit to close on the same statically selected PA.

Large raw artifacts may stay outside Git only when their experiment report
records their path, commit, configuration, size, and digest.

## Agentic Coding request-rate replay

Replay 128 deterministic, sequentially sampled conversations with Poisson
arrivals, a session concurrency cap of 64, and all authored inter-turn delays
removed. The default fixed subset keeps complete sessions with 5--32 turns;
it contains 1,630 turns and avoids the original trace's 68-turn drain tail:

```bash
bash benchmarks/pap/aiperf/run_agentic_code_profile.sh 2 128 64
```

The three positional parameters are `request_rate`, `num_conversations`, and
`concurrency`. The endpoint defaults to `http://127.0.0.1:9460`; override it
with `AIPERF_TARGET_URL`. Each run writes its no-delay input, source and input
SHA-256 digests, and effective workload settings under
`aiperf-artifacts/agentic-code/`.
