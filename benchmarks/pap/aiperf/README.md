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

## Files

| File | Purpose |
| --- | --- |
| `run_profile.sh` | Invoke one AIPerf profile against an existing endpoint |
| `run_capacity_matrix.sh` | Restart and run PAP/PD/DP capacity points |
| `generate_multiturn_dataset.py` | Generate fixed-turn randomized workloads |
| `build_synthetic_longctx_dataset.py` | Generate long-context workloads |
| `summarize_capacity_run.py` | Validate and summarize one run |
| `summarize_capacity_matrix.py` | Aggregate architecture/concurrency points |

The current default matrix is `dp_8,pd_6p2d,pap_7pa1p`. Removed PAP
multi-Projection topologies and retired custom-client presets are not
runnable lanes.

## Frozen S128 Graph baselines

The canonical PD baselines use the same 128-conversation, 455-request input,
conversation concurrency 32, Qwen3-8B FP16, eight L20 GPUs, 6P2D, a 2K token
budget on both worker roles, `max_num_seqs=256`, and piecewise CUDA Graphs.
Run either lane through the digest- and version-checked wrapper:

```bash
bash benchmarks/pap/scripts/run_s128_graph_baseline.sh pd
bash benchmarks/pap/scripts/run_s128_graph_baseline.sh dynamo
```

The project PD lane is the project conversation-pair proxy with workers from
`.venv-dynamo` vLLM 0.26.0. The Dynamo lane is Dynamo 1.4.1 with its KV-aware
router and the same vLLM 0.26.0 environment. Both enter through the project's
fail-closed same-node NIXL/UCX configuration. The wrapper requires source
SHA-256
`5421e2d4f9868d4b0dc3f36b5a9aa8e256fadfd929dffd789dbb62692591bd9a`
and 455 completed requests; AIPerf expands that source to input SHA-256
`f1da7ff22ef2446ddf9ae5670f28175fadd90fa37af8eba52d1d562fda22cc69`.

The frozen one-run observations are project PD: 7432.34-ms mean TTFT,
60.196-ms mean ITL, and 1.895 requests/s; Dynamo: 7229.00-ms mean TTFT,
61.391-ms mean ITL, and 1.949 requests/s. These are the default comparison
points, not repeated paper-ready estimates. The former project-source vLLM
0.23 PD result is retained only as a version-control observation.

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
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_6p2d,pap_7pa1p \
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

PAP uses `max_model_len=32768`, static 80/12-SM MPS, and a Projection memory
budget of 120% of checkpoint weight bytes per TP rank. PAP's vLLM processes
run eager because PAP owns the separate whole-step CUDA Graph.

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
