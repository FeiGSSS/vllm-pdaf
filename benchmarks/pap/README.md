# PAP benchmarks

This directory contains benchmark and profiling code for the current PAP
runtime. Immutable workload inputs live under `datasets/`. Experiment records
are separated into end-to-end and microbenchmark classes under `experiments/`.
Historical implementations are recovered from Git, not kept as runnable code.

## Current boundary

The supported PAP benchmark topology is same-host `xPA1P`, TP1, with:

- static 80/12-SM Prefill/Attention MPS partitions;
- synchronous v0.26 Prefill scheduling for immediate KV handoff;
- CUDA IPC for colocated Prefill--Attention KV sharing;
- NVSHMEM P2P inside the whole-step Projection--Attention CUDA Graph;
- initial-context-balanced `conversation_affinity` placement, sticky subsequent
  turns, and no PA-to-PA KV relocation;
- Qwen3-8B static YaRN with a 131,072-token serving limit;
- AIPerf as the serving workload client.

The primary eight-GPU comparison is PAP 7PA1P against three baselines behind
the same Dynamo frontend and KV-aware router: eight aggregated workers
(`dp8`), six Prefill plus two Decode workers (`6p2d`), and four Prefill plus
four Decode workers (`4p4d`).

The fixed protocol and first full comparison are recorded in
`experiments/e2e/PAP-20260824-DYNAMO-ARCH-BASELINES/report.md`.

## Active entry points

| Purpose | Entry point |
| --- | --- |
| One PAP E2E run | `scripts/run_pap_workload.sh` |
| AIPerf client wrapper | `scripts/run_aiperf_profile.sh` |
| Dynamo DP8/6P2D/4P4D baseline | `scripts/run_dynamo_workload.sh` |

Formal workload settings belong to each E2E experiment's `experiment.env`.
Shared runners intentionally provide no canonical request rate, concurrency,
or duration.

A targeted PAP run uses environment configuration:

```bash
PAP_TOPOLOGY=7pa1p \
PAP_PREFILL_GPUS=0,1,2,3,4,5,6 \
PAP_PROJECTION_GPUS=7 \
PAP_ROUTING_POLICY=conversation_affinity \
PAP_AIPERF_INPUT_FILE=/path/to/workload.jsonl \
PAP_AIPERF_SESSIONS=128 \
PAP_AIPERF_VARIABLE_TURNS=1 \
PAP_AIPERF_EXPECTED_REQUESTS=455 \
PAP_AIPERF_CONCURRENCY=32 \
MAX_MODEL_LEN=131072 \
  bash benchmarks/pap/scripts/run_pap_workload.sh
```

The direct PAP runner supplies Qwen's official static-YaRN configuration to
both Prefill and Projection by default:

```json
{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}
```

Set `PAP_HF_OVERRIDES=` and an explicit `MAX_MODEL_LEN` to run a native-context
control. Frozen comparison scripts may continue to pin their historical 32K
configuration explicitly.

## Runtime setup

- `scripts/aiperf-requirements.txt` pins the PyPI AIPerf client installed in
  `.venv-aiperf`.
- `scripts/run_aiperf_profile.sh` invokes one client profile through the local
  tokenizer and DCGM compatibility entry.
- `scripts/configure_nvshmem.sh` validates the current NVSHMEM runtime.
- `scripts/build_nvshmem_device_bridge.sh` builds the device-side bridge.
- `scripts/configure_same_node_nixl.sh` validates the NIXL/UCX runtime used
  by Prefill KV lifecycle bookkeeping and the Dynamo 6P2D baseline.
- `scripts/setup_same_node_nixl.sh` builds that local NIXL/UCX environment.
- `scripts/DYNAMO.md` documents the isolated Dynamo baseline environment.

NIXL is not a PAP Attention--Projection transport and is not used for
PA-to-PA request relocation.

## Active diagnostics

- `experiments/microbench/PAP-20260824-ATTENTION-LATENCY-SURFACE/probe.py`:
  production paged-decode Attention probe.
- `tooling/attention_latency_table.py`: exports measured Attention matrices;
  it deliberately performs no latency fitting.
- `experiments/microbench/PAP-20260824-ATTENTION-LATENCY-SURFACE/run.sh`:
  shards the measured Attention matrix across identical 12-SM GPU partitions
  and merges it fail-closed.
- `tooling/validate_deferred_trace.py`: current deferred-trace audit.
- `tooling/component_gpu_metrics.py`: Nsight Systems component metrics.
- `tooling/merge_attention_scaling.py`: validates and merges all measured
  workload/config shards without interpolation.
- `tooling/nixl_read_write_probe.py`: fail-closed NIXL transfer diagnosis.

Machine-specific and work-in-progress diagnostics may remain untracked until
their experiment boundary is reviewed.

## Evidence and registry

- `datasets/`: immutable, checksum-addressed workload inputs.
- `experiments/e2e/`: service-level experiments and AIPerf results.
- `experiments/microbench/`: isolated kernel and component experiments.
- `experiments/*/_runs/`: ignored raw logs, traces, and incomplete attempts.

Historical reports and manifests are evidence, not current launch
instructions. A report's source commit is the authority for reproducing its
retired implementation.
