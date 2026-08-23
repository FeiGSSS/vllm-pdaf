# PAP benchmarks

This directory contains the benchmark and profiling code for the current PAP
runtime. Historical measurements live under `experiments/`; their original
implementation is recovered from Git, not kept as runnable code here.

## Current boundary

The supported PAP benchmark topology is same-host `xPA1P`, TP1, with:

- static 80/12-SM Prefill/Attention MPS partitions;
- CUDA IPC for colocated Prefill--Attention KV sharing;
- NVSHMEM P2P inside the whole-step Projection--Attention CUDA Graph;
- static `conversation_affinity` placement and no PA-to-PA KV relocation;
- AIPerf as the serving workload client.

The primary eight-GPU comparison is PAP 7PA1P, PD 6P2D, and fused DP8.

## Active entry points

| Purpose | Entry point |
| --- | --- |
| One PAP E2E run | `scripts/run_pap_workload.sh` |
| PAP/PD/DP capacity matrix | `aiperf/run_capacity_matrix.sh` |
| AIPerf client wrapper | `aiperf/run_profile.sh` |
| One-run validation and summary | `aiperf/summarize_capacity_run.py` |
| Matrix aggregation | `aiperf/summarize_capacity_matrix.py` |
| PD baseline | `scripts/run_pd_multiturn_topology.sh` |
| DP baseline | `scripts/run_dp_multiturn.sh` |

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
MAX_MODEL_LEN=32768 \
  bash benchmarks/pap/scripts/run_pap_workload.sh
```

Run the current three-architecture matrix with:

```bash
PAP_CAPACITY_ARCHITECTURES=dp_8,pd_6p2d,pap_7pa1p \
  bash benchmarks/pap/aiperf/run_capacity_matrix.sh
```

## Runtime setup

- `scripts/configure_nvshmem.sh` validates the current NVSHMEM runtime.
- `scripts/build_nvshmem_device_bridge.sh` builds the device-side bridge.
- `scripts/configure_same_node_nixl.sh` validates the NIXL/UCX runtime used
  by Prefill KV lifecycle bookkeeping and the PD baseline.
- `scripts/setup_same_node_nixl.sh` builds that local NIXL/UCX environment.

NIXL is not a PAP Attention--Projection transport and is not used for
PA-to-PA request relocation.

## Active diagnostics

- `microbench/nvshmem_gpu_graph.cu`: NVSHMEM Graph protocol.
- `microbench/attention_scaling.py`: production paged-decode Attention.
- `microbench/projection_scaling.py`: Projection-side decode kernels.
- `microbench/prefill_saturation.py`: Prefill saturation.
- `scripts/run_mps_admission_latency.sh` and
  `run_mps_real_prefill_admission.sh`: static-MPS admission controls retained
  for `PAP-20260822-MPS-ADMISSION-MICRO`.
- `tooling/validate_deferred_trace.py`: current deferred-trace audit.
- `tooling/component_gpu_metrics.py`: Nsight Systems component metrics.
- `tooling/nixl_read_write_probe.py`: fail-closed NIXL transfer diagnosis.

Machine-specific and work-in-progress diagnostics may remain untracked until
their experiment boundary is reviewed.

## Evidence and registry

- `experiments/INDEX.md`: experiment index.
- `experiments/HISTORY.md`: historical result ledger.
- `experiments/PAP-*/report.md`: immutable experiment conclusions.
- `experiments/legacy/`: evidence predating the normalized record format.
- `profiles/` and `schemas/`: normalized tracked-run metadata.
- `validate_registry.py`: registry consistency checks.

Historical reports and manifests are evidence, not current launch
instructions. A report's source commit is the authority for reproducing its
retired implementation.
