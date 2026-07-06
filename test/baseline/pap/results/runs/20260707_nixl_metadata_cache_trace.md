# PAP NIXL Metadata Cache Trace Result - 2026-07-07

## Setup

- Code change: unified paged FlashAttention metadata cache.
- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Topology: `1pa1p`
- Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`
- GPUs: PA/Attention on GPU1, Projection on GPU2
- Workload launcher: `benchmarks/disagg_benchmarks/run_pap_128_testbed.sh`
- Workload: sonnet input length 128, output length 32, 64 prompts,
  request rate 16, 16 warmups
- Raw external run directory:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_003816`

This run is not a strict apples-to-apples replacement for
`20260706_nixl_root_trace_v3.md`, which used the random dataset. It is still a
useful warmed NIXL validation because the model, topology, transport, input and
output lengths, request rate, prompt count, and warmup count match.

## End-To-End Metrics

| Metric | Median | P99 |
|---|---:|---:|
| TTFT | `325.56 ms` | `644.16 ms` |
| TPOT | `93.01 ms` | `96.01 ms` |
| ITL | `89.98 ms` | `124.34 ms` |

Additional serving metrics:

| Metric | Value |
|---|---:|
| Successful requests | `64` |
| Failed requests | `0` |
| Request throughput | `8.76 req/s` |
| Output token throughput | `274.95 tok/s` |
| Total token throughput | `1361.63 tok/s` |
| Peak concurrent requests | `60` |

## Trace Comparison

| Field | 2026-07-06 baseline median | 2026-07-07 cache median | Change |
|---|---:|---:|---:|
| Attention `metadata_build_ms` | `1.84 ms` | `0.023 ms` | `-98.8%` |
| Attention `total_ms` | `4.25 ms` | `1.946 ms` | `-54.2%` |
| Projection `remote_total_ms` | `3.70 ms` | `1.505 ms` | `-59.3%` |
| Projection `recv_ms` | `3.56 ms` | `1.343 ms` | `-62.3%` |
| Attention `recv_qkv_ms` | `1.60 ms` | `1.456 ms` | `-9.0%` |
| Attention FA2 kernel | `0.095 ms` | `0.0955 ms` | unchanged |

## Communication Breakdown

The mailbox trace shows that raw NIXL payload handoff is not the median
`recv_qkv_ms` bottleneck in this run:

| Direction | Field | Median |
|---|---|---:|
| Projection to Attention QKV | mailbox read total | `0.017 ms` |
| Projection to Attention QKV | mailbox materialize | `0.012 ms` |
| Projection to Attention QKV | mailbox transfer | `0.000 ms` |
| Projection to Attention QKV | NIXL write transfer | `0.054 ms` |
| Attention to Projection output | mailbox read total | `0.015 ms` |
| Attention to Projection output | mailbox materialize | `0.010 ms` |
| Attention to Projection output | NIXL write transfer | `0.054 ms` |

`recv_qkv_ms` remains around `1.456 ms`, while mailbox read/materialize is only
tens of microseconds. This supports the earlier caveat: attention-side
`recv_qkv_ms` includes waiting for projection to produce the next layer QKV,
not just NVLink data transfer.

## Decision

Keep and commit Rank 1. It has:

- focused unit-test coverage for metadata cache hit behavior;
- an operation-level microbenchmark showing `1.425653 ms/build` to
  `0.053741 ms/build`;
- a warmed NIXL trace showing `metadata_build_ms` median reduced from
  `1.84 ms/layer` to `0.023 ms/layer`.

Next idea: split `recv_qkv_ms` into projection idle wait, notification decode,
payload materialization, and transfer/sync so the next communication optimization
targets true NVLink/NIXL overhead rather than projection-side work.
