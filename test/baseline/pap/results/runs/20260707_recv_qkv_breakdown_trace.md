# PAP NIXL Receive-QKV Breakdown Trace - 2026-07-07

## Setup

- Code change: propagate NIXL mailbox receive trace into attention
  `recv_qkv_ms` logs and trace summary.
- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Topology: `1pa1p`
- Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`
- GPUs: PA/Attention on GPU1, Projection on GPU2
- Workload launcher: `benchmarks/disagg_benchmarks/run_pap_128_testbed.sh`
- Workload: sonnet input length 128, output length 32, 64 prompts,
  request rate 16, 16 warmups
- Raw external run directory:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_005951`

## End-To-End Metrics

| Metric | Median | P99 |
|---|---:|---:|
| TTFT | `336.21 ms` | `532.67 ms` |
| TPOT | `93.71 ms` | `97.04 ms` |
| ITL | `92.62 ms` | `121.98 ms` |

Additional serving metrics:

| Metric | Value |
|---|---:|
| Successful requests | `64` |
| Failed requests | `0` |
| Request throughput | `8.92 req/s` |
| Output token throughput | `285.44 tok/s` |
| Total token throughput | `1391.50 tok/s` |
| Peak concurrent requests | `62` |

## Attention Receive Breakdown

Trace summary used `max_total_ms=10.0`, matching the default outlier cutoff for
attention trace rows.

| Field | Median | P99 | Meaning |
|---|---:|---:|---|
| `recv_qkv_ms` | `1.515 ms` | `4.396 ms` | Outer attention receive wall time |
| `recv_wait_ms` | `1.413 ms` | `4.212 ms` | Time spent waiting for mailbox message availability |
| `recv_read_ms` | `0.016 ms` | `0.105 ms` | Mailbox receiver read/materialize path |
| `recv_materialize_ms` | `0.012 ms` | `0.102 ms` | Tensor view/copy materialization from receive slot |
| `recv_transfer_ms` | `0.000 ms` | `0.000 ms` | Explicit NIXL read transfer on receive side |
| `recv_wait_other_ms` | `1.393 ms` | `4.188 ms` | Wait not explained by mailbox read/materialize |
| `recv_unaccounted_ms` | `0.091 ms` | `0.287 ms` | Executor wrapper time outside mailbox wait |

## Mailbox Read/Wait Cross-Check

| Direction | Field | Median | P99 |
|---|---|---:|---:|
| Projection to Attention QKV | read total | `0.016 ms` | `0.116 ms` |
| Projection to Attention QKV | materialize | `0.012 ms` | `0.111 ms` |
| Projection to Attention QKV | transfer | `0.000 ms` | `0.000 ms` |
| Projection to Attention QKV | wait | `1.414 ms` | `4.309 ms` |
| Projection to Attention QKV | payload bytes | `196608` | `614400` |
| Attention to Projection output | read total | `0.016 ms` | `0.046 ms` |
| Attention to Projection output | materialize | `0.011 ms` | `0.039 ms` |
| Attention to Projection output | transfer | `0.000 ms` | `0.000 ms` |
| Attention to Projection output | wait | `1.342 ms` | `7.173 ms` |
| Attention to Projection output | payload bytes | `131072` | `409600` |

## Other Attention Fields

| Field | Median | P99 |
|---|---:|---:|
| `metadata_build_ms` | `0.023 ms` | `2.247 ms` |
| `paged_flash_kernel_ms` | `0.095 ms` | `0.310 ms` |
| `compute_ms` | `0.426 ms` | `4.112 ms` |
| `send_output_ms` | `0.018 ms` | `0.031 ms` |
| `total_ms` | `1.996 ms` | `8.041 ms` |

## Decision

Keep and commit Rank 2 tracing. It does not directly reduce TPOT, but it
prevents optimizing the wrong component:

- Median receive-side data handling is only tens of microseconds.
- Receive-side explicit NIXL transfer is `0.000 ms` because this run uses the
  materialized receive-slot path.
- The dominant median component is `recv_wait_other_ms ~= 1.39 ms`, which is
  attention waiting for the next mailbox message after the previous layer output
  is sent.

Next optimization should target the handoff path between projection producing
the next QKV and attention observing the mailbox message: low-latency polling,
notification handling, ACK/slot policy, and possibly reduced per-layer control
metadata.
