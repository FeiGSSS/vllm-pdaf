# PAP NIXL Rank 4 Batch-Plan Notification - 2026-07-07

## Setup

- Code base starts after Rank 3 inline publish default.
- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Topology: `1pa1p`
- Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`
- GPUs: PA/Attention on GPU1, Projection on GPU2
- Workload launcher: `benchmarks/disagg_benchmarks/run_pap_128_testbed.sh`
- Workload: sonnet input length 128, output length 32, 64 prompts,
  request rate 16, 16 warmups
- Trace summary used `max_total_ms=None` for the comparison table.

End-to-end TPOT is noisy because this benchmark is not run with
`--ignore-eos`. The keep decision is based on repeated warmed runs plus the
targeted per-layer trace fields.

## Change

The NIXL mailbox OFFLOAD_EXEC batch control metadata now defaults to a
batch-plan protocol:

- First layer for a decode batch sends metadata version `4`: layer name,
  plan id, request ids, steps, scales, decode token ids, and batch suffix.
- Later layers for the same decode batch send metadata version `5`: layer name
  plus plan id only.
- The attention receiver caches the plan payload after version `4` and
  reconstructs later layer descriptors from version `5`.
- `PAP_NIXL_MAILBOX_BATCH_PLAN=0` can disable the behavior.

This only affects the NIXL mailbox transport. The local IPC path is unchanged.

## Runs

| Variant | Raw run directory |
|---|---|
| Rank 3 baseline A | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_012044` |
| Rank 3 baseline B | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_012228` |
| Batch plan A | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_013730` |
| Batch plan B | `/home/fei/research/PD/test/baseline/pap/results/runs/20260707_013953` |

## Per-Run Table

| Run | Output tokens | Mean TPOT | Median TPOT | `recv_qkv_ms` | Attention `total_ms` | Projection remote total | Projection recv | Task send total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rank 3 A | `2041` | `94.927 ms` | `96.208 ms` | `1.561 ms` | `2.755 ms` | `2.317 ms` | `1.617 ms` | `0.262 ms` |
| Rank 3 B | `2023` | `89.742 ms` | `90.825 ms` | `1.513 ms` | `2.680 ms` | `2.237 ms` | `1.558 ms` | `0.252 ms` |
| Batch plan A | `2023` | `86.177 ms` | `87.054 ms` | `1.473 ms` | `2.605 ms` | `2.179 ms` | `1.495 ms` | `0.225 ms` |
| Batch plan B | `2023` | `88.601 ms` | `89.194 ms` | `1.512 ms` | `2.625 ms` | `2.175 ms` | `1.491 ms` | `0.225 ms` |

## Mean A/B Comparison

| Metric | Rank 3 baseline | Batch plan | Delta |
|---|---:|---:|---:|
| Mean TPOT | `92.335 ms` | `87.389 ms` | `-4.946 ms (-5.36%)` |
| Median TPOT | `93.517 ms` | `88.124 ms` | `-5.392 ms (-5.77%)` |
| Mean TTFT | `364.846 ms` | `345.015 ms` | `-19.831 ms (-5.44%)` |
| `recv_qkv_ms` | `1.537 ms` | `1.492 ms` | `-0.045 ms (-2.91%)` |
| `recv_wait_other_ms` | `1.364 ms` | `1.341 ms` | `-0.023 ms (-1.71%)` |
| Attention `total_ms` | `2.718 ms` | `2.615 ms` | `-0.103 ms (-3.78%)` |
| Attention `metadata_build_ms` | `0.112 ms` | `0.102 ms` | `-0.010 ms (-8.68%)` |
| Projection remote total | `2.277 ms` | `2.177 ms` | `-0.100 ms (-4.40%)` |
| Projection recv | `1.588 ms` | `1.493 ms` | `-0.094 ms (-5.95%)` |
| Projection task send total | `0.257 ms` | `0.225 ms` | `-0.032 ms (-12.42%)` |
| Projection task publish | `0.246 ms` | `0.214 ms` | `-0.032 ms (-13.00%)` |
| Projection task notify | `0.020 ms` | `0.016 ms` | `-0.003 ms (-16.21%)` |

## Decision

Keep and commit the batch-plan protocol as the default NIXL mailbox behavior.

The targeted operation-level win is visible in projection-to-attention task
publication and notification, and the repeated warmed runs also show an
end-to-end TPOT improvement. The benefit is smaller than Rank 1 metadata cache
and Rank 3 inline publish, but it is consistent and the rollback switch remains
available through `PAP_NIXL_MAILBOX_BATCH_PLAN=0`.
