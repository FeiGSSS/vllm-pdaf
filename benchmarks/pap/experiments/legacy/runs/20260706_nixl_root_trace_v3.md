# PAP NIXL Root-Cause Trace Result - 2026-07-06

## Setup

- Model: `/data/ssd1/llm-models/Qwen3-8B`
- Topology: `1pa1p`
- Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`
- Attention backend: FlashAttention varlen paged path, logged as
  `Using FlashAttention version 2`
- Workload: random input length 128, output length 32, 64 prompts,
  request rate 16, max concurrency 64, 16 warmups
- Raw local run directory:
  `/tmp/vllm-pap-root-trace-v3-20260706`

## End-To-End Metrics

| Metric | Median | P99 |
|---|---:|---:|
| TTFT | `721.94 ms` | `1344.38 ms` |
| TPOT | `308.52 ms` | `325.84 ms` |

Additional serving metrics:

| Metric | Value |
|---|---:|
| Successful requests | `64` |
| Request throughput | `4.04 req/s` |
| Output token throughput | `129.34 tok/s` |
| Total token throughput | `646.68 tok/s` |

## Overhead Breakdown

| Level | Overhead | Median | P90 | P99 | Notes |
|---|---:|---:|---:|---:|---|
| Projection engine step | `exec_and_sample_ms` | `293.16 ms` | `715.41 ms` | `777.34 ms` | Dominates steady-state TPOT |
| Projection runner/model forward | `model_forward_ms` | `153.70 ms` | `362.41 ms` | `405.00 ms` | Main per-step forward body |
| Per-layer remote attention | `remote_total_ms` | `3.70 ms` | `8.93 ms` | `15.04 ms` | Repeated across 36 layers |
| Per-layer projection wait | `recv_ms` | `3.56 ms` | `8.60 ms` | `14.88 ms` | Mostly waiting for attention result |
| Attention executor total | `total_ms` | `4.25 ms` | `9.39 ms` | `15.48 ms` | Total attention-side batch cost |
| Attention metadata build | `metadata_build_ms` | `1.84 ms` | `6.39 ms` | `8.14 ms` | Main attention-side compute overhead |
| Attention QKV receive | `recv_qkv_ms` | `1.60 ms` | `2.23 ms` | `6.62 ms` | NIXL/mailbox receive-side cost |
| Attention FA2 kernel | `paged_flash_kernel_ms` | `0.095 ms` | `0.226 ms` | `0.328 ms` | Not the bottleneck |
| Projection send | `send_ms` | `0.145 ms` | `0.320 ms` | `0.353 ms` | Not a primary bottleneck |
| QKV split | `qkv_split_ms` | `0.021 ms` | `0.025 ms` | `0.235 ms` | Negligible |
| Output reshape | `attention_output_reshape_ms` | `0.002 ms` | `0.002 ms` | `0.008 ms` | Negligible |
| Scheduler/update | `sched_ms` / `postprocess_ms` | `<0.2 ms` | `<0.3 ms` | `<0.3 ms` | Negligible |

## Cold-Start And First-Batch Tail

| Metric | Max |
|---|---:|
| First projection runner forward | `2242 ms` |
| First engine step | `2402 ms` |
| Attention `recv_qkv_ms` | `1262 ms` |
| Attention `compute_ms` | `1160 ms` |

This is attributed to cold-start, JIT/cache initialization, and first connection
setup. It inflates TTFT, but it is not representative of steady-state TPOT.

## Root-Cause Summary

The attention kernel itself is not slow: the measured FA2 paged varlen kernel
median is about `0.095 ms`. The steady-state overhead comes from synchronous
per-layer remote attention:

1. Projection sends one attention task per layer.
2. The attention executor receives QKV and rebuilds paged FlashAttention
   metadata for the current batch.
3. Projection waits for the attention result before continuing the layer.
4. The pattern repeats across all 36 decoder layers.

Per-layer remote attention is only a few milliseconds, but multiplying
`~3.7 ms` by 36 layers explains the `150-300 ms` scale observed in projection
forward and engine-step TPOT.
