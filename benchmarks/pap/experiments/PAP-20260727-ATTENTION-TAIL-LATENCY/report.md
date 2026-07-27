# PAP Attention tail-latency convergence

> Controlled 7PA1P development validation. Raw AIPerf and deferred-trace
> artifacts remain machine-local under `experiments/_staging/`.

Date: 2026-07-27

## Decision

Keep the three mechanical tail-latency fixes:

1. Reuse paged-decode workspaces instead of allocating scratch per step.
2. Reuse pinned-host/device buffers for slot plans, sequence lengths, and block
   tables instead of constructing CUDA tensors on the request path.
3. Compile the Triton paged-decode specialization in a background stream when
   the first Prefill KV catalog layer is registered.

These changes remove the identified allocator/compilation stalls without
changing routing, batching, or Attention results. They improve ITL tail
latency, but they do not eliminate the separate multi-PA join tail.

## Workload

- Qwen3-8B FP16, eager execution, 7PA1P on eight NVIDIA L20 GPUs.
- 128 conversations, five turns, 640 requests, concurrency 32.
- Long-tail randomized multi-turn input, randomized 16-64-token output.
- `max_model_len=32768`, PA memory utilization 0.90.
- Attention-load routing with the validated sparse-migration settings.
- Deferred CUDA tracing enabled for all compared full runs.

## Root causes

The phase trace separated the old `attention_step_prepare` span into context
lookup, lock wait, slot-plan construction, metadata construction, workspace
allocation, and event creation.

- Registry lookup and lock wait were small.
- Direct CUDA tensor construction for slot plans and block tables could block
  the host for hundreds of milliseconds to seconds under load.
- After those allocations were removed, every PA still had exactly one
  218-233 ms `paged_fa_gpu_ms` sample. This was the first Triton specialization
  compilation occurring between the CUDA event submissions.

The final path builds reusable buffers once and updates them with nonblocking
host-to-device copies. The kernel warmup uses the real cross-layer KV-cache
view, so its dtype, page layout, and strides match decode.

## End-to-end result

Both full runs completed 640/640 requests with zero AIPerf errors and drained
all seven Attention sessions.

| Metric | Before tail fixes | Final | Change |
| --- | ---: | ---: | ---: |
| Request throughput | 5.143 req/s | 5.078 req/s | -1.3% |
| Output throughput | 166.626 tok/s | 164.520 tok/s | -1.3% |
| TTFT average | 1,904.81 ms | 1,881.69 ms | -1.2% |
| TTFT P95 | 7,404.49 ms | 7,428.13 ms | +0.3% |
| ITL average | 49.08 ms | 45.51 ms | -7.3% |
| ITL P95 | 95.35 ms | 78.07 ms | -18.1% |
| ITL P99 | 183.65 ms | 129.17 ms | -29.7% |

This is a tail-latency improvement, not a throughput claim. The single final
run's 1.3% throughput difference is within the variance observed in adjacent
7PA1P runs and should be resolved by repetitions in the capacity scan.

## Trace result

The reusable-buffer full run is the immediate pre-warmup comparison.

| Deferred trace | Reusable buffers only | Final |
| --- | ---: | ---: |
| Paged FA maximum | 226.91 ms | 25.45 ms |
| Paged FA samples >=100 ms | 7 | 0 |
| Step prepare maximum | 52.05 ms | 16.08 ms |
| Step prepare samples >=100 ms | 0 | 0 |
| Projection join P50 | 0.399 ms | 0.397 ms |
| Projection join P99 | 0.954 ms | 0.992 ms |
| Projection join samples >=100 ms | 29 | 21 |
| Projection join samples >=1 s | 6 | 6 |

The remaining second-scale join samples are therefore not caused by paged
Attention execution or step-metadata allocation. They need separate
attribution to Prefill/KV readiness, migration, and PA scheduling before any
new optimization is justified.

## Validation

```text
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_kernels.py \
  tests/pap/test_pap_attention_runtime.py \
  tests/pap/test_pap_deferred_cuda_trace.py \
  tests/benchmarks/pap/test_validate_deferred_trace.py -q

88 passed
```

`git diff --check` passed.

Raw artifacts:

```text
benchmarks/pap/experiments/_staging/capacity/
  20260727_7pa1p_c32_tail_deferred_full_direct_v2/
  20260727_7pa1p_c32_prepare_phase_trace/
  20260727_7pa1p_c32_block_buffer_full/
  20260728_7pa1p_c32_kernel_warmup_smoke/
  20260728_7pa1p_c32_tail_fix_full/
```
