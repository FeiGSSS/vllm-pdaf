# Qwen3-32B PAP vs PD TPOT Breakdown

Date: 2026-06-30

## Goal

Break down the roughly 1s PAP TPOT observed on dense Qwen3-32B, and compare it
with a PD-disaggregated baseline where decode attention stays inside the decode
worker.

## Test Bed Contract

Use this run configuration as the default PAP optimization test bed for the next
iteration. The target optimizations are:

- attention executor compute path;
- NIXL mailbox queue/ack/read/prepare path.

Keep the load fixed unless the experiment is explicitly testing sensitivity to
load. The purpose is to make before/after comparisons attributable to code
changes rather than traffic shape changes.

Fixed PAP test-bed configuration:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export PAP_ROOT=/home/fei/research/PD/vllm-pap
export VLLM_BIN=/home/fei/research/PD/vllm-pap/.venv/bin/vllm
export PYTHON_BIN=/home/fei/research/PD/vllm-pap/.venv/bin/python
export MODEL_PATH=/data/ssd1/llm-models/Qwen3-32B
export PREFIX_LEN=0
export PAP_TP_SIZE=2
export MAX_MODEL_LEN=64
export MAX_NUM_SEQS=512
export MAX_NUM_BATCHED_TOKENS=16384
export PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.76
export PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.76
export PAP_ENABLE_MPS=0
export PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox
export PAP_DIRECT_MAILBOX_OUTPUT=1
export PAP_NIXL_MAILBOX_SLOT_COUNT=2
export PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=2
export PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1
export PAP_OFFLOAD_EXEC_TRACE=1
export PAP_ATTENTION_LOCAL_PAGED_CACHE=1
export PAP_OFFLOAD_EXEC_USE_PAGED_FLASH_ATTN=1
export PAP_RUNNER_MICROBATCH_COUNT=3
export PAP_RUNNER_MICROBATCH_DECODE_THRESHOLD=12
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_QWEN3_LAYER_PROFILE=1
export VLLM_QWEN3_LAYER_PROFILE_ASYNC=1
export VLLM_QWEN3_LAYER_PROFILE_ASYNC_FLUSH_THRESHOLD=2048

bash /home/fei/research/PD/test/baseline/run_benchmark.sh \
  --mode pap \
  --topology 1pa1p \
  --input-lens 32 \
  --output-lens 16 \
  --qps 256 \
  --num-prompts 128 \
  --model /data/ssd1/llm-models/Qwen3-32B \
  --proxy-port 9400
```

Equivalent checked-in wrapper:

```bash
bash benchmarks/disagg_benchmarks/run_pap_128_testbed.sh
```

Use `PAP_PROXY_PORT=<port>` to avoid a busy proxy port. Keep
`PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND` unset for the default baseline;
set it to `1` only for the native-cache-append experiment described below.

Baseline pass/fail gate:

- completed requests: 128
- failed requests: 0
- service shutdown leaves all GPUs idle

Primary comparison metrics for the active test-bed baseline:

| Metric | Baseline |
| --- | ---: |
| run dir | `20260630_201443` |
| median TPOT | 453.14 ms |
| p99 TPOT | 454.19 ms |
| output throughput | 154.74 tok/s |
| projection `remote_total_ms` | 5.01 ms |
| projection `yield_ms` | 4.51 ms |
| projection `recv_ms` | 0.11 ms |
| attention `compute_ms` | 0.58 ms |
| attention `append_kv_ms` | 0.36 ms |
| attention `pack_ms` | 0.02 ms |
| attention `shape_lookup_ms` | 0.05 ms |
| attention `qkv_split_ms` | 0.02 ms |
| attention `paged_metadata_ms` | 0.04 ms |
| attention `paged_flash_ms` | 0.05 ms |
| projection mailbox task `queue_ms` | 0.11 ms |
| projection mailbox task `ack_wait_ms` | 0.00 ms |
| attention mailbox task `prepare_ms` | 0.008-0.009 ms |
| attention mailbox task `transfer_ms` | 0.78-0.98 ms |
| projection result read total | 0.57 ms |

For attention-compute work, a meaningful improvement should reduce
`attention compute_ms` and, ideally, PAP median TPOT without increasing mailbox
queue/ack pressure.

For mailbox work, a meaningful improvement should reduce projection
`remote_total_ms`/`yield_ms`, projection task `queue_ms`, attention task
`transfer_ms`, result read time, or the correlation gap where projection resumes
after attention has already sent the result. If `remote_total_ms` does not move,
the change is not on the TPOT critical path for this test bed.

## Common Load

- Model: `/data/ssd1/llm-models/Qwen3-32B`
- Tensor parallel size: 2
- Input length: 32
- Output length: 16
- Number of prompts: 128
- Request rate: 256 QPS
- `max_model_len`: 64
- `max_num_seqs`: 512
- `max_num_batched_tokens`: 16384
- `PAP_ATTENTION_LOCAL_PAGED_CACHE=1`
- `PAP_OFFLOAD_EXEC_USE_PAGED_FLASH_ATTN=1`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- Qwen3 layer profile enabled with async JSONL flush.

This is a minimal reproducible load for the original ~1s PAP TPOT issue. It is
not a full parameter sweep.

## Runs

PAP:

- Topology: `1PA1P`, TP=2
- Transport: `PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox`
- Direct mailbox output: enabled
- Runner microbatch count: 3
- Run dir: `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_165147`
- Result JSON: `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_165147/1PA1P_i32_o16_q256.json`
- Profile JSONL: `pap_profile/`

PD:

- Topology: `1P1D`, TP=2
- KV connector: native NIXL disaggregated path
- Run dir: `/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/20260630_165339`
- Result JSON: `/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/20260630_165339/1P1D_i32_o16_q256.json`
- Profile JSONL: `pd_profile/`

## End-to-End Results

| Architecture | Completed | Failed | Req throughput | Output throughput | Median TTFT | Median TPOT | P99 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PAP 1PA1P TP2 | 128 | 0 | 6.78 req/s | 108.50 tok/s | 4786.86 ms | 907.74 ms | 917.74 ms |
| PAP 1PA1P TP2 active | 128 | 0 | 9.86 req/s | 157.69 tok/s | 5595.40 ms | 459.33 ms | 472.23 ms |
| PAP 1PA1P TP2 current | 128 | 0 | 9.67 req/s | 154.74 tok/s | 5994.40 ms | 453.14 ms | 454.19 ms |
| PD 1P1D TP2 | 128 | 0 | 20.58 req/s | 329.32 tok/s | 4521.33 ms | 79.98 ms | 95.02 ms |

Derived:

- PAP median TPOT / 64 layers = 14.18 ms per layer per output token.
- Active PAP median TPOT / 64 layers = 7.18 ms per layer per output token.
- Current PAP median TPOT / 64 layers = 7.08 ms per layer per output token.
- PD median TPOT / 64 layers = 1.25 ms per layer per output token.
- Original PAP median TPOT is 11.35x PD median TPOT.
- Active PAP median TPOT is 5.74x PD median TPOT.
- Current PAP median TPOT is 5.67x PD median TPOT.
- PD output throughput is 2.13x current PAP output throughput.

## Decode Worker CUDA Event Profile

The layer profile is CUDA-event based. It is the best signal here for kernel
compute time, but it does not include the full remote scheduling/transport path.

Filtered steady decode samples use `context_len >= 32` and `batch_size >= 16`.

PAP projection-side local kernels:

| Stage | Median |
| --- | ---: |
| input_layernorm | 0.0051 ms |
| qkv_proj | 0.0850 ms |
| qk_norm_rope | 0.0102 ms |
| o_proj | 0.1834 ms |
| post_attention_layernorm | 0.0140 ms |
| mlp | 0.7219 ms |
| Sum of stage medians | 1.0197 ms |

PD decode-side local kernels:

| Stage | Median |
| --- | ---: |
| input_layernorm | 0.0051 ms |
| qkv_proj | 0.0891 ms |
| qk_norm_rope | 0.0113 ms |
| attention | 0.0317 ms |
| o_proj | 0.1249 ms |
| post_attention_layernorm | 0.0051 ms |
| mlp | 0.6851 ms |
| Sum of stage medians | 0.9523 ms |

For large decode batches (`context_len >= 32`, `batch_size >= 40`), PD local
attention is still only 0.0437 ms median and the full per-layer kernel sum is
1.0660 ms median.

## PAP Remote-Attention Trace

This section is the original root-cause PAP run
`/home/fei/research/PD/test/baseline/pap/results/runs/20260630_165147`. The
current active test-bed baseline is summarized in the 128-request test-bed
section below.

`tools/pap_trace_summary.py` over the original PAP service logs:

Projection trace median:

- Calls per remote batch: 25
- `send_ms`: 0.223 ms
- `yield_ms`: 5.935 ms
- `recv_ms`: 0.706 ms
- `total_ms`: 6.653 ms

Projection timeline median:

- `pre_attn_compute_ms`: 0.125 ms
- `remote_total_ms`: 6.572 ms
- `o_proj_ms`: 0.137 ms
- `self_attn_total_ms`: 8.680 ms

Projection layer timeline median:

- `self_attn_ms`: 8.565 ms
- `input_norm_ms`: 0.038 ms
- `post_attention_layernorm_ms`: 0.044 ms
- `mlp_ms`: 0.111 ms
- `layer_total_ms`: 8.772 ms

Attention-side trace median:

- Calls per attention batch: 42
- `recv_qkv_ms`: 1.017 ms
- `compute_ms`: 3.519 ms
- `send_output_ms`: 0.018 ms
- `total_ms`: 4.588 ms

Mailbox details:

- Projection sending `attention_task_batch`: total 5.470 ms median
  - queue 1.203 ms
  - ack wait 4.347 ms
- Attention reading `attention_task_batch`: total 3.315-3.562 ms median
  - transfer 0.792-0.917 ms
  - prepare 2.263-2.654 ms
- Projection reading `attention_result_batch`: total 0.657 ms median

Correlation:

- Attention path after projection send: 6.163 ms median
- Attention ready after projection resumes: 0.102 ms median
- Projection resume after attention ready: 0.000 ms median
- Projection resume to recv done: 0.666 ms median

## Single-Ubatch Remote Path

The following table follows one projection-side ubatch after local QKV
projection has finished. The numbers use matched projection/attention trace
entries with `calls >= 16` and `remote_total <= 10 ms`, excluding startup and
large outliers.

| Segment | Median | Notes |
| --- | ---: | --- |
| projection pre-attention compute | 0.125 ms | Local qkv projection plus q/k norm and rope before sending. |
| projection send API | 0.227 ms | Projection thread enqueue/send call for the QKV task. |
| projection send done to attention recv done | 3.904 ms | QKV task becomes available on the attention side. This includes mailbox queue/ack/read/prepare effects seen from the end-to-end timestamp correlation. |
| attention compute | 3.519 ms | Attention executor compute after QKV has been received. This is not mailbox receive time. |
| attention output send API | 0.018 ms | Attention executor sends the output batch. |
| attention send done to projection recv done | 0.769 ms | Result return path to the projection worker. |
| projection recv API | 0.404 ms | Projection-side receive after the ubatch resumes. |
| projection o_proj | 0.137-0.140 ms | Local output projection after attention result is ready. |

The robust projection-side end-to-end remote metric is:

- `remote_total_ms`: 6.57 ms median from the standard summary, 6.73 ms median
  from the matched large-ubatch subset.

Do not add every mailbox-side number below as if they are disjoint wall-clock
segments. For example, projection-side `ack_wait_ms` and attention-side
`read/prepare_ms` are two observations of overlapping producer/consumer work.

Mailbox component observations:

| Component | Median | Interpretation |
| --- | ---: | --- |
| projection `attention_task_batch` queue | 1.203 ms | Time in the projection mailbox async send queue before the sender thread starts publishing. |
| projection `attention_task_batch` ack wait | 4.347 ms | Sender thread waiting for the peer-side ack/backpressure release. |
| projection `attention_task_batch` send total | 5.470 ms | Background mailbox send lifetime from enqueue to ack. |
| attention task read prepare | 2.263-2.654 ms | NIXL read setup: locate remote payload, reserve local slot, create/reuse dlist handles, and prepare the transfer handle. |
| attention task transfer | 0.792-0.917 ms | Actual NIXL READ transfer and polling. |
| attention task read total | 3.315-3.562 ms | Attention-side mailbox read lifetime. |
| attention result send total | 1.663-1.803 ms | Attention-side background send lifetime for output batch. |
| projection result read total | 0.657 ms | Projection-side mailbox read lifetime for output batch. |

### Current Bottleneck Interpretation

There are two confirmed bottleneck groups.

1. Mailbox path.
   - Projection `queue_ms` is long because mailbox send is asynchronous and
     single-queue/sender-thread based. The foreground projection call can return
     quickly, but the background sender can be blocked by previous messages and
     by `_wait_ack()`. In this run, `ack_wait_ms` is much larger than pack/copy,
     so queue growth is mostly a backpressure symptom rather than local tensor
     copy cost.
   - Attention `prepare_ms` is long because `_read_remote_message()` prepares a
     NIXL READ for each message: it resolves remote payload metadata, reserves a
     local receive location, gets local/remote descriptor handles, and creates a
     prepped transfer handle. This is setup/control overhead before the actual
     `transfer_ms`.

2. Attention executor compute path.
   - The `compute_ms` field starts after `recv_next_qkv_batch_message()` has
     returned and `qkv_batch` is available. It ends before
     `send_output_batch()`. Therefore it does not include mailbox QKV receive
     or result send. It can include the local receive-slot release callback for
     the consumed QKV message, but not the QKV transfer/read itself.
   - It is not a pure fused attention kernel time. The current executor loops
     over `descriptor.items` and calls `compute_offload_exec_output()` per
     request. That includes splitting packed QKV, resolving the session, reserve
     and append decode KV, constructing segment lists, moving query if needed,
     and then running `compute_segmented_attention_output()`. The helper uses
     per-request SDPA/einsum-style segmented attention rather than vLLM's local
     fused batch decode attention path. This explains why the attention-side
     compute median is several milliseconds even though PD's local fused decode
     attention kernel is only about 0.03-0.04 ms for this short-context workload.

## 128-Request Test Bed

This section fixes the `1PA1P`, TP=2, Qwen3-32B, input 32, output 16, 128
request, QPS 256 workload as the follow-on optimization test bed. Future
attention-compute and mailbox changes should be compared against this workload
without changing the load shape.

Fixed parameters:

- Model: `/data/ssd1/llm-models/Qwen3-32B`
- Tensor parallelism: `2`
- Architecture: PAP `1PA1P`
- Number of prompts: `128`
- Input length: `32`
- Output length: `16`
- Request rate: `256`
- Mailbox async-slot env:
  `PAP_NIXL_MAILBOX_SLOT_COUNT=2`,
  `PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=2`,
  `PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1`
- Attention-local paged-cache env:
  `PAP_ATTENTION_LOCAL_PAGED_CACHE=1`,
  `PAP_OFFLOAD_EXEC_USE_PAGED_FLASH_ATTN=1`
- Native cache append:
  `PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND` is intentionally unset in the
  default baseline. It is an opt-in experiment because it improves attention
  compute medians but did not improve end-to-end TPOT in the first run.
- Batch append control-plane shortcut:
  the current code skips redundant request-id resolution and prefill-layer wait
  when `compute_offload_exec_batch_output()` already passes canonical session
  ids and the attention-local paged state exists. This targets
  `append_prepare_ms` only; it is not considered an end-to-end win unless TPOT
  moves on the same test bed.
- Paged metadata scratch:
  `PAP_ATTENTION_PAGED_METADATA_SCRATCH_CUDA` is intentionally unset. Reusing
  CUDA metadata buffers through CPU-side scratch copies was tested and increased
  `paged_metadata_ms`, so CUDA scratch is only an explicit negative experiment.

Current verified PAP baseline on this test bed:

- Run:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_201443`
- Median TTFT: `5994.40 ms`
- Median TPOT: `453.14 ms`
- Output throughput: `154.74 tok/s`
- Median attention `compute_ms`: `0.575 ms`
- Median attention `append_kv_ms`: `0.356 ms`
- Median attention `pack_ms`: `0.019 ms`
- Median attention `shape_lookup_ms`: `0.046 ms`
- Median attention `qkv_split_ms`: `0.019 ms`
- Median attention `paged_metadata_ms`: `0.041 ms`
- Median attention `paged_flash_ms`: `0.046 ms`
- Median attention `sdpa_ms`: `0.000 ms`
- Median projection `remote_total_ms`: `5.007 ms`
- Median projection `yield_ms`: `4.509 ms`
- Median projection `recv_ms`: `0.115 ms`

Attention append substage medians in the current run:

| Substage | Median |
| --- | ---: |
| `append_lock_wait_ms` | 0.000 ms |
| `append_prepare_ms` | 0.065 ms |
| `append_record_ms` | 0.048 ms |
| `append_tensor_ms` | 0.109 ms |
| `append_copy_ms` | 0.075 ms |
| `append_state_ms` | 0.022 ms |

The previous active run
`/home/fei/research/PD/test/baseline/pap/results/runs/20260630_195141` remains a
useful comparison point: median TPOT `459.33 ms`, output throughput
`157.69 tok/s`, attention `compute_ms=0.545 ms`, and projection
`remote_total_ms=5.208 ms`. The current run has slightly lower TPOT but similar
or slightly worse attention compute medians, so the difference should be treated
as run-to-run noise plus a small mailbox-path shift rather than a proven
attention-compute win.

Optimization target:

- Reduce mailbox transfer/read/resume overhead next. Attention compute is now
  below `0.6 ms` median on this test bed, while projection `remote_total_ms`
  remains about `5.0 ms`.
- The remaining attention-side compute target is `append_kv_ms`, now about
  `0.35 ms`. `shape_lookup_ms`, `qkv_split_ms`, `pack_ms`, paged metadata, and
  paged FlashAttention are each tens of microseconds.
- Projection-side task ACK wait is already `0.000 ms` under the async-slot
  configuration, but projection `yield_ms` is still about `4.5 ms`.
- Re-run the same 128-request test bed after each change and compare median
  TPOT, output throughput, attention trace medians, and mailbox trace medians.

Comparison protocol:

```bash
.venv/bin/python tools/pap_trace_summary.py \
  /path/to/run/service_logs
```

Use the default outlier-filtered summary for table-to-table comparisons. Use
`--include-outliers` only when diagnosing startup skew, long stalls, or
mailbox wait tails.

Runs:

- Baseline PAP: `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_165147`
- Batched attention compute: `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_180136`
- Batched attention compute plus async mailbox slots:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_180529`
- Batched compute plus async mailbox slots plus combined decode-slot append:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_182142`
- Local paged KV plus paged FlashAttention:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_190103`
- Paged-only batch append plus local paged FlashAttention:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_191504`
- Batch-vectorized local paged append plus local paged FlashAttention:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_192526`
- Fine-grained trace only:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_193518`
- Batch session lookup/request-id cache:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_194028`
- Batched QKV split plus batched local paged append:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_195141`
- Append substage trace diagnostic:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_200829`
- Strided-source local paged copy helper:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_201443`
- Native vLLM `reshape_and_cache_flash` append, opt-in:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_203329`
- Batch append control-plane shortcut:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_204907`
- Batch append control-plane shortcut plus native cache append, opt-in:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_205111`
- CUDA paged metadata scratch plus native cache append, negative experiment:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_210338`
- Interrupted CUDA paged metadata scratch startup, no benchmark JSON:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_210152`
- Cached offload-exec session entries plus native cache append, opt-in:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_211321`
- Invalid mailbox inline-poll experiment:
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_201831`

Run hygiene:

- Keep `VLLM_USE_FLASHINFER_SAMPLER=0` for this test bed. The startup attempt in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_184940`
  failed in flashinfer sampling JIT compilation before PAP decode reached the
  measured path.

Code changes in the batched attention compute run:

- `run_offload_exec_batch_once()` and `run_offload_exec_mailbox_loop()` now call
  `compute_offload_exec_batch_output()` for batch descriptors.
- The new fast path batches same-shape decode requests into one SDPA call.
- The attention trace now breaks `compute_ms` into `append_kv_ms`, `pack_ms`,
  `sdpa_ms`, and `reshape_ms`.

The async mailbox slot run used the same code plus:

```bash
export PAP_NIXL_MAILBOX_SLOT_COUNT=2
export PAP_NIXL_MAILBOX_RECV_SLOT_COUNT=2
export PAP_NIXL_MAILBOX_ASYNC_SEND_SLOTS=1
```

End-to-end results:

| Variant | Completed | Failed | Median TTFT | Median TPOT | P99 TPOT | Output throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline PAP | 128 | 0 | 4786.86 ms | 907.74 ms | 917.74 ms | 108.50 tok/s |
| Batched attention compute | 128 | 0 | 4677.00 ms | 878.23 ms | 879.98 ms | 112.01 tok/s |
| Batched compute + async slots | 128 | 0 | 4551.70 ms | 848.00 ms | 850.32 ms | 115.74 tok/s |
| Batched compute + async slots + combined append | 128 | 0 | 4928.33 ms | 782.25 ms | 821.62 ms | 119.30 tok/s |
| + local paged KV + paged FlashAttention | 128 | 0 | 5832.11 ms | 707.90 ms | 711.52 ms | 121.25 tok/s |
| + paged-only batch append | 128 | 0 | 5606.98 ms | 631.46 ms | 657.59 ms | 124.21 tok/s |
| + batch-vectorized append | 128 | 0 | 5658.98 ms | 547.03 ms | 577.87 ms | 134.71 tok/s |
| + batched QKV split/append | 128 | 0 | 5595.40 ms | 459.33 ms | 472.23 ms | 157.69 tok/s |
| + strided-source copy helper | 128 | 0 | 5994.40 ms | 453.14 ms | 454.19 ms | 154.74 tok/s |
| + native cache append, opt-in | 128 | 0 | 5593.55 ms | 469.97 ms | 472.42 ms | 156.51 tok/s |
| + control-plane shortcut | 128 | 0 | 6268.11 ms | 455.84 ms | 484.63 ms | 149.98 tok/s |
| + control-plane shortcut + native cache append, opt-in | 128 | 0 | 5010.31 ms | 459.81 ms | 470.60 ms | 158.91 tok/s |
| + CUDA metadata scratch + native cache append, negative | 128 | 0 | 4889.32 ms | 444.55 ms | 451.42 ms | 163.34 tok/s |
| + cached session entries + native cache append, opt-in | 128 | 0 | 5489.94 ms | 490.28 ms | 490.59 ms | 154.22 tok/s |

Trace medians:

| Metric | Baseline | Batched compute | Batched compute + async slots | + combined append | + local paged FA | + paged-only append | + vectorized append | + batched QKV split/append |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attention calls | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| attention `compute_ms` | 3.519 ms | 3.117 ms | 3.944 ms | 3.686 ms | 3.093 ms | 2.403 ms | 1.963 ms | 0.545 ms |
| attention `append_kv_ms` | n/a | 0.732 ms | 1.127 ms | 0.926 ms | 1.577 ms | 0.879 ms | 0.456 ms | 0.342 ms |
| attention `pack_ms` | n/a | 1.291 ms | 1.695 ms | 1.600 ms | 0.381 ms | 0.369 ms | 0.331 ms | 0.015 ms |
| attention `shape_lookup_ms` | n/a | n/a | n/a | n/a | n/a | n/a | 0.722 ms | 0.044 ms |
| attention `qkv_split_ms` | n/a | n/a | n/a | n/a | n/a | n/a | 0.285 ms | 0.015 ms |
| attention `sdpa_ms` | n/a | 0.137 ms | 0.138 ms | 0.138 ms | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms |
| attention `paged_metadata_ms` | n/a | n/a | n/a | n/a | 0.052 ms | 0.057 ms | 0.052 ms | 0.039 ms |
| attention `paged_flash_ms` | n/a | n/a | n/a | n/a | 0.050 ms | 0.053 ms | 0.051 ms | 0.045 ms |
| projection `remote_total_ms` | 6.572 ms | 4.118 ms | 4.336 ms | 5.874 ms | 8.188 ms | 7.278 ms | 6.565 ms | 5.208 ms |
| projection `yield_ms` | 5.836 ms | 3.655 ms | 3.895 ms | 5.203 ms | 7.120 ms | 6.455 ms | 4.714 ms | 4.677 ms |
| projection `recv_ms` | 0.742 ms | 0.382 ms | 0.237 ms | 0.569 ms | 0.542 ms | 0.268 ms | 0.117 ms | 0.115 ms |
| projection task `queue_ms` | 1.203 ms | 0.370 ms | 0.079 ms | 0.055 ms | 0.067 ms | 0.053 ms | 0.083 ms | 0.119 ms |
| projection task `ack_wait_ms` | 4.347 ms | 2.838 ms | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms | 0.000 ms |
| projection task send total | 5.470 ms | 3.350 ms | 0.204 ms | 0.180 ms | 0.183 ms | 0.233 ms | 0.256 ms | 0.356 ms |
| attention task read prepare | 2.263-2.654 ms | 0.215-2.301 ms | 0.009-0.010 ms | 0.006-0.008 ms | 0.008-0.009 ms | 0.009-0.011 ms | 0.008 ms | 0.006-0.008 ms |
| attention task read total | 3.315-3.562 ms | 1.352-3.219 ms | 0.848-0.900 ms | 0.770-0.893 ms | 0.789-0.914 ms | 0.852-0.899 ms | 0.764-0.867 ms | 0.963-1.023 ms |
| projection result read total | 0.657 ms | 0.633 ms | 0.775 ms | not summarized | 0.649 ms | 0.634 ms | 0.574 ms | 0.617 ms |

Interpretation:

- Batched attention compute is a real but limited win. It reduces attention
  `compute_ms` by about 0.40 ms median and improves TPOT by about 29.5 ms. The
  SDPA kernel itself is only about 0.14 ms; most attention-side compute time is
  local KV append plus pack/pad/copy work.
- Async mailbox slots are a stronger win on this test bed. They remove the
  projection-side task ACK wait from the measured send path and reduce attention
  read prepare to about 0.01 ms. TPOT improves another about 30 ms, from
  878.23 ms to 848.00 ms.
- At the batched-compute plus async-slot stage, the gap to PD was still large:
  PAP was about 10.6x the PD median TPOT of 79.98 ms for the same 128-request
  load. That stage showed that `append_kv_ms` and `pack_ms`, not the SDPA
  kernel, were the next useful attention-side targets.
- Combining decode-slot reservation and KV append into one registry call reduced
  attention `append_kv_ms` from `1.127 ms` to `0.926 ms` under async slots, and
  reduced median TPOT from `848.00 ms` to `782.25 ms` in this run. The projection
  `remote_total_ms` median did not improve in the same run, so the end-to-end
  gain should be treated as a workload-level result rather than a simple
  per-remote-call reduction.
- Local paged KV plus paged FlashAttention was the first pure-kernel-path
  improvement. It reduces median TPOT from `782.25 ms` to `707.90 ms`. The
  attention kernel and paged metadata are each only about `0.05 ms`, and
  `pack_ms` drops from `1.600 ms` to `0.381 ms`. The remaining attention-side
  compute cost at that point is mostly decode KV append/local paged-cache
  maintenance (`1.577 ms`).
- Paged-only batch append was the next improvement. It avoids the legacy segment
  decode-buffer double write when the local paged FA path is active, reducing
  attention `append_kv_ms` from `1.577 ms` to `0.879 ms` and attention
  `compute_ms` from `3.093 ms` to `2.403 ms`. End-to-end median TPOT improves
  from `707.90 ms` to `631.46 ms`.
- Batch-vectorized append groups local paged decode KV writes by
  `(pool, block_offset)` and uses one batched write per key/value group instead
  of one write per request. This reduces
  `append_kv_ms` from `0.879 ms` to `0.456 ms`, attention `compute_ms` from
  `2.403 ms` to `1.963 ms`, and median TPOT from `631.46 ms` to `547.03 ms`.
- Fine-grained attention trace in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_193518`
  showed that the residual `1.9 ms` compute time was mostly not FlashAttention:
  median `shape_lookup_ms=0.722 ms`, `append_kv_ms=0.437 ms`,
  `qkv_split_ms=0.285 ms`, `pack_ms=0.309 ms`, and `paged_flash_ms=0.055 ms`.
- Batch session lookup plus request-id caching in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_194028`
  reduced `shape_lookup_ms` to `0.634 ms`, but median TPOT regressed/noised to
  `558.18 ms`. It is useful cleanup/instrumentation, but it is not counted as
  the active performance baseline.
- Batched QKV split plus batched local paged append was the previous active
  test-bed baseline. It splits the packed QKV tensor once per attention batch
  and passes batched K/V tensors directly into the local paged append path,
  avoiding per-request `split`, per-request append entries, and `torch.cat()`
  for query assembly. This reduces attention `compute_ms` from `1.963 ms` to
  `0.545 ms`, `qkv_split_ms` from `0.285 ms` to `0.015 ms`, `pack_ms` from
  `0.331 ms` to `0.015 ms`, and median TPOT from `547.03 ms` to `459.33 ms`.
- Append substage tracing in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_200829`
  broke `append_kv_ms` into lock/prepare/record/tensor/copy/state work:
  median `append_prepare_ms=0.065 ms`, `append_record_ms=0.051 ms`,
  `append_tensor_ms=0.114 ms`, `append_copy_ms=0.078 ms`, and
  `append_state_ms=0.021 ms`.
- The strided-source copy helper in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_201443`
  keeps matching-device/dtype strided K/V views instead of forcing
  `contiguous().to()` before the final KV-cache write. This is simpler and
  semantically correct, but it did not measurably remove the
  `append_tensor_ms` bucket: median `append_tensor_ms=0.109 ms`,
  `append_copy_ms=0.075 ms`, and `append_kv_ms=0.356 ms`. Treat the small TPOT
  change from `459.33 ms` to `453.14 ms` as noise-level unless repeated runs
  confirm it.
- Native vLLM `reshape_and_cache_flash` append was tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_203329` behind
  `PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND=1`. It is a real local
  attention-compute improvement: median attention `compute_ms` dropped from
  `0.575 ms` to `0.397 ms`, `append_kv_ms` from `0.356 ms` to `0.209 ms`,
  `append_tensor_ms` from `0.109 ms` to `0.028 ms`, and `append_copy_ms` from
  `0.075 ms` to `0.018 ms`. It is not the default because the same run regressed
  median TPOT from `453.14 ms` to `469.97 ms`: projection `remote_total_ms`
  moved from `5.007 ms` to `5.291 ms`, `yield_ms` from `4.509 ms` to
  `4.757 ms`, and attention `recv_qkv_ms` from `1.648 ms` to `1.881 ms`.
  This keeps native append as an opt-in compute experiment until mailbox/yield
  behavior also improves.
- The batch append control-plane shortcut was tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_204907`. It
  does what it targets locally: median `append_prepare_ms` drops from
  `0.065 ms` to `0.011 ms`. It is not an end-to-end win by itself: median TPOT
  is `455.84 ms` versus the `453.14 ms` active baseline, and median attention
  `compute_ms` is `0.605 ms` because this run's `append_tensor_ms` and
  `append_copy_ms` medians increased to `0.131 ms` and `0.099 ms`.
- Combining the control-plane shortcut with native cache append was tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_205111` with
  `PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND=1`. This is the best
  attention-compute result so far: median attention `compute_ms=0.339 ms`,
  `append_kv_ms=0.152 ms`, `append_prepare_ms=0.010 ms`,
  `append_tensor_ms=0.029 ms`, and `append_copy_ms=0.019 ms`. It still does not
  beat the active end-to-end baseline: median TPOT is `459.81 ms`, while
  projection `remote_total_ms=5.177 ms` and `yield_ms=4.653 ms` remain much
  larger than the local compute improvement.
- CUDA paged metadata scratch was tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_210338` with
  native cache append. Although median TPOT was `444.55 ms`, this is not an
  attention-compute win: median attention `compute_ms` regressed to `0.631 ms`
  and `paged_metadata_ms` regressed to `0.321 ms`. The TPOT improvement came
  from projection-side remote/yield movement (`remote_total_ms=4.899 ms`,
  `yield_ms=4.395 ms`) rather than the metadata change. CUDA metadata scratch is
  therefore disabled by default and kept only behind
  `PAP_ATTENTION_PAGED_METADATA_SCRATCH_CUDA=1` for future experiments.
- Cached offload-exec session entries were tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_211321` with
  native cache append. This is the best attention-compute result so far:
  median attention `compute_ms=0.311 ms`, down from `0.339 ms` in
  `20260630_205111` and `0.575 ms` in the default `20260630_201443` baseline.
  The targeted field moved as expected: `shape_lookup_ms` dropped from
  `0.044 ms` to `0.024 ms`. This is not an end-to-end TPOT win: median TPOT
  regressed to `490.28 ms` because projection-side remote wait moved the wrong
  way (`remote_total_ms=5.408 ms`, `yield_ms=4.850 ms`). Keep the session-entry
  cache as a local compute cleanup, but do not treat this run as the best
  serving result.
- The interrupted startup run
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_210152` has no
  benchmark JSON and should not be used as a performance result.

## Latest TPOT Breakdown

The latest attention-compute optimization run is
`/home/fei/research/PD/test/baseline/pap/results/runs/20260630_211321`. It uses
`PAP_ATTENTION_LOCAL_PAGED_NATIVE_CACHE_APPEND=1` and cached
`PAPOffloadExecSessionEntry` objects. The CUDA metadata scratch experiment is
disabled.

Key comparison points:

| Run | Meaning | Median TPOT | Output throughput | Attention compute | Projection remote total | Projection yield |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260630_201443` | default active baseline | 453.14 ms | 154.74 tok/s | 0.575 ms | 5.006 ms | 4.507 ms |
| `20260630_205111` | previous best attention compute | 459.81 ms | 158.91 tok/s | 0.339 ms | 5.177 ms | 4.653 ms |
| `20260630_210338` | best TPOT, negative compute experiment | 444.55 ms | 163.34 tok/s | 0.631 ms | 4.907 ms | 4.398 ms |
| `20260630_211321` | best attention compute | 490.28 ms | 154.22 tok/s | 0.311 ms | 5.408 ms | 4.850 ms |

Attention-side `compute_ms` breakdown for the best compute run
`20260630_211321`:

| Component | Median |
| --- | ---: |
| total attention `compute_ms` | 0.311 ms |
| `append_kv_ms` | 0.147 ms |
| `append_prepare_ms` | 0.009 ms |
| `append_record_ms` | 0.053 ms |
| `append_tensor_ms` | 0.028 ms |
| `append_copy_ms` | 0.018 ms |
| `append_state_ms` | 0.020 ms |
| `shape_lookup_ms` | 0.024 ms |
| `qkv_split_ms` | 0.014 ms |
| `pack_ms` | 0.014 ms |
| `paged_metadata_ms` | 0.042 ms |
| `paged_flash_ms` | 0.044 ms |
| `send_output_ms` | 0.016 ms |

Projection-side one-layer remote path in the same run:

| Component | Median |
| --- | ---: |
| projection pre-attention compute | 0.137 ms |
| projection send API | 0.405 ms |
| attention path after projection send | 1.678 ms |
| attention `recv_qkv_ms` | 2.033 ms |
| attention `compute_ms` | 0.311 ms |
| projection resume after attention ready | 3.219 ms |
| projection receive API | 0.111 ms |
| projection `o_proj_ms` | 0.146 ms |
| projection `remote_total_ms` | 5.408 ms |
| projection self-attention total | 6.898 ms |

Mailbox observations in the same run:

| Path | Median |
| --- | ---: |
| projection task send total | 0.318 ms |
| projection task queue | 0.072 ms |
| projection task publish | 0.186 ms |
| projection task copy | 0.077 ms |
| attention task read total, rank 0 | 0.915 ms |
| attention task read total, rank 1 | 1.017 ms |
| attention task transfer, rank 0 | 0.898 ms |
| attention task transfer, rank 1 | 0.999 ms |
| attention task prepare | 0.006-0.008 ms |
| projection result read total | 0.627 ms |
| projection result wait | 0.003 ms |

Conclusion for this iteration:

- Yes, attention executor compute has been optimized: the measured attention
  compute median went from `3.519 ms` in the original root-cause run to
  `0.575 ms` in the current default baseline and `0.311 ms` in the opt-in
  native-append/session-entry-cache run.
- This optimization is no longer the main TPOT lever for the 128-request test
  bed. The best compute run is slower end-to-end because projection
  `remote_total_ms` and `yield_ms` are larger than in the default run.
- The next obvious bottleneck is projection-side remote wait/resume, especially
  the `projection_resume_after_attention_ready_ms` gap. In the latest run,
  attention has usually produced the result, but projection resumes about
  `3.2 ms` later before doing a short `0.11 ms` receive. This gap is larger
  than the full attention compute body by about 10x.
- The next concrete mailbox bottleneck is QKV read transfer on the attention
  side: about `0.9-1.0 ms` median per rank. `prepare_ms` is already down to
  single-digit microseconds and is no longer the dominant mailbox-read
  substage on this test bed.

Mailbox slot alignment follow-up, 2026-07-01:

- `PAP_RUNNER_MICROBATCH_COUNT=3` means the 128-request test bed has three
  runner ubatches in flight. The test-bed mailbox defaults were changed from
  two send/receive slots to three send/receive slots so slot capacity matches
  the runner pipeline shape.
- The mailbox slot size calculation now rounds each slot down to a 16-byte
  boundary. Without this, `PAP_NIXL_MAILBOX_SLOT_COUNT=3` can create an odd byte
  offset inside the 16 MiB mailbox buffer, and viewing that byte slice as BF16
  fails with `storage_offset() must be divisible by 2`.
- Controlled slot-count runs on the same code:

| Config | Run root | Success | Median TPOT | Projection `yield_ms` | Projection remote total | Attention path after send | Projection resume after ready |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default 2 slots, prior active code | `/home/fei/research/PD/test/baseline/pap/results/runs/20260701_123739` | 128/128 | 258.35 ms | 2.525 ms | 2.874 ms | 0.994 ms | 1.480 ms |
| explicit 3 slots | `/home/fei/research/PD/test/baseline/pap/results/runs/20260701_125809` | 128/128 | 238.77 ms | 2.346 ms | 2.795 ms | 1.236 ms | 1.091 ms |
| explicit 4 slots | `/home/fei/research/PD/test/baseline/pap/results/runs/20260701_125939` | 128/128 | 293.54 ms | 2.833 ms | 3.287 ms | 1.403 ms | 0.861 ms |
| default 3 slots, rerun A | `/home/fei/research/PD/test/baseline/pap/results/runs/20260701_130423` | 128/128 | 300.95 ms | 2.664 ms | 3.133 ms | 1.520 ms | 1.084 ms |
| default 3 slots, rerun B | `/home/fei/research/PD/test/baseline/pap/results/runs/20260701_130642` | 128/128 | 259.27 ms | 2.572 ms | 2.980 ms | 1.225 ms | 1.398 ms |

- Interpretation: three slots is the correct configuration for the 3-way
  runner pipeline and can reduce the per-layer remote path in a clean run, but
  the end-to-end TPOT remains noisy. This is a correctness and pipeline-shape
  alignment change, not a standalone proof of a stable TPOT win.
- Four slots is not a good default for this test bed. It increased QKV wait and
  projection remote total in the measured run.

Negative follow-up:

- Directly copying prefill/decode KV segments into the final padded batch,
  instead of first concatenating each request's segments, was tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_181456`.
- It regressed median TPOT to `879.35 ms` and increased attention `pack_ms` from
  `1.695 ms` to `2.209 ms` under the async-slot configuration.
- The likely reason is that more Python-level segment loops and small copy calls
  are slower than the previous `torch.cat()` plus contiguous batch copy for this
  short-context workload. This path was not kept.
- Reusing registry-level scratch buffers for the padded key/value/mask tensors
  was tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_182949`.
- It regressed median TPOT to `857.60 ms`, increased attention `compute_ms` from
  `3.686 ms` to `3.969 ms`, and increased attention `pack_ms` from `1.600 ms`
  to `1.872 ms` compared with the combined-append baseline. This path was not
  kept.
- Mailbox inline polling with
  `PAP_NIXL_MAILBOX_INLINE_POLL=1` and `PAP_NIXL_MAILBOX_POLL_SECONDS=0` was
  tested in
  `/home/fei/research/PD/test/baseline/pap/results/runs/20260630_201831`.
  It stalled at the benchmark phase and was stopped manually; all 128 requests
  failed with incomplete streaming responses. This is not a valid performance
  result and should not be used as the default mailbox configuration.

## Interpretation

Runner microbatching is active and overlaps part of the remote wait. The
projection path sends QKV, calls `dbo_yield()`, and resumes later to receive the
attention result. The correlation data shows that, when projection resumes, the
attention result is usually already ready or almost ready.

In the current 128-request baseline, median `yield_ms` is about `4.5 ms`, while
median projection `recv_ms` after resume is only about `0.11 ms`. Therefore
`yield_ms` should not be read as pure mailbox transfer time. It is mostly the
projection worker running/yielding to other runner microbatches before this
ubatch comes back to receive its already-produced attention result. The mailbox
critical path that remains clearly visible is the attention-side QKV read
transfer, about `0.78-0.98 ms` median per attention rank, plus the smaller
send/read bookkeeping on both sides.

The remaining TPOT gap is therefore not mainly local dense compute. Local
projection-side CUDA kernels are about 1.02 ms per layer for the steady decode
samples, similar to PD's 0.95 ms per layer. The difference is the PAP remote
attention path. Each layer now pays a remote path with mailbox transfer,
attention-side preparation, remote attention compute, result send, and result
receive. In the original root-cause run that path was about `6.6 ms` median per
projection remote batch. In the current 128-request test-bed baseline it is
about `5.0 ms`, while PD's local attention kernel is only `0.03-0.04 ms` median
for this short-context load.

For this load, the original `907.7 ms` PAP TPOT corresponds to `14.18 ms/layer`.
The current `453.1 ms` verified baseline improves that to `7.08 ms/layer`,
still much slower than the PD `79.98 ms` TPOT baseline. The 3-way runner
microbatching hides some wait but does not make the remote path free. The
unavoidable layer-by-layer dependency remains: ubatch N cannot start layer
`i + 1` until its layer `i` remote attention output is available.

## Current Root Cause

Dense Qwen3-32B at input length 32 has very cheap local decode attention in PD.
PAP replaces that cheap local attention kernel with a millisecond-scale remote
attention path. Local paged FlashAttention removes most of the pure attention
kernel/padding overhead, paged-only batch append removes one major duplicate
decode KV write, batch-vectorized append removes much of the per-request
small-copy cost, and batched QKV split/append removes most of the remaining
Python-side batch preparation cost. The active baseline is now dominated much
more by projection-side remote wait and mailbox transfer/resume than by the
attention executor compute body. The current 3-way runner microbatch path
overlaps some waiting, but the per-layer
remote-attention dependency is still on the decode critical path, and the remote
path is much larger than the local attention kernel it replaces for this
workload.

Next optimization should focus on reducing the remote path itself:

- reducing mailbox transfer/read/resume overhead on the QKV and result paths;
- reducing projection-side `yield_ms` where attention has already produced the
  result but projection has not resumed/received it yet;
- further lowering `append_kv_ms`, now about `0.35 ms`;
- fewer per-layer/per-ubatch control-plane round trips;
- larger or more stable decode batches only if they improve arithmetic
  intensity without increasing mailbox path cost;
- re-evaluating with longer contexts where local attention compute is less
  negligible.
