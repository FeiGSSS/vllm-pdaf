# PAP 32B large-ubatch projection profile

Date: 2026-05-28

## Config

- Model: `/data/ssd1/llm-models/Qwen3-32B`
- Topology: `1pa1p`
- TP: `2`
- Transport: `nixl_mailbox`
- Microbatches: `3`
- Prompt/output: `32 / 64`
- Requests/concurrency: `512 / 512`
- `PAP_MAX_NUM_BATCHED_TOKENS=16384`
- `PAP_MAX_NUM_SEQS=512`
- `PAP_MAX_MODEL_LEN=96`
- `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.76`
- `PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.80`
- NCCL workaround: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0`
- CUDA event profiling: `VLLM_QWEN3_LAYER_PROFILE=1`, `VLLM_QWEN3_LAYER_PROFILE_ASYNC=1`, `VLLM_QWEN3_LAYER_PROFILE_ASYNC_FLUSH_THRESHOLD=32768`

## Benchmark

- Successful requests: `512 / 512`
- Duration: `374.48 s`
- TTFT median: `15095.54 ms`
- TPOT median: `5172.08 ms`
- Output throughput: `87.50 tok/s`

This run is for profiling, not final serving throughput, because CUDA event sampling periodically synchronizes.

## Large ubatch trace

Trace summary: `trace_summary.json`

- Projection calls: median `170`, p90 `172`, max `172`
- Attention calls: median `170`, p90 `172`, max `172`
- Projection-side remote attention path:
  - send median `1.267 ms`
  - yield median `50.317 ms`
  - recv median `11.818 ms`
  - remote_total median `62.808 ms`
  - self_attn_total median `77.708 ms`
- Attention executor:
  - recv_qkv median `3.328 ms`
  - compute median `20.291 ms`
  - send_output median `0.038 ms`
  - total median `23.874 ms`

Interpretation: the large ubatch is real (`170/170/172` split), but projection's observed self-attention section is dominated by yield/recv waiting, not by the remote attention kernel itself.

## CUDA event projection stages

Profile samples: `profile/samples_pid*.jsonl`

Filtered to projection workers and `batch_size in [170, 172]`:

- `qkv_proj`: median `0.109 ms`, p90 `0.110 ms`
- `qk_norm_rope`: median `0.014 ms`, p90 `0.014 ms`
- `o_proj`: median `0.378 ms`, p90 `15.570 ms`
- `mlp`: median `0.913 ms`, p90 `0.922 ms`
- Median `qkv + qk_norm_rope + o_proj`: `0.501 ms`
- Median full non-attention layer work sampled here: `1.429 ms`

`o_proj` has a long tail because it includes the TP row-parallel path/all-reduce behavior, so p90 is much larger than the median.

## Notes

- Earlier `pre_attn_compute_ms` in the wall-clock projection timeline is CPU-side launch timing and should not be treated as GPU compute time.
- The first failed large-batch run with `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.80` OOMed in the attention executor; `0.76` is the working point for this setup.
- `PAP_MAX_NUM_BATCHED_TOKENS` must be passed to vLLM. Without it, default `2048` caused projection calls around `6` for ctx128 and made the batch look artificially small.
