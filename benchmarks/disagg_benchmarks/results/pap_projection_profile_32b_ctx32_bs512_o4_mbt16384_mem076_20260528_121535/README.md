# Qwen3-32B PAP ctx32 bs512 o4 projection profile

Date: 2026-05-28

## Config

- Model: `/data/ssd1/llm-models/Qwen3-32B`
- Topology: `1pa1p`
- TP: `2`
- Transport: `nixl_mailbox`
- Microbatches: `3`
- Prompt/output: `32 / 4`
- Requests/concurrency: `512 / 512`
- `PAP_MAX_NUM_BATCHED_TOKENS=16384`
- `PAP_MAX_NUM_SEQS=512`
- `PAP_MAX_MODEL_LEN=96`
- `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.76`
- NCCL workaround: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0`

## Benchmark

- Successful requests: `512 / 512`
- Duration: `57.76 s`
- TTFT median: `18585.67 ms`
- TPOT median: `2001.36 ms`
- Output throughput: `35.46 tok/s`

This was a short-output profiling run. It is useful as a batch-formation and
trace sanity point, but the output length is too small for final PAP/PD
serving comparison.

## Trace

Trace summary: `trace_summary.json`

- Attention calls: median `18`, p90 `82`, p99/max `84`
- Attention compute: median `4.60 ms`, p90 `11.04 ms`
- Attention total: median `5.95 ms`, p90 `13.44 ms`

The run confirms the scheduler can form large ubatches after
`PAP_MAX_NUM_BATCHED_TOKENS` is raised, but this short-output profile should
not be used as the main performance conclusion.
