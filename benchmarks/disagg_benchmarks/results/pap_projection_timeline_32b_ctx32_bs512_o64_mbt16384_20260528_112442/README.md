# Qwen3-32B PAP ctx32 bs512 o64 mbt16384 partial run

Date: 2026-05-28

## Config

- Model: `/data/ssd1/llm-models/Qwen3-32B`
- Topology: 1PA1P
- TP: 2 for PA and Projection
- PA-P transport: `nixl_mailbox`
- Request load: 512 prompts, random input 32, random output 64, `request-rate=inf`, `max-concurrency=512`
- PAP runner microbatches: 3
- vLLM scheduler:
  - `max_model_len=96`
  - `max_num_seqs=512`
  - `max_num_batched_tokens=16384`
- Runtime workarounds:
  - `NCCL_P2P_DISABLE=1`
  - `NCCL_IB_DISABLE=1`
  - `NCCL_CUMEM_ENABLE=0`
  - `NCCL_NVLS_ENABLE=0`

## Result

The run did not complete. Benchmark was stopped after the PAP NIXL mailbox path
timed out around layer 54:

- Projection sender: `timed out waiting for PAP NIXL mailbox ACK`
- Attention side: `timed out waiting for PAP NIXL receive slot`

No complete serving benchmark JSON was produced. Partial trace summary is saved
in `trace_summary_partial.json`.

## Key Finding

This run confirms the previous `calls=6` result was caused by scheduler token
budget, not by projection compute capacity.

With `input_len=32` and explicit `max_num_batched_tokens=16384`, projection did
form the intended large ubatches:

- Projection `Running: 512 reqs`
- Projection timeline rows: 6594
- `calls` p90: 170
- `calls` p99/max: 172
- This matches `512 / 3 ~= 170` for 3-way microbatching.

Partial projection timeline:

- `calls` median 18, p90 170, p99 172, max 172
- `send_ms` median 0.180 ms, p90 1.047 ms
- `yield_ms` median 12.557 ms, p90 54.541 ms
- `recv_ms` median 1.692 ms, p90 19.881 ms
- `remote_total_ms` median 14.500 ms, p90 77.627 ms
- `self_attn_total_ms` median 18.602 ms, p90 83.819 ms

The median is lower than 170 because the partial run contains both large early
ubatches and many smaller rows after request progress skew appears. The p90/p99
show that the intended large ubatches were reached before the transport path
stalled.

## Interpretation

Increasing `max_num_batched_tokens` fixed the batch formation problem, but it
exposed a NIXL mailbox scaling bottleneck for 512 concurrent PAP decode requests.
The bottleneck is now in the per-layer mailbox queue/ACK/receive-slot path, not
in vLLM scheduler batch formation.

For future complete performance runs, either reduce per-ubatch size, increase
mailbox capacity/timeout and backpressure handling, or move this path to a more
batched/fused communication path before expecting stable bs512 PAP results.
