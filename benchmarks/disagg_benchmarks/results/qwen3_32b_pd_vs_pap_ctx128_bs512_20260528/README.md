# Qwen3-32B PD vs PAP ctx128 bs512 exploratory run

Date: 2026-05-28
Model: `/data/ssd1/llm-models/Qwen3-32B`

## Target comparison

The intended comparison was:

- PD baseline: 3 prefill instances + 1 decode instance, TP=2 for every instance, 8 GPUs total.
- PAP: 3 PA instances + 1 projection instance, TP=2 for every instance, 8 GPUs total.
- Load: random input 128, output 64, 512 prompts sent as one burst (`request-rate=inf`, `max-concurrency=512`).

This is the experiment motivated by the 32B decode profile: with ctx=128 and larger batch, projection should have enough per-layer compute time to make attention/projection overlap meaningful.

## Results

| Run | Status | Completed | Failed | Duration s | Req/s | Out tok/s | Mean TTFT ms | Mean TPOT ms | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PD 3P1D TP2 ctx128 bs512 o64 | success | 512 | 0 | 32.62 | 15.70 | 1004.51 | 12447.45 | 210.53 | Baseline completed. |
| PAP 3PA1P TP2 rankports ctx128 bs512 o64 | failed | 0 | 512 | 2.30 | 0 | 0 | 0 | 0 | Initial TP2 projection ZMQ port collision was fixed, but this run still used inherited shell proxy and attention register returned 403. |
| PAP 3PA1P TP2 noproxy ctx128 bs512 o64 | failed | 0 | 512 | 19.86 | 0 | 0 | 0 | 0 | Register/prefill worked; projection timed out waiting for remote attention TCP response and returned 500. |
| PAP 1PA1P TP2 sanity ctx128 bs4 o8 | success | 4 | 0 | 7.13 | 0.56 | 4.49 | 6193.39 | 111.86 | Proves the basic Qwen3-32B TP2 PAP pair can complete. |
| PAP 1PA1P TP2 sanity ctx128 bs16 o64 | success | 16 | 0 | 28.83 | 0.55 | 35.52 | 1364.62 | 431.13 | Single PA/projection pair is stable under longer decode, but very slow. |
| PAP 3PA1P TP2 NIXL mailbox sanity ctx128 bs12 o8 | success | 12 | 0 | 5.21 | 2.30 | 18.41 | 2697.16 | 354.91 | NIXL mailbox bypasses the multi-PA P2P/NCCL hang on a small load. |
| PAP 3PA1P TP2 NIXL mailbox ctx128 bs512 o64, PA mem 0.90 | invalid | 512 | 0 | 321.28 | 1.59 | 5.39 | 14462.07 | 3045.39 | Benchmark saw HTTP completions, but only 1731/32768 output tokens were generated; attention executor OOM killed the data path. |
| PAP 3PA1P TP2 NIXL mailbox ctx128 bs512 o64, PA mem 0.78 | success | 512 | 0 | 157.29 | 3.26 | 208.32 | 14282.20 | 2224.55 | Full 32768 output tokens generated. PA memory utilization must leave room for colocated attention executors. |

## Implementation fix made during this run

The first PAP TP=2 failure was a real launcher/runtime bug: both projection TP ranks bound the same `PAP_OFFLOAD_EXEC_ZMQ_PORT=11300`.

Fix:

- `examples/pap/launch_pap_nixl.sh` now exports comma-separated rank-local projection ZMQ ports, e.g. `11300,11301`.
- `vllm/model_executor/models/qwen3.py` now selects `PAP_OFFLOAD_EXEC_ZMQ_PORT` by TP rank before constructing the P2P/NCCL offload transport.

Verification:

- `bash -n examples/pap/launch_pap_nixl.sh`
- `.venv/bin/python -m py_compile vllm/model_executor/models/qwen3.py`
- PAP 1PA1P TP2 sanity runs above completed successfully.

## Current conclusion

This run now has a valid PAP 3PA1P TP2 completion through the NIXL mailbox OFFLOAD_EXEC path, but it is not yet competitive with PD.

The current best apples-to-apples point in this directory is:

- PD 3P1D TP2: 32.62 s, 15.70 req/s, 1004.51 output tok/s, mean TPOT 210.53 ms.
- PAP 3PA1P TP2 NIXL mailbox: 157.29 s, 3.26 req/s, 208.32 output tok/s, mean TPOT 2224.55 ms.

The previous PAP failures had two different root causes:

- The NCCL/P2P path still has a multi-PA transport/scheduling problem when one projection instance talks to multiple PA attention endpoint pairs. NIXL mailbox is the current stable bypass for this topology.
- The first bs512 NIXL run used `PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.90`. In PAP, the PA vLLM worker and the attention executor share the same GPU. At 0.90 the prefill worker consumed about 41.39 GiB on an L20, leaving only a few MiB free, and the attention executor OOMed around layer 56. The valid run used 0.78 and left about 5 GiB free during the benchmark.

Trace summary from the valid NIXL run:

- Projection trace rows: 27008. `batches` median 3, mean 2.94, p99 3.
- Projection `calls` median 170, mean 155.30, p99 172. This is the projection-side macro batch size before splitting across PA endpoints/ubatches.
- Attention trace rows: 79488. Attention-side `calls` median 57, mean 52.77, p99 65. This matches roughly one projection macro batch distributed across three PA groups.
- Attention median per-batch timing: recv QKV 2.454 ms, compute 6.057 ms, send output 0.019 ms, total 8.534 ms.
- Projection median remote-attention timing per layer trace: send 2.016 ms, yield 17.602 ms, recv 1.849 ms, total 21.640 ms.

The high TPOT is therefore not explained by raw output transfer time. The dominant cost is the repeated projection/attention alternation and mailbox/attention-side scheduling at every layer under the current eager PAP path.

## Next steps

1. Keep `PAP_PREFILL_GPU_MEMORY_UTILIZATION <= 0.78` for Qwen3-32B TP2 PAP on 48 GiB L20-class GPUs unless attention memory is reduced.
2. Treat the current NIXL mailbox 3PA1P result as a correctness/stability baseline, not a performance target.
3. Continue debugging the NCCL/P2P multi-PA path separately; it is still expected to be faster if the multi-peer transport is made robust.
4. Optimize the PAP hot path around fused/batched attention and fewer per-layer mailbox round trips before expecting PAP to approach PD throughput.
