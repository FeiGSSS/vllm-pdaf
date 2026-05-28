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

This run does not yet produce a valid PD-vs-PAP performance comparison for the intended 3:1 topology. PD completed, but PAP 3PA1P TP2 does not complete the 512-burst workload.

The important finding is that the failure is no longer model size or TP=2 support in general. A single 1PA1P TP2 PAP pair completes. The failing path is the 3PA1P topology where one projection instance talks to multiple PA attention endpoint pairs under concurrent decode. In that path, projection enters remote attention, waits on the TCP/NCCL response, and eventually times out or hangs. Logs show prefill KV import succeeded before projection stalled.

So the next blocker is the multi-PA remote attention data plane/scheduling path, not the high-level experiment design.

## Next steps

1. Add structured per-layer remote attention traces for TP2 multi-PA: projection send, attention TCP receive, NCCL recv_qkv, attention compute, NCCL send_o, projection recv_o.
2. Reproduce with 3PA1P TP2 at smaller loads: bs=4, 8, 16, output=8 first, then output=64.
3. Test whether serializing projection remote-attention calls by attention endpoint avoids the hang. If yes, the issue is concurrent multi-comm scheduling in the current P2P/NCCL transport.
4. After 3PA1P TP2 is stable, rerun the intended ctx128/bs512/o64 comparison and only then interpret overlap performance.
