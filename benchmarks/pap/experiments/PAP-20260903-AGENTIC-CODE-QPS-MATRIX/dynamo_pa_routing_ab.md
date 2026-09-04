# Dynamo-managed PAP routing A/B

This A/B uses the canonical 60-session, three-turn Agentic Coding workload:
180 requests, Poisson 0.9 req/s, concurrency 60, no warmup, Qwen3-8B, and
7PA1P. All runs completed 180/180 requests with no correctness, routing, CUDA
Graph, or lifecycle audit failure.

## Results

| Prefill budget | Router | TTFT (ms) | TBT (ms) | E2E (ms) | Output token/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 2K | PAP conversation affinity | 1735.40 | 52.44 | 25934.32 | 275.91 |
| 2K | Dynamo, Prefill scale 2 | 1592.77 | 43.88 | 21973.98 | 290.38 |
| 32K | PAP conversation affinity | 1598.56 | 48.28 | 23968.93 | 283.86 |
| 32K | Dynamo default, Prefill scale 1 | 1683.73 | 44.09 | 22179.46 | 288.65 |
| 32K | Dynamo, Prefill scale 2 | 1610.53 | 45.71 | 22833.86 | 285.37 |

Against the current PAP router, Dynamo with Prefill scale 2 improves the 2K
case by 8.2% TTFT, 16.3% TBT, 15.3% end-to-end latency, and 5.2% output
throughput. In the 32K case it changes TTFT by +0.75%, while improving TBT by
5.3% and end-to-end latency by 4.7%.

## Interpretation

Dynamo routes every turn independently. It subscribes directly to each
Prefill worker's vLLM KV events and combines device-local prefix overlap,
reserved Prefill tokens, and unique active Decode blocks. It does not migrate
KV between PA workers. When a later turn selects another PA, that PA reuses any
resident prefix it already has and computes the remaining prompt.

The default Dynamo weight favors Decode balance for this PAP workload. In the
32K run, 52 of 60 conversations changed PA at least once. On turn 2, requests
that changed PA missed 5845 prompt tokens on average versus 3290 for requests
that stayed, and their mean TTFT was 2332 ms versus 1707 ms. This explains the
default router's TTFT regression despite its 8.7% TBT improvement.

Raising `prefill_load_scale` from 1 to 2 moves the policy toward prefix reuse.
It retains useful Decode balancing while making lost-prefix work expensive
enough for PAP's long-prompt workload. The measured 2K and 32K results support
2.0 as the PAP-Dynamo default.

Raw runs are under:

- `results/pap_7pa1p_dynamo_32k/qps_0p9/attempt_001`;
- `results/pap_7pa1p_dynamo_s2_32k/qps_0p9/attempt_001`;
- `results/pap_7pa1p_dynamo_s2_2k/qps_0p9/attempt_001`.
