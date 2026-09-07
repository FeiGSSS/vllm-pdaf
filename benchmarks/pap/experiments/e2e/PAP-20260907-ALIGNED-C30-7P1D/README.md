# PD 7P1D at concurrency 30

This record extends `PAP-20260907-ALIGNED-C30` with a 7-Prefill / 1-Decode
Dynamo topology. It uses the same Qwen3-8B FP16 native-32K workload, 60
three-turn sessions, client concurrency 30, routing settings, scheduler
budgets, graph configuration, hardware, and dependency isolation.

Two independent service starts are measured. Each run must complete 180
requests and pass effective-configuration, process topology, GPU UUID, CUDA
Graph, KV transfer, correctness, and cleanup audits. Both runs passed.

The shared Dynamo runner did not previously accept 7P1D. This experiment adds
that topology to the production runner and validates it without changing the
DP/PD model implementation.

## Results

| Measurement | Run 1 | Run 2 | Combined |
| --- | ---: | ---: | ---: |
| TTFT mean | 8.932 s | 8.844 s | 8.888 s |
| Request-level ITL mean | 67.574 ms | 66.360 ms | 66.967 ms |
| E2E latency mean | 14.367 s | 14.193 s | 14.280 s |
| Request throughput | 1.826 req/s | 1.822 req/s | 1.824 req/s |
| Output throughput | 149.057 tok/s | 148.767 tok/s | 148.912 tok/s |

The combined percentiles use all 360 request records and the same linear
percentile method as AIPerf:

| Metric | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 9.063 s | 13.654 s | 14.603 s |
| Request-level ITL | 68.652 ms | 74.498 ms | 79.063 ms |
| E2E latency | 14.328 s | 22.545 s | 25.876 s |

ITL is AIPerf `inter_token_latency`: the mean generation interval for each
request. Its percentiles describe the distribution of request-level means,
not the distribution of every individual streamed token gap.

Against the existing concurrency-30 measurements, 7P1D has 55.98% higher
mean TTFT and 36.69% higher mean ITL than 6P2D. It has 65.69% higher mean TTFT
and 49.31% higher mean ITL than PAP 7PA1P. These are measured ratios, not a
mechanism diagnosis.

During the client request windows, mean physical-GPU utilization was 54.04%,
mean memory utilization was 17.94%, and mean power was 222.16 W per GPU. The
physical-GPU aggregate hides the role imbalance between seven Prefill GPUs and
the single Decode GPU; per-GPU time series remain in each raw AIPerf record.

No goodput SLO was configured. The raw request records contain TTFT, ITL, E2E
latency, and timestamps, so goodput can be computed later under a shared SLO.

Both runs materialized identical AIPerf input bytes with SHA256
`ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5`.
Conversation/turn identities and output lengths also match the prior C30
architectures. The second run replayed the first run's effective configuration;
only run IDs, output paths, namespace, and discovery paths changed.
