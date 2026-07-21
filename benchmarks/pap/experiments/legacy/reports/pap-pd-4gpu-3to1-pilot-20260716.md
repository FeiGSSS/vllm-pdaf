# PAP/PD 4-GPU 3:1 capacity pilot

Date: 2026-07-16
Status: controlled pilot, not a formal repeated result

## Question

Under the same four-GPU budget, can `3PA1P` benefit from distributing KV across
three PA nodes while one KV-unaware Projection node batches decode work?

## Matched workload

- Hardware: four NVIDIA L20 GPUs, GPUs 0--3.
- Model: Qwen3-8B, float16, eager V1 runner.
- Request shape: input 4096, output 64, 48 requests submitted as one burst.
- Scheduler limit: 128 sequences and 8192 batched tokens.
- PD: `3P1D`, Prefill GPUs 0--2, Decode GPU 3, NIXL.
- PAP: `3PA1P`, PA GPUs 0--2, Projection GPU 3, `local_fast`.
- PAP PA partition: 72 Prefill SMs and 20 Attention SMs.
- Routing: round-robin. No cache-aware routing was active.

The first PD startup attempt is invalid and excluded: it omitted
`VLLM_USE_FLASHINFER_SAMPLER=0` and failed while JIT-compiling the FlashInfer
sampler. The recorded PD run uses the same sampler setting as PAP.

## Result

| Metric | PD 3P1D | PAP 3PA1P | PAP / PD |
| --- | ---: | ---: | ---: |
| Successful requests | 48/48 | 48/48 | -- |
| Duration | 25.96 s | 14.33 s | 0.552x |
| Request throughput | 1.85 req/s | 3.35 req/s | 1.811x |
| Output throughput | 118.33 tok/s | 214.34 tok/s | 1.811x |
| Median TTFT | 19,985.48 ms | 8,294.36 ms | 0.415x |
| Median TPOT | 59.66 ms | 32.44 ms | 0.544x |

PAP strict correctness and routing audits passed. The Gateway sent 16 requests
to each PA and all 48 requests to the single Projection. All three Attention
services drained to zero sessions.

## Capacity evidence

- Each PAP PA exposed 127,920 KV tokens; aggregate PA capacity was 383,760
  tokens.
- The PD Decode exposed 173,200 KV tokens. Its runtime log reached 99.1% KV
  usage with 42 running and 3 waiting requests.
- PAP Projection logged metadata-only KV placeholders and 0.0% KV usage.
- Prefix-cache hit rate was only about 0.7--0.8% in both systems. The result is
  therefore not caused by prefix-cache affinity.

This pilot supports the 3:1 capacity hypothesis: PAP avoids concentrating all
request KV and the full Prefill-to-Decode KV transfer on one decode GPU. It is
not yet a formal claim; a short-context point and an alternating repeated run
are still needed to establish the crossover and run-to-run variance.

The follow-up conversation-affine, five-round 4K+3K comparison is documented
in
[`../../PAP-20260716-4GPU-CONV-AFFINITY/report.md`](../../PAP-20260716-4GPU-CONV-AFFINITY/report.md).

## Raw evidence

- PAP: `benchmarks/pap/experiments/legacy/runs/20260716_4gpu_pilot_pap_3pa1p_i4096_o64_n48`
- PD: `/home/fei/research/PD/test/baseline/nixl_disaggregated/results/runs/20260716_143826`
