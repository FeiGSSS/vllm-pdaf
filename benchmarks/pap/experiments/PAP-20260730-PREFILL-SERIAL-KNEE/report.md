# PAP Prefill serial saturation knee

Date: 2026-07-30

## Question

At what single-request prompt length does the current PAP Prefill partition
stop gaining token throughput from additional work?

This experiment complements
`PAP-20260730-PREFILL-SATURATION`, which compared multiple requests in one
batch. Here every point is strictly `B=1`: one request completes before the
next request is submitted.

## Method

- Qwen3-8B FP16 on one NVIDIA L20
- PAP Prefill static-MPS partition: 18 chunks, 72 visible SMs
- eager execution, chunked Prefill enabled, Prefix Cache disabled
- `max_num_batched_tokens=32768`, `max_num_seqs=256`
- exact random token-ID prompts and one generated token
- one warmup and three measured repetitions per length
- primary time: vLLM EngineCore context-iteration elapsed time
- saturation onset: first measured length reaching 95% of peak prompt tok/s

The runner uses the level-0 scheduling gate and audits each sample as exactly
one request in one context iteration. All 63 measured samples across the
coarse and refinement scans passed.

## Results

| Prompt tokens | Prefill median | Prompt throughput | Peak fraction |
| ---: | ---: | ---: | ---: |
| 64 | 20.08 ms | 3,187 tok/s | 48.9% |
| 128 | 21.84 ms | 5,861 tok/s | 89.8% |
| 144 | 28.78 ms | 5,003 tok/s | 76.7% |
| 160 | 29.07 ms | 5,504 tok/s | 84.4% |
| 176 | 29.10 ms | 6,048 tok/s | 92.7% |
| **192** | **29.43 ms** | **6,524 tok/s** | **100.0%** |
| 208 | 39.17 ms | 5,310 tok/s | 81.4% |
| 224 | 39.85 ms | 5,621 tok/s | 86.2% |
| 240 | 40.31 ms | 5,954 tok/s | 91.3% |
| 256 | 40.00 ms | 6,400 tok/s | 98.1% |
| 320 | 50.00 ms | 6,400 tok/s | 98.1% |
| 384 | 60.70 ms | 6,326 tok/s | 97.0% |
| 512 | 80.87 ms | 6,331 tok/s | 97.0% |
| 768 | 122.91 ms | 6,248 tok/s | 95.8% |
| 1K | 166.12 ms | 6,020 tok/s | 92.3% |
| 2K | 345.11 ms | 5,795 tok/s | 88.8% |
| 4K | 715.08 ms | 5,594 tok/s | 85.7% |
| 8K | 1549.26 ms | 5,164 tok/s | 79.2% |
| 10K | 2025.44 ms | 4,937 tok/s | 75.7% |
| 20K | 4844.90 ms | 4,128 tok/s | 63.3% |
| 30K | 8469.13 ms | 3,542 tok/s | 54.3% |

Peak throughput is 6,524 prompt tok/s at 192 tokens. The registered 95%
threshold is 6,198 tok/s, so the first measured point reaching it is 192
tokens.

The short-length curve is not monotonic: 192 is a favorable local shape, and
208--240 temporarily regress. This is consistent with discrete kernel
shape/tiling boundaries, but a kernel trace would be required to assign the
cause. A more robust interpretation is that the sustained high-efficiency
region begins around 256 tokens: every sampled point from 256 through 768
remains above 95% of peak. At 1K, long-sequence Attention cost has already
pulled throughput below that threshold.

## Conclusion

For the current 72-SM eager Prefill configuration:

- formal first-95% saturation point: **192 tokens**;
- robust sustained saturation region: approximately **256--768 tokens**;
- 10K requests are far beyond the utilization knee and achieve only 75.7% of
  peak token throughput because causal Attention increasingly dominates.

Therefore `max_num_batched_tokens=32768` is only a scheduler budget, not a
desirable Prefill batch target. The earlier `3x10K` experiment is now easier
to interpret: each individual request is already far past the saturation
onset, so combining several such requests cannot improve aggregate Prefill
throughput.

This knee is specific to Qwen3-8B, FP16, eager execution, and the 72-SM MPS
partition. CUDA Graph execution, another model, or another SM allocation
requires a separate curve.

## Evidence

- Combined structured result: `raw/result.json`
- Coarse scan result: `raw/coarse_result.json`
- Refinement result: `raw/refine_result.json`
- Coarse EngineCore log: `raw/coarse_engine.log`
- Refinement EngineCore log: `raw/refine_engine.log`
- Effective configurations: `raw/coarse_config.env`,
  `raw/refine_config.env`
- Reproduction:

  ```bash
  PAP_PREFILL_MICROBENCH_GROUPS=serial \
    benchmarks/pap/scripts/run_prefill_saturation.sh

  PAP_PREFILL_MICROBENCH_GROUPS=serial_refine \
    benchmarks/pap/scripts/run_prefill_saturation.sh
  ```
