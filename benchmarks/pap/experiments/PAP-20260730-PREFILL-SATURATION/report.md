# PAP Prefill compute saturation

Date: 2026-07-30

## Question

Does admitting several long Prefill requests into one PAP PA forward improve
aggregate Prefill efficiency, or does it only delay every request and interfere
with the colocated Decode Attention process?

## Method

- Qwen3-8B FP16 on one NVIDIA L20
- the PAP Prefill static-MPS partition: 18 chunks, 72 visible SMs
- eager execution, `max_num_batched_tokens=32768`, `max_num_seqs=256`
- chunked Prefill enabled; Prefix Cache disabled
- exact random token-ID prompts; one generated token per request
- one warmup and three measured repetitions per shape
- primary latency: vLLM EngineCore context-iteration elapsed time
- secondary latency: `wake_up()` through completion wall time

The runner pauses only scheduling with `LLM.sleep(level=0)`, enqueues the whole
shape, and resumes with `wake_up(tags=["scheduling"])`. Every measured sample
was audited as exactly one context iteration containing the requested number
of requests and prompt tokens. Decode, remote Attention, and NIXL transfer are
not part of this microbenchmark.

## 1. Fixed 10K tokens per request

| Requests | Total prompt | Prefill median | Prompt throughput | Time / B1 | Throughput / B1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10K | 2025.57 ms | 4,937 tok/s | 1.000x | 1.000x |
| 2 | 20K | 4064.65 ms | 4,920 tok/s | 2.007x | 0.997x |
| 3 | 30K | 6140.97 ms | 4,885 tok/s | 3.032x | 0.990x |

The 30K three-request batch is 1.06% slower than three isolated 10K Prefills
executed serially. Batching therefore provides no aggregate throughput benefit
at this request length.

For three simultaneously available requests, isolated serial service would
complete them at approximately 2.03, 4.05, and 6.08 seconds, for a 4.05-second
mean completion time. A single three-request forward completes all three at
6.14 seconds. Serial admission reduces mean Prefill completion time by about
34.0% without increasing the final makespan.

## 2. Batch-size scan at fixed request lengths

| Prompt/request | Scanned request counts | Best observed throughput | B1 throughput | B1 within 95% of peak |
| ---: | --- | ---: | ---: | :---: |
| 1K | 1, 2, 4, 8, 16, 24, 32 | 5,969 tok/s | 5,969 tok/s | yes |
| 2K | 1, 2, 4, 8, 12, 16 | 5,793 tok/s | 5,726 tok/s | yes |
| 5K | 1, 2, 3, 4, 6 | 5,454 tok/s | 5,454 tok/s | yes |
| 10K | 1, 2, 3 | 4,937 tok/s | 4,937 tok/s | yes |

Within the scanned range, even one 1K-token request is already within 95% of
the best throughput observed at that request length. This does not prove the
hardware-utilization knee is exactly 1K; it bounds the knee to at most 1K for
the current 72-SM eager configuration. A sub-1K scan is required to locate the
lower-length knee.

The follow-up `PAP-20260730-PREFILL-SERIAL-KNEE` experiment completed that
scan. Its first measured 95%-of-peak point is 192 tokens, with a sustained
high-efficiency region at approximately 256--768 tokens.

## 3. Approximately 30K total prompt tokens

| Prompt/request | Requests | Prefill median | Prompt throughput |
| ---: | ---: | ---: | ---: |
| 1K | 30 | 5108.45 ms | 5,873 tok/s |
| 2K | 15 | 5227.79 ms | 5,739 tok/s |
| 5K | 6 | 5569.83 ms | 5,386 tok/s |
| 10K | 3 | 6140.97 ms | 4,885 tok/s |

At nearly equal total tokens, `3x10K` is 20.2% slower than `30x1K`; prompt
throughput is 16.8% lower. Total scheduled tokens alone therefore do not
describe Prefill cost. Per-sequence length matters because causal Prefill
Attention grows quadratically with sequence length.

## Conclusion

For the current long-context workload, the admission=1 result is not explained
by a mysterious `32768` scheduler limit. A single 10K request already extracts
essentially all Prefill token throughput available from the 72-SM partition.
Admitting two or three such requests together does not finish the total work
faster; it delays every request to the end of the larger forward and increases
interference time with colocated Decode Attention.

This result directly supports retaining per-PA Prefill admission=1 for the
current 10K-per-turn testbed. It does not establish admission=1 as a universal
policy for short prompts, different models, CUDA Graph execution, or a
different MPS split.

## Validity and evidence

- All 69 measured samples passed the exact single-context-iteration audit.
- An earlier direct-`generate(list)` pilot was rejected because async
  scheduling split many requested shapes across two iterations; none of those
  numbers are used above.
- Primary analyzed data: `raw/result.json`
- Engine iteration log: `raw/engine.log`
- Effective runtime configuration: `raw/effective_config.env`
- Reproduction:

  ```bash
  benchmarks/pap/scripts/run_prefill_saturation.sh
  ```
