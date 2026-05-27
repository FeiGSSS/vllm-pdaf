# Qwen3-32B PD Decode-Side Layer Profile (TP2 P + TP2 D)

Date: 2026-05-27

Configuration:
- 4 GPUs total: prefill instance TP=2 on GPU0-1, decode instance TP=2 on GPU2-3.
- Model: /data/ssd1/llm-models/Qwen3-32B.
- KV connector: NixlConnector, kv_role=kv_both.
- Proxy: tests/v1/kv_connector/nixl_integration/toy_proxy_server.py.
- D-only profiler: VLLM_QWEN3_LAYER_PROFILE enabled only on the decode instance.
- Shell HTTP proxy variables were unset for server and benchmark commands.
- enforce_eager=True was used so CUDA-event layer scopes are visible; this disables CUDA Graph replay.

Why this replaces the single-instance run:
- The earlier single-process LLM.generate profile mixed chunked prefill and decode scheduling. Configured batch was not equal to actual decode batch, so it could not answer the D-side decode-layer question directly.
- In this run, only the decode instance emits Qwen3 layer samples, and raw_samples.csv records the actual D-side batch size observed by each layer scope.

Server capacity observed:
- ctx128/max_model_len=160/gpu_memory_utilization=0.90/max_num_batched_tokens=8192: D KV cache 42,640 tokens, max concurrency 266.5x.
- ctx2048/max_model_len=2112/gpu_memory_utilization=0.95/max_num_batched_tokens=8192: D KV cache 83,858 tokens, max concurrency 39.71x.

Benchmark summary:
| run | success | input tokens | output tokens | req/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| ctx=128, configured batch=512, output=16 | 512 | 65536 | 8192 | 18.16 | 17248.07 | 97.81 |
| ctx=2048, configured batch=64, output=16 | 64 | 131072 | 1024 | 1.22 | 29155.77 | 84.55 |
| ctx=2048, configured batch=64, output=64 | 64 | 131072 | 4096 | 1.13 | 28900.36 | 86.08 |

Actual D-side batch distribution:

- ctx128_bs512/decode_samples_8192: actual batch min=1, max=512, unique=32, top=[(3, 23296), (5, 20608), (2, 19712), (59, 16128), (84, 14336), (8, 8960), (54, 8960), (10, 7168), (75, 6272), (87, 6272), (71, 5376), (61, 5376)]
- ctx2048_bs64/decode_samples: actual batch min=1, max=64, unique=11, top=[(8, 316288), (4, 295680), (1, 74368), (7, 45696), (5, 39424), (3, 25984), (6, 11648), (11, 8064), (10, 4480), (64, 896), (2, 896)]

Actual-batch timing bins:
| run | actual batch range | sample count | actual batch mean | attention ms/layer | projection core ms/layer | projection+LN ms/layer | qkv ms | o_proj ms | mlp ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx128_bs512 | 1-15 | 91392 | 4.6 | 0.0918 | 0.8377 | 0.8709 | 0.0934 | 0.1031 | 0.5942 |
| ctx128_bs512 | 16-31 | 8064 | 29.1 | 0.0993 | 0.9844 | 1.0186 | 0.0993 | 0.1477 | 0.6888 |
| ctx128_bs512 | 32-63 | 51072 | 57.1 | 0.1082 | 1.0220 | 1.0546 | 0.0997 | 0.1593 | 0.7163 |
| ctx128_bs512 | 64-127 | 49280 | 76.4 | 0.1179 | 1.0962 | 1.1300 | 0.1093 | 0.1833 | 0.7558 |
| ctx128_bs512 | 256-512 | 896 | 512.0 | 0.2383 | 3.0621 | 3.1094 | 0.2693 | 0.5720 | 2.1708 |
| ctx2048_bs64 | 1-15 | 822528 | 5.6 | 0.1172 | 0.8410 | 0.8738 | 0.0926 | 0.1058 | 0.5956 |
| ctx2048_bs64 | 64-127 | 896 | 64.0 | 0.1290 | 1.0334 | 1.0677 | 0.1021 | 0.1622 | 0.7210 |

Main observations:
- Short-context projection stress did produce a transient actual D batch of 512. At that point projection core rose to 3.06 ms/layer and projection+LN to 3.11 ms/layer, while attention was 0.24 ms/layer.
- For ctx2048, even with output=64, actual D batch mostly stayed below 15 because prefill completion paced arrivals into D and the D capacity was about 40 concurrent requests. The rare actual batch=64 slice had attention 0.129 ms/layer and projection core 1.03 ms/layer.
- Therefore, on this 32B TP2 + L20 setup, projection can become materially longer only when actual D batch is very large; long-context attention did not become the dominant per-layer compute term in the observed PD decode batches.
