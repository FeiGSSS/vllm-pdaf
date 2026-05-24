# PAP Experiment Results

This file records local PAP validation runs for the `feature/pap-true-split`
branch. The raw service logs live under `examples/pap/logs/` and are ignored by
git; this document keeps the reproducible command shape and the key outcomes.

## Qwen3-0.6B 1PA1P High Input/Output

Command shape:

```bash
PAP_TOPOLOGY=1pa1p \
PAP_SERVICE_ONLY=1 \
PAP_SKIP_SMOKE_REQUEST=1 \
PAP_PROXY_PORT=9090 \
PAP_PREFILL_GPUS=4 \
PAP_PROJECTION_GPUS=5 \
PAP_MAX_MODEL_LEN=2048 \
PAP_MAX_NUM_SEQS=2 \
PAP_OFFLOAD_EXEC_TRACE=1 \
PAP_ENABLE_MPS=1 \
bash examples/pap/launch_pap_nixl.sh \
  --model /data/ssd1/llm-models/Qwen3-0.6B
```

Request:

- Prompt size: 5,891 characters.
- OpenAI usage: `prompt_tokens=1185`, `completion_tokens=256`,
  `total_tokens=1441`.
- `max_tokens=256`, `temperature=0`.

Result:

- HTTP status: `200`.
- Finish reason: `length`.
- End-to-end curl latency: `10206 ms`.
- Proxy prefill metrics: `prefill_ms=128`, `prefill_prefix_len=1185`.
- Projection reported `External prefix cache hit rate: 100.0%`.
- Attention OFFLOAD_EXEC traces: `7168`.
- Projection OFFLOAD_EXEC traces: `7168`.
- Trace count matches `256 output tokens * 28 layers`.
- Output started with normal model text:
  `"<think>\nOkay, let's tackle this..."`.
- No `ERROR`, `Traceback`, `Exception`, `RuntimeError`, or `HTTPStatusError`
  entries were found in the PAP service logs.

## Qwen3-0.6B 4PA4P Round-Robin Routing

Command shape:

```bash
PAP_TOPOLOGY=4pa4p \
PAP_SERVICE_ONLY=1 \
PAP_SKIP_SMOKE_REQUEST=1 \
PAP_PROXY_PORT=9090 \
PAP_MODEL_PATH=/data/ssd1/llm-models/Qwen3-0.6B \
PAP_MAX_MODEL_LEN=1024 \
PAP_MAX_NUM_SEQS=2 \
PAP_OFFLOAD_EXEC_TRACE=1 \
PAP_ENABLE_MPS=1 \
bash examples/pap/launch_pap_nixl.sh
```

Request set:

- Sent 8 sequential `/v1/chat/completions` requests through the multi-PAP
  proxy.
- Each request used `max_tokens=8`, `temperature=0`.
- Each response reported `prompt_tokens=24`, `completion_tokens=8`,
  `total_tokens=32`, and `finish_reason=length`.

Round-robin assignment observed in `proxy.log`:

| Request | Prefill | Attention | Projection | Latency |
| --- | --- | --- | --- | --- |
| 0 | `8190` | `8390` | `8290` | `5257 ms` |
| 1 | `8191` | `8391` | `8291` | `5269 ms` |
| 2 | `8192` | `8392` | `8292` | `5232 ms` |
| 3 | `8193` | `8393` | `8293` | `5262 ms` |
| 4 | `8190` | `8390` | `8290` | `212 ms` |
| 5 | `8191` | `8391` | `8291` | `219 ms` |
| 6 | `8192` | `8392` | `8292` | `212 ms` |
| 7 | `8193` | `8393` | `8293` | `225 ms` |

Result:

- All 8 requests returned HTTP `200`.
- Outputs were valid model text, starting with
  `"<think>\nOkay, the user is asking"`.
- First request per PA/P pair paid a warmup/initialization cost of about
  `5.2 s`; the second round completed in about `0.21 s`.
- Each Attention executor logged `448` PAP OFFLOAD_EXEC traces.
- Each Projection logged `448` PAP OFFLOAD_EXEC traces.
- Per-instance trace count matches `2 requests * 8 output tokens * 28 layers`.
- All four Projections reported `External prefix cache hit rate: 100.0%`.
- No `ERROR`, `Traceback`, `Exception`, `RuntimeError`, or `HTTPStatusError`
  entries were found in the PAP service logs.

## Qwen3-0.6B 1PA1P KV-Unaware Projection Phase 1

Command shape:

```bash
PAP_TOPOLOGY=1pa1p \
PAP_SERVICE_ONLY=1 \
PAP_SKIP_SMOKE_REQUEST=1 \
PAP_PROXY_PORT=9090 \
PAP_PREFILL_GPUS=4 \
PAP_PROJECTION_GPUS=5 \
PAP_MAX_MODEL_LEN=2048 \
PAP_MAX_NUM_SEQS=2 \
PAP_OFFLOAD_EXEC_TRACE=1 \
PAP_ENABLE_MPS=1 \
bash examples/pap/launch_pap_nixl.sh \
  --model /data/ssd1/llm-models/Qwen3-0.6B
```

Request:

- Endpoint: `POST /v1/chat/completions`.
- OpenAI usage: `prompt_tokens=628`, `completion_tokens=96`,
  `total_tokens=724`.
- `max_tokens=96`, `temperature=0`.

Result:

- HTTP status: `200`.
- End-to-end request latency: `7004 ms`.
- Proxy prefill metrics: `prefill_ms=77`, `prefill_prefix_len=628`.
- Output started with normal Chinese model text:
  `"<think>\n好的，用户让我用三段话解释PAP架构下..."`.
- Projection reported prompt throughput `0.0 tokens/s`, confirming it did not
  run the 628-token prompt prefill locally.
- Attention OFFLOAD_EXEC traces: `2688`.
- Projection OFFLOAD_EXEC traces: `2688`.
- Trace count matches `96 output tokens * 28 layers`.
- No `ERROR`, `Traceback`, `Exception`, `RuntimeError`, or `HTTPStatusError`
  entries were found in the PAP service logs.
