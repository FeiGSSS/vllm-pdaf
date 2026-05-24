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

## Qwen3-0.6B 1PA1P Metadata-Only Projection Without KVConnector

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

Runtime difference from the prior run:

- Projection was launched as `PAP Projection vLLM metadata-only`.
- Projection vLLM non-default args did not include `kv_transfer_config`.
- Proxy reported Projection KV metadata keys:
  `['pap_attention_endpoint', 'pap_attention_kv_installed',
  'pap_attention_tcp_endpoint', 'pap_offload_exec_zmq_endpoint',
  'pap_prefill_kv_handle', 'pap_projection_kv_unaware',
  'pap_remote_prefix_len']`.
- No Prefill KV transport keys were present in the Projection payload.

Request:

- Endpoint: `POST /v1/chat/completions`.
- OpenAI usage: `prompt_tokens=344`, `completion_tokens=64`,
  `total_tokens=408`.
- `max_tokens=64`, `temperature=0`.

Result:

- HTTP status: `200`.
- End-to-end request latency: `6310 ms`.
- Proxy prefill metrics: `prefill_ms=63`, `prefill_prefix_len=344`.
- Output started with normal Chinese model text:
  `"<think>\n嗯，用户让我简要说明PAP架构中Projection节点..."`.
- Projection OFFLOAD_EXEC traces: `1792`.
- Attention OFFLOAD_EXEC traces: `1792`.
- Trace count matches `64 output tokens * 28 layers`.
- Projection logs had no `Got kv_transfer_params, but no KVConnector found`
  warning for the PAP metadata-only request.
- No `ERROR`, `Traceback`, `Exception`, `RuntimeError`, or `HTTPStatusError`
  entries were found in the Projection/proxy logs.

## Qwen3-0.6B 4PA2P Metadata-Only Projection X:Y Routing

Command shape:

```bash
PAP_TOPOLOGY=4pa2p \
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

Runtime contract:

- Four PA groups used GPUs `0,1,2,3`.
- Two metadata-only Projection nodes used GPUs `4,5`.
- Projection vLLM non-default args did not include `kv_transfer_config`.
- Proxy reported Projection KV metadata keys:
  `['pap_attention_endpoint', 'pap_attention_kv_installed',
  'pap_attention_tcp_endpoint', 'pap_offload_exec_zmq_endpoint',
  'pap_prefill_kv_handle', 'pap_projection_kv_unaware',
  'pap_remote_prefix_len']`.
- No Prefill KV transport keys were present in any Projection payload.

Request set:

- Sent 8 sequential `/v1/chat/completions` requests through the multi-PAP
  proxy.
- Each request used `max_tokens=12`, `temperature=0`.
- Each response reported `prompt_tokens=22`, `completion_tokens=12`,
  `total_tokens=34`.

Round-robin assignment observed in `proxy.log`:

| Request | Prefill | Attention | Projection | Latency |
| --- | --- | --- | --- | --- |
| 0 | `8100` | `8300` | `8200` | `5383 ms` |
| 1 | `8101` | `8301` | `8201` | `5290 ms` |
| 2 | `8102` | `8302` | `8200` | `471 ms` |
| 3 | `8103` | `8303` | `8201` | `444 ms` |
| 4 | `8100` | `8300` | `8200` | `299 ms` |
| 5 | `8101` | `8301` | `8201` | `271 ms` |
| 6 | `8102` | `8302` | `8200` | `295 ms` |
| 7 | `8103` | `8303` | `8201` | `271 ms` |

Result:

- All 8 requests returned HTTP `200`.
- Outputs were valid model text, starting with
  `"<think>\n好的，用户让我用一句话说明metadata-only Projection"`.
- Projection 0 logged `1344` OFFLOAD_EXEC traces.
- Projection 1 logged `1344` OFFLOAD_EXEC traces.
- Each Attention executor logged `672` OFFLOAD_EXEC traces.
- Projection trace count matches `4 requests * 12 output tokens * 28 layers`.
- Attention trace count matches `2 requests * 12 output tokens * 28 layers`.
- Projection logs had no `Got kv_transfer_params, but no KVConnector found`
  warning for PAP metadata-only requests.
- No `ERROR`, `Traceback`, `Exception`, `RuntimeError`, or `HTTPStatusError`
  entries were found in the Projection/Attention/proxy logs.

## Qwen3-0.6B PAP OFFLOAD_KV CUDA IPC Validation

Code checkpoint:

- `312ae6fbb` added PAP OFFLOAD_KV IPC descriptors.
- `a418ae539` added Attention executor `import_prefill_kv_ipc` handling.
- `913db9dad` added Prefill/Profile CUDA IPC descriptor export.
- `e22316c9a` made `PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc` the launcher default.
- `f945899e4` made real PyTorch CUDA IPC handles JSON-safe with pickled
  metadata inside the control payload.
- `4a9d567d5` added explicit IPC import logging.

Focused unit verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `70 passed`.

### 1PA1P OFFLOAD_KV IPC

Command shape:

```bash
PAP_TOPOLOGY=1pa1p \
PAP_SERVICE_ONLY=1 \
PAP_SKIP_SMOKE_REQUEST=1 \
PAP_PROXY_PORT=9000 \
PAP_MAX_TOKENS=32 \
PAP_ATTENTION_KV_DEBUG=1 \
PAP_OFFLOAD_EXEC_TRACE=1 \
bash examples/pap/launch_pap_nixl.sh \
  --model /data/ssd1/llm-models/Qwen3-0.6B
```

Request result:

- Endpoint: `POST /v1/completions` through the multi-PAP proxy.
- HTTP status: `200`.
- Usage: `prompt_tokens=18`, `completion_tokens=32`, `total_tokens=50`.
- Output was normal model text, not binary garbage or decode failure.
- Proxy logged Projection metadata keys only:
  `['pap_attention_endpoint', 'pap_attention_kv_installed',
  'pap_attention_tcp_endpoint', 'pap_offload_exec_zmq_endpoint',
  'pap_prefill_kv_handle', 'pap_projection_kv_unaware',
  'pap_remote_prefix_len']`.
- Attention logged `28` `PAP prefill KV imported via IPC descriptor` entries,
  one per Qwen3-0.6B layer.
- Projection logged `896` OFFLOAD_EXEC traces.
- Attention logged `896` OFFLOAD_EXEC traces.
- Trace count matches `32 completion tokens * 28 layers`.
- `kv_transfer_config` appeared only in the Prefill producer log, not in the
  Projection vLLM startup arguments.
- No `Traceback`, `ERROR`, `rejected`, or `Got kv_transfer_params` log entries
  were found in the PAP service logs.

## Projection Local Block-State Tightening

### 2026-05-24 Position-Only PAP OFFLOAD_EXEC Descriptor

The next stateless Projection tightening pass removed a Projection-side
dependency on vLLM slot/block-table metadata from the Qwen3 PAP attention path:

- `Qwen3Attention._compute_pap_attention()` now derives the PAP decode
  `seq_len` directly from scheduler positions.
- It no longer reads `forward_context.slot_mapping`.
- It no longer computes or sends `block_id` / `slot` from Projection-local
  block tables.
- The Attention side already derives the decode block/slot from OFFLOAD_EXEC
  `step` / `seq_len`, so the active NCCL compact data path does not need
  Projection-local block identifiers.
- The scheduler PAP Projection admission path now sends only the newly allocated
  local blocks to the model runner for metadata-only PAP requests instead of
  handing off the full request block list that includes remote-prefix external
  block placeholders.

Verification:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_true_split_contract.py -q
```

Result: `19 passed`.

Focused PAP suite:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `71 passed`.

1PA1P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=1pa1p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=16 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Request returned HTTP `200`.
- Usage: `prompt_tokens=29`, `completion_tokens=16`, `total_tokens=45`.
- Proxy logged Projection metadata keys only:
  `pap_attention_endpoint`, `pap_attention_kv_installed`,
  `pap_attention_tcp_endpoint`, `pap_offload_exec_zmq_endpoint`,
  `pap_prefill_kv_handle`, `pap_projection_kv_unaware`,
  `pap_remote_prefix_len`.
- Attention logged `28` IPC descriptor imports.
- Projection logged `448` OFFLOAD_EXEC traces, matching
  `16 completion tokens * 28 layers`.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

4PA2P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=4pa2p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=8 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Sent 8 sequential `/v1/completions` requests; all returned HTTP `200`.
- Each response reported `prompt_tokens=19`, `completion_tokens=8`,
  `total_tokens=27`.
- Route coverage:
  - `8100/8300 -> 8200`
  - `8101/8301 -> 8201`
  - `8102/8302 -> 8200`
  - `8103/8303 -> 8201`
  - repeated once.
- Proxy logged Projection metadata keys only for all 8 requests.
- Attention logged `224` IPC descriptor imports: `56` per Attention executor.
- Projection logged `1792` OFFLOAD_EXEC traces: `896` per Projection executor,
  matching `8 requests * 8 tokens * 28 layers`.
- Attention logged `1792` OFFLOAD_EXEC traces: `448` per Attention executor,
  matching `2 requests * 8 tokens * 28 layers`.
- `kv_transfer_config` appeared only in Prefill logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

### 4PA2P OFFLOAD_KV IPC X:Y Routing

Command shape:

```bash
PAP_TOPOLOGY=4pa2p \
PAP_SERVICE_ONLY=1 \
PAP_SKIP_SMOKE_REQUEST=1 \
PAP_PROXY_PORT=9000 \
PAP_MAX_TOKENS=12 \
PAP_ATTENTION_KV_DEBUG=1 \
PAP_OFFLOAD_EXEC_TRACE=1 \
bash examples/pap/launch_pap_nixl.sh \
  --model /data/ssd1/llm-models/Qwen3-0.6B
```

Request set:

- Sent 8 sequential `/v1/completions` requests through the multi-PAP proxy.
- Each response returned HTTP `200`.
- Each response reported `prompt_tokens=15`, `completion_tokens=12`,
  `total_tokens=27`.

Observed route coverage:

| Request | Prefill | Attention | Projection |
| --- | --- | --- | --- |
| 0 | `8100` | `8300` | `8200` |
| 1 | `8101` | `8301` | `8201` |
| 2 | `8102` | `8302` | `8200` |
| 3 | `8103` | `8303` | `8201` |
| 4 | `8100` | `8300` | `8200` |
| 5 | `8101` | `8301` | `8201` |
| 6 | `8102` | `8302` | `8200` |
| 7 | `8103` | `8303` | `8201` |

Evidence:

- Proxy logged Projection metadata keys only for all 8 requests.
- Attention logged `224` `PAP prefill KV imported via IPC descriptor` entries:
  `56` per Attention executor, matching `2 requests * 28 layers`.
- Projection logged `2688` OFFLOAD_EXEC traces:
  `1344` per Projection executor, matching `4 requests * 12 tokens * 28 layers`.
- Attention logged `2688` OFFLOAD_EXEC traces:
  `672` per Attention executor, matching `2 requests * 12 tokens * 28 layers`.
- `kv_transfer_config` appeared only in the four Prefill producer logs, not in
  Projection startup logs.
- No `Traceback`, `ERROR`, `rejected`, or `Got kv_transfer_params` log entries
  were found in the PAP service logs.

## Qwen3-0.6B PAP Projection Without Remote Prefix Block Allocation

Code checkpoint:

- `KVCacheManager.allocate_slots()` now accepts
  `allocate_external_computed_blocks`.
- The default is `True`, preserving KVConnector receiver semantics where
  external computed tokens still require local receiver blocks.
- PAP metadata-only Projection passes
  `allocate_external_computed_blocks=False` when `pap_remote_prefix_len` is
  present.
- In that mode, remote prefix progress is still used for scheduling/admission,
  but Projection allocates local blocks only for its local/current token work
  instead of allocating prompt-prefix block placeholders.

Focused KV cache regression:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py::test_allocate_external_tokens_can_skip_local_prefix_blocks -q
```

Result: `1 passed`.

The regression test covers both paths:

- PAP-style allocation with `num_external_computed_tokens=8`,
  `num_new_tokens=1`, and `allocate_external_computed_blocks=False` allocates
  only `1` local block.
- Default KVConnector-style allocation for the same token counts allocates
  `3` blocks, preserving existing receiver behavior.

Focused PAP + KV suite:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py \
  tests/v1/core/test_kv_cache_utils.py::test_allocate_external_tokens_can_skip_local_prefix_blocks -q
```

Result: `73 passed`.

1PA1P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=1pa1p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=12 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Request returned HTTP `200`.
- Usage: `prompt_tokens=24`, `completion_tokens=12`, `total_tokens=36`.
- Output was valid text, not garbled; semantic quality was limited by the
  0.6B model and short `max_tokens`.
- Proxy logged Projection metadata keys only:
  `pap_attention_endpoint`, `pap_attention_kv_installed`,
  `pap_attention_tcp_endpoint`, `pap_offload_exec_zmq_endpoint`,
  `pap_prefill_kv_handle`, `pap_projection_kv_unaware`,
  `pap_remote_prefix_len`.
- Projection logged `336` PAP OFFLOAD_EXEC projection traces, matching
  `12 completion tokens * 28 layers`.
- Attention logged `336` PAP OFFLOAD_EXEC attention traces, excluding the two
  startup OFFLOAD_EXEC listener lines.
- `kv_transfer_config` appeared only in Prefill logs, not Projection logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

4PA2P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=4pa2p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=8 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Sent 8 sequential `/v1/completions` requests; all returned HTTP `200`.
- Each response reported `prompt_tokens=22`, `completion_tokens=8`,
  `total_tokens=30`.
- Outputs were valid text, not garbled; semantic quality was limited by the
  0.6B model and short `max_tokens`.
- Route coverage:
  - `8100/8300 -> 8200`
  - `8101/8301 -> 8201`
  - `8102/8302 -> 8200`
  - `8103/8303 -> 8201`
  - repeated once.
- Proxy logged Projection metadata keys only for all 8 requests.
- Projection logged `1792` PAP OFFLOAD_EXEC projection traces:
  `896` per Projection, matching `8 requests * 8 tokens * 28 layers`.
- Attention logged `1792` PAP OFFLOAD_EXEC attention traces:
  `448` per Attention executor, matching
  `2 requests * 8 tokens * 28 layers`.
- `kv_transfer_config` appeared only in the four Prefill producer logs, not in
  Projection startup logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

## Explicit PAP Projection Scheduler State

This phase keeps Projection as a normal vLLM production server: OpenAI API
handling, model weight loading, model runner setup, logits, and sampling are
still vLLM-owned. The change is inside scheduler state selection.

- Added `PAPProjectionScheduleState` in
  `vllm/v1/core/sched/scheduler.py`.
- For `pap_projection_kv_unaware=True`, the state records:
  - `remote_prefix_len`
  - `remote_computed_tokens = remote_prefix_len - 1`
  - `local_computed_token_offset = remote_prefix_len - 1`
  - `allocate_external_computed_blocks=False`
  - `allocate_local_slots=False`
- Waiting and running request scheduling now read this state instead of
  scattering `pap_remote_prefix_len is None` checks through the allocation
  calls.
- Qwen3 PAP forward has a contract test proving the PAP branch returns after
  `_compute_pap_attention()` and before local `self.attn(q, k, v)`, so PAP
  requests do not enter the local attention KV update path.

Focused TDD/contract verification:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py::test_pap_projection_schedule_state_is_explicit -q
```

Result: RED first with missing `_get_pap_projection_schedule_state`, then
`1 passed` after implementation.

Regression suite:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/v1/core/test_scheduler.py::test_pap_projection_remote_prefix_len_parser \
  tests/v1/core/test_scheduler.py::test_pap_projection_schedule_state_is_explicit -q
```

Result: `26 passed`.

Boundary still open:

- Projection still initializes vLLM KV cache tensors during process startup.
  Request-level scheduling is KV-unaware and slotless, but skipping startup KV
  tensor allocation is a separate production model-runner compatibility phase.

## Qwen3-0.6B PAP Projection Running Local Slot Offset

Code checkpoint:

- `KVCacheManager.allocate_slots()` now accepts
  `local_computed_token_offset`.
- PAP Projection running requests pass `pap_remote_prefix_len - 1` as that
  offset.
- The scheduler still advances global `request.num_computed_tokens` so vLLM
  token positions, sampling, and request lifecycle stay intact.
- Local block allocation uses `request.num_computed_tokens -
  local_computed_token_offset`, so Projection request block tables grow from the
  Projection-local token count instead of the remote prompt-prefix position.
- This is a request-level state tightening step. Projection still initializes
  vLLM KV cache tensors and still has local token slots; it no longer lets the
  PA/Attention-owned remote prefix inflate running decode local slot history.

Focused regression:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_running_slots_use_local_progress_offset \
  tests/pap/test_pap_true_split_contract.py::test_scheduler_offsets_running_pap_projection_local_progress -q
```

Result: `2 passed`.

The KV regression uses a block size of `4` with a remote prefix offset of `8`.
After the first local token, subsequent running steps at global computed tokens
`9` and `11` do not allocate extra blocks; the next block is allocated only at
global computed token `12`, which is the Projection-local fifth token.

Focused PAP + KV suite:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py \
  tests/v1/core/test_kv_cache_utils.py::test_allocate_external_tokens_can_skip_local_prefix_blocks \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_running_slots_use_local_progress_offset -q
```

Result: `75 passed`.

1PA1P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=1pa1p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=16 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Request returned HTTP `200`.
- Usage: `prompt_tokens=25`, `completion_tokens=16`, `total_tokens=41`.
- Output was valid text, not garbled; semantic quality was limited by the
  0.6B model and short `max_tokens`.
- Proxy logged Projection metadata keys only:
  `pap_attention_endpoint`, `pap_attention_kv_installed`,
  `pap_attention_tcp_endpoint`, `pap_offload_exec_zmq_endpoint`,
  `pap_prefill_kv_handle`, `pap_projection_kv_unaware`,
  `pap_remote_prefix_len`.
- Projection logged `448` PAP OFFLOAD_EXEC projection traces, matching
  `16 completion tokens * 28 layers`.
- Attention logged `448` PAP OFFLOAD_EXEC attention traces.
- `kv_transfer_config` appeared only in Prefill logs, not Projection logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

4PA2P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=4pa2p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=8 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Sent 8 sequential `/v1/completions` requests; all returned HTTP `200`.
- Each response reported `prompt_tokens=22`, `completion_tokens=8`,
  `total_tokens=30`.
- Outputs were valid text, not garbled; semantic quality was limited by the
  0.6B model and short `max_tokens`.
- Route coverage:
  - `8100/8300 -> 8200`
  - `8101/8301 -> 8201`
  - `8102/8302 -> 8200`
  - `8103/8303 -> 8201`
  - repeated once.
- Proxy logged Projection metadata keys only for all 8 requests.
- Projection logged `1792` PAP OFFLOAD_EXEC projection traces:
  `896` per Projection, matching `8 requests * 8 tokens * 28 layers`.
- Attention logged `1792` PAP OFFLOAD_EXEC attention traces:
  `448` per Attention executor, matching
  `2 requests * 8 tokens * 28 layers`.
- `kv_transfer_config` appeared only in the four Prefill producer logs, not in
  Projection startup logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

## Qwen3-0.6B PAP Projection Slotless Request-Level Allocation

Code checkpoint:

- `KVCacheManager.allocate_slots()` now accepts `allocate_local_slots`.
- When `allocate_local_slots=False`, the manager returns
  `empty_kv_cache_blocks` without touching the coordinator, without allocating
  request blocks, and without committing prefix-cache entries.
- PAP Projection waiting and running scheduler paths pass
  `allocate_local_slots=False` when `pap_remote_prefix_len` is present.
- Ordinary vLLM and KVConnector paths keep the default
  `allocate_local_slots=True`.
- This removes request-level KV block allocation for PAP Projection. The vLLM
  Projection process still initializes KV cache tensors as part of production
  model runner setup, but PAP Projection requests no longer reserve local KV
  blocks.

Focused regression:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_can_disable_local_slot_allocation \
  tests/pap/test_pap_true_split_contract.py::test_scheduler_disables_local_slot_allocation_for_pap_projection -q
```

Result: `2 passed`.

Focused PAP + KV suite:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py \
  tests/v1/core/test_kv_cache_utils.py::test_allocate_external_tokens_can_skip_local_prefix_blocks \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_running_slots_use_local_progress_offset \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_can_disable_local_slot_allocation -q
```

Result: `77 passed`.

1PA1P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=1pa1p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=12 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Request returned HTTP `200`.
- Usage: `prompt_tokens=21`, `completion_tokens=12`, `total_tokens=33`.
- Output was valid text, not garbled; semantic quality was limited by the
  0.6B model and short `max_tokens`.
- Proxy logged Projection metadata keys only:
  `pap_attention_endpoint`, `pap_attention_kv_installed`,
  `pap_attention_tcp_endpoint`, `pap_offload_exec_zmq_endpoint`,
  `pap_prefill_kv_handle`, `pap_projection_kv_unaware`,
  `pap_remote_prefix_len`.
- Projection logged `336` PAP OFFLOAD_EXEC projection traces, matching
  `12 completion tokens * 28 layers`.
- Attention logged `336` PAP OFFLOAD_EXEC attention traces.
- `kv_transfer_config` appeared only in Prefill logs, not Projection logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.

4PA2P Qwen3-0.6B E2E after the change:

- Command shape:
  `PAP_TOPOLOGY=4pa2p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=1
  PAP_PROXY_PORT=9000 PAP_MAX_TOKENS=8 PAP_ATTENTION_KV_DEBUG=1
  PAP_OFFLOAD_EXEC_TRACE=1 bash examples/pap/launch_pap_nixl.sh --model
  /data/ssd1/llm-models/Qwen3-0.6B`
- Sent 8 sequential `/v1/completions` requests; all returned HTTP `200`.
- Each response reported `prompt_tokens=22`, `completion_tokens=8`,
  `total_tokens=30`.
- Outputs were valid text, not garbled; semantic quality was limited by the
  0.6B model and short `max_tokens`.
- Route coverage:
  - `8100/8300 -> 8200`
  - `8101/8301 -> 8201`
  - `8102/8302 -> 8200`
  - `8103/8303 -> 8201`
  - repeated once.
- Proxy logged Projection metadata keys only for all 8 requests.
- Projection logged `1792` PAP OFFLOAD_EXEC projection traces:
  `896` per Projection, matching `8 requests * 8 tokens * 28 layers`.
- Attention logged `1792` PAP OFFLOAD_EXEC attention traces:
  `448` per Attention executor, matching
  `2 requests * 8 tokens * 28 layers`.
- `kv_transfer_config` appeared only in the four Prefill producer logs, not in
  Projection startup logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping` errors were found.
