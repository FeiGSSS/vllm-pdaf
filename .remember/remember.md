# Handoff

## State

Branch `feature/pap-true-split`. Current latest implementation checkpoint is
`4a9d567d5` (`Log PAP OFFLOAD_KV IPC imports`), after:

- `9f788d6ae` Plan PAP OFFLOAD_KV IPC implementation
- `312ae6fbb` Add PAP OFFLOAD_KV IPC descriptors
- `a418ae539` Import PAP prefill KV via IPC descriptors in Attention
- `913db9dad` Export PAP prefill KV with CUDA IPC metadata
- `e22316c9a` Default PAP OFFLOAD_KV to CUDA IPC
- `f945899e4` Serialize PAP CUDA IPC handles for control messages

Projection remains a normal vLLM production server. It has no
`kv_transfer_config`, receives only PAP metadata in `kv_transfer_params`, and
uses scheduler remote-prefix progress from `pap_remote_prefix_len`.

Prefill/Profile to Projection prompt KV transfer remains removed. The remaining
prompt-KV path is Prefill/Profile to colocated Attention, now defaulted to PAP
OFFLOAD_KV CUDA IPC descriptors instead of TCP tensor-bundle payloads.

## Verified

Focused unit tests:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `70 passed`.

1PA1P Qwen3-0.6B with `PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc` default:

- HTTP `200`.
- Usage: `prompt_tokens=18`, `completion_tokens=32`, `total_tokens=50`.
- Projection payload keys were only PAP metadata:
  `pap_attention_endpoint`, `pap_attention_kv_installed`,
  `pap_attention_tcp_endpoint`, `pap_offload_exec_zmq_endpoint`,
  `pap_prefill_kv_handle`, `pap_projection_kv_unaware`,
  `pap_remote_prefix_len`.
- Attention logged `28` `PAP prefill KV imported via IPC descriptor` entries.
- Projection and Attention each logged `896` OFFLOAD_EXEC traces, matching
  `32 * 28`.
- `kv_transfer_config` appeared only in Prefill, not Projection.
- No `Traceback`, `ERROR`, `rejected`, or `Got kv_transfer_params`.

4PA2P Qwen3-0.6B with `PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc` default:

- Sent 8 sequential requests; all returned HTTP `200`.
- Route coverage:
  - `8100/8300 -> 8200`
  - `8101/8301 -> 8201`
  - `8102/8302 -> 8200`
  - `8103/8303 -> 8201`
  - repeated once.
- Each response used `prompt_tokens=15`, `completion_tokens=12`.
- Projection payload keys were only PAP metadata for all 8 requests.
- Attention logged `224` IPC import entries:
  `4 Attention * 2 requests each * 28 layers`.
- Projection traces: `2688` total, `1344` per Projection.
- Attention traces: `2688` total, `672` per Attention.
- `kv_transfer_config` appeared only in the four Prefill producer logs.
- No `Traceback`, `ERROR`, `rejected`, or `Got kv_transfer_params`.

## Important Runtime Notes

- PyTorch `reduce_tensor` returns non-JSON-native CUDA IPC rebuild args. The
  descriptor stores those args in `ipc_handle_pickled` with base64 encoding
  inside the control metadata.
- The first IPC version still gathers Prefill paged KV into contiguous tensors
  and Attention copies opened IPC tensors into its existing registry storage.
  This removes TCP tensor payloads but is not yet zero-copy shared KV ownership.
- Projection still keeps vLLM internal scheduler/block-table state for current
  and generated tokens. It does not receive prompt KV bytes or prompt KV
  handles.

## Next

1. Decide whether to keep the first IPC copy-into-registry path as the stable
   baseline or move directly to zero-copy/shared KV ownership.
2. Add stronger runtime assertions/log checks for absence of TCP tensor-bundle
   prefill import in default PAP mode.
3. Tighten Projection local KV assumptions:
   audit whether generated-token KV on Projection can be avoided now that
   Attention owns attention history.
4. Continue using `/data/ssd1/llm-models/Qwen3-0.6B` for fast experiments.

## Rules

- Use `.venv/bin/python` / `uv`; do not use system `python3` or bare `pip`.
- Poll startup and logs frequently; do not wait silently on long E2E runs.
- Full restart PAP experiments instead of restarting individual services.
- Clean E2E processes by PID and verify with `pgrep` before starting another
  run.

## 2026-05-24 Projection Local Block-State Tightening

Latest local phase after `160da122c`:

- `Qwen3Attention._compute_pap_attention()` no longer reads
  `forward_context.slot_mapping`.
- Projection no longer derives or sends `block_id` / `slot` in the PAP
  OFFLOAD_EXEC call metadata; it sends Q/K/V plus `seq_len` through the existing
  descriptor path.
- Attention already derives block/slot from `seq_len` in
  `compute_offload_exec_output()`, so this aligns the active NCCL compact path
  with a Projection node that does not reason about prompt-prefix block tables.
- Scheduler waiting-request PAP branch now records `req_to_new_blocks = new_blocks`
  for PAP metadata-only Projection requests, while non-PAP requests keep
  `kv_cache_manager.get_blocks(request_id)`.

Focused verification:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_true_split_contract.py -q
```

Result: `19 passed`.

Full focused PAP unit suite:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `71 passed`.

E2E after the change:

- 1PA1P Qwen3-0.6B:
  - HTTP `200`.
  - Usage `prompt_tokens=29`, `completion_tokens=16`, `total_tokens=45`.
  - Proxy Projection payload keys were PAP metadata only.
  - Attention IPC imports `28`.
  - Projection OFFLOAD_EXEC traces `448` = `16 * 28`.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.
- 4PA2P Qwen3-0.6B:
  - 8 sequential requests, all HTTP `200`.
  - Route coverage `8100/8300 -> 8200`, `8101/8301 -> 8201`,
    `8102/8302 -> 8200`, `8103/8303 -> 8201`, repeated once.
  - Each response had `prompt_tokens=19`, `completion_tokens=8`,
    `total_tokens=27`.
  - Attention IPC imports `224` total, `56` per Attention.
  - Projection OFFLOAD_EXEC traces `1792` total, `896` per Projection.
  - Attention OFFLOAD_EXEC traces `1792` total, `448` per Attention.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.

## 2026-05-24 PAP Projection Metadata-Only Startup KV

Current local phase:

- Projection vLLM processes now run with `PAP_PROJECTION_KV_UNAWARE=1`.
- In that role, `GPUModelRunner.initialize_kv_cache()` binds zero-size
  per-layer metadata-only KV placeholders via `init_kv_cache_metadata_only()`
  instead of calling normal `init_kv_cache()` and allocating real KV backing
  tensors.
- `GPUWorker.compile_or_warm_up_model()` skips startup local-attention warmup
  for this role. Memory profiling still runs with attention skipped so vLLM can
  produce the logical KV cache config needed by scheduler/block-table metadata.
- This keeps production vLLM API/model loading/logits/sampling while making
  Projection process-level KV storage metadata-only.

Verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py \
  tests/v1/core/test_scheduler.py::test_pap_projection_remote_prefix_len_parser \
  tests/v1/core/test_scheduler.py::test_pap_projection_schedule_state_is_explicit \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_can_disable_local_slot_allocation -q
```

Result: `40 passed`.

1PA1P Qwen3-0.6B E2E:

- HTTP `200`; usage `prompt_tokens=11`, `completion_tokens=8`,
  `total_tokens=19`.
- Projection startup logged metadata-only KV placeholders and skipped
  local-attention warmup.
- Projection GPU memory was about `1.6 GiB`, while the PA GPU was about
  `21 GiB`.
- Projection payload keys were metadata only.
- Projection and Attention OFFLOAD_EXEC traces were both `224` =
  `8 tokens * 28 layers`; IPC imports were `28`.
- `kv_transfer_config` appeared only in Prefill logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping`.

4PA2P Qwen3-0.6B E2E:

- 8 sequential requests all HTTP `200`; each had `prompt_tokens=13`,
  `completion_tokens=8`, `total_tokens=21`.
- Route coverage:
  `8100/8300 -> 8200`, `8101/8301 -> 8201`,
  `8102/8302 -> 8200`, `8103/8303 -> 8201`, repeated once.
- Both Projection processes logged metadata-only KV placeholders and skipped
  local-attention warmup; Projection GPUs were about `1.6 GiB` each.
- Projection OFFLOAD_EXEC traces were `1792` total, `896` per Projection.
- Attention OFFLOAD_EXEC traces were `1792` total, `448` per Attention.
- IPC imports were `56` per Attention executor.
- `kv_transfer_config` appeared only in Prefill logs, not Projection logs.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
  `invalid slot`, or `slot_mapping`.

Open boundary:

- Projection still has logical KV metadata because vLLM scheduler, block
  tables, and attention metadata depend on it. It no longer has Projection KV
  data transfer, request-local KV slots, or startup KV backing storage in PAP
  mode.

## 2026-05-24 Explicit PAP Projection Scheduler State

Current local phase:

- Projection remains a vLLM production server. We are not replacing model
  loading, OpenAI API request handling, model runner setup, logits, or sampling.
- Scheduler now has an explicit `PAPProjectionScheduleState` for
  `pap_projection_kv_unaware=True` requests instead of scattering implicit
  `pap_remote_prefix_len is None` checks.
- The state carries `remote_prefix_len`, `remote_computed_tokens`,
  `local_computed_token_offset`, `allocate_external_computed_blocks=False`,
  and `allocate_local_slots=False`.
- Waiting and running request allocation both consume that state.
- Added a Qwen3 contract that the PAP forward branch returns after
  `_compute_pap_attention()` and before local `self.attn(q, k, v)`, so PAP
  requests do not enter the local attention KV update path.

Verification:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py::test_pap_projection_schedule_state_is_explicit -q
```

Result: RED first with missing `_get_pap_projection_schedule_state`, then
`1 passed` after implementation.

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/v1/core/test_scheduler.py::test_pap_projection_remote_prefix_len_parser \
  tests/v1/core/test_scheduler.py::test_pap_projection_schedule_state_is_explicit -q
```

Result: `26 passed`.

Open boundary:

- Projection still allocates vLLM KV cache tensors at process startup. That is
  not request data movement and PAP requests are slotless, but startup KV cache
  allocation remains the next compatibility audit.

## 2026-05-24 PAP Projection Running Local Slot Offset

Latest local phase after `0fb0c8bc3`:

- `KVCacheManager.allocate_slots()` now has `local_computed_token_offset`.
- PAP Projection running decode passes `pap_remote_prefix_len - 1` as the
  offset.
- Global `request.num_computed_tokens` still advances normally for vLLM token
  positions, sampling, and request lifecycle.
- Local block allocation now uses local progress
  `request.num_computed_tokens - local_computed_token_offset`, so remote prompt
  prefix progress no longer inflates Projection running-request block history.

Focused regression:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_running_slots_use_local_progress_offset \
  tests/pap/test_pap_true_split_contract.py::test_scheduler_offsets_running_pap_projection_local_progress -q
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
  tests/v1/core/test_kv_cache_utils.py::test_pap_projection_running_slots_use_local_progress_offset -q
```

Result: `75 passed`.

E2E after the change:

- 1PA1P Qwen3-0.6B:
  - HTTP `200`.
  - Usage `prompt_tokens=25`, `completion_tokens=16`, `total_tokens=41`.
  - Output was valid text, not garbled.
  - Projection payload keys were PAP metadata only.
  - Projection OFFLOAD_EXEC traces `448` = `16 * 28`.
  - Attention OFFLOAD_EXEC traces `448`.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.
- 4PA2P Qwen3-0.6B:
  - 8 sequential requests, all HTTP `200`.
  - Route coverage `8100/8300 -> 8200`, `8101/8301 -> 8201`,
    `8102/8302 -> 8200`, `8103/8303 -> 8201`, repeated once.
  - Each response had `prompt_tokens=22`, `completion_tokens=8`,
    `total_tokens=30`.
  - Outputs were valid text, not garbled.
  - Projection OFFLOAD_EXEC traces `1792` total, `896` per Projection.
  - Attention OFFLOAD_EXEC traces `1792` total, `448` per Attention.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.

Next work:

- Audit whether any Projection-local KV data writes remain now that request
  slot allocation can be disabled.
- Do not claim full stateless Projection yet: vLLM still initializes KV cache
  tensors as part of production model runner setup.

## 2026-05-24 PAP Projection Slotless Request-Level Allocation

Latest local phase after `0232ca462`:

- `KVCacheManager.allocate_slots()` now has `allocate_local_slots`.
- When `allocate_local_slots=False`, it returns `empty_kv_cache_blocks` without
  touching the coordinator, allocating request blocks, or committing prefix-cache
  entries.
- PAP Projection waiting and running scheduler paths pass
  `allocate_local_slots=False` when `pap_remote_prefix_len` is present.
- Ordinary vLLM and KVConnector paths keep default local slot allocation.
- This removes request-level KV block allocation for PAP Projection. The process
  still initializes vLLM KV cache tensors at startup.

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

E2E after the change:

- 1PA1P Qwen3-0.6B:
  - HTTP `200`.
  - Usage `prompt_tokens=21`, `completion_tokens=12`, `total_tokens=33`.
  - Output was valid text, not garbled.
  - Projection payload keys were PAP metadata only.
  - Projection OFFLOAD_EXEC traces `336` = `12 * 28`.
  - Attention OFFLOAD_EXEC traces `336`.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.
- 4PA2P Qwen3-0.6B:
  - 8 sequential requests, all HTTP `200`.
  - Route coverage `8100/8300 -> 8200`, `8101/8301 -> 8201`,
    `8102/8302 -> 8200`, `8103/8303 -> 8201`, repeated once.
  - Each response had `prompt_tokens=22`, `completion_tokens=8`,
    `total_tokens=30`.
  - Outputs were valid text, not garbled.
  - Projection OFFLOAD_EXEC traces `1792` total, `896` per Projection.
  - Attention OFFLOAD_EXEC traces `1792` total, `448` per Attention.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.

Next work:

- Run a completion audit for remaining Projection-local KV writes and startup
  KV tensor allocation. The request-level block manager is now slotless for PAP,
  but full statelessness still needs proof that model layers never write local
  KV for PAP requests.

Next work:

- Continue auditing remaining Projection-local KV structures for generated
  tokens/current-token model-runner metadata.

## 2026-05-24 PAP Projection Skips Remote Prefix Block Allocation

Latest local phase after `c6817d015`:

- `KVCacheManager.allocate_slots()` now has
  `allocate_external_computed_blocks=True` by default.
- Default KVConnector behavior is preserved: external computed tokens still
  allocate local receiver blocks.
- PAP metadata-only Projection passes
  `allocate_external_computed_blocks=False` when `pap_remote_prefix_len` is
  present.
- Projection still uses remote prefix progress for scheduler admission, but it
  no longer allocates local prompt-prefix block placeholders for that remote
  prefix.

Focused regression:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_kv_cache_utils.py::test_allocate_external_tokens_can_skip_local_prefix_blocks -q
```

Result: `1 passed`.

The test verifies PAP-style allocation only consumes `1` local block for
`8` external prefix tokens plus `1` new token, while the default KVConnector path
still consumes `3` blocks.

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

E2E after the change:

- 1PA1P Qwen3-0.6B:
  - HTTP `200`.
  - Usage `prompt_tokens=24`, `completion_tokens=12`, `total_tokens=36`.
  - Output was valid text, not garbled.
  - Projection payload keys were PAP metadata only.
  - Projection OFFLOAD_EXEC traces `336` = `12 * 28`.
  - Attention OFFLOAD_EXEC traces `336`, excluding startup listener lines.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.
- 4PA2P Qwen3-0.6B:
  - 8 sequential requests, all HTTP `200`.
  - Route coverage `8100/8300 -> 8200`, `8101/8301 -> 8201`,
    `8102/8302 -> 8200`, `8103/8303 -> 8201`, repeated once.
  - Each response had `prompt_tokens=22`, `completion_tokens=8`,
    `total_tokens=30`.
  - Outputs were valid text, not garbled.
  - Projection OFFLOAD_EXEC traces `1792` total, `896` per Projection.
  - Attention OFFLOAD_EXEC traces `1792` total, `448` per Attention.
  - `kv_transfer_config` appeared only in Prefill logs.
  - No `Traceback`, `ERROR`, `Got kv_transfer_params`, `rejected`,
    `invalid slot`, or `slot_mapping`.
