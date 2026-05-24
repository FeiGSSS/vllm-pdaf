# Handoff

## 2026-05-24 PAP Shared KV Owner Design

We agreed on the next architecture phase after metadata-only Projection:
remove KV duplication between Prefill and Attention on a PA node.

Terminology rule: use `Prefill`, not the older mistaken wording.

Current limitation:

- Prefill and Attention are colocated but do not share one KV owner.
- Prefill exports prompt KV with CUDA IPC descriptors.
- Attention currently opens those descriptors and copies KV into its own
  registry storage.
- Decode K/V arriving from Projection is appended to Attention-local buffers.
- Therefore later turns cannot see first-turn decode KV from Prefill's point of
  view.

Target:

- Prefill owns the single real vLLM paged KV pool on the PA node.
- Attention reads Prefill-owned blocks through CUDA IPC.
- Attention writes Projection-provided decode K/V directly into Prefill-owned
  block slots.
- Later turns can reuse first-turn prompt plus decode KV when scheduled on the
  same PA.

Design file:

- `docs/superpowers/specs/2026-05-24-pap-shared-kv-owner-design.md`

Design decisions:

- Use one vLLM-facing `PAPSharedKVConnector`, not two independent connectors.
- Internally split into:
  - `LocalResidentBackend`: same-PA attach, CUDA IPC, no KV copy.
  - `RemoteMigrationBackend`: future cross-PA migration via NIXL/RDMA/NCCL or
    external-store style transport.
- `PAKVOwner` is the source of truth for session -> blocks, seq_len,
  leases/refcounts, placement, IPC descriptors, and decode slot materialization.
- Do not copy LMCache semantics for the local path. LMCache-style connector
  behavior loads external KV into newly allocated vLLM blocks; PAP local shared
  KV must attach resident Prefill-owned blocks instead.

Initial implementation should require session stickiness or recompute for
remote sessions. Cross-PA migration is a later phase after local no-copy
correctness is proven.

## 2026-05-24 PAP Shared KV Owner Phase 1

First implementation checkpoint toward shared KV:

- Added an implementation plan at
  `docs/superpowers/plans/2026-05-24-pap-shared-kv-owner-implementation.md`.
- Added a TDD contract proving IPC-imported Prefill KV opened by Attention keeps
  the original tensor view instead of being copied into Attention storage.
- `PAPAttentionRegistry.import_prefill_kv()` now has `copy=True` by default.
- The `import_prefill_kv_ipc` binary command passes `copy=False`, so opened IPC
  tensors remain the source view in Attention.
- Non-IPC tensor-bundle import keeps copy semantics as the debug/fallback path.

Verified:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `78 passed, 16 warnings`.

Boundary:

- This removes the extra Attention-side copy for IPC-imported tensors.
- It does not yet make Prefill's vLLM paged KV blocks the directly attached
  resident source. Current Prefill export still gathers into exported tensors.
- Next phase is `PAKVOwner`/resident block descriptors so Attention reads
  Prefill-owned paged blocks and later writes decode K/V into those blocks.

## 2026-05-24 PAP Shared KV Owner Phase 2

Second implementation checkpoint toward resident Prefill-owned KV:

- Added `PAPOffloadKVPagedIPCDescriptor` for whole-layer paged KV backing
  storage.
- Added `paged_kv_segments()` so Attention can build per-block K/V views over a
  paged KV cache without concatenating prompt KV.
- Added Prefill-side `import_prefill_paged_kv()` that posts
  `import_prefill_paged_kv_ipc` with only descriptor metadata and no tensor
  payload.
- Added Attention-side `open_ipc_paged_kv_cache()` and binary command handling
  for `import_prefill_paged_kv_ipc`.
- `PAPAttentionRegistry.import_prefill_paged_kv()` stores resident paged block
  segment views as Prefill KV for a layer; `append_decode_kv()` now expands
  those segments before appending decode K/V.

Verified:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py -q
```

Result: `55 passed, 16 warnings`.

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `37 passed, 16 warnings`.

Boundary:

- Attention can now receive a paged KV backing descriptor and compute from
  resident block views without a Prefill-side gather into contiguous K/V.
- The Qwen3 Prefill export path still calls the old gathered
  `import_prefill_kv_from_paged_cache()` by default.
- Decode K/V is still appended to Attention-local decode buffers; write-back
  into Prefill-owned paged blocks remains next.

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

Prefill to Projection prompt KV transfer remains removed. The remaining
prompt-KV path is Prefill to colocated Attention, now defaulted to PAP
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

## 2026-05-24 PAP Shared KV Owner Phase 3

Latest local phase after `0b2fc04bb`:

- Qwen3 Prefill now exports prompt KV to Attention with
  `import_prefill_paged_kv()` instead of the gathered
  `import_prefill_kv_from_paged_cache()` path.
- The Qwen3 export path derives block ids from the vLLM block table and passes
  the layer `kv_cache` backing tensor into the paged CUDA IPC descriptor.
- The Qwen3 paged path explicitly requires `PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc`.
- Added a contract test that rejects regressing Qwen3 back to the gathered
  export helper.

Focused verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py -q
```

Result: `81 passed, 16 warnings`.

```bash
.venv/bin/python -m pytest tests/pap/test_pap_launch_files.py -q
```

Result: `12 passed, 16 warnings`.

E2E 1PA1P Qwen3-0.6B:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=10`, `completion_tokens=6`, `total_tokens=16`.
- Output text: ` Also, explain the difference between`
- Attention logged `28` `PAP prefill paged KV imported via IPC descriptor`
  entries.
- No old gathered IPC import log or error pattern matched in the logs.

Current boundary:

- Prefill prompt KV now reaches Attention through the resident paged descriptor
  path.
- Decode K/V is still appended to Attention-local decode buffers.
- Next phase is decode K/V write-back into Prefill-owned paged blocks plus
  same-PA multi-turn reuse verification.

## 2026-05-24 PAP Shared KV Owner Phase 4

Latest local phase after `0e2d4ffe3`:

- Attention now records `PAPResidentPagedKV` metadata for paged Prefill imports.
- For descriptor-backed decode steps whose target slot is covered by the
  attached Prefill-owned paged KV blocks, `append_decode_kv()` writes the
  Projection-provided decode K/V directly into the resident paged KV backing
  tensor.
- Covered resident-block decode steps return resident paged segments and do not
  keep an Attention-local decode KV copy for that layer.
- If the target block is not attached yet, or the attached block coverage is
  full, the existing Attention-local decode buffer remains the fallback.

Focused verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `96 passed, 16 warnings`.

E2E 1PA1P Qwen3-0.6B:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=12`, `completion_tokens=6`, `total_tokens=18`.
- Output text: ` PAP is a type of`
- Attention logged `28` `PAP prefill paged KV imported via IPC descriptor`
  entries.
- Projection and Attention each logged `168` OFFLOAD_EXEC traces.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, or `IndexError` matched
  in logs.

Current boundary:

- The shared write-back path is real for slots already covered by attached
  Prefill-owned paged KV blocks.
- Generated tokens that require new Prefill-owned blocks still fall back to
  Attention-local decode storage.
- Next phase is `PAKVOwner`-style decode slot/block reservation and publication
  so all generated K/V can be written into Prefill-owned blocks, followed by
  same-PA multi-turn reuse verification.

## 2026-05-24 PAP Shared KV Owner Phase 5

Latest local phase after `a62895ef7`:

- Attention now publishes a new resident decode block when the decode descriptor
  points to a block that is not in the imported prompt block list but is present
  in the already attached Prefill-owned paged KV backing tensor.
- The newly published block is added to `PAPResidentPagedKV.block_ids`, decode
  K/V is written into the backing tensor, and resident segments are rebuilt over
  prompt plus generated blocks.
- Added a regression test proving a cross-block decode step writes into the
  resident paged backing and does not create an Attention-local decode KV entry.

Focused verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `97 passed, 16 warnings`.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=12`, `completion_tokens=8`, `total_tokens=20`.
- Output text: ` Pell's equation, which is a`
- This crosses the 16-token boundary and exercises a generated-token block.
- Attention logged `28` paged Prefill KV descriptor imports.
- Projection and Attention each logged `224` OFFLOAD_EXEC traces.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Current boundary:

- New resident decode block publication works only when the block id maps into
  the already attached Prefill-owned paged KV backing tensor.
- Block ids still come from the active decode descriptor path, not from an
  explicit Prefill owner allocation API.
- Next phase is an explicit `PAKVOwner`-style decode slot/block reservation API
  and same-PA multi-turn reuse verification.

## 2026-05-24 PAP Shared KV Owner Phase 6

Latest local phase after `67647c5ec`:

- Added `vllm/pap/kv_owner.py` with a pure-metadata `PAKVOwner`.
- `PAKVOwner` tracks session leases, resident layer blocks, sequence length,
  backed block capacity, decode slot reservation, and materialized decode
  slots.
- Attention registry now creates an owner session on registration, registers
  paged layer backing metadata during `import_prefill_paged_kv()`, and records
  successful resident decode writes as materialized owner slots.
- Added owner unit tests plus an Attention registry contract proving
  materialized decode slots are visible in the owner state.

Focused verification:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `101 passed, 16 warnings`.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- This crosses the 16-token boundary and exercises a generated-token block.
- Attention logged `28` paged Prefill KV descriptor imports.
- Projection and Attention each logged `224` OFFLOAD_EXEC traces.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Current boundary:

- `PAKVOwner` exists and records materialized decode slots, but it is still pure
  metadata and not yet the source of physical block allocation.
- Decode block ids still come from the current descriptor path.
- Next phase should move decode slot/block reservation earlier into the
  Prefill-owner path and use owner state to verify same-PA multi-turn reuse.
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
