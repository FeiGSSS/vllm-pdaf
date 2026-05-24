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

Next work:

- Consider the remaining deeper cut: avoid allocating external
  remote-prefix blocks inside `KVCacheManager.allocate_slots()` for PAP
  Projection entirely.
