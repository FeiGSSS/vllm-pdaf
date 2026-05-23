# Handoff

## State
Branch `feature/pap-true-split`: removed 2 prototype PAP modes (`debug_remote_attention`, `true_split`), keeping only the NCCL performance path. `PAPMode` enum replaced with `is_pap_enabled()` boolean. 117 unit tests pass. PAP pipeline successfully ran E2E up to the attention executor — NCCL QKV send + TCP control trigger work. Fails at: `RuntimeError: prefill KV must be imported before stateful decode attention`.

Key files modified: `vllm/pap/mode.py`, `vllm/pap/shadow_attention.py`, `vllm/pap/data_plane.py`, `vllm/model_executor/models/qwen3.py`, `vllm/v1/worker/gpu/model_runner.py`, NIXL scheduler, launch scripts, all tests.

Critical venv fix: deleted stale `__editable__` finder (mapped to old `vllm-papf` path), replaced with simple `.pth` file (`/home/fei/research/PD/vllm-pap/.venv/lib/python3.12/site-packages/vllm.pth`). Also fixed shebangs in `.venv/bin/*` from `vllm-papf` → `vllm-pap`.

## Next
1. **Implement prefill KV import path** — `_maybe_import_pap_prefill_kv_to_attention()` is currently a no-op. Needs to take the NIXL prefill KV handle received by projection and forward it to attention executor via TCP `import-prefill-kv-binary` endpoint. This is the blocker for completing a full PAP request.
2. **Clean launch script updates** — `launch_pap_6pa2p_qwen3_8b_nixl.sh` partially updated but `launch_pap_qwen3_8b_nixl.sh`, `multi_pap_proxy_server.py`, `pap_proxy_server.py`, `pd_payloads.py` still have old mode references.
3. **E2E benchmark** after KV import fix.

## Context
GPUs 0-1 may have leaked memory from prior runs. Use GPUs 2+ for fresh tests.
The `_pap_enabled_for_batch()` method in model_runner checks both `kv_connector_extra_config.pap_enabled` (global) and per-request TCP endpoint tracking.
NCCL env vars required: `NCCL_P2P_DISABLE=1 NCCL_NUM_CHANNELS=1 NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=WARN`.
