# PAP OFFLOAD_KV IPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default Prefill/Profile-to-Attention prompt KV tensor-bundle path with a PAP OFFLOAD_KV CUDA IPC descriptor path while keeping Projection metadata-only.

**Architecture:** Projection remains a vLLM production server with PAP scheduler remote-prefix state. Prefill/Profile exports gathered prompt KV through CUDA IPC descriptors over the existing Attention control channel. Attention opens the IPC tensors and imports them into its existing registry storage, then acknowledges import.

**Tech Stack:** vLLM v1 scheduler/model runner, PyTorch CUDA IPC via `torch.multiprocessing.reductions.reduce_tensor` and `rebuild_cuda_tensor`, FastAPI/TestClient, `.venv/bin/python -m pytest`.

---

## File Structure

- Modify `vllm/pap/data_plane.py`: add OFFLOAD_KV IPC tensor-handle and descriptor dataclasses with `to_dict` / `from_dict` validation.
- Modify `vllm/pap/shadow_attention.py`: add IPC export/import helper, transport selection, and a TCP control command that carries descriptor metadata only.
- Modify `examples/pap/pap_attention_executor.py`: add IPC import handling in binary TCP/FastAPI path and call the existing `PAPAttentionRegistry.import_prefill_kv`.
- Modify `vllm/model_executor/models/qwen3.py`: pass `PAP_OFFLOAD_KV_TRANSPORT` into the Prefill/Profile import helper and default to `cuda_ipc`.
- Modify `examples/pap/launch_pap_nixl.sh`: export `PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"`.
- Modify `.remember/remember.md`: record the phase result after verification.
- Test `tests/pap/test_pap_data_plane.py`: descriptor validation and roundtrip serialization.
- Test `tests/pap/test_pap_attention_executor.py`: binary IPC import command calls registry without tensor-bundle payload.
- Test `tests/pap/test_pap_true_split_contract.py`: Qwen3 import path references OFFLOAD_KV transport selection.
- Test `tests/pap/test_pap_launch_files.py`: launcher defaults OFFLOAD_KV to CUDA IPC.

## Task 1: OFFLOAD_KV IPC Descriptor Contract

**Files:**
- Modify: `vllm/pap/data_plane.py`
- Test: `tests/pap/test_pap_data_plane.py`

- [ ] **Step 1: Write the failing descriptor roundtrip test**

Add this to `tests/pap/test_pap_data_plane.py`:

```python
def test_offload_kv_ipc_descriptor_roundtrip() -> None:
    from vllm.pap.data_plane import (
        PAPCudaIPCTensorHandle,
        PAPOffloadKVIPCDescriptor,
    )

    key_handle = PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(8, 2, 16),
        ipc_handle={"GPU-abc": ("storage", 1, 2, 3, 4, 5, 0)},
    )
    value_handle = PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(8, 2, 16),
        ipc_handle={"GPU-abc": ("storage", 1, 2, 3, 4, 5, 0)},
    )
    descriptor = PAPOffloadKVIPCDescriptor(
        request_id="cmpl-1",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=8,
        block_ids=(3, 4),
        key=key_handle,
        value=value_handle,
    )

    restored = PAPOffloadKVIPCDescriptor.from_dict(descriptor.to_dict())

    assert restored == descriptor
    assert restored.transport is PAPTensorTransport.CUDA_IPC
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py::test_offload_kv_ipc_descriptor_roundtrip -q
```

Expected: FAIL with an import error for `PAPCudaIPCTensorHandle` or `PAPOffloadKVIPCDescriptor`.

- [ ] **Step 3: Implement descriptor dataclasses**

Add `PAPCudaIPCTensorHandle` and `PAPOffloadKVIPCDescriptor` to `vllm/pap/data_plane.py`. The descriptor must set `transport=PAPTensorTransport.CUDA_IPC`, reject negative `seq_len`, preserve `block_ids`, and provide `to_dict` / `from_dict` methods.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/pap/data_plane.py tests/pap/test_pap_data_plane.py
git commit -m "Add PAP OFFLOAD_KV IPC descriptors"
```

## Task 2: Attention Executor IPC Import Command

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [ ] **Step 1: Write the failing binary command test**

Add a test that monkeypatches an IPC descriptor opener to return CPU tensors, posts a binary bundle whose metadata contains `command="import_prefill_kv_ipc"` and an IPC descriptor dict, and asserts:

```python
metadata["seq_len"] == 2
tensors == {}
session["prefill_seq_lens"] == {"model.layers.0.self_attn.attn": 2}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py::test_attention_executor_binary_imports_prefill_kv_ipc_descriptor -q
```

Expected: FAIL because `import_prefill_kv_ipc` is not handled.

- [ ] **Step 3: Implement binary IPC command**

In `compute_binary_attention_response`, detect `metadata["command"] == "import_prefill_kv_ipc"`, parse `metadata["descriptor"]` through `PAPOffloadKVIPCDescriptor.from_dict`, open key/value tensors through a helper, and call `registry.import_prefill_kv`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py::test_attention_executor_binary_imports_prefill_kv_ipc_descriptor -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/pap/pap_attention_executor.py tests/pap/test_pap_attention_executor.py
git commit -m "Import PAP prefill KV via IPC descriptors in Attention"
```

## Task 3: Prefill/Profile IPC Export Helper

**Files:**
- Modify: `vllm/pap/shadow_attention.py`
- Test: `tests/pap/test_pap_data_plane.py`

- [ ] **Step 1: Write the failing helper test**

Add a test that monkeypatches `reduce_tensor` and `_post_bytes_tcp`, calls `import_prefill_kv(..., transport=PAPTensorTransport.CUDA_IPC)`, and asserts the posted payload metadata uses `command="import_prefill_kv_ipc"` and does not include serialized `key` or `value` tensors.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py::test_import_prefill_kv_cuda_ipc_posts_descriptor_without_tensors -q
```

Expected: FAIL because `import_prefill_kv` does not accept `transport`.

- [ ] **Step 3: Implement IPC export path**

Add a `transport` argument to `import_prefill_kv` and `import_prefill_kv_from_paged_cache`. For `CUDA_IPC`, synchronize the current CUDA stream if needed, export key/value with `reduce_tensor`, build `PAPOffloadKVIPCDescriptor`, and post a tensor bundle containing descriptor metadata and `{}` tensors.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py::test_import_prefill_kv_cuda_ipc_posts_descriptor_without_tensors -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/pap/shadow_attention.py tests/pap/test_pap_data_plane.py
git commit -m "Export PAP prefill KV with CUDA IPC metadata"
```

## Task 4: Make CUDA IPC the Default PAP OFFLOAD_KV Path

**Files:**
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `examples/pap/launch_pap_nixl.sh`
- Test: `tests/pap/test_pap_true_split_contract.py`
- Test: `tests/pap/test_pap_launch_files.py`

- [ ] **Step 1: Write failing contract tests**

Update tests to assert:

```python
assert "PAP_OFFLOAD_KV_TRANSPORT" in method
assert "PAPTensorTransport.CUDA_IPC" in method
assert 'PAP_OFFLOAD_KV_TRANSPORT="${PAP_OFFLOAD_KV_TRANSPORT:-cuda_ipc}"' in text
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_true_split_contract.py::test_qwen3_pap_imports_prefill_kv_for_offload tests/pap/test_pap_launch_files.py::test_pap_launcher_configures_nccl_offload_exec -q
```

Expected: FAIL because OFFLOAD_KV transport is not wired yet.

- [ ] **Step 3: Wire default transport**

Read `PAP_OFFLOAD_KV_TRANSPORT`, default to `cuda_ipc`, convert to `PAPTensorTransport`, and pass it to `import_prefill_kv_from_paged_cache`. Export the env var in the launcher for both Attention and vLLM processes.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_true_split_contract.py tests/pap/test_pap_launch_files.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/model_executor/models/qwen3.py examples/pap/launch_pap_nixl.sh tests/pap/test_pap_true_split_contract.py tests/pap/test_pap_launch_files.py
git commit -m "Default PAP OFFLOAD_KV to CUDA IPC"
```

## Task 5: Verification, Docs, and Handoff

**Files:**
- Modify: `docs/design/pap_experiment_results.md`
- Modify: `.remember/remember.md`
- Local ignored update: `HANDOFF.md`

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py tests/pap/test_pap_attention_executor.py tests/pap/test_pap_true_split_contract.py tests/pap/test_pap_launch_files.py -q
```

Expected: PASS.

- [ ] **Step 2: Run 1PA1P E2E**

Run:

```bash
PAP_TOPOLOGY=1pa1p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=0 \
bash examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-0.6B
```

Expected: HTTP 200 response, Projection payload metadata-only, IPC import logs, no default TCP tensor-bundle prefill import.

- [ ] **Step 3: Run X:Y E2E**

Run:

```bash
PAP_TOPOLOGY=4pa2p PAP_SERVICE_ONLY=1 PAP_SKIP_SMOKE_REQUEST=0 \
bash examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-0.6B
```

Expected: multiple requests route across PA/Attention and Projection pairs, all HTTP 200, Projection metadata-only, IPC import logs, OFFLOAD_EXEC trace counts match output tokens.

- [ ] **Step 4: Update docs and memory**

Record unit/E2E results, commit ids, IPC status, and next risks in `docs/design/pap_experiment_results.md`, `.remember/remember.md`, and ignored local `HANDOFF.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/design/pap_experiment_results.md .remember/remember.md
git commit -m "Validate PAP OFFLOAD_KV IPC path"
```
