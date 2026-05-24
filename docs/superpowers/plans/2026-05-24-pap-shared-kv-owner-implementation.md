# PAP Shared KV Owner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PAP shared KV in phases so Prefill and Attention stop duplicating KV, starting with no-copy IPC import into Attention.

**Architecture:** The first implementation phase changes Attention IPC import semantics from "open then copy into Attention registry" to "open and retain Prefill-exported tensor views." Later phases add `PAKVOwner`, resident block attach, decode write-back, and same-PA multi-turn reuse. Each phase must preserve Projection metadata-only behavior.

**Tech Stack:** Python, PyTorch tensors/CUDA IPC descriptors, vLLM PAP helpers, FastAPI test client, pytest via `.venv/bin/python`.

---

## File Structure

- `examples/pap/pap_attention_executor.py`: Attention registry and IPC import behavior.
- `tests/pap/test_pap_attention_executor.py`: first-phase no-copy IPC import contract.
- `docs/superpowers/specs/2026-05-24-pap-shared-kv-owner-design.md`: source design.
- `.remember/remember.md`: phase summaries.
- `HANDOFF.md`: ignored local handoff, updated after each phase.

## Phase 1 Scope

Phase 1 proves a real step toward the final objective:

- IPC-imported Prefill KV tensors are not copied inside Attention.
- Attention registry stores the opened tensors as the Prefill-owned source view.
- Existing non-IPC tensor bundle import can keep copy semantics as a fallback.

Phase 1 does not yet prove full resident vLLM block attach because the current
Prefill export still gathers paged KV into contiguous exported tensors. That is
Phase 2.

### Task 1: No-Copy IPC Import Contract

**Files:**
- Modify: `tests/pap/test_pap_attention_executor.py`
- Modify: `examples/pap/pap_attention_executor.py`

- [ ] **Step 1: Write the failing test**

Add a test near `test_attention_executor_binary_imports_prefill_kv_ipc_descriptor`:

```python
def test_attention_executor_ipc_import_keeps_opened_tensor_views(
    monkeypatch,
) -> None:
    import torch
    from fastapi.testclient import TestClient

    from examples.pap import pap_attention_executor
    from examples.pap.pap_attention_executor import create_app
    from vllm.pap.remote_attention import (
        deserialize_tensor_bundle,
        serialize_tensor_bundle,
    )

    key = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    value = torch.tensor([[[2.0, 0.0]], [[0.0, 4.0]]])

    def fake_open_ipc_prefill_kv(descriptor):
        return key, value

    monkeypatch.setattr(
        pap_attention_executor,
        "open_ipc_prefill_kv",
        fake_open_ipc_prefill_kv,
    )

    descriptor = PAPOffloadKVIPCDescriptor(
        request_id="req-shared-ipc",
        layer_name="model.layers.0.self_attn.attn",
        seq_len=2,
        block_ids=(4,),
        key=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(key.shape),
            ipc_handle={"GPU-test": ("key", 1, 2, 3, 4, 5, 0)},
        ),
        value=PAPCudaIPCTensorHandle(
            dtype="float32",
            shape=tuple(value.shape),
            ipc_handle={"GPU-test": ("value", 1, 2, 3, 4, 5, 0)},
        ),
    )

    app = create_app()
    client = TestClient(app)
    client.post(
        "/v1/pap/attention/register",
        json={
            "request_id": "req-shared-ipc",
            "conversation_id": "conv-shared-ipc",
            "prefill_endpoint": "http://localhost:8100",
            "kv_transfer_params": {},
            "prefix_len": 2,
        },
    )

    imported = client.post(
        "/v1/pap/attention/import-prefill-kv-binary",
        content=serialize_tensor_bundle(
            {
                "command": "import_prefill_kv_ipc",
                "descriptor": descriptor.to_dict(),
            },
            {},
        ),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert imported.status_code == 200
    metadata, tensors = deserialize_tensor_bundle(imported.content)
    assert metadata["seq_len"] == 2
    assert tensors == {}

    registry = app.state.registry
    session_id = registry.resolve_session_request_id("req-shared-ipc")
    stored_key, stored_value = registry._prefill_kv[session_id][
        "model.layers.0.self_attn.attn"
    ]
    assert stored_key.data_ptr() == key.data_ptr()
    assert stored_value.data_ptr() == value.data_ptr()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py::test_attention_executor_ipc_import_keeps_opened_tensor_views -q
```

Expected failure:

- Assertion fails because `PAPAttentionRegistry.import_prefill_kv()` calls
  `detach().contiguous().to(self._storage_device)`, creating a new tensor.

- [ ] **Step 3: Implement minimal no-copy support**

Change `PAPAttentionRegistry.import_prefill_kv()` to accept a keyword-only
`copy: bool = True`.

Use:

```python
if copy:
    key_state = key.detach().contiguous().to(self._storage_device)
    value_state = value.detach().contiguous().to(self._storage_device)
else:
    key_state = key.detach()
    value_state = value.detach()
```

Then change the `import_prefill_kv_ipc` binary command handler to call:

```python
seq_len = registry.import_prefill_kv(
    request_id=descriptor.request_id,
    layer_name=descriptor.layer_name,
    key=key,
    value=value,
    seq_len=descriptor.seq_len,
    block_ids=list(descriptor.block_ids),
    copy=False,
)
```

Leave non-IPC `import_prefill_kv` paths unchanged so fallback tensor bundle
imports still copy into Attention-owned storage.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_ipc_import_keeps_opened_tensor_views \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_binary_imports_prefill_kv_ipc_descriptor \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_imports_prefill_kv_before_stateful_decode \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_binary_imports_prefill_kv_before_decode -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Run PAP focused suite**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Document and commit Phase 1**

Update:

- `.remember/remember.md`
- `HANDOFF.md`

Commit:

```bash
git add examples/pap/pap_attention_executor.py \
  tests/pap/test_pap_attention_executor.py \
  .remember/remember.md
git commit -m "Keep PAP IPC prefill KV shared in Attention"
```

Do not commit `HANDOFF.md`; it is ignored local handoff state.

### Task 2: Paged KV IPC Descriptor

**Files:**
- Modify: `vllm/pap/data_plane.py`
- Modify: `vllm/pap/remote_attention.py`
- Modify: `vllm/pap/shadow_attention.py`
- Test: `tests/pap/test_pap_data_plane.py`
- Test: `tests/pap/test_pap_remote_attention.py`

- [ ] **Step 1: Write failing descriptor roundtrip test**

Add `test_offload_kv_paged_ipc_descriptor_roundtrip()` to
`tests/pap/test_pap_data_plane.py`.

Expected behavior:

- A paged descriptor contains one CUDA IPC handle for the full layer KV cache
  backing tensor.
- It carries `block_ids`, `seq_len`, `block_size`, `num_kv_heads`, and `layout`.
- Roundtrip through `to_dict()` / `from_dict()` preserves all fields.

- [ ] **Step 2: Write failing paged view test**

Add `test_paged_kv_segments_match_gathered_kv()` to
`tests/pap/test_pap_remote_attention.py`.

Expected behavior:

- Given a paged KV cache, block ids, seq_len, and layout, the helper returns
  segment tensor views over the original `kv_cache`.
- Segment contents match the existing `gather_paged_kv()` output after
  concatenation.
- The segment view shares storage with the original paged KV cache.

- [ ] **Step 3: Implement data-plane descriptor**

Add `PAPOffloadKVPagedIPCDescriptor` to `vllm/pap/data_plane.py` with fields:

- `request_id: str`
- `layer_name: str`
- `seq_len: int`
- `block_ids: tuple[int, ...]`
- `block_size: int`
- `num_kv_heads: int`
- `layout: str`
- `kv_cache: PAPCudaIPCTensorHandle`
- `transport: PAPTensorTransport = PAPTensorTransport.CUDA_IPC`

- [ ] **Step 4: Implement paged segment helper**

Add `paged_kv_segments()` to `vllm/pap/remote_attention.py`.

It should return `list[tuple[key_segment, value_segment]]` using the same layout
rules as `gather_paged_kv()`, but must not concatenate segments.

- [ ] **Step 5: Add shadow-attention post helper**

Add a new `import_prefill_paged_kv()` helper in `vllm/pap/shadow_attention.py`
that posts command `import_prefill_paged_kv_ipc` with the paged descriptor and
no tensor payload.

This helper is the first Prefill-side entry point that can avoid gathering
prompt KV before export.

### Task 3: Attention Paged Descriptor Import

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [ ] **Step 1: Write failing Attention import test**

Add a test that monkeypatches `open_ipc_paged_kv_cache()` to return a fake
paged KV tensor, posts `import_prefill_paged_kv_ipc`, then verifies
`append_decode_kv()` computes from resident paged segments whose tensor storage
points at the fake paged KV cache.

- [ ] **Step 2: Implement command handler**

Add `open_ipc_paged_kv_cache()` and handle `import_prefill_paged_kv_ipc` in
`compute_binary_attention_response()`.

Registry should store paged Prefill KV as resident segments derived from the
opened paged KV cache rather than as copied contiguous K/V tensors.

- [ ] **Step 3: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py -q
```

Expected: all selected tests pass.
