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

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py::test_attention_executor_ipc_import_keeps_opened_tensor_views -q
```

Expected failure:

- Assertion fails because `PAPAttentionRegistry.import_prefill_kv()` calls
  `detach().contiguous().to(self._storage_device)`, creating a new tensor.

- [x] **Step 3: Implement minimal no-copy support**

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

- [x] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_ipc_import_keeps_opened_tensor_views \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_binary_imports_prefill_kv_ipc_descriptor \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_imports_prefill_kv_before_stateful_decode \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_binary_imports_prefill_kv_before_decode -q
```

Expected: all selected tests pass.

- [x] **Step 5: Run PAP focused suite**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_launch_files.py -q
```

Expected: all selected tests pass.

- [x] **Step 6: Document and commit Phase 1**

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

- [x] **Step 1: Write failing descriptor roundtrip test**

Add `test_offload_kv_paged_ipc_descriptor_roundtrip()` to
`tests/pap/test_pap_data_plane.py`.

Expected behavior:

- A paged descriptor contains one CUDA IPC handle for the full layer KV cache
  backing tensor.
- It carries `block_ids`, `seq_len`, `block_size`, `num_kv_heads`, and `layout`.
- Roundtrip through `to_dict()` / `from_dict()` preserves all fields.

- [x] **Step 2: Write failing paged view test**

Add `test_paged_kv_segments_match_gathered_kv()` to
`tests/pap/test_pap_remote_attention.py`.

Expected behavior:

- Given a paged KV cache, block ids, seq_len, and layout, the helper returns
  segment tensor views over the original `kv_cache`.
- Segment contents match the existing `gather_paged_kv()` output after
  concatenation.
- The segment view shares storage with the original paged KV cache.

- [x] **Step 3: Implement data-plane descriptor**

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

- [x] **Step 4: Implement paged segment helper**

Add `paged_kv_segments()` to `vllm/pap/remote_attention.py`.

It should return `list[tuple[key_segment, value_segment]]` using the same layout
rules as `gather_paged_kv()`, but must not concatenate segments.

- [x] **Step 5: Add shadow-attention post helper**

Add a new `import_prefill_paged_kv()` helper in `vllm/pap/shadow_attention.py`
that posts command `import_prefill_paged_kv_ipc` with the paged descriptor and
no tensor payload.

This helper is the first Prefill-side entry point that can avoid gathering
prompt KV before export.

### Task 3: Attention Paged Descriptor Import

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write failing Attention import test**

Add a test that monkeypatches `open_ipc_paged_kv_cache()` to return a fake
paged KV tensor, posts `import_prefill_paged_kv_ipc`, then verifies
`append_decode_kv()` computes from resident paged segments whose tensor storage
points at the fake paged KV cache.

- [x] **Step 2: Implement command handler**

Add `open_ipc_paged_kv_cache()` and handle `import_prefill_paged_kv_ipc` in
`compute_binary_attention_response()`.

Registry should store paged Prefill KV as resident segments derived from the
opened paged KV cache rather than as copied contiguous K/V tensors.

- [x] **Step 3: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py -q
```

Expected: all selected tests pass.

### Task 4: Wire Qwen3 Prefill Export To Paged Descriptor

**Files:**
- Modify: `vllm/model_executor/models/qwen3.py`
- Test: `tests/pap/test_pap_true_split_contract.py`

- [x] **Step 1: Write contract test**

Added `test_qwen3_prefill_uses_paged_kv_import_for_cuda_ipc()` to verify
Qwen3 Prefill export calls `import_prefill_paged_kv()`, passes `kv_cache`
directly, derives `block_ids`, and no longer calls the gathered
`import_prefill_kv_from_paged_cache()` path.

- [x] **Step 2: Implement Qwen3 paged export**

`Qwen3Attention._maybe_import_pap_prefill_kv_to_attention()` now requires
`PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc`, derives Prefill block ids from the vLLM
block table, and sends a paged KV IPC descriptor through
`import_prefill_paged_kv()`.

- [x] **Step 3: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_attention_executor.py -q
```

Result: `81 passed, 16 warnings`.

Run:

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
- No old gathered IPC import log or error pattern matched in the E2E logs.

Boundary:

- Qwen3 Prefill now uses the resident paged descriptor path.
- Decode K/V still lands in Attention-local decode buffers.
- Next phase must write Projection-provided decode K/V into Prefill-owned
  paged blocks and add same-PA multi-turn reuse verification.

### Task 5: Decode K/V Write-Back To Attached Resident Blocks

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write resident write-back tests**

Added tests proving that when a layer was imported through the paged IPC
descriptor path, `append_decode_kv()` writes the next decode K/V token into the
attached paged KV backing tensor and returns resident segments rather than an
extra Attention-local decode segment.

Also added a regression for the runtime boundary where the attached resident
blocks are full: the registry must fall back to the Attention-local decode
buffer instead of writing past the resident backing tensor or crashing while
rebuilding resident segments.

- [x] **Step 2: Implement resident write-back metadata**

`PAPAttentionRegistry` now records `PAPResidentPagedKV` metadata for paged
imports. During descriptor-backed decode, it writes one-token K/V into the
attached paged KV cache when the target slot is covered by the attached blocks.
The resident segments are rebuilt over the same backing tensor, so no additional
Attention-local KV copy is kept for covered slots.

- [x] **Step 3: Preserve fallback behavior**

If a decode token targets a block not currently attached from Prefill, or the
attached block coverage is already full, the existing Attention-local decode
buffer path remains the fallback. This keeps current single-request E2E
functional while making the covered resident-block path no-copy.

- [x] **Step 4: Verify**

Run:

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
  `slot_mapping`, `rejected`, `block ids do not cover`, or `IndexError`
  matched in logs.

Boundary:

- Decode K/V write-back is implemented for slots already covered by the attached
  Prefill-owned paged KV blocks.
- Decode K/V that needs a new, not-yet-attached Prefill-owned block still falls
  back to Attention-local storage.
- Next phase must introduce `PAKVOwner`-style decode-slot/block allocation and
  publication so Attention can write all generated K/V into Prefill-owned
  blocks, then verify same-PA multi-turn reuse.

### Task 6: Publish New Resident Decode Blocks In Attached Backing

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write new-block resident write-back test**

Added `test_attention_registry_publishes_new_resident_decode_block()` to prove
that when decode crosses from the imported prompt block into a new block that
already exists in the attached Prefill-owned paged KV backing tensor, Attention
publishes that block into the resident set, writes K/V into it, and does not
fall back to `_decode_kv`.

- [x] **Step 2: Implement minimal publication**

`PAPAttentionRegistry._try_write_decode_to_resident_paged_kv()` now accepts a
decode `block_id` not present in `resident.block_ids` when that block is within
the attached paged KV backing tensor. It adds the block to the resident block
list, writes the decode K/V into the backing tensor, and rebuilds resident
segments over prompt plus generated blocks.

- [x] **Step 3: Verify**

Run:

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
- The run crosses the 16-token boundary and exercises a generated-token block.
- Attention logged `28` `PAP prefill paged KV imported via IPC descriptor`
  entries.
- Projection and Attention each logged `224` OFFLOAD_EXEC traces.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- This publishes new resident decode blocks only when the block id already maps
  into the attached Prefill-owned paged KV backing tensor.
- It is still not a full scheduler/worker `PAKVOwner`: block ids currently come
  from the active decode descriptor path, not from an explicit Prefill owner
  allocation API.
- Next phase should make decode slot/block reservation explicit in a
  `PAKVOwner`-style component and then verify same-PA multi-turn reuse.

### Task 7: PAKVOwner Metadata Skeleton

**Files:**
- Add: `vllm/pap/kv_owner.py`
- Test: `tests/pap/test_pap_kv_owner.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write owner metadata tests**

Added tests for `PAKVOwner` session registration, layer block registration,
decode slot reservation, decode slot materialization, backed-block rejection,
and lease/refcount release behavior.

- [x] **Step 2: Implement metadata owner**

Added `PAKVOwner`, `PAKVSessionState`, `PAKVLayerState`, and `PAKVDecodeSlot`.
The owner is pure metadata in this phase: it tracks session leases, resident
layer block ids, sequence length, backed block capacity, reserved decode slots,
and materialized decode slots.

- [x] **Step 3: Connect Attention registry**

`PAPAttentionRegistry` now creates a PA KV owner session on registration,
registers paged layer backing metadata on `import_prefill_paged_kv()`, and marks
resident decode writes as materialized in `PAKVOwner`.

- [x] **Step 4: Verify**

Run:

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
- The run crosses the 16-token boundary and exercises a generated-token block.
- Attention logged `28` paged Prefill KV descriptor imports.
- Projection and Attention each logged `224` OFFLOAD_EXEC traces.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- `PAKVOwner` is now present and records materialized decode slots, but it is
  not yet the allocator that decides physical block ids for vLLM.
- Decode block ids still arrive through the active descriptor path and are then
  validated/materialized by the owner.
- Next phase should move decode slot/block reservation earlier into the
  Prefill-owner path and use the owner state to drive same-PA multi-turn reuse.

### Task 8: Reserve OFFLOAD_EXEC Decode Slots Through PAKVOwner

**Files:**
- Modify: `vllm/pap/kv_owner.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_kv_owner.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write next-slot owner test**

Added a `PAKVOwner.reserve_next_decode_slot()` contract that derives the next
decode seq_len from the owner layer state, validates the target block against
the resident backing capacity, publishes the new block id in owner metadata,
and returns the block/slot descriptor.

- [x] **Step 2: Write OFFLOAD_EXEC owner-reservation test**

Added an Attention executor test proving `compute_offload_exec_output()` calls
the owner reservation path after paged Prefill KV has been imported, writes the
Projection-provided decode K/V into the resident paged backing tensor, records a
materialized owner slot, and avoids an Attention-local decode KV copy.

- [x] **Step 3: Implement minimal owner reservation path**

`PAPAttentionRegistry.reserve_decode_slot()` now resolves the session and, when
resident owner layer state exists, reserves the next decode slot through
`PAKVOwner`. If OFFLOAD_EXEC is replaying the current Prefill token
(`seq_len == owner_layer.seq_len`), it returns the existing resident block slot
without advancing owner state. Non-resident/fallback paths keep the previous
`seq_len -> block_id/slot` arithmetic.

`compute_offload_exec_output()` now obtains block/slot descriptors from the
registry reservation method before calling `append_decode_kv()`.

- [x] **Step 4: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py::test_compute_offload_exec_output_reserves_decode_slot_from_owner \
  tests/pap/test_pap_attention_executor.py::test_compute_offload_exec_output_uses_step_block_descriptor \
  tests/pap/test_pap_attention_executor.py::test_compute_offload_exec_output_from_packed_qkv \
  tests/pap/test_pap_kv_owner.py -q
```

Result: `7 passed, 16 warnings`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `103 passed, 16 warnings`.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- Attention logged `28` paged Prefill KV descriptor imports.
- Projection logged `224` OFFLOAD_EXEC traces.
- Attention logged `224` OFFLOAD_EXEC traces plus the `28` paged imports.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- OFFLOAD_EXEC shared-KV decode slot selection now enters `PAKVOwner`.
- `PAKVOwner` still computes the next block from logical seq_len and resident
  capacity; it is not yet integrated with a real vLLM physical block allocator.
- Same-PA multi-turn reuse remains unverified.

### Task 9: Separate Reserved and Materialized Resident Coverage

**Files:**
- Modify: `vllm/pap/kv_owner.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_kv_owner.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write resident coverage owner test**

Added a test proving `PAKVOwner` does not expose a merely reserved decode slot
as reusable resident prefix coverage. `get_resident_prefix_coverage()` reports
the prompt length after reservation and advances only after
`materialize_decode_slot()`.

- [x] **Step 2: Split owner reservation/materialization state**

`PAKVLayerState` now tracks `reserved_slots` separately from
`materialized_slots`. Reserving a decode slot no longer advances the layer's
resident `seq_len`; materialization advances the resident sequence length after
Attention has written K/V into the resident paged backing tensor.

- [x] **Step 3: Expose resident coverage through Attention registry**

`PAPAttentionRegistry.get_resident_prefix_coverage()` resolves wrapped request
ids and returns the owner coverage. A registry test proves resident coverage for
a same-PA session advances from prompt to prompt+decode after decode write-back
into Prefill-owned paged blocks.

- [x] **Step 4: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py::test_attention_registry_reports_resident_prefix_after_decode_writeback \
  tests/pap/test_pap_attention_executor.py::test_compute_offload_exec_output_reserves_decode_slot_from_owner -q
```

Result: `7 passed, 16 warnings`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `105 passed, 16 warnings`.

Run:

```bash
pre-commit run ruff-check --files \
  examples/pap/pap_attention_executor.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_kv_owner.py \
  vllm/pap/kv_owner.py
```

Result: passed.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- Projection logged `224` OFFLOAD_EXEC traces.
- Attention logged `224` OFFLOAD_EXEC traces plus `28` paged Prefill KV
  descriptor imports.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- The owner now has a safe same-PA resident coverage query for later-turn
  reuse: reserved slots are invisible until materialized.
- This is still an in-process metadata/query surface; vLLM scheduler
  integration for attaching resident blocks on the next turn is not implemented
  yet.
- The owner still does not allocate physical vLLM blocks.

### Task 10: Expose Resident Prefix Coverage Control Plane

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `examples/pap/pap_proxy_server.py`
- Test: `tests/pap/test_pap_attention_executor.py`
- Test: `tests/pap/test_pap_proxy_server.py`

- [x] **Step 1: Write resident-prefix endpoint test**

Added an Attention executor API test proving
`GET /v1/pap/attention/sessions/{request_id}/resident-prefix` returns
JSON-safe owner coverage after paged Prefill import and decode write-back.

- [x] **Step 2: Write proxy helper test**

Added a proxy helper test proving `get_attention_resident_prefix()` queries the
new internal Attention endpoint and returns the response JSON. This is the
control-plane call the proxy can use before later-turn route-back/reuse
scheduling.

- [x] **Step 3: Implement endpoint/helper**

`PAPAttentionRegistry.get_resident_prefix_coverage()` is now exposed through the
Attention FastAPI app. `pap_proxy_server.py` now provides
`get_attention_resident_prefix()`.

- [x] **Step 4: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py::test_attention_executor_resident_prefix_endpoint_reports_coverage \
  tests/pap/test_pap_proxy_server.py::test_get_attention_resident_prefix_queries_internal_endpoint \
  tests/pap/test_pap_proxy_server.py -q
```

Result: `11 passed, 16 warnings`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `126 passed, 16 warnings`.

Run:

```bash
pre-commit run ruff-check --files \
  examples/pap/pap_attention_executor.py \
  examples/pap/pap_proxy_server.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_proxy_server.py
```

Result: passed.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- Projection logged `224` OFFLOAD_EXEC traces.
- Attention logged `224` OFFLOAD_EXEC traces plus `28` paged Prefill KV
  descriptor imports.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- Same-PA resident coverage is now queryable over the internal Attention API and
  from the proxy helper layer.
- The proxy still does not maintain a conversation/session placement table or
  use resident coverage to skip Prefill work on a later turn.
- vLLM scheduler attach for resident blocks remains unimplemented.

### Task 11: Track Conversation Placement in Multi PAP Proxy

**Files:**
- Modify: `examples/pap/multi_pap_proxy_server.py`
- Test: `tests/pap/test_multi_pap_proxy_server.py`

- [x] **Step 1: Write placement route-back test**

Added a unit test for `select_conversation_instances()` proving an existing
conversation routes back to the original PA group and Projection instance
instead of following round-robin selection.

- [x] **Step 2: Write proxy placement/coverage contract**

Added a contract test that the multi proxy initializes a conversation placement
table, selects through `select_conversation_instances()`, queries Attention
resident prefix coverage, and updates the placement table after a request.

- [x] **Step 3: Implement minimal placement tracking**

`PAPConversationPlacement` now records conversation id, current request id, PA
group, Projection instance, Prefill KV handle, and last known resident seq_len.
`_handle_openai_request()` now:

- routes same-conversation requests back to their previous PA/Projection;
- queries Attention resident-prefix coverage for an existing placement;
- stores/updates the placement after Prefill returns.

- [x] **Step 4: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_pap_attention_executor.py -q
```

Result: `63 passed, 16 warnings`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `128 passed, 16 warnings`.

Run:

```bash
pre-commit run ruff-check --files \
  examples/pap/multi_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py
```

Result: passed.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- Projection logged `224` OFFLOAD_EXEC traces.
- Attention logged `224` OFFLOAD_EXEC traces plus `28` paged Prefill KV
  descriptor imports.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- The proxy now has same-PA route-back state and observes resident coverage for
  existing conversations.
- It still performs a normal Prefill request for later turns; skipping already
  resident tokens and attaching resident blocks in vLLM scheduler remains next.

### Task 12: Reuse Conversation Prefill KV Handle

**Files:**
- Modify: `examples/pap/multi_pap_proxy_server.py`
- Test: `tests/pap/test_multi_pap_proxy_server.py`

- [x] **Step 1: Write handle reuse test**

Added a unit test for `pap_prefill_kv_handle_for_request()` proving an existing
conversation placement reuses the original Prefill KV handle instead of the new
request's freshly registered Attention session handle.

- [x] **Step 2: Apply handle reuse in request path**

The multi proxy now uses `pap_prefill_kv_handle_for_request()` when attaching
PAP Prefill->Attention parameters, building the Projection payload, and updating
conversation placement metadata.

- [x] **Step 3: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_pap_attention_executor.py -q
```

Result: `65 passed, 16 warnings`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `130 passed, 16 warnings`.

Run:

```bash
pre-commit run ruff-check --files \
  examples/pap/multi_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py
```

Result: passed.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- Projection logged `224` OFFLOAD_EXEC traces.
- Attention logged `224` OFFLOAD_EXEC traces plus `28` paged Prefill KV
  descriptor imports.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- Existing conversation requests now keep pointing Prefill and Projection at
  the original owner session handle.
- The proxy still registers a new Attention request shell and still sends a
  normal Prefill request; skipping resident prefix work and scheduler attach
  remain next.

### Task 13: Preserve Resident Decode Coverage on Reimport

**Files:**
- Modify: `vllm/pap/kv_owner.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_pap_kv_owner.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [x] **Step 1: Write owner re-registration test**

Added an owner test proving a repeated layer registration for the same session
does not drop previously materialized decode coverage.

- [x] **Step 2: Write Attention reimport test**

Added an Attention registry test proving a second paged Prefill import for the
same session/layer keeps owner resident coverage, `PAPResidentPagedKV.block_ids`,
and resident Prefill segments at prompt+decode instead of regressing to prompt
only.

- [x] **Step 3: Implement merge semantics**

`PAKVOwner.register_layer_blocks()` now merges repeated layer registrations with
existing block ids, sequence length, backed capacity, reserved slots, and
materialized slots.

`PAPAttentionRegistry.import_prefill_paged_kv()` now preserves existing resident
block publication and rebuilds resident segments from owner coverage after
registering layer blocks.

- [x] **Step 4: Verify**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_remote_attention.py -q
```

Result: `59 passed, 16 warnings`.

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_true_split_contract.py \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_remote_attention.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_proxy_server.py \
  tests/pap/test_multi_pap_proxy_server.py \
  tests/pap/test_pap_launch_files.py -q
```

Result: `132 passed, 16 warnings`.

Run:

```bash
pre-commit run ruff-check --files \
  vllm/pap/kv_owner.py \
  examples/pap/pap_attention_executor.py \
  tests/pap/test_pap_kv_owner.py \
  tests/pap/test_pap_attention_executor.py
```

Result: passed.

E2E 1PA1P Qwen3-0.6B with `max_tokens=8`:

- Request returned HTTP `200`.
- Usage: `prompt_tokens=11`, `completion_tokens=7`, `total_tokens=18`.
- Output text: ` P P\n\n\n\n.\n\n`
- Projection logged `224` OFFLOAD_EXEC traces.
- Attention logged `224` OFFLOAD_EXEC traces plus `28` paged Prefill KV
  descriptor imports.
- No `Traceback`, `ERROR`, `Got kv_transfer_params`, `invalid slot`,
  `slot_mapping`, `rejected`, `block ids do not cover`, `IndexError`, or
  `500 Internal` matched in logs.

Boundary:

- Reusing the original owner session handle no longer risks losing previously
  materialized decode coverage when Prefill reimports prompt paged KV.
- The system still performs a normal Prefill request for later turns; scheduler
  resident attach/skip remains next.
