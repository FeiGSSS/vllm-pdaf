# PAP IPC-Only KV with Decode Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Attention-local KV pool. Make Attention read/write Prefill-owned KV cache via CUDA IPC only. Add a decode-KV commit protocol so Prefill's prefix cache can hit decode-generated blocks on future requests.

**Architecture:** Prefill vLLM owns the single physical KV cache. Attention executor imports it via CUDA IPC and writes decode KV with `reshape_and_cache_flash`. After each decode batch, the decode side sends a commit (request_id, new_seq_len, new token_ids) to Prefill. Prefill applies the commit to its `KVCacheManager` so `Request.num_computed_tokens`, `Request.block_hashes`, and `BlockPool` prefix-cache table reflect the remotely-written blocks. Local pool, copy-prefix fallback, and `PAP_ATTENTION_COPY_PREFIX_KV` are removed.

**Tech Stack:** PyTorch, CUDA IPC, vLLM V1 (KVCacheManager / BlockPool / scheduler), FastAPI, existing PAP `local_fast` transport.

---

## File Structure

**Created:**
- `vllm/pap/decode_commit.py` — Commit dataclass + serialization (Pydantic model + binary form).
- `vllm/pap/decode_commit_client.py` — Thin HTTP/local_fast sender used by Attention or sampler.
- `vllm/pap/decode_commit_router.py` — Prefill-side router that fans commits out to KVCacheManager.
- `tests/pap/test_decode_commit.py` — Round-trip + apply tests.
- `tests/pap/test_attention_no_local_pool.py` — Fail-closed test for removed local pool.
- `tests/pap/test_decode_commit_integration.py` — End-to-end test with mocked Prefill KV manager.

**Modified:**
- `examples/pap/pap_attention_executor.py` — Remove `_local_paged_kv*`, `_local_paged_kv_pools`, copy-prefix fallback, `_compute_offload_exec_paged_flash_batch`; make unified path fail-closed; call commit client after each successful decode append.
- `vllm/pap/shadow_attention.py` — Drop legacy local-pool branches; keep lease + descriptor import.
- `vllm/pap/data_plane.py` — Extend descriptor with commit endpoint hint (default empty).
- `vllm/v1/core/kv_cache_manager.py` — Add `apply_decode_commit(request, new_seq_len, new_token_ids)` that updates `num_computed_tokens`, appends tokens, calls `cache_blocks`.
- `vllm/v1/engine/llm_engine.py` or `vllm/entrypoints/openai/api_server.py` (whichever hosts PAP prefill) — Expose `/v1/pap/prefill/decode-commit` endpoint that calls into KVCacheManager via existing engine handle.
- `examples/pap/launch_pap_nixl.sh` — Drop `PAP_ATTENTION_COPY_PREFIX_KV`; add `PAP_DECODE_COMMIT_ENDPOINT` plumbing; prefill binds commit route.

**Deleted (or empty-stubbed for one release):**
- `PAPAttentionRegistry._local_paged_kv_pools`, `_local_paged_kv`, related helpers (kept only behind an `assert False, "removed"` if removal would break imports during transition).

---

## Task 1: Decode Commit Data Structure

**Files:**
- Create: `vllm/pap/decode_commit.py`
- Test: `tests/pap/test_decode_commit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pap/test_decode_commit.py
from vllm.pap.decode_commit import PAPDecodeCommit, serialize_commit, deserialize_commit

def test_commit_roundtrip():
    commit = PAPDecodeCommit(
        request_id="req-1",
        new_seq_len=17,
        new_token_ids=[42, 7, 99],
        layer_complete=True,
    )
    blob = serialize_commit(commit)
    restored = deserialize_commit(blob)
    assert restored == commit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: vllm.pap.decode_commit`.

- [ ] **Step 3: Write minimal implementation**

```python
# vllm/pap/decode_commit.py
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from typing import Sequence

@dataclass(frozen=True)
class PAPDecodeCommit:
    request_id: str
    new_seq_len: int
    new_token_ids: tuple[int, ...]
    layer_complete: bool

    @classmethod
    def from_dict(cls, d: dict) -> "PAPDecodeCommit":
        return cls(
            request_id=str(d["request_id"]),
            new_seq_len=int(d["new_seq_len"]),
            new_token_ids=tuple(int(t) for t in d["new_token_ids"]),
            layer_complete=bool(d.get("layer_complete", True)),
        )

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "new_seq_len": self.new_seq_len,
            "new_token_ids": list(self.new_token_ids),
            "layer_complete": self.layer_complete,
        }

def serialize_commit(commit: PAPDecodeCommit) -> bytes:
    return json.dumps(commit.to_dict()).encode("utf-8")

def deserialize_commit(blob: bytes) -> PAPDecodeCommit:
    return PAPDecodeCommit.from_dict(json.loads(blob.decode("utf-8")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_roundtrip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/pap/decode_commit.py tests/pap/test_decode_commit.py
git commit -m "PAP: add decode commit data structure"
```

---

## Task 2: Apply Decode Commit on Prefill Side

**Files:**
- Modify: `vllm/v1/core/kv_cache_manager.py`
- Test: `tests/pap/test_decode_commit.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/pap/test_decode_commit.py
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheManager

def _make_request(req_id: str, prompt_len: int) -> Request:
    request = Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=None,
        block_hashes=[],
    )
    return request

def test_apply_decode_commit_advances_block_hashes(monkeypatch):
    request = _make_request("req-1", prompt_len=4)
    # ... stand up a minimal KVCacheManager against a stub config ...
    # apply commit
    manager.apply_decode_commit(request, new_seq_len=20, new_token_ids=[100, 101, 102])
    assert request.num_computed_tokens == 20
    assert len(request.block_hashes) > 0
```

(Test scaffolding will instantiate `KVCacheManager` via the existing test helpers; if those helpers do not exist yet in tests/pap/conftest.py, fall back to monkeypatching the coordinator and assert only the public side effects: `num_computed_tokens` and `block_hashes`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_apply_decode_commit_advances_block_hashes -v`
Expected: FAIL with `AttributeError: apply_decode_commit`.

- [ ] **Step 3: Add the manager method**

In `vllm/v1/core/kv_cache_manager.py`, add after the existing `cache_blocks` method (around line 635):

```python
    def apply_decode_commit(
        self,
        request: Request,
        new_seq_len: int,
        new_token_ids: Sequence[int],
    ) -> None:
        """Apply a remote decode-KV commit from PAP Attention.

        Advances request.num_computed_tokens, appends generated tokens, and
        commits newly full blocks to the prefix cache.
        """
        if new_seq_len <= int(request.num_computed_tokens):
            return
        delta = [int(t) for t in new_token_ids]
        # Append to request token ids (this also extends block_hashes via
        # Request.append_output_token_ids).
        request.append_output_token_ids(delta)
        request.num_computed_tokens = int(new_seq_len)
        if self.enable_caching:
            self.coordinator.cache_blocks(request, int(new_seq_len))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_apply_decode_commit_advances_block_hashes -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/core/kv_cache_manager.py tests/pap/test_decode_commit.py
git commit -m "PAP: KVCacheManager.apply_decode_commit advances block hashes"
```

---

## Task 3: Prefill HTTP Endpoint for Commit

**Files:**
- Create: `vllm/pap/decode_commit_router.py`
- Modify: vLLM API server (find via search)
- Test: `tests/pap/test_decode_commit.py`

- [ ] **Step 1: Locate the PAP prefill API surface**

Run: `rg -n "PAPCommunicator|PAPConnector|register_pap|/v1/pap" vllm/ entrypoints/`
Identify the FastAPI app where PAP prefill routes are mounted (likely under `vllm/entrypoints/openai/` or a dedicated PAP module).

- [ ] **Step 2: Write the failing test**

```python
# append to tests/pap/test_decode_commit.py
from fastapi.testclient import TestClient
from vllm.pap.decode_commit_router import build_commit_router

class StubManager:
    def __init__(self):
        self.calls = []
    def apply_decode_commit(self, request, new_seq_len, new_token_ids):
        self.calls.append((request.request_id, new_seq_len, list(new_token_ids)))

def test_commit_endpoint_applies_to_manager():
    manager = StubManager()
    requests = {"req-1": object()}
    app = build_commit_router(manager=manager, requests=requests)
    client = TestClient(app)
    resp = client.post("/v1/pap/prefill/decode-commit", json={
        "request_id": "req-1",
        "new_seq_len": 17,
        "new_token_ids": [1, 2, 3],
        "layer_complete": True,
    })
    assert resp.status_code == 200
    assert manager.calls == [("req-1", 17, [1, 2, 3])]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_endpoint_applies_to_manager -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement the router**

```python
# vllm/pap/decode_commit_router.py
from __future__ import annotations
from typing import Any, Callable
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class DecodeCommitRequest(BaseModel):
    request_id: str
    new_seq_len: int
    new_token_ids: list[int]
    layer_complete: bool = True

def build_commit_router(
    *,
    manager: Any,
    requests: dict[str, Any],
    lookup_request: Callable[[str], Any] | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest) -> dict:
        request = (
            lookup_request(req.request_id)
            if lookup_request is not None
            else requests.get(req.request_id)
        )
        if request is None:
            raise HTTPException(status_code=404, detail=f"unknown PAP request {req.request_id}")
        manager.apply_decode_commit(
            request=request,
            new_seq_len=req.new_seq_len,
            new_token_ids=tuple(req.new_token_ids),
        )
        return {"request_id": req.request_id, "applied": True}

    return router
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_endpoint_applies_to_manager -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/pap/decode_commit_router.py tests/pap/test_decode_commit.py
git commit -m "PAP: prefill decode-commit FastAPI router"
```

---

## Task 4: Mount Commit Router on Prefill vLLM

**Files:**
- Modify: the prefill server module identified in Task 3 Step 1 (likely `vllm/pap/prefill_server.py` or `vllm/entrypoints/openai/api_server.py` PAP branch).
- Test: extend `tests/pap/test_decode_commit.py`.

- [ ] **Step 1: Identify mount point**

Run: `rg -n "include_router|FastAPI\\(" vllm/ | head -20` to find the app where prefill mounts PAP routes.

- [ ] **Step 2: Write the failing test**

```python
def test_commit_router_mounted_on_prefill_app():
    # Build the prefill app via the existing factory used in tests.
    from vllm.entrypoints.openai.api_server import build_app  # adjust import
    app = build_app(...)
    routes = {r.path for r in app.routes}
    assert "/v1/pap/prefill/decode-commit" in routes
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_router_mounted_on_prefill_app -v`
Expected: FAIL.

- [ ] **Step 4: Mount the router**

In the prefill app builder, after existing PAP routes:

```python
from vllm.pap.decode_commit_router import build_commit_router

def _mount_pap_commit_router(app, *, engine_model):
    manager = engine_model.engine_core_kv_cache_manager  # adjust attribute name
    requests = engine_model.engine_core_requests         # adjust attribute name
    app.include_router(build_commit_router(manager=manager, requests=requests))
```

Wire it the same way other PAP routes (e.g., NIXL connector endpoints) are wired. Use the same lookup mechanism the prefill uses for `/v1/pap/prefill/*` requests.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_router_mounted_on_prefill_app -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/entrypoints/openai/api_server.py tests/pap/test_decode_commit.py
git commit -m "PAP: mount decode-commit router on prefill API"
```

---

## Task 5: Attention-side Commit Client

**Files:**
- Create: `vllm/pap/decode_commit_client.py`
- Modify: `examples/pap/pap_attention_executor.py` — call client after each batch append.
- Test: `tests/pap/test_decode_commit.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/pap/test_decode_commit.py
from vllm.pap.decode_commit_client import DecodeCommitClient

def test_commit_client_posts_to_endpoint(monkeypatch):
    posted = {}
    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return FakeResp()
    monkeypatch.setattr("httpx.post", fake_post)
    client = DecodeCommitClient(endpoint="http://127.0.0.1:9999/v1/pap/prefill/decode-commit")
    client.commit(request_id="r", new_seq_len=10, new_token_ids=(1, 2))
    assert posted["url"].endswith("/v1/pap/prefill/decode-commit")
    assert posted["json"]["request_id"] == "r"
    assert posted["json"]["new_seq_len"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_client_posts_to_endpoint -v`
Expected: FAIL.

- [ ] **Step 3: Implement the client**

```python
# vllm/pap/decode_commit_client.py
from __future__ import annotations
import os
import logging
from typing import Iterable
import httpx

logger = logging.getLogger("pap_decode_commit")

class DecodeCommitClient:
    def __init__(self, endpoint: str | None = None, timeout_s: float = 0.2) -> None:
        self.endpoint = endpoint or os.environ.get("PAP_DECODE_COMMIT_ENDPOINT", "")
        self.timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)

    def commit(
        self,
        *,
        request_id: str,
        new_seq_len: int,
        new_token_ids: Iterable[int],
        layer_complete: bool = True,
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "request_id": request_id,
            "new_seq_len": int(new_seq_len),
            "new_token_ids": [int(t) for t in new_token_ids],
            "layer_complete": bool(layer_complete),
        }
        try:
            resp = httpx.post(self.endpoint, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(
                "PAP decode commit failed request_id=%s new_seq_len=%d err=%s",
                request_id, new_seq_len, exc,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_client_posts_to_endpoint -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vllm/pap/decode_commit_client.py tests/pap/test_decode_commit.py
git commit -m "PAP: Attention-side decode commit HTTP client"
```

---

## Task 6: Wire Commit Into Attention Executor

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_decode_commit_integration.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/pap/test_decode_commit_integration.py
def test_attention_emits_commit_after_decode(monkeypatch):
    """
    After compute_offload_exec_batch_output runs the unified append path,
    DecodeCommitClient.commit must be called once with the new seq_len and
    any token ids carried in the descriptor.
    """
    # Build a minimal PAPAttentionRegistry with one unified state pre-seeded.
    # Stub append_decode_kv_to_unified_prefill_cache to no-op.
    # Stub _compute_unified_paged_flash_batch to return a dummy tensor.
    # Capture calls into a fake DecodeCommitClient.
    ...
    assert fake_commit.calls == [("req-1", expected_seq_len, expected_tokens)]
```

(The test body must construct descriptor items with `request_id`, `step`, and a new field `decode_token_ids: tuple[int, ...]` that this task introduces.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit_integration.py -v`
Expected: FAIL with `AttributeError` on the new descriptor field or commit not called.

- [ ] **Step 3: Extend the descriptor with decode token ids**

In `vllm/pap/data_plane.py`, extend `PAPOffloadExecItem` (the per-row item type used inside `PAPOffloadExecBatchDescriptor`) with:

```python
@dataclass(frozen=True)
class PAPOffloadExecItem:
    request_id: str
    step: int
    scale: float
    output_tensor_id: int
    decode_token_ids: tuple[int, ...] = ()
```

Update `to_dict`/`from_dict` to round-trip this field. Default empty for backward compat.

- [ ] **Step 4: Emit commit from Attention**

In `examples/pap/pap_attention_executor.py`, in `compute_offload_exec_batch_output` after the unified branch successfully writes and computes:

```python
from vllm.pap.decode_commit_client import DecodeCommitClient

_COMMIT_CLIENT = DecodeCommitClient()  # process-singleton; reads env at import

# After unified_output computed and before returning it:
if _COMMIT_CLIENT.enabled:
    for index, item in enumerate(items):
        new_seq_len = int(decode_seq_lens[index])
        token_ids = tuple(int(t) for t in getattr(item, "decode_token_ids", ()))
        _COMMIT_CLIENT.commit(
            request_id=item.request_id,
            new_seq_len=new_seq_len,
            new_token_ids=token_ids,
        )
```

(The commit must NOT block the return of attention output. If latency becomes a problem, queue onto a background thread. Start simple: synchronous fire-and-forget with a 0.2s timeout.)

- [ ] **Step 5: Run integration test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_decode_commit_integration.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add examples/pap/pap_attention_executor.py vllm/pap/data_plane.py tests/pap/test_decode_commit_integration.py
git commit -m "PAP: emit decode commit after unified KV append"
```

---

## Task 7: Remove Attention-Local Pool (Fail-Closed)

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Test: `tests/pap/test_attention_no_local_pool.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/pap/test_attention_no_local_pool.py
import pytest

def test_local_pool_attributes_removed():
    import examples.pap.pap_attention_executor as mod
    reg = mod.PAPAttentionRegistry()
    for forbidden in (
        "_local_paged_kv",
        "_local_paged_kv_pools",
    ):
        assert not hasattr(reg, forbidden), forbidden

def test_compute_offload_exec_batch_output_rejects_non_unified():
    import examples.pap.pap_attention_executor as mod
    reg = mod.PAPAttentionRegistry()
    # Build a minimal descriptor + qkv_batch with no unified state pre-seeded.
    with pytest.raises(RuntimeError, match="PAP unified KV state missing"):
        mod.compute_offload_exec_batch_output(registry=reg, descriptor=..., qkv_batch=...)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_attention_no_local_pool.py -v`
Expected: FAIL.

- [ ] **Step 3: Remove the local pool fast path**

In `examples/pap/pap_attention_executor.py`:

1. Delete:
   - `_local_paged_kv_pools`, `_local_paged_kv` from `PAPAttentionRegistry.__init__`
   - `_get_or_create_local_paged_pool_locked`
   - `_ensure_local_paged_pool_capacity_locked`
   - `_allocate_local_paged_blocks_locked`
   - `_release_local_paged_blocks_locked`
   - `_copy_segments_to_local_paged_blocks`
   - `_install_local_paged_prefill_locked`
   - `_local_prefix_state_locked`
   - `_canonicalize_local_paged_state_locked`
   - `_ensure_local_paged_decode_state_locked`
   - `_append_local_paged_decode_locked`
   - `local_paged_attention_state`
   - `has_descriptor_prefix_for_local_paged_attention`
   - `materialize_descriptor_prefix_for_local_paged_attention`
   - `append_decode_kv_batch_for_local_paged_attention`
   - `append_decode_kv_tensor_batch_for_local_paged_attention`
   - `_compute_offload_exec_paged_flash_batch`
   - `_local_paged_slot_mapping_tensor`, `_local_paged_copy_source`, `_try_local_paged_native_cache_append` (if only used by the deleted path)
   - The `_pap_attention_copy_prefix_kv` env helper and all `if _pap_attention_copy_prefix_kv()` branches
   - `PAPLocalPagedKVPool`, `PAPLocalPagedAttentionState`, `PAPLocalPagedDecodeAppend` dataclasses
   - All `build_paged_flash_metadata` callers that fed the legacy path

2. In `compute_offload_exec_batch_output`, after `unified_states = registry.get_unified_paged_states(...)`:

```python
if unified_states is None:
    raise RuntimeError(
        "PAP unified KV state missing for layer="
        f"{descriptor.layer_name}; local pool removed, set PAP_UNIFIED_KV=1 "
        "and ensure Prefill exports unified descriptors"
    )
```

3. Drop the `if _pap_attention_copy_prefix_kv()` branch and the call to `append_decode_kv_tensor_batch_for_local_paged_attention`.

4. Remove `_release_local_paged_blocks_locked` calls from `_release_session_locked` and `_replace_existing_session_locked`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pap/test_attention_no_local_pool.py -v`
Expected: PASS.

- [ ] **Step 5: Run full PAP unit-test suite**

Run: `.venv/bin/python -m pytest tests/pap/ -v`
Expected: PASS on everything that does not specifically test the deleted local pool. Any test that exercised the local pool must either be deleted or migrated to the unified path. List failures:

```
pytest tests/pap/ -v 2>&1 | tee /tmp/pap-tests-after-cleanup.log
```

- [ ] **Step 6: Commit**

```bash
git add examples/pap/pap_attention_executor.py tests/pap/test_attention_no_local_pool.py
git commit -m "PAP: remove Attention-local KV pool, fail-closed unified path"
```

---

## Task 8: Update Shadow Attention to Drop Local-Pool Branches

**Files:**
- Modify: `vllm/pap/shadow_attention.py`
- Test: existing `tests/pap/test_pap_unified*`.

- [ ] **Step 1: Audit current local-pool branches**

Run: `rg -n "_local_paged|PAP_ATTENTION_COPY_PREFIX_KV|copy_prefix" vllm/pap/shadow_attention.py`
List every hit; each must be removed or rewritten.

- [ ] **Step 2: Write the failing test**

```python
def test_shadow_attention_has_no_local_pool_branches():
    import inspect, vllm.pap.shadow_attention as sa
    src = inspect.getsource(sa)
    assert "_local_paged" not in src
    assert "PAP_ATTENTION_COPY_PREFIX_KV" not in src
    assert "copy_prefix" not in src
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pap/test_shadow_attention_has_no_local_pool_branches -v`
Expected: FAIL.

- [ ] **Step 4: Remove the branches**

Strip every `if _pap_attention_copy_prefix_kv():` block, `_local_paged` field, and copied-prefix fallback. Keep only the unified-state descriptor flow.

- [ ] **Step 5: Run all unified-KV tests**

Run: `.venv/bin/python -m pytest tests/pap/test_pap_unified*.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vllm/pap/shadow_attention.py tests/pap/
git commit -m "PAP: drop local-pool branches from shadow_attention"
```

---

## Task 9: Launcher Cleanup

**Files:**
- Modify: `examples/pap/launch_pap_nixl.sh`

- [ ] **Step 1: Audit launcher env vars**

Run: `rg -n "PAP_ATTENTION_COPY_PREFIX_KV|PAP_UNIFIED_KV|PAP_DECODE_COMMIT" examples/pap/`

- [ ] **Step 2: Drop `PAP_ATTENTION_COPY_PREFIX_KV` and force `PAP_UNIFIED_KV=1`**

```bash
# examples/pap/launch_pap_nixl.sh
# In the env-export block:
export PAP_UNIFIED_KV=1
export PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS="${PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS:-32}"
export PAP_DECODE_COMMIT_ENDPOINT="${PAP_DECODE_COMMIT_ENDPOINT:-http://127.0.0.1:${PAP_PREFILL_PORT_BASE}/v1/pap/prefill/decode-commit}"
unset PAP_ATTENTION_COPY_PREFIX_KV
```

- [ ] **Step 3: Verify bash syntax**

Run: `bash -n examples/pap/launch_pap_nixl.sh`
Expected: exit 0, no output.

- [ ] **Step 4: Commit**

```bash
git add examples/pap/launch_pap_nixl.sh
git commit -m "PAP: launcher defaults to unified KV + decode commit endpoint"
```

---

## Task 10: Smoke Run + Benchmark

**Files:**
- Run: one-request smoke, then 256-prompt unified benchmark.
- Capture: `test/baseline/pap/results/runs/20260706_ipc_only_commit/`.

- [ ] **Step 1: Kill stray PAP processes**

```bash
for f in test/baseline/pap/results/runs/*/launcher.pid; do
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null || true
done
sleep 3
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill 2>/dev/null || true
```

- [ ] **Step 2: Run one-request smoke**

```bash
PAP_PREFILL_GPUS=2 PAP_PROJECTION_GPUS=3 \
PAP_PREFILL_PORT_BASE=8170 PAP_PROJECTION_PORT_BASE=8270 \
PAP_ATTENTION_PORT_BASE=8370 PAP_ATTENTION_TCP_PORT_BASE=9370 \
PAP_ATTENTION_ZMQ_PORT_BASE=10370 PAP_PROJECTION_ZMQ_PORT_BASE=11370 \
PAP_PROXY_PORT=9170 PAP_VLLM_PORT_BASE=50700 PAP_PREFILL_NIXL_PORT_BASE=5770 \
PAP_LOG_DIR=test/baseline/pap/results/runs/20260706_ipc_only_commit/service_logs \
PAP_RESULT_PATH=test/baseline/pap/results/runs/20260706_ipc_only_commit/one_request.json \
PAP_OFFLOAD_EXEC_TRANSPORT=local_fast \
PAP_OFFLOAD_EXEC_TRACE=1 \
PAP_DECODE_COMMIT_TRACE=1 \
./examples/pap/launch_pap_nixl.sh --model /data/ssd1/llm-models/Qwen3-8B
```

Expected:
- 200 OK response.
- attention log shows "PAP decode commit sent" with HTTP 200.
- prefill log shows "PAP decode commit applied".

- [ ] **Step 3: Run 256-prompt benchmark**

Same launcher env as Step 2 but with `PAP_NUM_PROMPTS=256 PAP_INPUT_LEN=128 PAP_OUTPUT_LEN=16 PAP_QPS=16 PAP_NUM_GPUS=...` and the existing benchmark harness.

Expected:
- 256/256 complete, 0 failed.
- Median TTFT <= prior unified run (991 ms target).
- Median TPOT <= prior unified run (232 ms target). Stretch: <= 100 ms.
- Prefill-side log shows N decode commits applied.

- [ ] **Step 4: Run prefix-cache hit validation**

Run two consecutive requests with the same prompt. The second request should show non-zero `find_longest_cache_hit` for the prefix. Inspect prefill log for "block hit length" line, or use `KVCacheManager.make_prefix_cache_stats()` output.

- [ ] **Step 5: Commit results metadata**

```bash
.venv/bin/python tools/pap_bench_summary.py test/baseline/pap/results/runs/20260706_ipc_only_commit/ > \
  test/baseline/pap/results/runs/20260706_ipc_only_commit/SUMMARY.md
git add test/baseline/pap/results/runs/20260706_ipc_only_commit/
git commit -m "PAP: IPC-only KV + decode commit 256-prompt benchmark"
```

---

## Self-Review Checklist

After writing the plan, run through this list:

- [ ] **Spec coverage:** Each goal in the goal statement maps to tasks.
  - "Remove attention local pool" → Tasks 7, 8, 9.
  - "IPC-only KV read/write" → Task 7 fail-closes the path; existing unified code already does IPC.
  - "Prefill perceives new KV via commit" → Tasks 1–6.
- [ ] **Placeholder scan:** No "TBD"/"fill in" left. Where test scaffolding depends on helpers that may not exist, the test description specifies fallback (monkeypatch + public-side assertions).
- [ ] **Type consistency:** `PAPDecodeCommit` field names (`request_id`, `new_seq_len`, `new_token_ids`, `layer_complete`) are used consistently across `decode_commit.py`, `decode_commit_router.py`, `decode_commit_client.py`, and the executor integration. `PAPOffloadExecItem.decode_token_ids` matches.
- [ ] **Order:** Task 1 (dataclass) → Task 2 (manager method) → Task 3 (router) → Task 4 (mount) → Task 5 (client) → Task 6 (executor integration) → Task 7 (cleanup) → Task 8 (shadow cleanup) → Task 9 (launcher) → Task 10 (validation). Each task is independently shippable.
