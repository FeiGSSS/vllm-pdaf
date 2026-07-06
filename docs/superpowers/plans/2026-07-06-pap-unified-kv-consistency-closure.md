# PAP Unified KV Consistency Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Prefill/Projection/Attention consistency loop for PAP IPC-only unified KV so Attention reads/writes Prefill-owned KV, Projection carries the right logical metadata, Prefill commits remote decode state, and leases are released deterministically.

**Architecture:** Prefill owns physical paged KV and scheduler metadata. Attention owns transient CUDA IPC views keyed by `(request_id, layer_name)` and writes decode KV directly into Prefill-owned pages. Projection owns decode execution and routes OFFLOAD_EXEC with logical metadata only: request id, layer, seq len, endpoint, and decode token id.

**Tech Stack:** vLLM V1 EngineCore utility calls, FastAPI routers, CUDA IPC descriptors, PAP NIXL mailbox transport, pytest.

---

## Current Target Protocol

```text
Prefill                         Attention                         Projection
   | prefill computes KV             |                                  |
   | pin lease + reserve capacity    |                                  |
   | OFFLOAD_KV descriptor --------> | import CUDA IPC unified state     |
   | <--------- ready/queued status  |                                  |
   | publish routing metadata ---------------------------------------> |
   |                                decode step OFFLOAD_EXEC(Q,K,V,t) |
   |                                <-------------------------------  |
   |                                append K/V into Prefill KV         |
   |                                run paged attention                |
   |                                commit(request, seq, token) -----> |
   | apply commit in EngineCore      |                                  |
   |                                O ------------------------------> |
   | request finished                | release local state               |
   | <---------------- lease release |                                  |
   | free/keep KV by Prefill policy  |                                  |
```

Key invariants:

- `lease` protects physical KV blocks from reuse.
- `commit` advances Prefill logical state: request tokens, computed length, and prefix-cache block hashes.
- `Attention descriptor discard` removes only Attention's local IPC/index state.
- `lease release` is the only signal that allows Prefill to free or reuse leased blocks.
- Projection never owns CUDA IPC KV handles or block ids. It only carries logical routing and decode metadata.

---

## Files Map

- Modify: `vllm/pap/data_plane.py`
  - Validate unified paged KV descriptors.
  - Serialize `decode_token_ids` through OFFLOAD_EXEC metadata.
- Modify: `vllm/pap/shadow_attention.py`
  - Make unified-KV lease pinning fail closed with useful logs.
- Modify: `vllm/v1/worker/gpu_model_runner.py`
  - Add scheduled decode `input_ids` to PAP forward context.
- Modify: `vllm/v1/worker/gpu/model_runner.py`
  - Mirror PAP forward-context token changes for the alternate model runner.
- Modify: `vllm/model_executor/models/qwen3.py`
  - Attach per-request decode token ids to OFFLOAD_EXEC descriptors and compact metadata templates.
  - Gate decode on actual Attention readiness, not only optimistic request membership.
- Modify: `examples/pap/pap_attention_executor.py`
  - Commit non-empty decode token ids.
  - Return/query descriptor readiness.
  - Release Prefill lease when Attention session is deleted.
- Modify: `vllm/v1/core/kv_cache_manager.py`
  - Validate commit token delta before mutating request state.
- Modify: `vllm/v1/engine/core.py`
  - Add EngineCore utility methods for PAP decode commit and lease release.
- Modify: `vllm/v1/engine/core_client.py`
  - Expose sync and async EngineCore client wrappers for those PAP utilities.
- Create: `vllm/pap/prefill_control_router.py`
  - FastAPI router for Prefill-side commit/release endpoints.
- Modify: `vllm/entrypoints/openai/api_server.py`
  - Mount the PAP Prefill control router when PAP unified KV is enabled.
- Create: `vllm/pap/lease_release_client.py`
  - Small HTTP client used by Attention to release Prefill leases.
- Modify: `vllm/pap/decode_commit_client.py`
  - Either make the blocking behavior explicit or switch to a bounded background queue.
- Modify: `tests/pap/test_pap_data_plane.py`
  - Descriptor validation and OFFLOAD_EXEC token metadata tests.
- Modify: `tests/pap/test_decode_commit.py`
  - Real `apply_decode_commit` validation tests and EngineCore utility tests.
- Modify: `tests/pap/test_pap_attention_executor.py`
  - Attention readiness, commit token propagation, and lease release tests.
- Modify: `tests/pap/test_pap_qwen3_tp_routing.py`
  - Projection metadata template includes token ids.
- Modify: `tests/pap/test_pap_launch_files.py`
  - Launcher exports the control endpoints consistently.

---

### Task 1: Fail Closed For Unified KV Export

**Files:**
- Modify: `tests/pap/test_pap_data_plane.py`
- Modify: `vllm/pap/data_plane.py`
- Modify: `vllm/pap/shadow_attention.py`

- [ ] **Step 1: Add descriptor validation tests**

Add `pytest` to imports in `tests/pap/test_pap_data_plane.py` and append:

```python
def _paged_ipc_handle() -> PAPCudaIPCTensorHandle:
    return PAPCudaIPCTensorHandle(
        dtype="float16",
        shape=(2, 4, 1, 8),
        ipc_handle={"GPU-test": ("storage", 1, 2, 3, 4, 5, 0)},
    )


def test_unified_paged_ipc_descriptor_requires_lease() -> None:
    with pytest.raises(ValueError, match="requires lease_id"):
        PAPOffloadKVPagedIPCDescriptor(
            request_id="req-1",
            layer_name="layer0",
            seq_len=4,
            block_ids=(0,),
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            kv_cache=_paged_ipc_handle(),
            unified_kv_mode=True,
            prefix_len=4,
            writable_start_token=4,
            writable_end_token=8,
        )


def test_unified_paged_ipc_descriptor_requires_capacity_for_writable_end() -> None:
    with pytest.raises(ValueError, match="lease_capacity_tokens"):
        PAPOffloadKVPagedIPCDescriptor(
            request_id="req-1",
            layer_name="layer0",
            seq_len=4,
            block_ids=(0, 1),
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            kv_cache=_paged_ipc_handle(),
            lease_id="lease-1",
            leased_block_ids=(0, 1),
            lease_capacity_tokens=4,
            unified_kv_mode=True,
            prefix_len=4,
            writable_start_token=4,
            writable_end_token=8,
        )


def test_unified_paged_ipc_descriptor_requires_blocks_covering_writable_end() -> None:
    with pytest.raises(ValueError, match="block_ids"):
        PAPOffloadKVPagedIPCDescriptor(
            request_id="req-1",
            layer_name="layer0",
            seq_len=4,
            block_ids=(0,),
            block_size=4,
            num_kv_heads=1,
            layout="NHD",
            kv_cache=_paged_ipc_handle(),
            lease_id="lease-1",
            leased_block_ids=(0,),
            lease_capacity_tokens=8,
            unified_kv_mode=True,
            prefix_len=4,
            writable_start_token=4,
            writable_end_token=8,
        )
```

- [ ] **Step 2: Verify tests fail before implementation**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py -v -k unified_paged_ipc_descriptor
```

Expected: failures because descriptors currently allow missing lease/capacity/block coverage.

- [ ] **Step 3: Enforce unified descriptor invariants**

In `PAPOffloadKVPagedIPCDescriptor.__post_init__`, after writable token normalization, add:

```python
            if self.lease_id is None:
                raise ValueError("unified_kv_mode requires lease_id")
            if self.leased_block_ids is None or not self.leased_block_ids:
                raise ValueError("unified_kv_mode requires leased_block_ids")
            if self.lease_capacity_tokens is None:
                raise ValueError("unified_kv_mode requires lease_capacity_tokens")
            if int(self.lease_capacity_tokens) < int(self.writable_end_token):
                raise ValueError(
                    "lease_capacity_tokens must cover writable_end_token"
                )
            required_blocks = (
                int(self.writable_end_token) + self.block_size - 1
            ) // self.block_size
            if len(self.block_ids) < required_blocks:
                raise ValueError("block_ids must cover writable_end_token")
```

- [ ] **Step 4: Make lease pin failure fail closed**

In `vllm/pap/shadow_attention.py`, replace the silent fallback in `import_prefill_paged_kv()` with:

```python
    except Exception as exc:
        logger.exception(
            "PAP unified KV lease pin failed request_id=%s layer=%s blocks=%d",
            request_id,
            layer_name,
            len(block_ids),
        )
        raise RuntimeError(
            f"PAP unified KV lease pin failed for request_id={request_id}"
        ) from exc
```

This keeps non-unified mode unchanged because the `try` body only pins when `unified_kv_mode` is true.

- [ ] **Step 5: Verify descriptor tests pass**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py -v -k unified_paged_ipc_descriptor
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm/pap/data_plane.py vllm/pap/shadow_attention.py tests/pap/test_pap_data_plane.py
git commit -m "fix(pap): fail closed for unified kv descriptors"
```

---

### Task 2: Carry Decode Token IDs Through OFFLOAD_EXEC

**Files:**
- Modify: `tests/pap/test_pap_data_plane.py`
- Modify: `tests/pap/test_pap_qwen3_tp_routing.py`
- Modify: `vllm/pap/data_plane.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Modify: `vllm/v1/worker/gpu/model_runner.py`
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `examples/pap/pap_attention_executor.py`

- [ ] **Step 1: Add data-plane token metadata roundtrip tests**

Append to `tests/pap/test_pap_data_plane.py`:

```python
def test_offload_exec_batch_metadata_roundtrips_decode_token_ids() -> None:
    descriptor = PAPOffloadExecBatchDescriptor(
        layer_name="layer0",
        items=(
            PAPOffloadExecDescriptor(
                "req-a", "layer0", 7, 0.125, decode_token_ids=(42,)
            ),
            PAPOffloadExecDescriptor(
                "req-b", "layer0", 8, 0.25, decode_token_ids=(99,)
            ),
        ),
    )

    metadata = _offload_exec_batch_descriptor_to_metadata(descriptor)
    restored = _offload_exec_batch_descriptor_from_metadata(metadata)

    assert metadata["v"] == 3
    assert metadata["t"] == [[42], [99]]
    assert restored.items[0].decode_token_ids == (42,)
    assert restored.items[1].decode_token_ids == (99,)


def test_offload_exec_batch_metadata_v2_remains_backward_compatible() -> None:
    metadata = {
        "v": 2,
        "l": "layer0",
        "r": ["req-a"],
        "s": [7],
        "a": [0.125],
    }

    restored = _offload_exec_batch_descriptor_from_metadata(metadata)

    assert restored.items[0].decode_token_ids == ()
```

- [ ] **Step 2: Verify metadata tests fail**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py -v -k decode_token_ids
```

Expected: the v3 metadata test fails because `t` is not serialized yet.

- [ ] **Step 3: Add compact metadata v3**

In `PAPOffloadExecBatchDescriptor.__post_init__`, normalize optional template key `t`:

```python
            if "t" in template:
                token_rows = tuple(
                    tuple(int(token_id) for token_id in row)
                    for row in template["t"]
                )
                normalized_template["t"] = token_rows
                lengths.add(len(token_rows))
```

In `_offload_exec_batch_descriptor_to_metadata()`, emit v3 when token ids exist:

```python
    token_rows = [list(item.decode_token_ids) for item in descriptor.items]
    has_tokens = any(token_rows)
    metadata = {
        "v": 3 if has_tokens else 2,
        "l": descriptor.layer_name,
        "r": [item.request_id for item in descriptor.items],
        "s": [int(item.step) for item in descriptor.items],
        "a": [float(item.scale) for item in descriptor.items],
    }
    if has_tokens:
        metadata["t"] = token_rows
    return metadata
```

In the template path, read `t` from `metadata_template` and emit `v: 3` plus `t` when present.

In `_offload_exec_batch_descriptor_from_metadata()`, accept both v2 and v3:

```python
        token_rows = metadata.get("t")
        if token_rows is None:
            token_rows = [()] * len(request_ids)
        if not (len(request_ids) == len(steps) == len(scales) == len(token_rows)):
            raise ValueError("compact PAP OFFLOAD_EXEC batch metadata length mismatch")
```

Then pass `decode_token_ids=tuple(int(t) for t in token_row)` when constructing each `PAPOffloadExecDescriptor`.

- [ ] **Step 4: Add input token ids to PAP forward context**

In both model runners, update `_pap_forward_context_kwargs()` to accept `input_ids: torch.Tensor | None` and include:

```python
            "pap_input_token_ids": (
                tuple(
                    int(token_id)
                    for token_id in input_ids.detach().reshape(-1).to(
                        device="cpu", dtype=torch.long
                    ).tolist()
                )
                if input_ids is not None
                else ()
            ),
```

Update the call sites to pass the local `input_ids` object.

- [ ] **Step 5: Attach decode token ids in Qwen3 OFFLOAD_EXEC descriptors**

In `vllm/model_executor/models/qwen3.py`, read once near `pap_positions`:

```python
        input_token_ids = tuple(
            int(token_id)
            for token_id in additional_kwargs.get("pap_input_token_ids") or ()
        )
```

When constructing each `PAPOffloadExecDescriptor`, add:

```python
                decode_token_ids=(
                    (input_token_ids[req_index],)
                    if req_index < len(input_token_ids)
                    else ()
                ),
```

When constructing compact route templates, add:

```python
                    "t": tuple(
                        (input_token_ids[req_index],)
                        if req_index < len(input_token_ids)
                        else ()
                        for req_index in req_indices
                    ),
```

- [ ] **Step 6: Verify Attention commits non-empty token ids**

Add a test in `tests/pap/test_pap_attention_executor.py` that monkeypatches `_get_commit_client()` to capture commits, calls the unified batch path with `PAPOffloadExecDescriptor(..., decode_token_ids=(42,))`, and asserts:

```python
assert commits == [("req-1", expected_new_seq_len, (42,))]
```

- [ ] **Step 7: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py tests/pap/test_pap_qwen3_tp_routing.py tests/pap/test_pap_attention_executor.py -v -k "decode_token_ids or offload_exec"
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add vllm/pap/data_plane.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu/model_runner.py vllm/model_executor/models/qwen3.py examples/pap/pap_attention_executor.py tests/pap/test_pap_data_plane.py tests/pap/test_pap_qwen3_tp_routing.py tests/pap/test_pap_attention_executor.py
git commit -m "feat(pap): propagate decode token ids to attention commits"
```

---

### Task 3: Apply Decode Commit In Real Prefill EngineCore

**Files:**
- Modify: `tests/pap/test_decode_commit.py`
- Modify: `vllm/v1/core/kv_cache_manager.py`
- Modify: `vllm/v1/engine/core.py`
- Modify: `vllm/v1/engine/core_client.py`
- Create: `vllm/pap/prefill_control_router.py`
- Modify: `vllm/entrypoints/openai/api_server.py`
- Modify: `vllm/pap/decode_commit_client.py`

- [ ] **Step 1: Replace the hanging TestClient coverage**

Keep `test_commit_endpoint_applies_to_manager` only if it is rewritten without `fastapi.testclient.TestClient`. Add a router unit test that calls the route endpoint through `httpx.AsyncClient` and `ASGITransport`, or delete it after EngineCore utility tests cover the same behavior.

Use this command before and after the change:

```bash
timeout 20s .venv/bin/python -m pytest tests/pap/test_decode_commit.py::test_commit_endpoint_applies_to_manager -vv -s
```

Expected before fix in this environment: timeout. Expected after replacement: no timeout.

- [ ] **Step 2: Add token delta validation test**

Append to `tests/pap/test_decode_commit.py`:

```python
def test_apply_decode_commit_rejects_token_delta_mismatch() -> None:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    request = Request(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        block_hasher=lambda req: [b"h"],
    )
    request.num_computed_tokens = 4

    manager = object.__new__(KVCacheManager)
    manager.enable_caching = False

    with pytest.raises(ValueError, match="new_token_ids"):
        manager.apply_decode_commit(request, new_seq_len=6, new_token_ids=[100])
```

- [ ] **Step 3: Validate commit deltas**

In `KVCacheManager.apply_decode_commit()`, before appending:

```python
        old_seq_len = int(request.num_computed_tokens)
        if new_seq_len <= old_seq_len:
            return
        delta = [int(t) for t in new_token_ids]
        expected_delta = int(new_seq_len) - old_seq_len
        if len(delta) != expected_delta:
            raise ValueError(
                "new_token_ids length must match new_seq_len delta: "
                f"expected {expected_delta}, got {len(delta)}"
            )
```

- [ ] **Step 4: Add EngineCore utility methods**

In `vllm/v1/engine/core.py`, add methods on `EngineCore`:

```python
    def pap_apply_decode_commit(
        self,
        request_id: str,
        new_seq_len: int,
        new_token_ids: Sequence[int],
    ) -> dict[str, Any]:
        request = self.scheduler.requests.get(str(request_id))
        if request is None:
            return {
                "request_id": str(request_id),
                "applied": False,
                "reason": "unknown_request",
            }
        old_seq_len = int(request.num_computed_tokens)
        self.scheduler.kv_cache_manager.apply_decode_commit(
            request=request,
            new_seq_len=int(new_seq_len),
            new_token_ids=tuple(int(t) for t in new_token_ids),
        )
        return {
            "request_id": str(request_id),
            "applied": True,
            "old_seq_len": old_seq_len,
            "new_seq_len": int(request.num_computed_tokens),
        }
```

Import `Sequence` and `Any` if not already available in this module.

- [ ] **Step 5: Expose EngineCore client wrappers**

In `vllm/v1/engine/core_client.py`, add abstract methods:

```python
    def pap_apply_decode_commit(
        self, request_id: str, new_seq_len: int, new_token_ids: Sequence[int]
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def pap_apply_decode_commit_async(
        self, request_id: str, new_seq_len: int, new_token_ids: Sequence[int]
    ) -> dict[str, Any]:
        raise NotImplementedError
```

Implement sync MP with:

```python
        return self.call_utility(
            "pap_apply_decode_commit",
            request_id,
            int(new_seq_len),
            tuple(int(t) for t in new_token_ids),
        )
```

Implement async MP with:

```python
        return await self.call_utility_async(
            "pap_apply_decode_commit",
            request_id,
            int(new_seq_len),
            tuple(int(t) for t in new_token_ids),
        )
```

For in-process client, forward directly to `self.engine_core.pap_apply_decode_commit(...)`.

- [ ] **Step 6: Create Prefill control router**

Create `vllm/pap/prefill_control_router.py`:

```python
from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


class DecodeCommitRequest(BaseModel):
    request_id: str
    new_seq_len: int
    new_token_ids: list[int]
    layer_complete: bool = True


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def build_prefill_control_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/pap/prefill/decode-commit")
    async def commit(req: DecodeCommitRequest, raw_request: Request) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        method = getattr(engine_client, "pap_apply_decode_commit_async", None)
        if method is not None:
            result = await method(req.request_id, req.new_seq_len, req.new_token_ids)
        else:
            result = await _maybe_await(
                engine_client.pap_apply_decode_commit(
                    req.request_id,
                    req.new_seq_len,
                    tuple(req.new_token_ids),
                )
            )
        if not result.get("applied", False):
            raise HTTPException(status_code=404, detail=result)
        return result

    return router
```

- [ ] **Step 7: Mount router in OpenAI API server**

In `vllm/entrypoints/openai/api_server.py`, import `os` if needed and include the router in `build_app()` or immediately after app creation:

```python
    if os.environ.get("PAP_UNIFIED_KV", "").lower() in {"1", "true", "yes", "on"}:
        from vllm.pap.prefill_control_router import build_prefill_control_router

        app.include_router(build_prefill_control_router())
```

- [ ] **Step 8: Run commit tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_decode_commit.py -v
```

Expected: all tests pass without TestClient hanging.

- [ ] **Step 9: Commit**

```bash
git add vllm/v1/core/kv_cache_manager.py vllm/v1/engine/core.py vllm/v1/engine/core_client.py vllm/pap/prefill_control_router.py vllm/entrypoints/openai/api_server.py vllm/pap/decode_commit_client.py tests/pap/test_decode_commit.py
git commit -m "feat(pap): apply decode commits through engine core"
```

---

### Task 4: Release Prefill Leases From Attention Cleanup

**Files:**
- Modify: `vllm/pap/prefill_control_router.py`
- Modify: `vllm/v1/engine/core.py`
- Modify: `vllm/v1/engine/core_client.py`
- Create: `vllm/pap/lease_release_client.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `tests/pap/test_pap_attention_executor.py`
- Modify: `tests/pap/test_decode_commit.py`

- [ ] **Step 1: Add EngineCore lease release utility**

In `vllm/v1/engine/core.py`, add:

```python
    def pap_release_kv_lease(
        self,
        request_id: str,
        lease_id: str,
    ) -> dict[str, Any]:
        from vllm.pap.kv_lease import pap_release_lease

        released = pap_release_lease(str(lease_id))
        return {
            "request_id": str(request_id),
            "lease_id": str(lease_id),
            "released": True,
            "block_count": len(released),
        }
```

- [ ] **Step 2: Expose client wrappers**

In `vllm/v1/engine/core_client.py`, mirror Task 3 with sync and async wrappers calling utility method `pap_release_kv_lease`.

- [ ] **Step 3: Add release endpoint**

Extend `vllm/pap/prefill_control_router.py`:

```python
class LeaseReleaseRequest(BaseModel):
    request_id: str
    lease_id: str


    @router.post("/v1/pap/prefill/lease-release")
    async def release_lease(
        req: LeaseReleaseRequest,
        raw_request: Request,
    ) -> dict[str, Any]:
        engine_client = raw_request.app.state.engine_client
        method = getattr(engine_client, "pap_release_kv_lease_async", None)
        if method is not None:
            return await method(req.request_id, req.lease_id)
        return await _maybe_await(
            engine_client.pap_release_kv_lease(req.request_id, req.lease_id)
        )
```

- [ ] **Step 4: Create Attention-side release client**

Create `vllm/pap/lease_release_client.py`:

```python
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


class LeaseReleaseClient:
    def __init__(
        self,
        endpoint: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("PAP_LEASE_RELEASE_ENDPOINT")
        self.timeout_s = (
            float(timeout_s)
            if timeout_s is not None
            else float(os.environ.get("PAP_LEASE_RELEASE_TIMEOUT", "0.2"))
        )

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint)

    def release(self, *, request_id: str, lease_id: str) -> None:
        if not self.enabled:
            return
        try:
            response = httpx.post(
                str(self.endpoint),
                json={"request_id": request_id, "lease_id": lease_id},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except Exception:
            logger.warning(
                "PAP lease release failed request_id=%s lease_id=%s",
                request_id,
                lease_id,
                exc_info=True,
            )
```

- [ ] **Step 5: Call release client outside registry lock**

In `examples/pap/pap_attention_executor.py`, change `release_session()` so it removes local state under the registry lock, captures `(request_id, lease_id)`, then invokes `LeaseReleaseClient.release()` after the lock is released.

The expected user-visible behavior:

```text
DELETE /v1/pap/attention/sessions/{request_id}
  -> Attention drops local descriptor/unified state
  -> Attention POSTs Prefill /v1/pap/prefill/lease-release
  -> Prefill releases or frees deferred blocks by lease id
```

- [ ] **Step 6: Add release test**

In `tests/pap/test_pap_attention_executor.py`, create a session with `_session_lease_ids["req-1"] = "lease-1"`, monkeypatch the release client, call `registry.release_session("req-1")`, and assert:

```python
assert released == [("req-1", "lease-1")]
assert registry.get_session("req-1") is None
```

- [ ] **Step 7: Run release tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py tests/pap/test_decode_commit.py -v -k "release or lease"
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add vllm/pap/prefill_control_router.py vllm/v1/engine/core.py vllm/v1/engine/core_client.py vllm/pap/lease_release_client.py examples/pap/pap_attention_executor.py tests/pap/test_pap_attention_executor.py tests/pap/test_decode_commit.py
git commit -m "feat(pap): release prefill kv leases from attention cleanup"
```

---

### Task 5: Make Attention KV Readiness An Explicit Barrier

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `vllm/pap/shadow_attention.py`
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Modify: `vllm/v1/worker/gpu/model_runner.py`
- Modify: `tests/pap/test_pap_attention_executor.py`
- Modify: `tests/pap/test_pap_qwen3_tp_routing.py`

- [ ] **Step 1: Add readiness endpoint**

In `examples/pap/pap_attention_executor.py`, add:

```python
    @app.get("/v1/pap/attention/sessions/{request_id}/prefill-readiness")
    async def get_prefill_readiness(request_id: str) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "layers": [
                readiness.__dict__
                for readiness in registry.get_prefill_readiness(request_id)
            ],
        }
```

Add `PAPAttentionRegistry.get_prefill_readiness(request_id)` returning copies of all readiness records for the resolved session id.

- [ ] **Step 2: Return ready status from OFFLOAD_KV import**

For synchronous imports in `handle_binary_request`, ensure the response metadata includes:

```python
{
    "request_id": descriptor.request_id,
    "layer_name": descriptor.layer_name,
    "seq_len": descriptor.seq_len,
    "status": "ready",
    "unified_kv_mode": descriptor.unified_kv_mode,
}
```

For async imports, keep `status: "queued"` but make Projection treat that as not ready.

- [ ] **Step 3: Stop marking KV installed before readiness**

In both model runners, replace the optimistic behavior:

```python
        if kv_transfer_params.get("pap_attention_kv_installed"):
            self.pap_attention_kv_installed_by_req_id.add(req_id)
```

with a state that records only the prefill export request. Move `pap_attention_kv_installed_by_req_id.add(req_id)` to the code path that observes all required layer readiness.

- [ ] **Step 4: Gate Qwen3 decode on readiness**

In `vllm/model_executor/models/qwen3.py`, keep the current fail-fast behavior:

```python
            if prefix_len > 0 and request_id not in attention_kv_installed_by_request:
                if not prefill_kv_handle:
                    raise RuntimeError("PAP missing local prefill KV handle")
                raise RuntimeError("PAP attention KV is not installed")
```

Add tests that the error is raised until readiness is confirmed.

- [ ] **Step 5: Add readiness tests**

In `tests/pap/test_pap_attention_executor.py`, assert the transition:

```python
readiness = registry.prefill_layer_readiness(
    request_id="req-1",
    layer_name="layer0",
)
assert readiness is None

registry.import_prefill_paged_kv(... unified_kv_mode=True, lease_id="lease-1", ...)
readiness = registry.prefill_layer_readiness(
    request_id="req-1",
    layer_name="layer0",
)
assert readiness.descriptor_received
assert readiness.descriptor_opened
assert readiness.ready
```

- [ ] **Step 6: Run readiness tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py tests/pap/test_pap_qwen3_tp_routing.py -v -k "readiness or installed"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add examples/pap/pap_attention_executor.py vllm/pap/shadow_attention.py vllm/model_executor/models/qwen3.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu/model_runner.py tests/pap/test_pap_attention_executor.py tests/pap/test_pap_qwen3_tp_routing.py
git commit -m "fix(pap): gate projection decode on attention kv readiness"
```

---

### Task 6: Bound Commit Client Latency

**Files:**
- Modify: `vllm/pap/decode_commit_client.py`
- Modify: `tests/pap/test_decode_commit.py`
- Modify: `examples/pap/pap_attention_executor.py`

- [ ] **Step 1: Decide behavior**

Use a bounded background queue for decode commits:

- `commit()` enqueues and returns immediately.
- A daemon worker posts commits with the existing timeout.
- If the queue is full, log a warning and drop only when `PAP_DECODE_COMMIT_FAIL_CLOSED` is false.
- When `PAP_DECODE_COMMIT_FAIL_CLOSED=1`, raise `RuntimeError` on enqueue failure.

- [ ] **Step 2: Add nonblocking client test**

In `tests/pap/test_decode_commit.py`, monkeypatch `httpx.post` to block on an event and assert `client.commit(...)` returns before the event is released.

- [ ] **Step 3: Implement bounded queue**

In `DecodeCommitClient.__init__`, initialize:

```python
self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1024)
self._worker = threading.Thread(
    target=self._run_worker,
    name="pap-decode-commit-client",
    daemon=True,
)
self._worker.start()
```

Keep the existing synchronous POST code inside `_post_commit(payload)`.

- [ ] **Step 4: Run commit client tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_decode_commit.py -v -k commit_client
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm/pap/decode_commit_client.py tests/pap/test_decode_commit.py examples/pap/pap_attention_executor.py
git commit -m "perf(pap): make decode commit notifications nonblocking"
```

---

### Task 7: Single-Request Smoke Before Benchmark

**Files:**
- Modify: `examples/pap/launch_pap_nixl.sh`
- Modify: `tests/pap/test_pap_launch_files.py`
- Optional modify: `tools/pap_remote_attention_diagnostics.py`

- [ ] **Step 1: Export release endpoint in launcher**

In `examples/pap/launch_pap_nixl.sh`, add:

```bash
PAP_LEASE_RELEASE_ENDPOINT="${PAP_LEASE_RELEASE_ENDPOINT:-http://127.0.0.1:${PREFILL_PORT_BASE}/v1/pap/prefill/lease-release}"
export PAP_LEASE_RELEASE_ENDPOINT
```

Keep `PAP_DECODE_COMMIT_ENDPOINT` exported.

- [ ] **Step 2: Add launch file test**

In `tests/pap/test_pap_launch_files.py`, assert the script contains:

```python
assert "PAP_DECODE_COMMIT_ENDPOINT" in text
assert "PAP_LEASE_RELEASE_ENDPOINT" in text
assert "PAP_UNIFIED_KV=\"${PAP_UNIFIED_KV:-1}\"" in text
```

- [ ] **Step 3: Run launcher tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_launch_files.py -v
```

Expected: pass.

- [ ] **Step 4: Run local PAP unit suite**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_data_plane.py tests/pap/test_decode_commit.py tests/pap/test_pap_attention_executor.py tests/pap/test_pap_qwen3_prefill_kv.py tests/pap/test_pap_qwen3_tp_routing.py tests/pap/test_pap_launch_files.py -v
```

Expected: pass.

- [ ] **Step 5: Run one-request service smoke**

Start the PAP launcher in the normal local setup, then issue one prompt with output length 1. Verify logs contain these events in order:

```text
PAP KV lease pin request_id=...
PAP unified KV state stored request_id=... layer=...
PAP attention KV is ready request_id=...
PAP unified KV append ...
PAP decode commit request_id=... new_seq_len=... new_token_ids=[...]
pap_apply_decode_commit applied=True
PAP lease release request_id=... lease_id=...
```

Failure criteria:

- `PAP unified KV state missing` appears.
- Commit payload has `new_token_ids=[]`.
- Projection starts OFFLOAD_EXEC before the Attention readiness event.
- Request finish occurs without Prefill lease release or TTL fallback.

- [ ] **Step 6: Commit**

```bash
git add examples/pap/launch_pap_nixl.sh tests/pap/test_pap_launch_files.py tools/pap_remote_attention_diagnostics.py
git commit -m "test(pap): add unified kv smoke readiness checks"
```

---

### Task 8: Benchmark Only After Correctness Invariants Hold

**Files:**
- Modify: benchmark notes or result metadata under `test/baseline/pap/results/` only after successful smoke.

- [ ] **Step 1: Freeze benchmark preconditions**

Before running the 256-prompt benchmark, record:

```text
PAP_UNIFIED_KV=1
PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS=32
PAP_DECODE_COMMIT_ENDPOINT=http://127.0.0.1:${PREFILL_PORT_BASE}/v1/pap/prefill/decode-commit
PAP_LEASE_RELEASE_ENDPOINT=http://127.0.0.1:${PREFILL_PORT_BASE}/v1/pap/prefill/lease-release
PAP_OFFLOAD_EXEC_TRANSPORT=nixl_mailbox
PAP_OFFLOAD_KV_TRANSPORT=cuda_ipc
```

- [ ] **Step 2: Run benchmark**

Use the existing PAP benchmark command for the project. Do not change workload shape during this run.

- [ ] **Step 3: Inspect correctness logs before performance numbers**

For the benchmark log set, verify:

```bash
rg -n "PAP unified KV state missing|new_token_ids=\\[\\]|lease release failed|decode commit failed" <log-dir>
```

Expected: no matches.

- [ ] **Step 4: Compare latency/throughput**

Only after Step 3 passes, compare:

- TTFT
- TPOT
- end-to-end latency
- request success count
- GPU memory footprint on Prefill/Projection/Attention

- [ ] **Step 5: Commit results metadata**

```bash
git add test/baseline/pap/results
git commit -m "bench(pap): record unified kv ipc-only smoke results"
```

---

## Recommended Execution Order

1. Task 1: fail closed and descriptor validation.
2. Task 2: decode token id propagation.
3. Task 3: real EngineCore commit endpoint.
4. Task 4: explicit lease release.
5. Task 5: readiness barrier.
6. Task 6: nonblocking commit client.
7. Task 7: single-request smoke.
8. Task 8: benchmark.

Do not run the 256-prompt benchmark until Tasks 1-7 pass. The benchmark is not informative while commits can be empty, leases can remain unreleased, or Projection can race ahead of Attention descriptor import.

## Self-Review

- Spec coverage: The plan covers Prefill descriptor timing, Projection metadata, Attention lookup, commit, descriptor discard, and lease release.
- Placeholder scan: No task contains TBD-style placeholders. Optional diagnostics are explicitly marked optional and not required for correctness.
- Type consistency: The same names are used across tasks: `decode_token_ids`, `pap_input_token_ids`, `pap_apply_decode_commit`, `pap_release_kv_lease`, `PAP_LEASE_RELEASE_ENDPOINT`.
