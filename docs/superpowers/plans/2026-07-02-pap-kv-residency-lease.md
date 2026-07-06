# PAP KV Residency Lease Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix PAP 1PA1P `local_fast` TTFT and OOM by replacing cumulative synchronous Prefill-KV duplication with lifecycle-managed KV residency.

**Architecture:** Implement in four dependent stages. Stage A makes the current local paged KV pool bounded by live sessions via release + block recycling. Stage B stops copying prefix KV into a second Attention-local pool where possible by keeping Prefill-owned paged KV descriptors resident and using local blocks only for decode extension. Stage C moves Prefill-KV import/install off the Prefill HTTP response critical path or makes it lazy/async with explicit readiness fences. Stage D restores PAP prefill capacity knobs to PD-comparable values and validates TTFT/TPOT/OOM with fresh experiments.

**Tech Stack:** Python, PyTorch CUDA tensors, CUDA IPC descriptors, vLLM V1 request/attention metadata, existing PAP proxy/attention executor/local_fast transport, repo-local `.venv/bin/python` and `.venv/bin/vllm` only.

---

## Global Constraints

- Do not add unit tests. The user explicitly requested no unit tests for this early-development phase.
- Do not commit unless the user explicitly asks.
- Use only `/home/fei/research/PD/vllm-pap/.venv/bin/python` and `/home/fei/research/PD/vllm-pap/.venv/bin/vllm` for Python/vLLM commands.
- Always set `VLLM_USE_FLASHINFER_SAMPLER=0` for benchmarks.
- Preserve the latest TPOT path: paged FlashAttention, native cache append, direct QKV send, `local_fast` transport.
- Validate each stage with source parsing and the smallest relevant PAP experiment, not broad test suites.

---

### Task A: Session Release + Local Paged-KV Block Recycling

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `examples/pap/multi_pap_proxy_server.py`
- Modify: `examples/pap/pap_proxy_server.py` if the single-proxy path still matters

**Target behavior:** Attention local paged KV memory grows with live concurrency, not cumulative completed requests. On request completion, cancellation, or projection error, proxy calls Attention `DELETE /v1/pap/attention/sessions/{request_id}`. Attention returns all local paged blocks for that session to each layer pool free list.

- [ ] Add a free-list field to `PAPLocalPagedKVPool`.
  - Expected shape: `free_blocks: list[int]` or equivalent.
  - Keep `next_block` as high-water mark for new allocation.

- [ ] Change `_allocate_local_paged_blocks_locked()` to reuse free blocks before bumping `next_block`.
  - Preserve current behavior when `free_blocks` is empty.
  - Ensure returned block ids are unique within the allocation.

- [ ] Add `_release_local_paged_blocks_locked(session_request_id: str)`.
  - Iterate `self._local_paged_kv.get(session_request_id, {})`.
  - For each `PAPLocalPagedAttentionState`, append its `block_ids` back to `state.pool.free_blocks` exactly once.
  - Then remove `self._local_paged_kv[session_request_id]`.
  - Do not shrink `pool.kv_cache` during request handling.

- [ ] Call `_release_local_paged_blocks_locked()` from `release_session()` before dropping session dictionaries.
  - Keep existing metadata cleanup.
  - Keep `self._attention_sessions.free_session(request_id)`.

- [ ] Add INFO-level diagnostic logs gated by `PAP_ATTENTION_POOL_PROFILE=1`.
  - On grow: layer, old capacity, new capacity, next_block, free_count.
  - On release: request_id, layers released, blocks released, free_count summary.

- [ ] Add proxy cleanup in `multi_pap_proxy_server.py`.
  - Add helper to call each selected Attention endpoint `DELETE /v1/pap/attention/sessions/{request_id}`.
  - Invoke it in `finally` after projection response/stream completes or errors.
  - For streaming responses, wrap the generator so cleanup happens after stream exhaustion or client disconnect.

- [ ] Mirror cleanup in `pap_proxy_server.py` if still used by current launch paths.

- [ ] Validate with static checks.
  - Run: `.venv/bin/python -m py_compile examples/pap/pap_attention_executor.py examples/pap/multi_pap_proxy_server.py examples/pap/pap_proxy_server.py`
  - Expected: exit 0.

- [ ] Validate with a memory-profile PAP run.
  - Run a 1PA1P `local_fast` short benchmark with `PAP_ATTENTION_POOL_PROFILE=1`, i128/o16/qps16/c64, preferably 128 or 256 prompts if safe.
  - Expected: no OOM; release logs appear; pool grows only to a live-concurrency-sized plateau and reuses free blocks.
  - Report TTFT/TPOT and whether TPOT changed significantly versus `20260702_220019_pap_local_fast_1pa1p_64req`.

---

### Task B: Descriptor-Resident Prefix KV / Avoid Second Prefix Copy

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `vllm/pap/remote_attention.py` if paged FlashAttention helpers require descriptor composition
- Modify: `vllm/model_executor/models/qwen3.py` only if the Projection side must pass extra descriptor metadata

**Target behavior:** Prefill prefix KV is not copied into Attention local paged pool. Attention stores a descriptor-backed prefix state pointing at Prefill-owned paged KV via CUDA IPC. Local paged pool stores only decode extension blocks.

- [ ] Introduce a prefix state representation in `pap_attention_executor.py`.
  - It should preserve `kv_cache`, remote `block_ids`, `seq_len`, `block_size`, `num_kv_heads`, and layout.
  - It should be reference-counted or session-owned so release closes/drops descriptors.

- [ ] Change `import_prefill_paged_kv()` to store descriptor-resident prefix state without `_install_local_paged_prefill_locked()` by default.
  - Keep a temporary env fallback `PAP_ATTENTION_COPY_PREFIX_KV=1` for comparison only if needed.
  - Default should be descriptor-resident prefix.

- [ ] Update local paged attention state assembly for decode.
  - For first decode token, combine remote prefix blocks with local decode extension blocks.
  - If current FlashAttention call cannot consume two block tables/caches, implement the minimal bridge that uses remote prefix cache for prefix and local pool for appended decode tokens without duplicating all prefix blocks.
  - If the backend hard-requires a single kv_cache tensor, stop and report the exact API blocker rather than reintroducing full prefix copy.

- [ ] Ensure `release_session()` drops descriptor-resident prefix state and returns only local decode blocks to free lists.

- [ ] Validate with static checks.
  - Run: `.venv/bin/python -m py_compile examples/pap/pap_attention_executor.py vllm/pap/remote_attention.py vllm/model_executor/models/qwen3.py`
  - Expected: exit 0.

- [ ] Validate correctness with a short PAP smoke run.
  - Use 1PA1P `local_fast`, i128/o16, small prompt count.
  - Expected: requests complete without shape/seq_len mismatch.
  - Report TTFT/TPOT and GPU0 memory versus Stage A.

---

### Task C: Async/Lazy Prefill KV Readiness Outside TTFT Critical Path

**Files:**
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `vllm/pap/shadow_attention.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `examples/pap/multi_pap_proxy_server.py`

**Target behavior:** The Prefill HTTP response is not blocked by all 36 layer imports/installations. Prefill publishes descriptor metadata quickly; Attention imports/opens descriptors asynchronously or lazily, and Projection/Attention waits only at the first decode layer if readiness is not complete.

- [ ] Add a readiness state per session/layer in Attention registry.
  - States: descriptor_received, descriptor_opened, ready, failed.
  - Expose a lightweight status or wait primitive used by decode path.

- [ ] Change Prefill-side `qwen3.py` import path to avoid synchronous per-layer TCP roundtrip when `PAP_PREFILL_KV_ASYNC=1`.
  - Publish descriptor metadata and return without waiting for full local install.
  - Keep default off until validated, then decide whether to flip.

- [ ] In Attention executor, move descriptor open/install into a background worker or lazy first-use path.
  - Background worker should preserve per-session/layer errors.
  - Lazy path should block only the layer currently needed by decode.

- [ ] Add profile logs gated by `PAP_PREFILL_IPC_PROFILE=1`.
  - Prefill side: descriptor build, publish, response wait.
  - Attention side: queue delay, descriptor open, ready time.
  - Proxy: prefill_ms and first projection chunk/response latency.

- [ ] Validate with a short benchmark.
  - Compare Stage B sync mode vs Stage C async/lazy mode on i128/o16/qps16/c64.
  - Expected: TTFT median and p75 drop; TPOT does not significantly regress.

---

### Task D: Restore Capacity Knobs + Final 1:1 Benchmark Matrix

**Files:**
- Modify: `examples/pap/launch_pap_nixl.sh`
- Modify: benchmark invocation scripts only if required
- No core model changes expected unless Stage C requires an env default flip

**Target behavior:** PAP 1PA1P can run the same comparable benchmark shape as PD without OOM, and TTFT improves without significant TPOT regression.

- [ ] Restore PAP prefill capacity knobs toward PD-comparable values.
  - `PAP_PREFILL_GPU_MEMORY_UTILIZATION`: move from 0.65 back toward 0.8 after Stage A/B memory fixes.
  - `PAP_MAX_NUM_BATCHED_TOKENS`: move from 4096 toward 8192.
  - Keep `PAP_MAX_MODEL_LEN=512` for i128/o16 microbench unless comparing long contexts.

- [ ] Run fresh PD 1P1D baseline.
  - i128/o16/qps16/c64, 256 prompts, GPUs 2/3 if available.
  - Save under `test/baseline/pap/results/runs/`.

- [ ] Run fresh PAP 1PA1P `local_fast`.
  - i128/o16/qps16/c64, 256 prompts, GPUs 0/1 if available.
  - Save under `test/baseline/pap/results/runs/`.
  - Enable relevant profile flags only when they do not materially perturb timing, or run profile and clean timing separately.

- [ ] Produce final comparison table.
  - completed/failed
  - median/mean/p99 TTFT
  - median/mean/p99 TPOT
  - request throughput
  - output throughput
  - total token throughput
  - GPU0 memory evidence that pool plateaued/recycled

- [ ] Decide default flips.
  - Stage A release/recycle should be default-on if stable.
  - Stage B descriptor-resident prefix should be default-on only if correctness and TPOT are acceptable.
  - Stage C async/lazy should be default-on only if TTFT improves and TPOT does not significantly regress.

---

## Review Gates

After each task:

1. Implementation subagent reports status, files changed, commands run, and artifact paths.
2. Controller verifies actual diff and runs/reads the stated validation output.
3. Codex review checks root-cause fit, correctness risks, and simplification opportunities.
4. A fresh code-reviewer subagent checks code quality.
5. Only then proceed to the next stage.

## Success Criteria

- PAP 256-prompt i128/o16/qps16/c64 no longer OOMs.
- Attention local paged pool memory is bounded by live concurrency rather than cumulative requests.
- PAP median TTFT improves materially from 363.79 ms clean 64req baseline.
- PAP median TPOT does not significantly regress from 45.02 ms clean 64req baseline.
- All changed Python files parse successfully.
