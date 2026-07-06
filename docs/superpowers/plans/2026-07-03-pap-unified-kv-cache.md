# PAP Unified KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor PAP from split Prefill-owned prefix KV plus Attention-local suffix KV into a single Prefill-owned leased paged KV cache used by both Prefill and Attention, reducing KV copies and making TTFT/OOM improvements measurable.

**Architecture:** Implement a lease-first migration. First add Prefill-side KV block lease/pin semantics without changing compute, then add preallocated writable decode capacity to descriptors, then make Attention append decode K/V into Prefill-owned IPC-opened KV blocks, and finally simplify Attention compute to one paged KV source. Each stage has a static validation gate and a small runtime smoke before any larger benchmark.

**Tech Stack:** Python, PyTorch CUDA IPC, vLLM V1 scheduler/KVCacheManager/BlockPool, PAP proxy/Attention executor/local_fast transport, repo-local `.venv/bin/python` and `.venv/bin/vllm` only.

---

## Global Constraints

- Do not add unit tests. This project is in early-development mode; use source parsing, static validation, focused smoke runs, and benchmark artifacts instead.
- Do not commit unless explicitly requested.
- Use only `/home/fei/research/PD/vllm-pap/.venv/bin/python` and `/home/fei/research/PD/vllm-pap/.venv/bin/vllm` for Python/vLLM commands.
- Always set `VLLM_USE_FLASHINFER_SAMPLER=0` for benchmarks.
- Preserve current fast TPOT path unless the task explicitly replaces it: paged FlashAttention, native cache append where applicable, direct QKV send, `local_fast` transport.
- Do not run broad test suites. Each stage validates with `py_compile`/`bash -n`, one-request smoke, then a small PAP benchmark only after the one-request smoke passes.
- If a stage hits a backend/API blocker, stop that stage, record the blocker with file:line evidence, and do not paper over it with a hidden prefix copy.

---

## Current Source Facts To Preserve

- Prefill exports its vLLM-owned paged KV through CUDA IPC:
  - `vllm/model_executor/models/qwen3.py:_maybe_import_pap_prefill_kv_to_attention()` reads `attn_metadata.block_table`, `attn_metadata.seq_lens`, and `self.attn.kv_cache`.
  - `vllm/pap/shadow_attention.py:import_prefill_paged_kv()` builds `PAPOffloadKVPagedIPCDescriptor`.
  - `vllm/pap/data_plane.py:PAPOffloadKVPagedIPCDescriptor` carries `request_id`, `layer_name`, `seq_len`, `block_ids`, `block_size`, `num_kv_heads`, `layout`, and CUDA IPC tensor handle.
- Attention currently has split state:
  - Prefix descriptor view: `examples/pap/pap_attention_executor.py:PAPPrefillPagedKV`.
  - Attention-local pool: `PAPLocalPagedKVPool` and `_local_paged_kv_pools`.
  - Decode suffix append currently writes to Attention-local KV in `append_decode_kv_tensor_batch_for_local_paged_attention()`.
- Prefill-side vLLM owns block lifecycle:
  - Scheduler finish path frees request blocks through `kv_cache_manager.free(request)`.
  - `SingleTypeKVCacheManager.req_to_blocks` maps request IDs to blocks and `free()` returns blocks to the block pool.
- Main correctness gap for unified KV: no Prefill-side remote lease/pin primitive currently protects exported blocks after Prefill scheduler considers a request finished.

---

## Version and Experiment Management

For each implementation stage:

- Record the git diff scope before and after the stage:
  - `git status --short`
  - `git diff --stat -- <stage files>`
- Do not commit automatically. If the user requests a commit, create one commit per completed stage with a message of the form:
  - `PAP: add KV lease scaffolding`
  - `PAP: add leased decode capacity descriptors`
  - `PAP: append Attention decode KV into leased Prefill blocks`
  - `PAP: use single-source Prefill KV for Attention FA`
- Store runtime artifacts under `test/baseline/pap/results/runs/YYYYMMDD_<stage>_<shape>/`.
- Each run directory should contain:
  - benchmark JSON from `.venv/bin/vllm bench serve` when applicable;
  - `service_logs/`;
  - `run_metadata.json` with at least: stage, git short hash if committed or `dirty`, model, topology, transport, GPUs, input/output len, qps, max concurrency, num prompts, env flags, and known limitations;
  - optional profile output if enabled.
- Summaries should be generated with `tools/pap_bench_summary.py` and must include model, TP size, dtype, prompt count, warmups, completed/failed, TTFT, TPOT, throughput, observed max concurrency, artifact path, and server-side batch evidence note.
- Update `docs/design/pap-pd-comparison-methodology-20260701.md` only after a stage has at least one successful smoke or benchmark artifact. Mark failures as observations/blockers, not conclusions.

---

## Stage 0: Baseline Snapshot and Guardrails

**Purpose:** Freeze the current state before touching allocator/lifecycle code.

**Files:**
- Inspect: `examples/pap/pap_attention_executor.py`
- Inspect: `vllm/model_executor/models/qwen3.py`
- Inspect: `vllm/pap/shadow_attention.py`
- Inspect: `vllm/pap/data_plane.py`
- Inspect: `vllm/v1/core/sched/scheduler.py`
- Inspect: `vllm/v1/core/kv_cache_manager.py`
- Inspect: `vllm/v1/core/single_type_kv_cache_manager.py`
- Inspect: `vllm/v1/core/block_pool.py`
- Inspect: `vllm/v1/worker/gpu_model_runner.py`

- [ ] Record working-tree state.
  - Run: `git status --short`
  - Save notable uncommitted files in the stage notes.

- [ ] Capture current PAP failure baseline.
  - Use the latest Stage B one-request failure artifact if present.
  - Expected current blocker: descriptor-resident prefix runtime fails before useful benchmark because compute path is still split-cache/prefix-only sensitive.

- [ ] Confirm static validity before new edits.
  - Run: `.venv/bin/python -m py_compile examples/pap/pap_attention_executor.py vllm/model_executor/models/qwen3.py vllm/pap/shadow_attention.py vllm/pap/data_plane.py`
  - Expected: exit 0.

---

## Stage 1: Prefill-Side KV Lease / Pin Scaffolding

**Purpose:** Make exported Prefill KV blocks safe to read after Prefill-side request completion by preventing premature block-pool reuse.

**Files:**
- Modify: `vllm/v1/core/block_pool.py`
- Modify: `vllm/v1/core/single_type_kv_cache_manager.py`
- Modify: `vllm/v1/core/kv_cache_manager.py`
- Modify: `vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/pap/data_plane.py`
- Modify: `vllm/pap/shadow_attention.py`
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `examples/pap/multi_pap_proxy_server.py`

**Design:** Add PAP-only remote lease state keyed by `lease_id`/`request_id`. Exporting paged KV creates or refreshes a lease over the request's Prefill block IDs. Scheduler finish detaches request scheduling state but defers returning leased blocks to the block pool. Proxy/Attention session release sends a release signal so Prefill can unlease and free.

- [ ] Add a PAP lease metadata type.
  - Extend `PAPOffloadKVPagedIPCDescriptor` with optional fields:
    - `lease_id: str | None`
    - `leased_block_ids: tuple[int, ...] | None`
    - `lease_seq_len: int | None`
    - `lease_capacity_tokens: int | None`
  - Preserve backward compatibility by making fields optional in `from_dict()`.

- [ ] Add Prefill-side lease bookkeeping in KV manager/coordinator layer.
  - Track request/block lease counts without affecting non-PAP paths.
  - Minimum API shape:
    - `pap_pin_blocks(request_id: str, block_ids: Sequence[int], lease_id: str) -> None`
    - `pap_release_lease(lease_id: str) -> list[int]`
    - `pap_has_active_lease(request_id: str) -> bool`
  - Lease state must be guarded by the same scheduler/KV-manager thread assumptions as normal block allocation.

- [ ] Change request finish behavior to defer leased blocks.
  - In scheduler free path, if PAP lease exists, do not return those blocks to the block pool.
  - Still remove the request from active scheduling queues so it no longer runs locally.
  - Add diagnostic logs gated by `PAP_KV_LEASE_PROFILE=1`:
    - request_id, lease_id, block count, defer/free action.

- [ ] Create lease during paged KV export.
  - In `qwen3.py` or `shadow_attention.py`, after deriving `block_ids`, request a lease before sending descriptor.
  - Descriptor should include `lease_id` and leased block list.
  - If lease creation fails, fail closed and keep existing copied/local path disabled rather than exposing unsafe IPC.

- [ ] Add release path from proxy/Attention to Prefill.
  - Extend existing Attention session DELETE cleanup flow to also notify Prefill lease release.
  - Prefer a small Prefill-side HTTP endpoint or existing proxy-to-Prefill control path.
  - Release should be idempotent.

- [ ] Add lease leak guard.
  - Add TTL/env fallback such as `PAP_KV_LEASE_TTL_SECONDS` for crash cleanup.
  - Log expired leases but do not silently free active leases during first implementation unless clearly safe.

- [ ] Validate statically.
  - Run: `.venv/bin/python -m py_compile vllm/v1/core/block_pool.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/sched/scheduler.py vllm/pap/data_plane.py vllm/pap/shadow_attention.py examples/pap/pap_attention_executor.py examples/pap/multi_pap_proxy_server.py`
  - Expected: exit 0.

- [ ] Validate with a lease-only one-request smoke.
  - Keep Attention compute on the known-working copied/local fallback if needed.
  - Enable `PAP_KV_LEASE_PROFILE=1`.
  - Expected: lease create log before descriptor export; request completion does not free leased blocks until DELETE/release; release log appears; no block reuse before release.

- [ ] Record artifacts.
  - Save logs under `test/baseline/pap/results/runs/YYYYMMDD_stage1_lease_smoke/`.
  - Add `run_metadata.json` with stage, env flags, and result.

**Review gate:** Independent code review must confirm non-PAP free path is unchanged and leased blocks cannot be returned to the free queue before release.

---

## Stage 2: Leased Decode Capacity Descriptor

**Purpose:** Avoid remote dynamic allocation in the MVP by preallocating or reserving the full decode capacity needed for a request, then exposing a single complete block list to Attention.

**Files:**
- Modify: `vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/v1/core/kv_cache_manager.py`
- Modify: `vllm/v1/core/single_type_kv_cache_manager.py`
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `vllm/pap/data_plane.py`
- Modify: `vllm/pap/shadow_attention.py`
- Modify: `examples/pap/pap_attention_executor.py`

**Design:** In PAP unified-KV mode, Prefill leases enough blocks for `prefix_len + planned_decode_tokens` before descriptor export. Attention receives `writable_start`, `writable_end`, and the full `leased_block_ids`; it can write suffix K/V only inside that range.

- [ ] Add env gate `PAP_UNIFIED_KV=1`.
  - Default off until Stage 4 smoke passes.
  - When off, existing split/local behavior remains available.

- [ ] Add capacity fields to descriptor.
  - Required in unified mode:
    - `prefix_len`
    - `leased_capacity_tokens`
    - `writable_start_token`
    - `writable_end_token`
    - `leased_block_ids`
  - Validate `prefix_len <= writable_start_token <= writable_end_token <= leased_capacity_tokens`.

- [ ] Determine planned decode capacity.
  - Use per-request max tokens if available from scheduler/request metadata.
  - For MVP, allow env fallback `PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS`.
  - Fail closed if capacity cannot cover the configured smoke output length.

- [ ] Allocate/reserve blocks in Prefill KV manager.
  - Extend request block allocation before export so `leased_block_ids` covers prefix plus decode capacity.
  - Ensure newly allocated blocks follow vLLM zeroing/init semantics or document why Attention writes every slot before read.

- [ ] Export descriptor with full block list.
  - `block_ids` should become the full leased block list in unified mode.
  - Preserve prefix-only `seq_len` separately as `prefix_len` so Attention knows what is already populated.

- [ ] Attention import should store unified state.
  - Add a state representation such as `PAPUnifiedPagedKVState`:
    - `kv_cache`
    - `block_ids`
    - `prefix_len`
    - `seq_len`
    - `capacity_tokens`
    - `writable_start_token`
    - `writable_end_token`
    - `lease_id`
  - Do not allocate `PAPLocalPagedKVPool` in unified mode.

- [ ] Validate statically.
  - Run py_compile on all touched Python files.

- [ ] Validate with descriptor-only smoke.
  - Use one request with `PAP_UNIFIED_KV=1` but keep copied/local compute fallback disabled if unsafe.
  - Expected: descriptor imports with full leased capacity; Attention stores unified state; release frees lease.

**Review gate:** Confirm descriptor has enough information for Attention to compute block/slot for every generated suffix token without calling a remote allocator.

---

## Stage 3: Attention Remote Append Into Prefill-Owned KV

**Purpose:** Move decode suffix K/V writes from Attention-local `PAPLocalPagedKVPool` into Prefill-owned IPC-opened KV cache blocks.

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `vllm/pap/remote_attention.py` only if slot/block helpers should be shared
- Modify: `vllm/pap/data_plane.py` only if descriptor/state fields need adjustment

**Design:** Replace local suffix append in unified mode with remote append into `PAPUnifiedPagedKVState.kv_cache`. The block table is the single leased block list from Prefill. Attention must never allocate local paged blocks for unified-mode requests.

- [ ] Add unified-mode append helper.
  - Example API:
    - `_append_decode_kv_to_unified_prefill_cache_locked(session_request_id, layer_name, key_batch, value_batch, seq_lens)`
  - Compute:
    - `position = seq_len - 1`
    - validate `writable_start_token <= position < writable_end_token`
    - `logical_block = position // block_size`
    - `block_offset = position % block_size`
    - `physical_block = block_ids[logical_block]`

- [ ] Use existing native append where possible.
  - Reuse `reshape_and_cache_flash`/`_try_local_paged_native_cache_append` logic with the Prefill-owned `kv_cache` tensor if layout matches.
  - Otherwise use explicit tensor copy into the correct `[block, k/v, offset, head, dim]` view.

- [ ] Update session seq_len after remote append.
  - Unified state `seq_len` should monotonically increase.
  - Do not mutate Prefill scheduler request state; this state is Attention-side logical decode progress under the lease.

- [ ] Disable local suffix pool for unified requests.
  - In unified mode, any call path that would allocate `PAPLocalPagedKVPool` for that request should raise a clear error.
  - Keep old local path for non-unified mode.

- [ ] Add profile logs gated by `PAP_UNIFIED_KV_PROFILE=1`.
  - per-layer append: request_id, layer, seq_len, block_id, block_offset, append_ms.
  - error logs for out-of-range writes.

- [ ] Validate statically.
  - Run py_compile on touched files.

- [ ] Validate with one-request remote append smoke.
  - Use output length 1 first, then output length 4.
  - Do not use long timeout waits; run client in background and inspect result file/logs promptly.
  - Expected: no local pool allocation logs for unified request; remote append logs appear; request completes.

**Review gate:** Independent reviewer checks block/slot math and verifies no hidden prefix/suffix local copy remains in unified mode.

---

## Stage 4: Single-Source Paged FlashAttention Compute

**Purpose:** Run Attention compute over one Prefill-owned IPC-opened paged KV cache and one full block table, eliminating split prefix/suffix FlashAttention and LSE merge.

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `vllm/pap/remote_attention.py` if metadata helpers move there

**Design:** In unified mode, Attention builds paged FlashAttention metadata from `PAPUnifiedPagedKVState.block_ids` and `seq_len`, and calls `flash_attn_varlen_func` once with `kv_cache` from Prefill IPC.

- [ ] Add `build_unified_paged_flash_metadata()`.
  - Input: list of unified states.
  - Output: block_table, seq_lens, cu_seqlens_q, max_seq_len.
  - It should match existing `build_paged_flash_metadata()` semantics but use Prefill-owned block IDs.

- [ ] Add unified compute path.
  - If all batch rows are unified and share compatible `kv_cache`, call one FA invocation.
  - If rows span different `kv_cache` tensors, group by cache and compute per group, then scatter outputs.
  - Do not use prefix/suffix merge in unified mode.

- [ ] Validate layout constraints.
  - Use existing `layout`/`kv_cache` shape checks.
  - If layout is unsupported, fail clearly with instruction to use non-unified fallback.

- [ ] Remove or bypass Stage B split-cache blocker in unified mode.
  - No prefix-only/suffix-only special case should be needed because the block table is single-source.

- [ ] Validate statically.
  - Run py_compile on touched files.

- [ ] Validate runtime in increasing steps.
  - One request, output length 1.
  - One request, output length 4.
  - 16 prompts, i128/o16/qps4/c8.
  - Expected: completed=all, failed=0, no Attention local pool allocation for unified requests, lease release at end.

**Review gate:** Independent reviewer checks single-cache FA assumptions and verifies no copied-prefix fallback was silently triggered.

---

## Stage 5: Cleanup and Remove Redundant Fast-Path State

**Purpose:** Remove duplicate state only after unified runtime is proven stable.

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `vllm/pap/remote_attention.py` if dense segment helpers are no longer needed on unified path

- [ ] Keep legacy split/local path behind explicit env fallback.
  - Suggested envs:
    - `PAP_UNIFIED_KV=1` for new path.
    - `PAP_ATTENTION_COPY_PREFIX_KV=1` for old copy-prefix comparison only.

- [ ] Make `_prefill_kv` dense segment storage lazy in unified mode.
  - Unified path should not need dense `paged_kv_segments` except for fallback.

- [ ] Ensure release cleans both unified leases and any local fallback state.

- [ ] Validate with one-request and 16-prompt smoke.

**Review gate:** Confirm memory profile shows no cumulative local pool growth and no prefix copy unless explicit fallback env is set.

---

## Stage 6: Benchmark Matrix and Experiment Record

**Purpose:** Establish whether unified KV improves TTFT/OOM without materially regressing TPOT.

**Files:**
- Use: `examples/pap/launch_pap_nixl.sh`
- Use: `tools/pap_bench_summary.py`
- Update after successful runs: `docs/design/pap-pd-comparison-methodology-20260701.md`

- [ ] Run fresh PAP unified one-request smoke.
  - `PAP_UNIFIED_KV=1`
  - `PAP_PREFILL_KV_ASYNC=1`
  - `PAP_ATTENTION_POOL_PROFILE=1`
  - `PAP_KV_LEASE_PROFILE=1`
  - `PAP_UNIFIED_KV_PROFILE=1`

- [ ] Run PAP 16-prompt smoke.
  - i128/o16/qps4/c8.
  - Expected: completed all, no local prefix copy, lease release logs.

- [ ] Run PAP 64-prompt comparison.
  - i128/o16/qps16/c64 if safe.
  - Compare against prior `20260702_220019_pap_local_fast_1pa1p_64req`.

- [ ] Run final PAP 256-prompt comparison.
  - i128/o16/qps16/c64.
  - Use restored capacity knobs: prefill GPU memory 0.80, max batched tokens 8192, max model len 512 for microbench.

- [ ] Run fresh PD 1P1D baseline only after PAP unified path is stable.
  - Same model, i128/o16/qps16/c64, 256 prompts.

- [ ] Summarize with `tools/pap_bench_summary.py`.
  - Include completed/failed, median/mean/p99 TTFT, median/mean/p99 TPOT, throughput, observed max concurrency, and server-side batch evidence.

- [ ] Update design/experiment docs.
  - Record command, env, artifact path, git state, and conclusion.
  - If results are mixed, state observations only; do not claim improvement without comparable baseline.

---

## Success Criteria

- PAP no longer needs an Attention-local prefix KV copy in unified mode.
- Decode suffix KV is written into Prefill-owned leased KV blocks in unified mode.
- Prefill-side blocks are not freed/reused until Attention/Proxy releases the lease.
- Attention compute uses one paged KV source in unified mode; no split-cache FA merge required.
- 256-prompt i128/o16/qps16/c64 PAP run completes without OOM.
- PAP median TTFT improves materially versus the clean 64-request PAP baseline while TPOT does not significantly regress.
- All changed Python files parse successfully.

---

## Known Risks and Stop Conditions

- **Lease safety risk:** If leased blocks can still return to the block pool before release, stop immediately and fix lifecycle before any remote append work.
- **Remote write risk:** If Attention writes outside the leased writable range, stop and add stricter validation before continuing.
- **Scheduler invariant risk:** If vLLM scheduler assumes finished requests always free all blocks immediately, isolate PAP unified mode behind env gates and avoid changing non-PAP behavior.
- **FA layout risk:** If Prefill-owned KV layout cannot be passed to paged FlashAttention as a single source, record exact backend blocker and keep unified append disabled.
- **Benchmark validity risk:** Do not claim performance wins from output length 1 or from different concurrency/load shapes.
