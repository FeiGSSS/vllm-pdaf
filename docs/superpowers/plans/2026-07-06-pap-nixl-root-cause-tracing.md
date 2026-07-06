# PAP NIXL Root Cause Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add operation-level PAP NIXL tracing that can explain high TTFT and TPOT by separating attention compute, data movement, result packaging, and request first-output overhead.

**Architecture:** Keep execution behavior unchanged and extend existing PAP trace logs plus `vllm.pap.trace_summary`. Attention executor logs must expose the real compute path breakdown. Projection/core logs must expose first-output step costs so TTFT can be tied to scheduler, model forward, logits, sampling, and postprocess operations.

**Tech Stack:** Python, vLLM V1 engine/worker/model runner, PAP OFFLOAD_EXEC logs, pytest.

---

### Task 1: Attention Compute Detail Trace

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `vllm/pap/trace_summary.py`
- Test: `tests/pap/test_pap_trace_summary.py`
- Test: `tests/pap/test_pap_attention_executor.py`

- [ ] **Step 1: Write parser tests for operation-level attention fields**

Add fixture fields for `metadata_build_ms`, `paged_flash_kernel_ms`,
`attention_output_reshape_ms`, `compute_unaccounted_ms`,
`pre_compute_done_ns`, `paged_flash_done_ns`, and `reshape_done_ns`.

- [ ] **Step 2: Verify the parser test fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_trace_summary.py -q
```

Expected: FAIL because the new fields are not parsed.

- [ ] **Step 3: Implement parser support**

Extend `_ATTENTION_COMPUTE_DETAIL_RE`, the `attention` field map, and summary
append logic in `vllm/pap/trace_summary.py`.

- [ ] **Step 4: Verify parser test passes**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_trace_summary.py -q
```

Expected: PASS.

- [ ] **Step 5: Write executor tests for populated detail fields**

Use the existing monkeypatch pattern in `tests/pap/test_pap_attention_executor.py`
to make `_compute_unified_paged_flash_batch()` report non-zero metadata/kernel
fields and assert the emitted log contains those fields.

- [ ] **Step 6: Verify executor test fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py -q
```

Expected: FAIL because the executor does not emit the new fields.

- [ ] **Step 7: Implement executor trace field population**

Rename/bridge the current `unified_metadata_ms` and `unified_paged_flash_ms`
writes to the existing logged names, add explicit operation-level aliases, and
write timestamps around metadata build, paged FlashAttention, and output reshape.

- [ ] **Step 8: Verify executor test passes**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py -q
```

Expected: PASS.

### Task 2: TTFT First-Output Operation Trace

**Files:**
- Modify: `vllm/v1/engine/core.py`
- Modify: `vllm/v1/worker/gpu_worker.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Modify: `vllm/pap/trace_summary.py`
- Test: `tests/pap/test_pap_trace_summary.py`

- [ ] **Step 1: Write parser test for TTFT operation fields**

Add fixture log lines for a `PAP OFFLOAD_EXEC projection ttft path` record with
`request_id`, `step_start_ns`, `sched_done_ns`, `worker_exec_start_ns`,
`runner_forward_start_ns`, `model_forward_done_ns`, `logits_done_ns`,
`runner_done_ns`, `worker_exec_done_ns`, `sample_done_ns`,
`postprocess_done_ns`, and `first_output_done_ns`.

- [ ] **Step 2: Verify TTFT parser test fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_trace_summary.py -q
```

Expected: FAIL because `projection_ttft_path` does not exist.

- [ ] **Step 3: Implement TTFT parser support**

Add a regex and summary bucket for first-output operation costs, including
scheduler, queue-to-worker, model forward, logits, runner postprocess, sampling,
scheduler update, and unaccounted overhead.

- [ ] **Step 4: Verify TTFT parser test passes**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_trace_summary.py -q
```

Expected: PASS.

- [ ] **Step 5: Emit TTFT path logs from runtime**

Thread only timestamp data already collected under `PAP_OFFLOAD_EXEC_TRACE=1`;
avoid changing scheduling behavior. Log once per request when its first generated
token appears in `EngineCoreOutputs`.

- [ ] **Step 6: Run focused tests and lint**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_trace_summary.py tests/pap/test_pap_attention_executor.py -q
pre-commit run ruff-check --files examples/pap/pap_attention_executor.py vllm/pap/trace_summary.py vllm/v1/engine/core.py vllm/v1/worker/gpu_worker.py vllm/v1/worker/gpu_model_runner.py tests/pap/test_pap_trace_summary.py tests/pap/test_pap_attention_executor.py
```

Expected: PASS.

### Task 3: NIXL Re-run and Root Cause Report

**Files:**
- Read: `test/baseline/pap/results/runs/.../service_logs/*.log`
- Read: generated `trace_summary.json`

- [ ] **Step 1: Run NIXL benchmark with tracing enabled**

Use the same 1PA1P Qwen3-8B benchmark settings as the previous NIXL trace run.

- [ ] **Step 2: Summarize trace logs**

Run:

```bash
.venv/bin/python -m vllm.pap.trace_summary <run>/service_logs --include-outliers
```

- [ ] **Step 3: Identify root cause**

Compare TTFT and TPOT against operation-level fields. Root cause is only proven
when the dominant median/p99 costs are assigned to concrete operations rather
than modules.
