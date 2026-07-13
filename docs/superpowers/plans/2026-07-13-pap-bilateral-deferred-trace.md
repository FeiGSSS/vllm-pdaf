# PAP/PD Bilateral Deferred Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default-off, drain-flushed bilateral timing for PAP Projection and PD Decode, then run a fixed C4 PAP off/on and PD-twoway off/on diagnostic matrix.

**Architecture:** Extend the existing process-local deferred CUDA collector with host-duration recording and a diagnostic-only trigger-file exporter. Instrument identical Qwen3 QKV boundaries in PAP Projection and PD Decode, instrument PAP local-fast source/wait boundaries and the PD paged FlashAttention call, then make both benchmark runners capture and fail-closed validate the JSON artifacts.

**Tech Stack:** Python 3.12 through `.venv/bin/python`, PyTorch CUDA Events, vLLM Qwen3/FlashAttention, Bash benchmark runners, pytest.

## Global Constraints

- `PAP_DEFERRED_CUDA_TRACE=0` remains the default.
- No hot-path CUDA synchronize, per-layer JSON write, or per-layer log.
- C4 remains Qwen3-8B FP16, 16K, five rounds, four conversations, o256, QPS 2, GPU1/GPU2, MPS 70:30.
- Do not run pre-commit; run focused pytest and shell syntax checks.
- All Python commands use `.venv/bin/python`.
- Preserve unrelated tracked and untracked user files.

---

### Task 1: Host Durations and Trigger-File Exporter

**Files:**
- Modify: `tests/pap/test_pap_deferred_cuda_trace.py`
- Modify: `vllm/pap/deferred_cuda_trace.py`

**Interfaces:**
- Produces: `DeferredCudaTraceCollector.record_duration(name: str, duration_ms: float) -> None`.
- Produces: `record_deferred_host_duration(name: str, duration_ms: float) -> None`.
- Produces: `DeferredTraceFileExporter(output_path: str, scope: str, role: str, snapshot_fn: Callable[..., dict[str, Any]], poll_interval_s: float = 0.05)`.
- Produces: `ensure_deferred_trace_file_exporter() -> None`.

- [ ] **Step 1: Write failing collector and exporter tests**

```python
def test_deferred_trace_records_host_duration_without_cuda_event() -> None:
    collector = DeferredCudaTraceCollector(event_factory=lambda: None)
    collector.record_duration("token_boundary_input_ids_d2h_wall_ms", 0.25)
    assert collector.raw_snapshot(blocking=False)["durations"] == {
        "token_boundary_input_ids_d2h_wall_ms": [0.25]
    }


def test_deferred_trace_exporter_flushes_on_trigger(tmp_path: Path) -> None:
    output = tmp_path / "trace.json"
    exporter = DeferredTraceFileExporter(
        output_path=str(output),
        scope="projection_process_critical_chain",
        role="projection",
        snapshot_fn=lambda *, blocking: {
            "enabled": True,
            "pending_records": 0,
            "dropped_records": 0,
            "error_records": 0,
            "spans": {},
        },
        poll_interval_s=0.005,
    )
    exporter.start()
    Path(f"{output}.flush").touch()
    assert _wait_until(output.exists)
    exporter.stop()
    assert json.loads(output.read_text())["scope"] == (
        "projection_process_critical_chain"
    )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_deferred_cuda_trace.py -q
```

Expected: FAIL because `record_duration` and `DeferredTraceFileExporter` do not exist.

- [ ] **Step 3: Implement host aggregation and exporter**

Add a lock-protected `record_duration()` that appends finite non-negative milliseconds to
the existing duration store. Add a daemon exporter which polls `<output>.flush`, performs
one blocking process snapshot, adds `scope`, `role`, and `pid`, writes a temporary JSON,
and publishes it with `os.replace()`. Start it lazily only when tracing is enabled and an
output path exists.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 pytest command and expect all tests to pass.

### Task 2: Fail-Closed Artifact Validator

**Files:**
- Create: `benchmarks/multi_turn/validate_deferred_trace.py`
- Create: `tests/benchmarks/test_validate_deferred_trace.py`

**Interfaces:**
- Produces: `validate_trace(payload: Mapping[str, Any], scope: str, num_layers: int = 36, reference_peer_batches: int | None = None) -> dict[str, int]`.
- CLI: `--trace PATH --scope projection_process_critical_chain|pd_decode_process_critical_chain --num-layers 36 [--attention-stats PATH]`.

- [ ] **Step 1: Write failing validator tests**

```python
def test_projection_trace_requires_matching_layer_and_forward_counts() -> None:
    payload = _projection_trace(layer_count=72, token_count=2)
    counts = validate_trace(
        payload,
        scope="projection_process_critical_chain",
        num_layers=36,
        reference_peer_batches=72,
    )
    assert counts["decode_forwards"] == 2


def test_pd_trace_rejects_qkv_fa_count_mismatch() -> None:
    payload = _pd_trace(layer_count=72)
    payload["spans"]["pd_paged_fa_gpu_ms"]["count"] = 71
    with pytest.raises(ValueError, match="count mismatch"):
        validate_trace(
            payload,
            scope="pd_decode_process_critical_chain",
            num_layers=36,
        )
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/benchmarks/test_validate_deferred_trace.py -q
```

Expected: FAIL because the validator module does not exist.

- [ ] **Step 3: Implement validator and CLI**

Validate enabled state, exact scope, positive collector count, zero pending/drop/error,
required spans, equal per-layer counts, divisibility by 36, token-boundary count, and the
optional Attention peer-batch reference. Print a compact JSON count summary on success.

- [ ] **Step 4: Verify GREEN**

Run the Task 2 pytest command and expect all tests to pass.

### Task 3: PAP Projection and Local-Fast Spans

**Files:**
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `vllm/pap/local_fast_transport.py`
- Modify: `vllm/v1/worker/gpu/model_runner.py`
- Modify: `tests/pap/test_pap_qwen3_tp_routing.py`

**Interfaces:**
- Consumes: Task 1 deferred span and host-duration APIs.
- Produces spans: `qkv_norm_rope_gpu_ms`, `projection_qk_repack_gpu_ms`, `qkv_p2p_copy_gpu_ms`, `output_doorbell_wait_wall_ms`, `output_ready_wait_gpu_ms`, and `token_boundary_input_ids_d2h_wall_ms`.

- [ ] **Step 1: Write failing role/boundary tests**

Add tests proving the Qwen3 trace predicate enables Projection spans only for a PAP decode
batch and PD spans only for a `max_query_len == 1` decode batch. Add a static runner-facing
test that all six span names are present in their intended source files.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_qwen3_tp_routing.py -q
```

Expected: FAIL because the bilateral trace predicate and spans are missing.

- [ ] **Step 3: Add minimal Projection instrumentation**

Use one Qwen3 CUDA span around QKV/norm/RoPE and a separate span around PAP Q/K direct-buffer
repack. In local-fast, time QKV source copy and split output receive into host doorbell wait
and GPU ready wait. In ModelRunner, time only the existing input-ID D2H/list conversion.
Every branch must check the default-off trace role before creating timers or Events.

- [ ] **Step 4: Verify GREEN**

Run the Task 3 pytest command plus:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_local_fast_transport.py -q
```

Expect both files to pass.

### Task 4: PD Decode QKV and FlashAttention Spans

**Files:**
- Modify: `vllm/model_executor/models/qwen3.py`
- Modify: `vllm/v1/attention/backends/flash_attn.py`
- Create: `tests/pap/test_pap_bilateral_trace_contract.py`

**Interfaces:**
- Consumes: Task 1 deferred CUDA span API and role predicate.
- Produces PD spans: `qkv_norm_rope_gpu_ms` and `pd_paged_fa_gpu_ms`.

- [ ] **Step 1: Write failing source-boundary contract tests**

```python
def test_pd_trace_uses_same_qkv_span_and_main_paged_fa_boundary() -> None:
    qwen = QWEN3.read_text(encoding="utf-8")
    flash = FLASH_ATTN.read_text(encoding="utf-8")
    assert '"qkv_norm_rope_gpu_ms"' in qwen
    assert '"pd_paged_fa_gpu_ms"' in flash
    assert 'PAP_DEFERRED_TRACE_ROLE", ""' in qwen
    assert "max_query_len" in flash
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_bilateral_trace_contract.py -q
```

Expected: FAIL because the PD paged-FA span is missing.

- [ ] **Step 3: Instrument PD main paged FlashAttention call**

Record the same Qwen3 QKV span when role is `pd_decode`. Wrap only the non-cascade main paged
`flash_attn_varlen_func` call when role is `pd_decode` and `max_query_len == 1`; do not include
KV append or Prefill.

- [ ] **Step 4: Verify GREEN**

Run Task 4 pytest and expect it to pass.

### Task 5: PAP and PD Runner Capture Contracts

**Files:**
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh`
- Modify: `tests/pap/test_pap_launch_files.py`
- Modify: `tests/benchmarks/test_pd_multiturn_runner_contract.py`

**Interfaces:**
- Consumes: Task 1 exporter and Task 2 validator CLI.
- Produces PAP artifact: `projection_deferred_trace.json`.
- Produces PD artifact: `pd_decode_deferred_trace.json`.

- [ ] **Step 1: Write failing runner contract tests**

Assert both runners keep trace default-off, pass explicit roles/output paths only to the
target process, create `.flush` after workload completion, wait for the JSON artifact, and
invoke `validate_deferred_trace.py` before finalization.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_launch_files.py tests/benchmarks/test_pd_multiturn_runner_contract.py -q
```

Expected: FAIL because bilateral capture is not wired.

- [ ] **Step 3: Implement runner capture and metadata**

Add default-off env handling, effective-config fields, role-scoped process env, trigger-file
creation, bounded artifact wait, validator invocation, and finalizer artifact registration.
The PAP Attention HTTP capture remains unchanged.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run Task 5 pytest plus:

```bash
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh
```

Expect zero failures and both shell checks to exit 0.

### Task 6: Focused Regression Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes all prior tasks.
- Produces a verified diagnostic build.

- [ ] **Step 1: Run focused suite**

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_deferred_cuda_trace.py \
  tests/pap/test_pap_qwen3_tp_routing.py \
  tests/pap/test_pap_local_fast_transport.py \
  tests/pap/test_pap_bilateral_trace_contract.py \
  tests/pap/test_pap_launch_files.py \
  tests/benchmarks/test_validate_deferred_trace.py \
  tests/benchmarks/test_finalize_pap_pd_multiturn.py \
  tests/benchmarks/test_pd_multiturn_runner_contract.py \
  tests/benchmarks/test_pap_multiturn_mps_contract.py \
  tests/benchmarks/test_pd_three_lane_testbed_contract.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Review tracked diff and compile modified Python**

```bash
.venv/bin/python -m py_compile \
  vllm/pap/deferred_cuda_trace.py \
  benchmarks/multi_turn/validate_deferred_trace.py \
  vllm/model_executor/models/qwen3.py \
  vllm/v1/attention/backends/flash_attn.py \
  vllm/v1/worker/gpu/model_runner.py
git diff --check
git status --short
```

Expected: compile and diff checks exit 0; status contains only intended tracked changes plus
pre-existing untracked artifacts.

### Task 7: Fixed C4 Diagnostic Matrix and Result Report

**Files:**
- Create after measurement: `docs/design/pap-bilateral-deferred-trace-results-20260713.md`
- Modify after measurement: `docs/design/pap-experiment-history-index.md`

**Interfaces:**
- Consumes the PAP and PD C4 runners.
- Produces four run roots and one evidence-backed next-step recommendation.

- [ ] **Step 1: Run PAP trace-off and trace-on C4 quick**

```bash
PAP_DEFERRED_CUDA_TRACE=0 \
PAP_LOAD_RUN_ID=20260713_pap_bilateral_trace_off_c4 \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh quick c4

PAP_DEFERRED_CUDA_TRACE=1 \
PAP_LOAD_RUN_ID=20260713_pap_bilateral_trace_on_c4 \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh quick c4
```

Expected: 20/20 requests, strict/cache/routing/drain gates pass; trace-on has zero pending,
drop, and error records.

- [ ] **Step 2: Run PD-twoway trace-off and trace-on C4 quick**

```bash
PAP_DEFERRED_CUDA_TRACE=0 \
PD_LOAD_RUN_ID=20260713_pd_twoway_bilateral_trace_off_c4 \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh twoway

PAP_DEFERRED_CUDA_TRACE=1 \
PD_LOAD_RUN_ID=20260713_pd_twoway_bilateral_trace_on_c4 \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh twoway
```

Expected: 20/20 requests, official two-way reuse and correctness gates pass; trace-on has
matching QKV/FA counts and zero pending/drop/error records.

- [ ] **Step 3: Calculate budgets and perturbation**

Use `.venv/bin/python` to load both `aggregate.json` files and trace JSON files. Report PAP
and PD trace perturbation, each mean/p50/p90/p99 span, `mean_ms * 36`, and the residual against
trace-on TPOT. Do not add overlapping CPU and GPU waits as independent time.

- [ ] **Step 4: Write the result report and history row**

Record exact commands, commit/dirty state, run roots, correctness gates, counts, perturbation,
budget table, limitations, and the next optimization selected by the thresholds in the design.

- [ ] **Step 5: Fresh verification before reporting completion**

Re-run Task 6 verification, read all four result validity fields and validator summaries, and
run `git diff --check`. Report any failed gate instead of claiming completion.
