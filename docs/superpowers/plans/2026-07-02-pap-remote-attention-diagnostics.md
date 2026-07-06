# PAP Remote Attention Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 diagnostic loop for PAP remote-attention optimization: benchmark/log summary, theoretical lower-bound calculation, fast-path detection, and offline validation on existing 20260701 runs.

**Architecture:** Add one focused diagnostics module under `vllm/pap/` that composes existing trace parsing from `vllm/pap/trace_summary.py` with benchmark JSON parsing and lower-bound math. Add one CLI wrapper under `tools/` for run directories. Keep existing profiling changes untouched and avoid changing PAP runtime behavior in this phase.

**Tech Stack:** Python 3.12, stdlib `json`/`argparse`/`dataclasses`/`pathlib`, existing `vllm.pap.trace_summary`, pytest.

---

## File Structure

- Create `vllm/pap/remote_attention_diagnostics.py`
  - Owns result JSON discovery, benchmark metric extraction, lower-bound calculation, fast-path detection from trace summaries, and row rendering.
- Create `tools/pap_remote_attention_diagnostics.py`
  - Thin CLI wrapper around `vllm.pap.remote_attention_diagnostics.main`.
- Create `tests/pap/test_pap_remote_attention_diagnostics.py`
  - Unit tests using synthetic JSON/log files.
- Modify `docs/design/pap-pd-comparison-methodology-20260701.md`
  - Append a short Phase A diagnostics section only after the offline CLI has been run.
- Do not modify current profiling files in this phase:
  - `examples/pap/multi_pap_proxy_server.py`
  - `examples/pap/pap_attention_executor.py`
  - `vllm/model_executor/models/qwen3.py`
  - `vllm/pap/shadow_attention.py`

---

### Task 1: Add diagnostic row model and lower-bound calculator

**Files:**
- Create: `vllm/pap/remote_attention_diagnostics.py`
- Test: `tests/pap/test_pap_remote_attention_diagnostics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/pap/test_pap_remote_attention_diagnostics.py` with:

```python
from pathlib import Path

from vllm.pap.remote_attention_diagnostics import (
    LowerBoundConfig,
    estimate_remote_attention_lower_bound,
)


def test_estimate_remote_attention_lower_bound_qwen3_8b_batch64() -> None:
    estimate = estimate_remote_attention_lower_bound(
        LowerBoundConfig(
            batch_size=64,
            q_size=4096,
            kv_size=1024,
            output_size=4096,
            dtype_bytes=2,
            p2p_bandwidth_gbps=21.0,
            attention_compute_ms=0.12,
            num_layers=36,
        )
    )

    assert estimate.bytes_per_layer == 1310720
    assert round(estimate.transfer_ms_per_layer, 3) == 0.062
    assert round(estimate.lower_bound_ms_per_layer, 3) == 0.182
    assert round(estimate.lower_bound_ms_per_token, 3) == 6.559
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py::test_estimate_remote_attention_lower_bound_qwen3_8b_batch64 -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.pap.remote_attention_diagnostics'`.

- [ ] **Step 3: Implement the lower-bound model**

Create `vllm/pap/remote_attention_diagnostics.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PAP remote-attention benchmark and trace diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LowerBoundConfig:
    batch_size: int
    q_size: int
    kv_size: int
    output_size: int
    dtype_bytes: int
    p2p_bandwidth_gbps: float
    attention_compute_ms: float
    num_layers: int


@dataclass(frozen=True)
class LowerBoundEstimate:
    bytes_per_layer: int
    transfer_ms_per_layer: float
    lower_bound_ms_per_layer: float
    lower_bound_ms_per_token: float


def estimate_remote_attention_lower_bound(
    config: LowerBoundConfig,
) -> LowerBoundEstimate:
    elements_per_request = config.q_size + config.kv_size + config.kv_size + config.output_size
    bytes_per_layer = config.batch_size * elements_per_request * config.dtype_bytes
    bytes_per_ms = config.p2p_bandwidth_gbps * 1_000_000.0
    transfer_ms = bytes_per_layer / bytes_per_ms
    per_layer = transfer_ms + config.attention_compute_ms
    return LowerBoundEstimate(
        bytes_per_layer=bytes_per_layer,
        transfer_ms_per_layer=transfer_ms,
        lower_bound_ms_per_layer=per_layer,
        lower_bound_ms_per_token=per_layer * config.num_layers,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py::test_estimate_remote_attention_lower_bound_qwen3_8b_batch64 -v
```

Expected: PASS.

---

### Task 2: Parse benchmark JSON and service logs into one diagnostic row

**Files:**
- Modify: `vllm/pap/remote_attention_diagnostics.py`
- Modify: `tests/pap/test_pap_remote_attention_diagnostics.py`

- [ ] **Step 1: Add failing test for a synthetic PAP run directory**

Append to `tests/pap/test_pap_remote_attention_diagnostics.py`:

```python
import json

from vllm.pap.remote_attention_diagnostics import summarize_run_directory


def test_summarize_run_directory_combines_benchmark_trace_and_lower_bound(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pap_1pa1p_i128_o32_q16_c64_w32"
    service_logs = run_dir / "service_logs"
    service_logs.mkdir(parents=True)
    (run_dir / "1PA1P_i128_o32_q16_c64_w32.json").write_text(
        json.dumps(
            {
                "completed": 256,
                "failed": 0,
                "request_throughput": 6.11,
                "output_throughput": 195.536,
                "median_ttft_ms": 884.675,
                "median_tpot_ms": 294.765,
                "p99_tpot_ms": 306.697,
                "max_concurrent_requests": 64,
            }
        )
    )
    (service_logs / "projection_0.log").write_text(
        "PAP OFFLOAD_EXEC projection timeline "
        "layer=model.layers.1.self_attn.attn batches=1 calls=64 "
        "pre_attn_compute_ms=0.400 send_ms=0.040 trigger_ms=0.000 "
        "yield_ms=0.200 recv_ms=1.010 o_proj_ms=0.300 "
        "remote_total_ms=1.050 self_attn_total_ms=1.750\n"
    )
    (service_logs / "attention_0.log").write_text(
        "PAP OFFLOAD_EXEC attention mailbox batch trace "
        "layer=model.layers.1.self_attn.attn calls=64 recv_qkv_ms=0.820 "
        "compute_ms=0.120 send_output_ms=0.010 total_ms=0.970 "
        "append_kv_ms=0.050 pack_ms=0.030 sdpa_ms=0.040 reshape_ms=0.020 "
        "paged_metadata_ms=0.000 paged_flash_ms=0.000 "
        "qkv_shape=(64, 6144) output_shape=(64, 4096) batch_key=abc "
        "recv_done_ns=1003900000 compute_done_ns=1004100000 "
        "send_done_ns=1004200000\n"
    )

    row = summarize_run_directory(run_dir)

    assert row.topology == "1PA1P"
    assert row.input_len == "128"
    assert row.output_len == "32"
    assert row.qps == "16"
    assert row.max_concurrency == "64"
    assert row.completed == 256
    assert row.failed == 0
    assert row.median_tpot_ms == 294.765
    assert row.projection_remote_total_median_ms == 1.05
    assert row.attention_compute_median_ms == 0.12
    assert round(row.lower_bound_ms_per_layer, 3) == 0.182
    assert round(row.e2e_ms_per_layer, 3) == 8.188
    assert row.fast_path_status["paged_flash"] == "inactive"
    assert row.fast_path_status["attention_batch_calls_median"] == "64.000"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py::test_summarize_run_directory_combines_benchmark_trace_and_lower_bound -v
```

Expected: FAIL with `ImportError` or `AttributeError` for `summarize_run_directory`.

- [ ] **Step 3: Implement run summarization**

Add to `vllm/pap/remote_attention_diagnostics.py`:

```python
import json
import re
from pathlib import Path
from typing import Any

from vllm.pap.trace_summary import summarize_pap_trace_logs

_RESULT_NAME_RE = re.compile(
    r"(?P<topology>[A-Za-z0-9]+)_i(?P<input_len>\d+)_o(?P<output_len>\d+)"
    r"_q(?P<qps>[^_.]+)(?:_c(?P<max_concurrency>\d+))?"
    r"(?:_w(?P<num_warmups>\d+))?"
)


@dataclass(frozen=True)
class DiagnosticRow:
    path: str
    topology: str
    input_len: str
    output_len: str
    qps: str
    max_concurrency: str
    num_warmups: str
    completed: int
    failed: int
    request_throughput: float
    output_throughput: float
    median_ttft_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    projection_remote_total_median_ms: float
    projection_recv_median_ms: float
    attention_compute_median_ms: float
    attention_total_median_ms: float
    lower_bound_ms_per_layer: float
    lower_bound_ms_per_token: float
    e2e_ms_per_layer: float
    fast_path_status: dict[str, str]


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _result_json_for_run(run_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in run_dir.glob("*.json")
        if path.name != "run_metadata.json" and not path.name.startswith(".")
    )
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one result JSON in {run_dir}, found {len(candidates)}")
    return candidates[0]


def _stat_median(summary: dict[str, Any], section: str, field: str) -> float:
    value = summary.get(section, {}).get(field)
    if value is None:
        return 0.0
    return float(getattr(value, "median", 0.0))


def _fmt_stat(value: float) -> str:
    return f"{value:.3f}"


def _fast_path_status(trace_summary: dict[str, Any]) -> dict[str, str]:
    paged_flash = _stat_median(trace_summary, "attention_trace", "paged_flash_ms")
    fallback = _stat_median(trace_summary, "attention_trace", "fallback_ms")
    calls = _stat_median(trace_summary, "attention_trace", "calls")
    return {
        "paged_flash": "active" if paged_flash > 0.0 else "inactive",
        "fallback": "active" if fallback > 0.0 else "inactive",
        "attention_batch_calls_median": _fmt_stat(calls),
    }


def summarize_run_directory(
    run_dir: str | Path,
    *,
    lower_bound_config: LowerBoundConfig | None = None,
) -> DiagnosticRow:
    run_path = Path(run_dir)
    result_path = _result_json_for_run(run_path)
    result = _load_json(result_path)
    match = _RESULT_NAME_RE.search(result_path.stem)
    groups = match.groupdict() if match else {}
    service_logs = run_path / "service_logs"
    trace_summary = summarize_pap_trace_logs(service_logs) if service_logs.exists() else {}
    attention_compute = _stat_median(trace_summary, "attention_trace", "compute_ms")
    config = lower_bound_config or LowerBoundConfig(
        batch_size=64,
        q_size=4096,
        kv_size=1024,
        output_size=4096,
        dtype_bytes=2,
        p2p_bandwidth_gbps=21.0,
        attention_compute_ms=attention_compute or 0.12,
        num_layers=36,
    )
    lower_bound = estimate_remote_attention_lower_bound(config)
    median_tpot = float(result.get("median_tpot_ms") or 0.0)
    return DiagnosticRow(
        path=str(run_path),
        topology=str(groups.get("topology") or ""),
        input_len=str(groups.get("input_len") or ""),
        output_len=str(groups.get("output_len") or ""),
        qps=str(groups.get("qps") or ""),
        max_concurrency=str(groups.get("max_concurrency") or ""),
        num_warmups=str(groups.get("num_warmups") or ""),
        completed=int(result.get("completed") or 0),
        failed=int(result.get("failed") or 0),
        request_throughput=float(result.get("request_throughput") or 0.0),
        output_throughput=float(result.get("output_throughput") or 0.0),
        median_ttft_ms=float(result.get("median_ttft_ms") or 0.0),
        median_tpot_ms=median_tpot,
        p99_tpot_ms=float(result.get("p99_tpot_ms") or 0.0),
        projection_remote_total_median_ms=_stat_median(
            trace_summary, "projection_timeline", "remote_total_ms"
        ),
        projection_recv_median_ms=_stat_median(
            trace_summary, "projection_timeline", "recv_ms"
        ),
        attention_compute_median_ms=attention_compute,
        attention_total_median_ms=_stat_median(trace_summary, "attention_trace", "total_ms"),
        lower_bound_ms_per_layer=lower_bound.lower_bound_ms_per_layer,
        lower_bound_ms_per_token=lower_bound.lower_bound_ms_per_token,
        e2e_ms_per_layer=median_tpot / config.num_layers if config.num_layers else 0.0,
        fast_path_status=_fast_path_status(trace_summary),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py -v
```

Expected: both tests PASS.

---

### Task 3: Add markdown/CSV rendering and CLI

**Files:**
- Modify: `vllm/pap/remote_attention_diagnostics.py`
- Create: `tools/pap_remote_attention_diagnostics.py`
- Modify: `tests/pap/test_pap_remote_attention_diagnostics.py`

- [ ] **Step 1: Add failing tests for markdown rendering**

Append to `tests/pap/test_pap_remote_attention_diagnostics.py`:

```python
from vllm.pap.remote_attention_diagnostics import DiagnosticRow, rows_to_markdown


def test_rows_to_markdown_includes_lower_bound_and_fast_path_status() -> None:
    row = DiagnosticRow(
        path="run",
        topology="1PA1P",
        input_len="128",
        output_len="32",
        qps="16",
        max_concurrency="64",
        num_warmups="32",
        completed=256,
        failed=0,
        request_throughput=6.11,
        output_throughput=195.536,
        median_ttft_ms=884.675,
        median_tpot_ms=294.765,
        p99_tpot_ms=306.697,
        projection_remote_total_median_ms=1.05,
        projection_recv_median_ms=1.01,
        attention_compute_median_ms=0.12,
        attention_total_median_ms=0.97,
        lower_bound_ms_per_layer=0.182,
        lower_bound_ms_per_token=6.559,
        e2e_ms_per_layer=8.188,
        fast_path_status={
            "paged_flash": "inactive",
            "fallback": "inactive",
            "attention_batch_calls_median": "64.000",
        },
    )

    markdown = rows_to_markdown([row])

    assert "| topology | input | output | qps | warmup | max conc |" in markdown
    assert "| 1PA1P | 128 | 32 | 16 | 32 | 64 |" in markdown
    assert "0.182" in markdown
    assert "8.188" in markdown
    assert "paged_flash=inactive" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py::test_rows_to_markdown_includes_lower_bound_and_fast_path_status -v
```

Expected: FAIL because `rows_to_markdown` is missing.

- [ ] **Step 3: Implement rendering and CLI main**

Add to `vllm/pap/remote_attention_diagnostics.py`:

```python
import argparse


def _row_values(row: DiagnosticRow) -> list[str]:
    fast_paths = ",".join(
        f"{key}={value}" for key, value in sorted(row.fast_path_status.items())
    )
    return [
        row.topology,
        row.input_len,
        row.output_len,
        row.qps,
        row.num_warmups,
        row.max_concurrency,
        f"{row.completed}/{row.failed}",
        f"{row.output_throughput:.3f}",
        f"{row.median_tpot_ms:.3f}",
        f"{row.projection_remote_total_median_ms:.3f}",
        f"{row.projection_recv_median_ms:.3f}",
        f"{row.attention_compute_median_ms:.3f}",
        f"{row.lower_bound_ms_per_layer:.3f}",
        f"{row.e2e_ms_per_layer:.3f}",
        fast_paths,
        row.path,
    ]


def rows_to_markdown(rows: list[DiagnosticRow]) -> str:
    headers = [
        "topology",
        "input",
        "output",
        "qps",
        "warmup",
        "max conc",
        "done/failed",
        "out tok/s",
        "median TPOT",
        "proj remote",
        "proj recv",
        "attn compute",
        "lb/layer",
        "e2e/layer",
        "fast paths",
        "path",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_row_values(row)) + " |")
    return "\n".join(lines)


def _candidate_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            run_dirs.append(path.parent)
        elif path.is_dir() and any(child.suffix == ".json" for child in path.iterdir()):
            run_dirs.append(path)
        elif path.is_dir():
            run_dirs.extend(
                child.parent
                for child in path.rglob("*.json")
                if child.name != "run_metadata.json"
            )
    return sorted(set(run_dirs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    rows = [summarize_run_directory(path) for path in _candidate_run_dirs(args.paths)]
    print(rows_to_markdown(rows))
    return 0
```

Create `tools/pap_remote_attention_diagnostics.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI wrapper for PAP remote-attention diagnostics."""

from vllm.pap.remote_attention_diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py -v
```

Expected: PASS.

---

### Task 4: Run the diagnostics on existing 20260701 artifacts

**Files:**
- No code changes expected unless Task 4 exposes a parser bug.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py tests/pap/test_pap_trace_summary.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the diagnostic CLI on the latest methodology directory**

Run:

```bash
.venv/bin/python tools/pap_remote_attention_diagnostics.py test/baseline/pap/results/runs/pd_pap_methodology_20260701
```

Expected: markdown table with one row per JSON result. Rows with missing service logs should show `0.000` trace fields rather than crashing.

- [ ] **Step 3: Save the CLI output for experiment bookkeeping**

Run:

```bash
.venv/bin/python tools/pap_remote_attention_diagnostics.py \
  test/baseline/pap/results/runs/pd_pap_methodology_20260701 \
  > test/baseline/pap/results/runs/pd_pap_methodology_20260701/remote_attention_diagnostics.md
```

Expected: `remote_attention_diagnostics.md` exists and contains `lb/layer` and `e2e/layer` columns.

- [ ] **Step 4: Inspect the diagnostic output**

Run:

```bash
grep -n "1PA1P\|1P1D\|lb/layer\|e2e/layer" test/baseline/pap/results/runs/pd_pap_methodology_20260701/remote_attention_diagnostics.md
```

Expected: rows for the 20260701 results. If PAP trace fields are zero for runs with copied logs outside `service_logs`, record that as an instrumentation gap instead of editing runtime code.

---

### Task 5: Record Phase 1 findings in the methodology note

**Files:**
- Modify: `docs/design/pap-pd-comparison-methodology-20260701.md`

- [ ] **Step 1: Append the diagnostics section**

Append this section to `docs/design/pap-pd-comparison-methodology-20260701.md` after the current interpretation section:

```markdown

## Phase A Remote-Attention Diagnostics

The Phase A diagnostic loop adds a per-run table that joins benchmark metrics,
PAP trace summaries, and a simple remote-attention lower bound:

```text
T_lb_layer = bytes(QKV + attention_output) / P2P_bandwidth + attention_compute
```

For Qwen3-8B bf16 at batch 64, the default diagnostic assumption is
`q_size=4096`, `kv_size=1024`, `output_size=4096`, `P2P=21 GB/s`, and
`36` layers. This gives a rough lower bound near `0.18 ms/layer`, or
`6.6 ms/token`, before scheduler and queueing effects.

The generated table is stored at:
`test/baseline/pap/results/runs/pd_pap_methodology_20260701/remote_attention_diagnostics.md`.
Use it to decide which existing fast-path flag to test next. If trace columns are
zero for a row, that run lacks the required trace logs and should not be used for
micro-path conclusions.
```
```

- [ ] **Step 2: Check markdown syntax around the inserted fenced block**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
text = Path('docs/design/pap-pd-comparison-methodology-20260701.md').read_text()
assert text.count('```') % 2 == 0
assert 'Phase A Remote-Attention Diagnostics' in text
PY
```

Expected: command exits successfully.

---

### Task 6: Final verification and git status review

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_remote_attention_diagnostics.py tests/pap/test_pap_trace_summary.py -v
```

Expected: PASS.

- [ ] **Step 2: Run py_compile on new modules**

Run:

```bash
.venv/bin/python -m py_compile \
  vllm/pap/remote_attention_diagnostics.py \
  tools/pap_remote_attention_diagnostics.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Review git status without committing**

Run:

```bash
git status --short
```

Expected: new diagnostic module, CLI, test, plan/spec docs, and generated diagnostic markdown are visible. Do not commit unless the user explicitly asks.

---

## Self-Review

- Spec coverage: Task 1 implements lower-bound math; Tasks 2-3 implement benchmark/log summary and fast-path detection; Task 4 validates on existing run artifacts; Task 5 records experiment details; Task 6 verifies tests and git state.
- Placeholder scan: no placeholders remain; every code step includes exact snippets and commands.
- Type consistency: `LowerBoundConfig`, `LowerBoundEstimate`, `DiagnosticRow`, `estimate_remote_attention_lower_bound`, `summarize_run_directory`, and `rows_to_markdown` are defined before use.
