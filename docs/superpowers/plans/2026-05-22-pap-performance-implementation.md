# PAP Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Begin implementing the PAP performance architecture and run PAP 6PA2P against the existing 6P2D PD benchmark harness.

**Architecture:** First make the existing PAP deployment benchmarkable through the external baseline runner, then separate `debug_remote_attention` from the new `true_split` path. The first measurable version may still be debug-mode PAP, but performance-mode work must move toward Projection without local decode attention and Attention owning decode KV.

**Tech Stack:** vLLM V1, Qwen3-8B, FastAPI proxy, NIXL KV transfer, CUDA MPS, pytest, `/home/fei/research/PD/test/baseline/run_benchmark.sh`.

---

## File Map

- `examples/pap/launch_pap_6pa2p_qwen3_8b_nixl.sh`: add service-only mode so benchmark launchers can start PAP and leave it running without the one-request smoke path.
- `tests/pap/test_pap_launch_files.py`: assert service-only mode, status file, and benchmark-safe defaults exist.
- `/home/fei/research/PD/test/baseline/pap/config.sh`: new external baseline mode config for PAP.
- `/home/fei/research/PD/test/baseline/pap/launch_service.sh`: new external baseline mode launcher that delegates to the PAP 6PA2P service-only launcher and writes the proxy port status file.
- `tests/pap/test_pap_baseline_integration.py`: local tests that inspect the external launcher files when present and verify the contract expected by `run_benchmark.sh`.
- `vllm/pap/mode.py`: new mode helper for `debug_remote_attention` vs `true_split`.
- `tests/pap/test_pap_mode.py`: failing tests for mode parsing and for preventing performance benchmarks from accidentally using the debug path.
- Later: `vllm/pap/projection_runner.py`, `vllm/pap/attention_executor.py`, model hook changes in `vllm/model_executor/models/qwen3.py`.

## Task 1: Make The PAP Launcher Service-Aware

- [ ] **Step 1: Write the failing launcher test**

Add to `tests/pap/test_pap_launch_files.py`:

```python
def test_pap_6pa2p_launch_supports_benchmark_service_mode() -> None:
    script = ROOT / "examples" / "pap" / "launch_pap_6pa2p_qwen3_8b_nixl.sh"
    text = script.read_text()

    assert "PAP_SERVICE_ONLY" in text
    assert "PAP_STATUS_FILE" in text
    assert "PAP_SKIP_SMOKE_REQUEST" in text
    assert 'echo "$PROXY_PORT" >"$STATUS_FILE"' in text
    assert 'if [[ "${SERVICE_ONLY}" == "1" ]]' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_launch_files.py::test_pap_6pa2p_launch_supports_benchmark_service_mode -v
```

Expected: FAIL because the service-only variables are not implemented.

- [ ] **Step 3: Implement the launcher service mode**

Modify `examples/pap/launch_pap_6pa2p_qwen3_8b_nixl.sh`:

```bash
SERVICE_ONLY="${PAP_SERVICE_ONLY:-0}"
STATUS_FILE="${PAP_STATUS_FILE:-}"
SKIP_SMOKE_REQUEST="${PAP_SKIP_SMOKE_REQUEST:-0}"
```

After the proxy health check:

```bash
if [[ -n "$STATUS_FILE" ]]; then
    echo "$PROXY_PORT" >"$STATUS_FILE"
fi

if [[ "${SERVICE_ONLY}" == "1" ]]; then
    echo "PAP_SERVICE_ONLY=1; services remain running. Logs are in $LOG_DIR"
    wait
fi
```

Wrap the smoke request:

```bash
if [[ "${SKIP_SMOKE_REQUEST}" != "1" ]]; then
    echo "Running one PAP 6PA:2P request through proxy"
    .venv/bin/python examples/pap/run_one_request.py ...
fi
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_launch_files.py::test_pap_6pa2p_launch_supports_benchmark_service_mode -v
```

Expected: PASS.

## Task 2: Add Baseline pap Mode

- [ ] **Step 1: Write the failing integration test**

Create `tests/pap/test_pap_baseline_integration.py`:

```python
from pathlib import Path

BASELINE = Path("/home/fei/research/PD/test/baseline")
PAP = BASELINE / "pap"


def test_external_baseline_has_pap_mode_contract() -> None:
    config = PAP / "config.sh"
    launcher = PAP / "launch_service.sh"

    assert config.exists()
    assert launcher.exists()

    config_text = config.read_text()
    launcher_text = launcher.read_text()

    assert "PAP_PROXY_PORT" in config_text
    assert "VLLM_BIN" in config_text
    assert "/home/fei/research/PD/vllm-pap/.venv/bin/vllm" in config_text
    assert "PAP_SERVICE_ONLY=1" in launcher_text
    assert "PAP_SKIP_SMOKE_REQUEST=1" in launcher_text
    assert "PAP_STATUS_FILE" in launcher_text
    assert "launch_pap_6pa2p_qwen3_8b_nixl.sh" in launcher_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_baseline_integration.py::test_external_baseline_has_pap_mode_contract -v
```

Expected: FAIL because `/home/fei/research/PD/test/baseline/pap` does not exist.

- [ ] **Step 3: Create external baseline config**

Create `/home/fei/research/PD/test/baseline/pap/config.sh`:

```bash
#!/bin/bash

export BASELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="/home/fei/research/PD"
export PAP_ROOT="${PAP_ROOT:-${PROJECT_ROOT}/vllm-pap}"
export VLLM_BIN="${VLLM_BIN:-${PAP_ROOT}/.venv/bin/vllm}"
export PYTHON_BIN="${PYTHON_BIN:-${PAP_ROOT}/.venv/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/data/ssd1/llm-models/Qwen3-8B}"
export BENCH_DIR="${PROJECT_ROOT}/refer_codes/vllm/benchmarks"
export DATASET_PATH="${DATASET_PATH:-${BENCH_DIR}/sonnet_4x.txt}"
export DATASET_NAME="sonnet"

export NUM_PROMPTS="${NUM_PROMPTS:-100}"
export PREFIX_LEN="${PREFIX_LEN:-50}"
export PAP_PROXY_PORT="${PAP_PROXY_PORT:-9000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-10000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
export BENCH_TIMEOUT="${BENCH_TIMEOUT:-600}"
export CLUSTER_READY_WAIT_SECONDS="${CLUSTER_READY_WAIT_SECONDS:-30}"

export RESULTS_ROOT="${BASELINE_ROOT}/results"
export LOG_DIR="${RESULTS_ROOT}/logs"
export RAW_DIR="${RESULTS_ROOT}/raw"
export RUNS_DIR="${RESULTS_ROOT}/runs"
```

- [ ] **Step 4: Create external baseline launcher**

Create `/home/fei/research/PD/test/baseline/pap/launch_service.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

TOPOLOGY_RAW="${1:?topology is required}"
STATUS_FILE="${2:-/tmp/pap_status}"
TOPOLOGY="$(printf '%s' "${TOPOLOGY_RAW}" | tr '[:upper:]' '[:lower:]')"
FAILURE_FILE="${STATUS_FILE}.failed"

if [[ "${TOPOLOGY}" != "6pa2p" ]]; then
  echo "Unsupported pap topology: ${TOPOLOGY_RAW}" >&2
  exit 1
fi

TOPOLOGY_TAG="$(printf '%s' "${TOPOLOGY}" | tr '[:lower:]' '[:upper:]')"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_DIR}/${TOPOLOGY_TAG}}"
mkdir -p "${RUN_LOG_DIR}"
rm -f "${STATUS_FILE}" "${FAILURE_FILE}"

cleanup() {
  set +e
  rm -f "${STATUS_FILE}"
}

trap cleanup EXIT
trap 'exit 0' INT TERM

cd "${PAP_ROOT}"

PAP_SERVICE_ONLY=1 \
PAP_SKIP_SMOKE_REQUEST=1 \
PAP_STATUS_FILE="${STATUS_FILE}" \
PAP_PROXY_PORT="${PAP_PROXY_PORT}" \
PAP_MODEL_PATH="${MODEL_PATH}" \
PAP_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
PAP_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
PAP_LOG_DIR="${RUN_LOG_DIR}" \
bash examples/pap/launch_pap_6pa2p_qwen3_8b_nixl.sh
```

- [ ] **Step 5: Run syntax and integration tests**

Run:

```bash
bash -n /home/fei/research/PD/test/baseline/pap/config.sh
bash -n /home/fei/research/PD/test/baseline/pap/launch_service.sh
.venv/bin/python -m pytest tests/pap/test_pap_baseline_integration.py -v
```

Expected: PASS.

## Task 3: Smoke Benchmark PAP 6PA2P

- [ ] **Step 1: Verify no leftover PAP/vLLM processes**

Run:

```bash
ps -ef | rg -n "examples/pap|vllm serve|multi_pap_proxy_server|pap_attention_executor|vllm bench|benchmark_serving"
```

Expected: no active service processes except the `rg` command.

- [ ] **Step 2: Run a tiny PAP smoke benchmark**

Run:

```bash
PAP_PREFILL_GPU_MEMORY_UTILIZATION=0.45 \
PAP_PROJECTION_GPU_MEMORY_UTILIZATION=0.80 \
bash /home/fei/research/PD/test/baseline/run_benchmark.sh \
  --mode pap \
  --topology 6pa2p \
  --input-lens 1024 \
  --output-lens 16 \
  --qps 1 \
  --num-prompts 4
```

Expected: service starts, benchmark JSON is written under
`/home/fei/research/PD/test/baseline/pap/results/runs/<run-id>`.

- [ ] **Step 3: If launch fails, debug by logs**

Inspect:

```bash
latest=$(ls -1dt /home/fei/research/PD/test/baseline/pap/results/runs/* | head -n1)
find "$latest/service_logs" -maxdepth 1 -type f -print
tail -n 120 "$latest/service_logs/proxy.log"
tail -n 120 "$latest/service_logs/prefill_0.log"
tail -n 120 "$latest/service_logs/projection_0.log"
tail -n 120 "$latest/service_logs/attention_0.log"
```

Use systematic debugging before changing code.

## Task 4: Add PAP Mode Guard

- [ ] **Step 1: Write failing tests**

Create `tests/pap/test_pap_mode.py`:

```python
import pytest

from vllm.pap.mode import PAPMode, parse_pap_mode, is_debug_remote_attention


def test_parse_pap_mode_defaults_to_debug_remote_attention() -> None:
    assert parse_pap_mode(None) is PAPMode.DEBUG_REMOTE_ATTENTION


def test_parse_pap_mode_accepts_true_split() -> None:
    assert parse_pap_mode("true_split") is PAPMode.TRUE_SPLIT


def test_parse_pap_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unsupported PAP mode"):
        parse_pap_mode("shadow-but-fast")


def test_debug_remote_attention_helper() -> None:
    assert is_debug_remote_attention("debug_remote_attention")
    assert not is_debug_remote_attention("true_split")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_mode.py -v
```

Expected: FAIL because `vllm.pap.mode` does not exist.

- [ ] **Step 3: Implement mode helper**

Create `vllm/pap/mode.py`:

```python
from __future__ import annotations

from enum import StrEnum


class PAPMode(StrEnum):
    DEBUG_REMOTE_ATTENTION = "debug_remote_attention"
    TRUE_SPLIT = "true_split"


def parse_pap_mode(value: str | None) -> PAPMode:
    if value is None or value == "":
        return PAPMode.DEBUG_REMOTE_ATTENTION
    try:
        return PAPMode(value)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in PAPMode)
        raise ValueError(f"unsupported PAP mode {value!r}; supported: {supported}") from exc


def is_debug_remote_attention(value: str | None) -> bool:
    return parse_pap_mode(value) is PAPMode.DEBUG_REMOTE_ATTENTION
```

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_mode.py -v
```

Expected: PASS.

## Task 5: Start True-Split Interfaces

- [ ] **Step 1: Write interface tests for Attention session lifecycle**
- [ ] **Step 2: Implement a CPU/no-op lifecycle object that tracks sessions and block ownership**
- [ ] **Step 3: Add tests for Qwen3 PAP true-split path refusing to fall back to local attention**
- [ ] **Step 4: Wire config so `pap_mode=true_split` is separate from debug remote attention**

This task deliberately starts with contracts and no performance claims. The first
true-split milestone is correctness: Projection does not call local decode
attention and Attention owns decode KV.

## Task 6: Run Comparable Benchmarks

- [ ] **Step 1: Run PD 6P2D workload or collect existing authoritative JSON**
- [ ] **Step 2: Run PAP 6PA2P on the same workload**
- [ ] **Step 3: Summarize TTFT, TPOT, ITL, request throughput, token throughput**
- [ ] **Step 4: If PAP is slower, separate debug-path overhead from architectural bottlenecks**

Do not claim PAP beats PD unless current JSON results prove it on the same
workload and model.
