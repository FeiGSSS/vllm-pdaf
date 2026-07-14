# PAP Attention Runtime Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vllm.pap.attention_executor` the authoritative PAP Attention
runtime without changing runtime behavior.

**Architecture:** Move the existing executor intact into the installed `vllm`
package. Keep the old example path as a thin script launcher, and make all
repository launchers and tests consume the packaged module.

**Tech Stack:** Python 3.12, PyTorch/CUDA, FastAPI, pytest, Bash.

## Global Constraints

- Do not split the 5,918-line executor in this phase.
- Do not change protocols, scheduling, kernels, transports, environment
  variables, or defaults.
- Use `.venv/bin/python`; never use system Python.
- Do not run pre-commit.
- Preserve all unrelated untracked benchmark output.
- Produce one behavior-preserving structural commit with an `Assisted-by:`
  trailer.

---

### Task 1: Add the Package-Boundary Contract

**Files:**

- Modify: `tests/pap/test_pap_launch_files.py`
- Inspect: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`

**Interfaces:**

- Consumes: repository-relative `ROOT` already defined by the test module.
- Produces: a contract requiring the packaged executor, compatibility launcher,
  and module-based process startup.

- [ ] **Step 1: Write the failing architecture test**

Add this test to `tests/pap/test_pap_launch_files.py`:

```python
def test_pap_attention_runtime_is_packaged() -> None:
    runtime = ROOT / "vllm" / "pap" / "attention_executor.py"
    compatibility_launcher = (
        ROOT / "examples" / "pap" / "pap_attention_executor.py"
    )
    launcher = ROOT / "examples" / "pap" / "launch_pap_nixl.sh"
    testbed = (
        ROOT
        / ".claude"
        / "skills"
        / "vllm-pap-benchmark"
        / "scripts"
        / "run_pap_same_pd_workload.sh"
    )

    assert runtime.is_file()
    compatibility_text = compatibility_launcher.read_text()
    assert "from vllm.pap.attention_executor import main" in compatibility_text
    assert len(compatibility_text.splitlines()) < 20
    assert "-m vllm.pap.attention_executor" in launcher.read_text()
    assert "-m vllm.pap.attention_executor" in testbed.read_text()
```

- [ ] **Step 2: Run the test and verify the old layout fails**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_launch_files.py::test_pap_attention_runtime_is_packaged -q
```

Expected: `FAILED` because `vllm/pap/attention_executor.py` does not exist.

### Task 2: Relocate the Runtime and Preserve Script Compatibility

**Files:**

- Create by rename: `vllm/pap/attention_executor.py`
- Replace: `examples/pap/pap_attention_executor.py`

**Interfaces:**

- Consumes: all existing public and test-consumed names from the executor.
- Produces: `vllm.pap.attention_executor.main() -> None` and the unchanged
  executor module API.

- [ ] **Step 1: Move the complete executor implementation**

Move the existing file to `vllm/pap/attention_executor.py` without changing its
classes, functions, globals, or initialization behavior.

- [ ] **Step 2: Give the packaged runtime an explicit entry function**

Replace the current bottom-level script block with:

```python
def main() -> None:
    """Run the PAP Attention executor service."""
    import uvicorn

    args = parse_args()
    if args.tcp_port is not None:
        start_attention_tcp_server(
            app.state.registry,
            host=args.host,
            port=args.tcp_port,
            app=app,
        )
    if args.offload_exec_zmq_port is not None:
        logger.info(
            "PAP Attention OFFLOAD_EXEC ZMQ endpoint reserved at %s:%d",
            args.host,
            args.offload_exec_zmq_port,
        )
    maybe_start_offload_exec_transport(
        app=app,
        host=args.host,
        zmq_port=args.offload_exec_zmq_port,
    )
    write_runtime_cuda_context_audit(role="attention")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add the old-path compatibility launcher**

Replace `examples/pap/pap_attention_executor.py` with:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility launcher for the packaged PAP Attention runtime."""

from vllm.pap.attention_executor import main


if __name__ == "__main__":
    main()
```

### Task 3: Redirect Repository Consumers

**Files:**

- Modify: `examples/pap/launch_pap_nixl.sh`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`
- Modify: `tests/pap/test_pap_attention_executor.py`
- Modify: `tests/pap/test_pap_launch_files.py`

**Interfaces:**

- Consumes: `vllm.pap.attention_executor` from Task 2.
- Produces: launch and test paths that no longer treat `examples` as runtime
  package code.

- [ ] **Step 1: Switch process startup to module execution**

Replace every Attention process command of this form:

```bash
python examples/pap/pap_attention_executor.py
```

with its existing Python executable followed by:

```bash
-m vllm.pap.attention_executor
```

Keep every environment assignment and command-line argument unchanged.

- [ ] **Step 2: Redirect test imports mechanically**

In `tests/pap/test_pap_attention_executor.py`, replace only these prefixes:

```python
from examples.pap.pap_attention_executor import
from examples.pap import pap_attention_executor
```

with:

```python
from vllm.pap.attention_executor import
from vllm.pap import attention_executor as pap_attention_executor
```

Where the old import already used `as executor_module`, preserve that alias
instead. Keep every local imported name and monkeypatch target unchanged.

- [ ] **Step 3: Update the existing launcher assertion**

Replace:

```python
assert "pap_attention_executor.py" in text
```

with:

```python
assert "-m vllm.pap.attention_executor" in text
```

- [ ] **Step 4: Run the architecture test and focused tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_launch_files.py::test_pap_attention_runtime_is_packaged \
  tests/pap/test_pap_attention_executor.py -q
```

Expected: all selected tests pass.

### Task 4: Verify Runtime Equivalence and Commit

**Files:**

- Verify: all files changed by Tasks 1-3.
- Preserve: untracked benchmark result directories and local notes.

**Interfaces:**

- Consumes: packaged executor and redirected consumers.
- Produces: one tested structural commit.

- [ ] **Step 1: Compile and syntax-check entry paths**

Run:

```bash
.venv/bin/python -m py_compile \
  vllm/pap/attention_executor.py \
  examples/pap/pap_attention_executor.py
bash -n examples/pap/launch_pap_nixl.sh
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
```

Expected: each command exits zero with no output.

- [ ] **Step 2: Run the established combined PAP regression suite**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_data_plane.py \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_contract.py \
  tests/pap/test_pap_launch_files.py \
  tests/benchmarks/test_pap_pd_multiturn_client.py \
  tests/benchmarks/test_pap_pd_multiturn_load_client.py -q
```

Expected: the suite passes with no failures.

- [ ] **Step 3: Run a C1 quick startup/correctness smoke if GPUs are idle**

Use the fixed testbed with the same C1 quick settings as the latest sealed KV
smoke, changing only `RUN_ID` to
`20260714_attention_runtime_relocation_c1_quick`.

Expected: services start through `-m vllm.pap.attention_executor`, all requests
and turn transitions complete, and correctness/session-drain gates pass.

- [ ] **Step 4: Check the final diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the planned tracked files are modified,
while pre-existing raw results remain untracked.

- [ ] **Step 5: Commit the structural change**

Stage only the planned files and commit with:

```text
Move PAP Attention runtime into vllm package

Relocate the authoritative PAP Attention executor from examples into the
installed vllm package, retain a thin compatibility launcher, and redirect
launchers and tests to the packaged module without changing runtime behavior.

Assisted-by: Codex
```
