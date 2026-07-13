# PAP MPS 80:20 Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible PAP-only 80:20 MPS diagnostic and measure it against the frozen 70:30 PAP and PD results.

**Architecture:** The PAP-only wrapper maps a named profile to exact Prefill and Attention MPS percentages. The underlying runner validates the name/value tuple and records it; the three-lane formal entry remains on the default profile.

**Tech Stack:** Bash benchmark runners, pytest contract tests, Qwen3-8B on two L20 GPUs.

## Global Constraints

- Do not rerun or modify either frozen PD baseline.
- `baseline_70_30` remains the default and the basis for future TPOT work.
- `diagnostic_80_20` is the only permitted non-default profile.
- Keep the C4 five-turn request trace and all correctness gates unchanged.
- Do not run pre-commit.

---

### Task 1: Add the named diagnostic profile

**Files:**
- Create: `tests/benchmarks/test_pap_multiturn_mps_contract.py`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`

**Interfaces:**
- Consumes: `PAP_LOAD_MPS_PROFILE` with default `baseline_70_30`.
- Produces: exact `PAP_BENCH_MPS_PROFILE`, `PAP_PREFILL_MPS_PERCENT`, and `PAP_ATTENTION_MPS_PERCENT` values in the child runner and `effective_config.env`.

- [ ] **Step 1: Write a failing contract test**

```python
def test_pap_only_runner_has_explicit_80_20_diagnostic_profile() -> None:
    wrapper = PAP_WRAPPER.read_text(encoding="utf-8")
    runner = PAP_RUNNER.read_text(encoding="utf-8")
    assert 'PAP_LOAD_MPS_PROFILE:-baseline_70_30' in wrapper
    assert 'diagnostic_80_20' in wrapper
    assert 'PAP_BENCH_MPS_PROFILE="${MPS_PROFILE}"' in wrapper
    assert 'PAP_BENCH_MPS_PROFILE:-baseline_70_30' in runner
    assert 'PAP_BENCH_MPS_PROFILE=%q' in runner
```

- [ ] **Step 2: Verify the test fails because the profile is absent**

Run: `.venv/bin/python -m pytest tests/benchmarks/test_pap_multiturn_mps_contract.py -v`

Expected: one failed assertion for the missing profile contract.

- [ ] **Step 3: Implement the two exact profile mappings**

In the wrapper, map `baseline_70_30` to `70/30` and
`diagnostic_80_20` to `80/20`, reject all other values, and pass the selected
name and percentages to the child. In the child runner, validate those two
exact tuples for `multiturn_load` and record the name in `effective_config.env`.

- [ ] **Step 4: Verify tests and shell syntax**

Run:

```bash
.venv/bin/python -m pytest tests/benchmarks/test_pap_multiturn_mps_contract.py tests/benchmarks/test_pd_three_lane_testbed_contract.py -v
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
```

Expected: all tests pass and both syntax checks exit 0.

### Task 2: Run and report the diagnostic

**Files:**
- Create: one run directory below `test/baseline/pap/results/runs/`
- Modify: `docs/design/pap-experiment-history-index.md` only if the result is retained
- Create: `docs/design/pap-mps-80-20-diagnostic-results-20260713.md` only after valid measurements exist

**Interfaces:**
- Consumes: `PAP_LOAD_MPS_PROFILE=diagnostic_80_20` and the frozen C4 workload.
- Produces: strict-audited PAP result JSON, aggregate JSON, and a comparison table against frozen metrics.

- [ ] **Step 1: Check GPUs, processes, proxy environment, and UCX runtime**

Run the repository skill's required read-only preflight checks. Do not kill
unrelated processes.

- [ ] **Step 2: Run one C4 quick diagnostic**

Run:

```bash
PAP_LOAD_MPS_PROFILE=diagnostic_80_20 \
PAP_LOAD_RUN_ID=20260713_pap_mps_80_20_c4_quick \
bash .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh quick c4
```

Expected: 20 completed requests, no failures, strict audit passed, and zero
active sessions after drain.

- [ ] **Step 3: Decide whether repetition is warranted**

Compare the quick result with the three frozen PAP 70:30 cells. If the apparent
R1 TTFT or TPOT change exceeds their spread, run the PAP-only `formal c4` entry
with the same profile; otherwise record that the change is within noise.

- [ ] **Step 4: Verify and report**

Read the result, aggregate, effective configuration, correctness audit, routing
audit, and session-drain artifacts. Report all four latency scopes and ratios,
then explicitly restore `baseline_70_30` as the subsequent optimization basis.
