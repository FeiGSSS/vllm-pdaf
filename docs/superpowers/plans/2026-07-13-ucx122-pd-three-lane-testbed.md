# UCX 1.22 PD Three-Lane Test Bed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make repo-local UCX 1.22 the fail-closed default for same-node NIXL
GET and produce one repeatable comparison of PD-oneway, PD-twoway, and PAP.

**Architecture:** Both PD lanes share `NixlConnector`, the official multi-turn
proxy, one runner, and one auditor; only `bidirectional_kv_xfer` and its required
threshold differ. A repo-local UCX/NIXL runtime helper supplies and verifies the
same UCX 1.22 libraries for both PD lanes. The existing aggregator remains the
source of per-lane statistics and a new matrix function composes three pairwise
comparisons into one report.

**Tech Stack:** Bash, UCX 1.22.0, NIXL 1.3.0, CUDA IPC, vLLM V1,
Prometheus text metrics, Python 3.12, pytest.

## Global Constraints

- Work in `/home/fei/research/PD/vllm-pap` on `feature/pap`.
- Use `.venv/bin/python`; never use system Python or bare pip.
- Do not access Hugging Face; use `/data/ssd1/llm-models/Qwen3-8B`.
- Never touch GPU0; experiments use GPU1 and GPU2 serially.
- Do not scan MPS; PAP remains 70/30.
- Keep `UCX_PROTO_EMULATION_ENABLE=n` and fail closed on fallback.
- Do not modify NIXL scheduler/worker algorithms or the PAP hot path.
- Do not run pre-commit and do not commit without a later explicit approval.
- Preserve unrelated untracked files and results.

---

### Task 1: Return D-side metadata in the terminal chat stream chunk

**Files:**
- Modify: `vllm/entrypoints/openai/chat_completion/protocol.py`
- Modify: `vllm/entrypoints/openai/chat_completion/serving.py`
- Create: `tests/entrypoints/openai/chat_completion/test_streaming_kv_transfer_params.py`

**Interfaces:**
- Consumes: `RequestOutput.kv_transfer_params` from the completed D request.
- Produces: optional `ChatCompletionStreamResponse.kv_transfer_params`, present
  only on a chunk whose choice has a non-null `finish_reason`.

- [x] **Step 1: Add a failing streaming regression test**

```python
assert terminal_chunks[0]["kv_transfer_params"] == {
    "remote_engine_id": "decode-engine",
    "remote_block_ids": [[1, 2]],
}
```

- [x] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/entrypoints/openai/chat_completion/test_streaming_kv_transfer_params.py -q
```

Expected before implementation: `KeyError: 'kv_transfer_params'`.

- [x] **Step 3: Add the optional protocol field and terminal-only assignment**

```python
kv_transfer_params: dict[str, Any] | None = None
```

```python
kv_transfer_params=(
    res.kv_transfer_params if choice_data.finish_reason is not None else None
)
```

- [x] **Step 4: Verify GREEN**

Expected: `1 passed`.

### Task 2: Add repo-local UCX 1.22 installation and runtime verification

**Files:**
- Modify: `.gitignore`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/ucx122_runtime_env.sh`
- Create: `tests/benchmarks/test_ucx122_runtime_contract.py`

**Interfaces:**
- Consumes: optional `PAP_UCX122_ROOT`, `PAP_NIXL_UCX122_ROOT`,
  `PAP_UCX122_SOURCE_DIR`, and `PAP_NIXL130_SOURCE_DIR`.
- Produces: `configure_ucx122_runtime` and `verify_ucx122_runtime` Bash
  functions plus `.local/ucx-1.22` and `.local/nixl-ucx122` artifacts.

- [x] **Step 1: Write contract tests that require pinned versions and paths**

```python
assert 'UCX_VERSION="1.22.0"' in setup_text
assert 'NIXL_VERSION="1.3.0"' in setup_text
assert "UCX_PROTO_EMULATION_ENABLE=n" in runtime_text
assert "libplugin_UCX.so" in runtime_text
assert ".local/" in gitignore_text
```

- [x] **Step 2: Run the contract test and verify it fails because scripts are missing**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_ucx122_runtime_contract.py -q
```

- [x] **Step 3: Implement the runtime helper**

The helper must export exactly:

```bash
UCX_TLS=cuda_ipc,cuda_copy,tcp
UCX_PROTO_EMULATION_ENABLE=n
UCX_MODULE_DIR="${PAP_UCX122_ROOT}/lib/ucx"
NIXL_PLUGIN_DIR="${PAP_NIXL_UCX122_ROOT}/src/plugins/ucx"
LD_LIBRARY_PATH="${PAP_UCX122_ROOT}/lib:${PAP_UCX122_ROOT}/lib/ucx:${LD_LIBRARY_PATH:-}"
```

`verify_ucx122_runtime` must require UCX `1.22.0`, resolve the plugin's UCX
libraries under `PAP_UCX122_ROOT`, and instantiate a NIXL UCX agent using
`.venv/bin/python`.

- [x] **Step 4: Implement idempotent `install` and read-only `verify` commands**

The install command must configure UCX with:

```text
--enable-shared --disable-static --enable-cma --enable-devel-headers
--with-cuda=/usr --without-verbs --without-rdmacm --without-gdrcopy
```

It must configure the NIXL 1.3.0 build against that UCX prefix and build the
`plugin_UCX` target. Missing sources are downloaded only by explicit `install`;
benchmark runners call only `verify`.

- [x] **Step 5: Verify contracts, shell syntax, and the persistent runtime**

```bash
.venv/bin/python -m pytest tests/benchmarks/test_ucx122_runtime_contract.py -q
bash -n .claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh
bash -n .claude/skills/vllm-pap-benchmark/scripts/ucx122_runtime_env.sh
bash .claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh verify
```

### Task 3: Parameterize the PD runner and make the auditor mode-aware

**Files:**
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh`
- Modify: `benchmarks/multi_turn/pd_multiturn_load_reuse_metrics.py`
- Modify: `tests/benchmarks/test_pd_multiturn_load_reuse_metrics.py`
- Create: `tests/benchmarks/test_pd_multiturn_runner_contract.py`

**Interfaces:**
- Consumes: positional `oneway|twoway` or `PD_LOAD_TRANSFER_MODE`, default
  `oneway`; the UCX runtime functions from Task 2.
- Produces: result fields `pd_transfer_mode`, `bidirectional_kv_xfer`, and
  direction-specific `nixl_transfers`; mode-aware cache validation.

- [x] **Step 1: Add failing auditor tests for both modes**

```python
oneway = validate_pd_multiturn_load_reuse(..., transfer_mode="oneway")
assert oneway["proxy_cache"]["hits"] == 0
twoway = validate_pd_multiturn_load_reuse(..., transfer_mode="twoway")
assert twoway["proxy_cache"]["hits"] == conversations * 4
assert twoway["nixl_transfers"]["d_to_p"]["transfer_count"] > 0
```

Also require rejection when effective config, proxy hit/miss counts, or
directional transfer counts contradict the selected mode.

- [x] **Step 2: Add a failing runner contract test**

```python
assert "NixlConnector" in runner_text
assert "NixlPushConnector" not in runner_text
assert "bidirectional_kv_xfer" in runner_text
assert "configure_ucx122_runtime" in runner_text
assert "disagg_proxy_multiturn.py" in runner_text
```

- [x] **Step 3: Run both test files and verify RED**

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_pd_multiturn_load_reuse_metrics.py \
  tests/benchmarks/test_pd_multiturn_runner_contract.py -q
```

- [x] **Step 4: Implement one shared NixlConnector runner**

For `oneway`, emit `bidirectional_kv_xfer=false`. For `twoway`, emit:

```json
{
  "bidirectional_kv_xfer": true,
  "kv_recompute_threshold": 0,
  "decoder_kv_blocks_ttl": 480,
  "enable_cross_layers_blocks": "True"
}
```

Use the official multi-turn proxy for both modes. Record UCX version, plugin
path, emulation policy, protocol log setting, and complete KV configs.

- [x] **Step 5: Implement mode-aware metrics and proxy validation**

Keep prompt-source conservation common. Parse P-side metrics as D→P and D-side
metrics as P→D. Oneway requires P transfer count zero and all proxy requests
MISS. Twoway requires one MISS plus four HITs per conversation and positive
D→P transfers.

- [x] **Step 6: Run tests and shell syntax checks**

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_pd_multiturn_load_reuse_metrics.py \
  tests/benchmarks/test_pd_multiturn_runner_contract.py -q
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_load.sh
```

### Task 4: Add three-lane aggregation, report, and orchestration

**Files:**
- Modify: `benchmarks/multi_turn/compare_pap_pd_multiturn_load.py`
- Modify: `tests/benchmarks/test_compare_pap_pd_multiturn_load.py`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh`
- Create: `tests/benchmarks/test_pd_three_lane_testbed_contract.py`

**Interfaces:**
- Consumes: three valid aggregates named `pd_oneway`, `pd_twoway`, and `pap`.
- Produces: `compare_three_aggregates(...)`, a `compare-three` CLI command,
  `comparison.json`, and one Markdown matrix.

- [x] **Step 1: Add failing three-way comparison tests**

```python
matrix = compare_three_aggregates(oneway, twoway, pap)
assert matrix["ratios"]["pd_twoway_over_pd_oneway"]
assert matrix["ratios"]["pap_over_pd_oneway"]
assert matrix["ratios"]["pap_over_pd_twoway"]
```

Require profile, hardware, repetition mode, prompt shape, and prompt digest
parity across all three aggregates.

- [x] **Step 2: Add a failing orchestrator contract test**

```python
assert "pd-oneway" in script_text
assert "pd-twoway" in script_text
assert "PD-oneway → PD-twoway → PAP" not in script_text
assert "pd_oneway pd_twoway pap pd_twoway pap pd_oneway" in normalized_text
assert "compare-three" in script_text
```

- [x] **Step 3: Run comparison and contract tests to verify RED**

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_compare_pap_pd_multiturn_load.py \
  tests/benchmarks/test_pd_three_lane_testbed_contract.py -q
```

- [x] **Step 4: Implement `compare_three_aggregates` and Markdown rendering**

Reuse `compare_aggregates` for each pair and expose absolute medians plus the
three requested ratios for Round 1 and steady Round 2–5 TTFT/TPOT/latency.

- [x] **Step 5: Implement quick and Latin-square formal orchestration**

Quick order:

```text
pd_oneway pd_twoway pap
```

Formal order:

```text
pd_oneway pd_twoway pap
pd_twoway pap pd_oneway
pap pd_oneway pd_twoway
```

Create separate lane directories and aggregate each lane before calling
`compare-three`.

- [x] **Step 6: Run tests and syntax checks**

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_compare_pap_pd_multiturn_load.py \
  tests/benchmarks/test_pd_three_lane_testbed_contract.py -q
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh
```

### Task 5: Run staged GPU validation and archive evidence

**Files:**
- Modify: `benchmarks/pap/experiments/legacy/reports/pd-same-node-nixl-transfer-root-cause-20260713.md`
- Create: `benchmarks/pap/experiments/legacy/reports/pd-oneway-twoway-pap-five-turn-results-20260713.md`
- Modify: `benchmarks/pap/experiments/HISTORY.md`
- Modify: `.codex/skills/vllm-pap-benchmark/SKILL.md`

**Interfaces:**
- Consumes: valid C2 quick, C4 quick, and C4 formal result roots.
- Produces: durable root-cause explanation, result report, history index entry,
  and updated benchmark instructions.

- [x] **Step 1: Install/verify UCX 1.22 and run NIXL agent smoke**

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh install
bash .claude/skills/vllm-pap-benchmark/scripts/setup_ucx122_nixl.sh verify
```

- [x] **Step 2: Run C2 quick**

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh quick c2
```

Require all three lanes valid before continuing.

- [x] **Step 3: Run C4 quick**

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh quick c4
```

- [x] **Step 4: Run C4 formal**

```bash
bash .claude/skills/vllm-pap-benchmark/scripts/run_pd_pap_multiturn_load.sh formal c4
```

- [x] **Step 5: Write the root-cause and experiment records from actual artifacts**

The root-cause record must include UCX 1.21 default versus UCX 1.22 strict
throughput and protocol evidence. The experiment report must list every result
root, effective config, validity gate, repetition value, aggregate median, ratio,
and output-digest warning. The history index must link both documents.

- [x] **Step 6: Run final verification without pre-commit**

```bash
.venv/bin/python -m pytest \
  tests/entrypoints/openai/chat_completion/test_streaming_kv_transfer_params.py \
  tests/benchmarks/test_ucx122_runtime_contract.py \
  tests/benchmarks/test_pd_multiturn_load_reuse_metrics.py \
  tests/benchmarks/test_pd_multiturn_runner_contract.py \
  tests/benchmarks/test_compare_pap_pd_multiturn_load.py \
  tests/benchmarks/test_pd_three_lane_testbed_contract.py -q
git diff --check
```

Review `effective_config.env`, correctness audits, transfer metrics, proxy logs,
session drain, aggregate JSON, report Markdown, GPU cleanup, and tracked diff.
