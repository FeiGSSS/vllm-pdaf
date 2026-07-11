# PAP/PD Multi-turn North-star Test Bed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed 1P1D-PD versus 1PA1P-PAP two-turn 16K test bed that
reports TTFT/TPOT, compares against tracked references, and fails closed on cache
or lifecycle errors.

**Architecture:** Add one deterministic streaming client and one pure comparison/
reference module. Reuse the existing self-contained PAP lifecycle runner through a
new client mode, then wrap it with a serial quick/formal orchestrator. Keep official
PD, NIXL, and the official multi-turn proxy unchanged; a separate one-time bootstrap
script only launches them and invokes the common client.

**Tech Stack:** Python 3.12 via `.venv/bin/python`, pytest, httpx, Transformers local
tokenizer, Bash, vLLM OpenAI-compatible streaming API, NIXL, JSON and Markdown.

## Global Constraints

- Work on `feature/pap`; preserve all unrelated tracked and untracked files.
- Never use system `python3`, bare `python`, `pip`, or `pip install`.
- Do not run pre-commit. Commits use `git commit --no-verify` and include
  `Assisted-by: OpenAI Codex`.
- Do not modify PD, NIXL, or
  `examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py`.
- Use only local model `/data/ssd1/llm-models/Qwen3-8B` and local corpus
  `/home/fei/research/PD/refer_codes/vllm/benchmarks/sonnet_4x.txt`.
- Freeze profile `qwen3_8b_chat_16k_2turn_o256_c1_v1`: 16,000 document tokens,
  120-token append, two 256-token outputs, thinking on, concurrency one.
- Freeze FP16, TP1, eager, max model length 20,000, max batched tokens 4,096,
  max sequences 2, GPU 1/2, and PAP MPS 70/30.
- Performance runs set `PAP_PREFIX_CACHE_AUDIT=0` and read SSE through HTTP EOF.
- Quick is one repetition and diagnostic only. Formal is three serial repetitions.
- Formal improvement requires valid gates and round-two TPOT median at least 3%
  below the tracked PAP reference.

---

## File Structure

- `benchmarks/multi_turn/pap_pd_multiturn_client.py`: fixed workload, streaming
  parser, cache-LCP validation, per-repetition result.
- `benchmarks/multi_turn/compare_pap_pd_multiturn.py`: result validation,
  aggregation, classification, report rendering, explicit reference writes.
- `tests/benchmarks/test_pap_pd_multiturn_client.py`: client and metric unit tests.
- `tests/benchmarks/test_compare_pap_pd_multiturn.py`: comparator/reference tests.
- `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`: add the
  north-star client mode without duplicating service lifecycle code.
- `.claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh`: quick/
  formal PAP orchestration and report generation.
- `.claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh`:
  one-time official PD launcher/reference candidate generation.
- `tests/pap/test_pap_launch_files.py`: static runner safety and frozen-contract
  tests.
- `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/`: tracked
  profile and PD/PAP references.
- `test/baseline/pap/README.md`: entry commands and artifact policy.

### Task 1: Deterministic Streaming Client

**Files:**
- Create: `tests/benchmarks/test_pap_pd_multiturn_client.py`
- Create: `benchmarks/multi_turn/pap_pd_multiturn_client.py`

**Interfaces:**
- Produces:

  ```python
  calculate_tpot_ms(latency_ms: float, ttft_ms: float,
                    completion_tokens: int) -> float
  parse_prefill_headers(headers: Mapping[str, str]) -> dict[str, int | None]
  block_aligned_prefix_metrics(first_prompt_ids: Sequence[int],
                               first_output_ids: Sequence[int],
                               second_prompt_ids: Sequence[int],
                               block_size: int = 16) -> dict[str, int]
  profile_fingerprint(profile: Mapping[str, object]) -> str
  run_two_turn(args: argparse.Namespace) -> dict[str, object]
  ```

- CLI writes one `result.json` and accepts `--base-url`, `--model`, `--corpus`,
  `--result`, `--architecture`, `--topology`, `--conversation-id`, and the frozen
  engine/profile fields required for fingerprinting.

- [ ] **Step 1: Write failing metric and LCP tests**

  ```python
  def test_tpot_excludes_first_token():
      assert calculate_tpot_ms(110.0, 10.0, 5) == 25.0

  def test_lcp_excludes_unmaterialized_final_sample():
      metrics = block_aligned_prefix_metrics(
          list(range(32)), list(range(100, 133)),
          [*range(32), *range(100, 132), 999], block_size=16,
      )
      assert metrics["decode_derived_hit_tokens"] == 32
      assert metrics["expected_cached_tokens"] == 64
  ```

- [ ] **Step 2: Run RED tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_pap_pd_multiturn_client.py -q
  ```

  Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement pure metric, header, digest and fingerprint helpers**

  Use canonical JSON (`sort_keys=True`, compact separators) and SHA-256. Reject
  non-finite/negative latency, absent token IDs, accounting inconsistencies, and
  unsupported block sizes with `ValueError`.

- [ ] **Step 4: Add failing streaming parser tests**

  Feed a fake response with an initial `prompt_token_ids` chunk, token chunks,
  terminal finish chunk, usage-only chunk, `[DONE]`, and one sentinel line after
  `[DONE]`. Assert all token IDs are captured, TTFT starts at the first output token,
  usage wins over chunk count, and the sentinel is consumed.

- [ ] **Step 5: Implement SSE consumption through EOF**

  The loop records `[DONE]` but continues iterating until the response iterator
  ends. Set `return_token_ids=true` and `stream_options.include_usage=true`; fail if
  the token-ID count differs from usage completion tokens.

- [ ] **Step 6: Add failing two-turn workload tests**

  Use a fake tokenizer/client to assert the first message uses corpus tokens
  `[0:16000]`, the append uses `[16000:16120]`, both requests share conversation ID
  and cache salt, the second contains the first assistant text, and thinking is off.

- [ ] **Step 7: Implement `run_two_turn` and atomic result writing**

  Store per-round prompt/output token digests, TTFT/TPOT/latency, actual prefill
  headers, finish reason, and the true retokenized LCP. For PAP require actual
  cached tokens to equal the expected block-aligned LCP and require at least one
  decode-derived block.

- [ ] **Step 8: Run GREEN tests and focused existing chat tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_pap_pd_multiturn_client.py \
    tests/pap/test_pap_multiturn_chat_prefix_cache.py -q
  ```

  Expected: all pass.

- [ ] **Step 9: Commit**

  ```bash
  git add benchmarks/multi_turn/pap_pd_multiturn_client.py \
    tests/benchmarks/test_pap_pd_multiturn_client.py
  git commit --no-verify -m "Add PAP PD multi-turn north-star client" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 2: Comparator and Explicit Reference Management

**Files:**
- Create: `tests/benchmarks/test_compare_pap_pd_multiturn.py`
- Create: `benchmarks/multi_turn/compare_pap_pd_multiturn.py`

**Interfaces:**
- Produces:

  ```python
  validate_repetition(result: Mapping[str, object]) -> None
  aggregate_repetitions(results: Sequence[Mapping[str, object]]) -> dict[str, object]
  classify_tpot(candidate_ms: float, reference_ms: float,
                threshold: float = 0.03) -> str
  compare_candidate(candidate: Mapping[str, object],
                    pd_reference: Mapping[str, object],
                    pap_reference: Mapping[str, object]) -> dict[str, object]
  render_markdown(comparison: Mapping[str, object]) -> str
  write_reference_atomic(path: Path, reference: Mapping[str, object]) -> None
  ```

- CLI subcommands: `aggregate`, `compare`, and `write-reference`. `compare` never
  writes either reference. `write-reference` requires `--allow-reference-write`.

- [ ] **Step 1: Write failing validation and median tests**

  Cover two valid rounds, missing token IDs, wrong output length, failed cache gate,
  mixed fingerprints, one-repetition diagnostic aggregation and three-repetition
  median aggregation.

- [ ] **Step 2: Run RED tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_compare_pap_pd_multiturn.py -q
  ```

  Expected: module-not-found failure.

- [ ] **Step 3: Implement validation and aggregation**

  Aggregate round-one/round-two TTFT, TPOT and latency plus conversation latency.
  Preserve raw repetition metrics and use `statistics.median` for formal values.
  A failed validity gate raises instead of dropping the bad repetition.

- [ ] **Step 4: Add failing 3% boundary and report tests**

  Assert `96.999/100 -> improved`, `97/100 -> improved`, `103/100 -> regressed`,
  and values inside the interval are neutral. Assert Markdown contains per-round PD,
  PAP reference, candidate, both ratios, target status and warnings.

- [ ] **Step 5: Implement compare and Markdown/JSON output**

  Require identical profile fingerprints and hardware signatures. Quick candidates
  always classify as `diagnostic`; formal candidates use round-two TPOT. TTFT and
  conversation regressions above 3% become warnings.

- [ ] **Step 6: Add failing explicit-write tests and implement atomic promotion**

  `write-reference` without `--allow-reference-write` exits nonzero. With the flag,
  write a temporary sibling, `fsync`, and `replace`; reject non-formal or invalid
  aggregates.

- [ ] **Step 7: Run GREEN tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_compare_pap_pd_multiturn.py -q
  ```

  Expected: all pass.

- [ ] **Step 8: Commit**

  ```bash
  git add benchmarks/multi_turn/compare_pap_pd_multiturn.py \
    tests/benchmarks/test_compare_pap_pd_multiturn.py
  git commit --no-verify -m "Add multi-turn north-star comparison" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 3: Integrate the Fixed Client into the PAP Lifecycle

**Files:**
- Modify: `tests/pap/test_pap_launch_files.py`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`

**Interfaces:**
- New mode: `PAP_BENCH_CLIENT_MODE=multiturn_north_star`.
- New recorded setting: `PAP_VLLM_DTYPE`, default `auto`; north-star wrapper passes
  `float16`.
- Produces `${RUN_ROOT}/result.json` plus existing audit artifacts.

- [ ] **Step 1: Write failing static runner contract tests**

  Assert the mode is allow-listed, invokes the repo client, passes architecture/
  topology/frozen engine settings, refuses non-1PA1P, forces prefix audit off,
  enables prompt token details, passes dtype to both vLLM servers, and still calls
  session drain before correctness audit.

- [ ] **Step 2: Run RED test**

  ```bash
  .venv/bin/python -m pytest \
    tests/pap/test_pap_launch_files.py::test_pap_runner_supports_multiturn_north_star -q
  ```

  Expected: failure because the mode is absent.

- [ ] **Step 3: Add the minimal runner mode**

  Extend the existing case statement and client dispatch. Keep all service startup,
  direct-QKV, batched route, mailbox, ACK/lease and cleanup defaults unchanged.
  Add `--dtype "${PAP_VLLM_DTYPE}"` to Prefill and Projection and record it.

- [ ] **Step 4: Add result validation**

  Use `.venv/bin/python` to require `validity.status=passed`, exactly two rounds,
  256 tokens/round, PAP cache evidence passed, and the expected fingerprint fields.

- [ ] **Step 5: Run static tests and shell syntax**

  ```bash
  bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
  .venv/bin/python -m pytest tests/pap/test_pap_launch_files.py -q
  ```

  Expected: all pass.

- [ ] **Step 6: Commit**

  ```bash
  git add .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh \
    tests/pap/test_pap_launch_files.py
  git commit --no-verify -m "Integrate PAP multi-turn north-star mode" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 4: Quick/Formal Orchestration and PD Bootstrap

**Files:**
- Modify: `tests/pap/test_pap_launch_files.py`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh`
- Modify: `test/baseline/pap/README.md`

**Interfaces:**
- `bash .../run_multiturn_north_star.sh quick`
- `bash .../run_multiturn_north_star.sh formal`
- `bash .../bootstrap_pd_multiturn_reference.sh`
- PD bootstrap writes its aggregate candidate to
  `/tmp/pap_pd_multiturn_reference_candidate.json`.
- PAP formal writes its aggregate candidate to
  `/tmp/pap_multiturn_reference_candidate.json`; if the initial PAP reference is
  absent, comparison status is `uninitialized` rather than a fabricated verdict.

- [ ] **Step 1: Write failing orchestration safety tests**

  Assert quick/formal counts are 1/3, repetitions are a normal serial loop, GPU
  1/2 are checked but never killed, fixed profile parameters are exported, formal
  requires a clean tracked tree, the comparer is always called, and neither script
  contains `pkill`, bare Python/pip or writes a reference implicitly.

- [ ] **Step 2: Run RED tests**

  ```bash
  .venv/bin/python -m pytest tests/pap/test_pap_launch_files.py -q
  ```

  Expected: new script-existence assertions fail.

- [ ] **Step 3: Implement the PAP orchestrator**

  Check GPU 1/2 process occupancy with `nvidia-smi`, allocate deterministic disjoint
  port blocks, call the existing runner once per repetition, collect result paths,
  aggregate, and compare against tracked PD/PAP references. Preserve failed run
  directories and return nonzero.

- [ ] **Step 4: Implement the one-time official PD bootstrap**

  Launch the unchanged official multi-turn proxy and producer/consumer vLLM
  services in scoped process groups. Use the same model/profile/GPUs/FP16 settings,
  run three serial repetitions with full restarts, audit official logs/metrics for
  bidirectional cache reuse, then emit a PD reference candidate. Never call the
  reference-write command automatically.

- [ ] **Step 5: Document exact commands and artifact locations**

  Add quick, formal, PD bootstrap, compare and explicit promotion commands plus the
  meaning of `diagnostic`, `improved`, `neutral`, `regressed`, and `invalid`.

- [ ] **Step 6: Run GREEN static tests and syntax checks**

  ```bash
  bash -n .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh
  bash -n \
    .claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh
  .venv/bin/python -m pytest tests/pap/test_pap_launch_files.py -q
  ```

  Expected: all pass.

- [ ] **Step 7: Commit**

  ```bash
  git add .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh \
    .claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh \
    tests/pap/test_pap_launch_files.py test/baseline/pap/README.md
  git commit --no-verify -m "Add multi-turn north-star test bed" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 5: Live Validation and Reference Bootstrap

**Files:**
- Create:
  `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/profile.json`
- Create:
  `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/pd_reference.json`
- Create:
  `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/pap_reference.json`
- Create:
  `test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/README.md`

- [ ] **Step 1: Run all CPU/static tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_pap_pd_multiturn_client.py \
    tests/benchmarks/test_compare_pap_pd_multiturn.py \
    tests/pap/test_pap_launch_files.py \
    tests/pap/test_pap_multiturn_chat_prefix_cache.py -q
  ```

  Expected: all pass.

- [ ] **Step 2: Commit code before formal measurement**

  Confirm `git diff --quiet` and `git diff --cached --quiet`. Untracked historical
  artifacts remain untouched.

- [ ] **Step 3: Bootstrap three-run PD candidate**

  ```bash
  bash .claude/skills/vllm-pap-benchmark/scripts/bootstrap_pd_multiturn_reference.sh
  ```

  Expected: three valid results, bidirectional reuse evidence, no fatal logs.

- [ ] **Step 4: Explicitly write the PD reference**

  ```bash
  .venv/bin/python benchmarks/multi_turn/compare_pap_pd_multiturn.py \
    write-reference --architecture pd \
    --aggregate /tmp/pap_pd_multiturn_reference_candidate.json \
    --output test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/pd_reference.json \
    --allow-reference-write
  ```

- [ ] **Step 5: Run three-run PAP formal candidate**

  ```bash
  bash .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh formal
  ```

  Expected: three valid results, Decode-derived hit, active sessions zero, report
  generated.

- [ ] **Step 6: Explicitly write the initial PAP reference**

  ```bash
  .venv/bin/python benchmarks/multi_turn/compare_pap_pd_multiturn.py \
    write-reference --architecture pap \
    --aggregate /tmp/pap_multiturn_reference_candidate.json \
    --output test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1/pap_reference.json \
    --allow-reference-write
  ```

  Create `profile.json` and README from the same validated aggregate metadata.

- [ ] **Step 7: Re-run quick against both tracked references**

  ```bash
  bash .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_north_star.sh quick
  ```

  Expected: `diagnostic`, valid gates, complete PD/PAP ratios, no service leak.

- [ ] **Step 8: Commit references**

  ```bash
  git add test/baseline/pap/references/qwen3_8b_chat_16k_2turn_o256_c1_v1
  git commit --no-verify -m "Establish multi-turn north-star references" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 6: Optimization Handoff

**Files:**
- Modify: `docs/design/pap-experiment-history-index.md`
- Create: a dated tracked experiment note under `test/baseline/pap/docs/`.

- [ ] **Step 1: Record the north-star baseline**

  Link design, plan, code commits, reference artifacts and raw run roots. Report
  each round's TTFT/TPOT and the PAP/PD ratio without averaging TTFT and TPOT into
  one score.

- [ ] **Step 2: Select the first profile lane**

  Use a diagnostic-only trace with the exact profile and classify wall time into
  Prefill, projection compute, remote Attention compute, PA↔P communication, CPU/
  scheduler gaps and queueing. Do not compare trace latency as a performance run.

- [ ] **Step 3: Commit the handoff note**

  ```bash
  git add docs/design/pap-experiment-history-index.md test/baseline/pap/docs
  git commit --no-verify -m "Record multi-turn north-star baseline" \
    -m "Assisted-by: OpenAI Codex"
  ```
