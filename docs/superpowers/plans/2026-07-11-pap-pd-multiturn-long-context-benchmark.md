# PAP/PD Multi-turn Long-context Benchmark Implementation Plan

> **Status: superseded — do not execute.** On 2026-07-12 the project boundary was
> corrected: PD, NIXL, and the official PD proxy remain unmodified; official logs
> and `/metrics` are the only PD observation surfaces. The token-accounting API,
> PD proxy, and standalone OOM-admission implementation were reverted by
> `9fde5ff6d`. This file is retained only for development-history traceability.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate an OOM-safe, reproducible benchmark that compares
`PAP-native`, `PD-oneway`, and `PD-bidir` under identical 1:1, multi-turn,
long-context workloads.

**Architecture:** Expose vLLM's existing local/external prefill-token split at the
OpenAI API, then consume it through one common multi-turn client and result schema.
The existing PAP and PD lifecycle scripts remain the service-launch foundation;
dedicated long-context wrappers freeze safe settings, parse actual KV capacity from
startup logs, and refuse a cell before traffic when its live-token estimate exceeds
70% of capacity. Raw runs stay outside Git, while the manifest, code, audits, and
summary are tracked.

**Tech Stack:** Python 3.12 through `.venv/bin/python`, pytest, httpx/FastAPI,
OpenAI-compatible Completions and Chat Completions, Bash lifecycle scripts, vLLM
V1 scheduler and NIXL connector, NVIDIA L20 GPUs.

## Global Constraints

- Work only on `feature/pap`; preserve unrelated tracked and untracked user files.
- Never use system `python3`, bare `python`, `pip`, or `pip install`; use
  `.venv/bin/python` and `uv` only.
- Do not run pre-commit; commits use `git commit --no-verify` and include an
  AI-assistance trailer.
- Use only `/data/ssd1/llm-models/Qwen3-8B` with `local_files_only=True`; no
  Hugging Face network access.
- The three lanes are exactly `PAP-native` (1PA1P), `PD-oneway` (1P1D), and
  `PD-bidir` (1P1D); PD modes share one proxy implementation and differ only by
  the explicit reuse switch and connector configuration.
- Use TP=1, `float16`, `--enforce-eager`, `max_model_len=40960`, block size 16,
  chunked prefill, prefix caching, temperature 0, fixed seed, `ignore_eos=true`,
  and `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Default `max_num_batched_tokens=4096`; 8192 is forbidden unless all three
  32K/C1 preflights prove equal headroom and the whole comparison group changes
  together.
- Context-specific `max_num_seqs` is 4K:8, 16K:4, 32K:2. A conditional 32K/C4
  run requires all three lanes to restart with `max_num_seqs=4` and re-pass C1.
- PAP uses MPS 70/30 without scanning. Initial memory-utilization values are the
  already validated lane defaults: PAP prefill/projection 0.76/0.76 and PD
  prefill/decode 0.80/0.80; after preflight they are frozen in effective config.
- A cell is admitted only when
  `active_conversations * max_rendered_context_tokens_per_conversation <=
  floor(0.70 * reported_usable_kv_token_capacity)` for every KV-owning service.
- Failure to parse capacity is fail-closed. CUDA OOM, EngineDeadError, traceback,
  accounting mismatch, output mismatch, lifecycle failure, or unexpected process/
  port ownership makes the run invalid and stops all higher points in that lane.
- Never use `pkill`, never kill an unrelated PID, and never touch GPU 0 while the
  unrelated service seen there remains active. Every launched service, client,
  proxy, and sampler must have a recorded process group and be terminated only
  through that group. The CUDA MPS daemon is the sole exception: record its unique
  pipe/log directories and stop it only through that pipe's control endpoint.
- Formal performance runs require tracked code to be committed and clean. Existing
  untracked raw artifacts do not block the tracked-clean check and must not be added.
- Formal performance can run in parallel only after the 16K/R4/C2 solo-vs-parallel
  interference Gate shows every primary metric within 2%; otherwise it is serial.
- Do not claim a 32K result until a real 32K request completes and all correctness,
  accounting, lifecycle, and OOM audits pass.

---

## File Structure

- `vllm/outputs.py`: carry local and external cached-token counts to serving code.
- `vllm/v1/engine/output_processor.py`: copy the scheduler's exclusive prefill
  accounting into `RequestOutput`.
- `vllm/entrypoints/openai/engine/protocol.py`: add backward-compatible detailed
  fields under `prompt_tokens_details`.
- `vllm/entrypoints/openai/completion/serving.py`: expose the fields for
  Completions.
- `vllm/entrypoints/openai/chat_completion/serving.py`: expose the fields for Chat.
- `benchmarks/multi_turn/long_context_common.py`: schemas, capacity parsing,
  admission, accounting validation, IDs, digests, and metric formulas.
- `benchmarks/multi_turn/benchmark_multiturn_long_context.py`: deterministic exact
  and Chat workload generation, audit and streaming execution, and JSONL output.
- `benchmarks/multi_turn/summarize_multiturn_long_context.py`: run validation,
  three-lane comparison, `best_PD`, and stable-improvement classification.
- `benchmarks/multi_turn/manifests/pap_pd_multiturn_long_context.json`: frozen Gate
  and matrix profiles from the approved design.
- `examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py`: the
  shared PD one-way/bidirectional proxy and observability adapter.
- `examples/pap/multi_pap_proxy_server.py`: generic exclusive token-accounting
  response headers for the common client.
- `.claude/skills/vllm-pap-benchmark/scripts/long_context_runner_common.sh`: shared
  tracked-clean, capacity, admission, OOM audit, and artifact helpers.
- `.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_long_context.sh`:
  safe 1PA1P wrapper and client invocation.
- `.claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_long_context.sh`:
  safe 1P1D wrapper for both PD modes.
- `.claude/skills/vllm-pap-benchmark/scripts/run_multiturn_long_context_cell.sh`:
  manifest-driven single-cell dispatcher; it never schedules an entire matrix
  without an explicit list of cell IDs.
- `tests/pap/test_prompt_token_accounting.py`: serving propagation tests.
- `tests/benchmarks/test_multiturn_long_context_common.py`: schema/admission tests.
- `tests/benchmarks/test_benchmark_multiturn_long_context.py`: deterministic
  workload and SSE/metric tests.
- `tests/benchmarks/test_summarize_multiturn_long_context.py`: comparison tests.
- `tests/pap/test_disagg_proxy_multiturn.py`: PD mode and response-accounting tests.
- `tests/pap/test_pap_launch_files.py`: static safety and runner contract checks.

### Task 1: Expose Exclusive Prompt-token Accounting

**Files:**
- Modify: `vllm/outputs.py`
- Modify: `vllm/v1/engine/output_processor.py`
- Modify: `vllm/entrypoints/openai/engine/protocol.py`
- Modify: `vllm/entrypoints/openai/completion/serving.py`
- Modify: `vllm/entrypoints/openai/chat_completion/serving.py`
- Create: `tests/pap/test_prompt_token_accounting.py`

**Interfaces:**
- Consumes: `PrefillStats.num_local_cached_tokens` and
  `PrefillStats.num_external_cached_tokens`, already populated by the scheduler.
- Produces: `RequestOutput.num_local_cached_tokens`,
  `RequestOutput.num_external_cached_tokens`, and optional OpenAI usage fields
  `prompt_tokens_details.local_cached_tokens` and
  `prompt_tokens_details.external_cached_tokens`.

- [ ] **Step 1: Write failing propagation tests**

  Construct a `RequestOutput` with total/local/external counts and assert the two
  new attributes survive. Exercise `_make_prompt_tokens_details(True, 96, ..., 64,
  32)` and assert the serialized object contains total 96, local 64, external 32.
  Add a negative test asserting the helper rejects `64 + 48 != 96`.

- [ ] **Step 2: Verify the tests fail before implementation**

  Run:

  ```bash
  .venv/bin/python -m pytest tests/pap/test_prompt_token_accounting.py -q
  ```

  Expected: collection or assertion failure because the new fields/signature do
  not exist.

- [ ] **Step 3: Carry the split through `RequestOutput`**

  Add keyword-only optional arguments and attributes:

  ```python
  num_local_cached_tokens: int | None = None,
  num_external_cached_tokens: int | None = None,
  ```

  Add the same two integer fields to the per-request state in
  `output_processor.py`; when first prefill stats arrive, assign all three counts,
  and pass them from `_new_request_output()` into `RequestOutput`.

- [ ] **Step 4: Expose backward-compatible OpenAI details**

  Extend `PromptTokenUsageInfo` with optional fields:

  ```python
  local_cached_tokens: int | None = None
  external_cached_tokens: int | None = None
  ```

  Update completion and Chat serving helpers/call sites to fill them only when
  prompt-token details are enabled. Validate non-negative values and
  `local + external == cached`; retain the existing `cached_tokens` field.

- [ ] **Step 5: Run focused and compatibility tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/pap/test_prompt_token_accounting.py \
    tests/pap/test_pap_multiturn_prefix_cache.py \
    tests/pap/test_pap_multiturn_chat_prefix_cache.py -q
  ```

  Expected: all pass.

- [ ] **Step 6: Commit the accounting surface**

  ```bash
  git add vllm/outputs.py vllm/v1/engine/output_processor.py \
    vllm/entrypoints/openai/engine/protocol.py \
    vllm/entrypoints/openai/completion/serving.py \
    vllm/entrypoints/openai/chat_completion/serving.py \
    tests/pap/test_prompt_token_accounting.py
  git commit --no-verify -m "Expose exclusive prompt token accounting" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 2: Add Common Schemas and OOM-safe Admission

**Files:**
- Create: `benchmarks/multi_turn/long_context_common.py`
- Create: `tests/benchmarks/test_multiturn_long_context_common.py`

**Interfaces:**
- Produces:

  ```python
  parse_kv_capacities(log_paths: Mapping[str, Path]) -> dict[str, int]
  decide_capacity_admission(
      capacities: Mapping[str, int],
      required_services: Sequence[str],
      active_conversations: int,
      max_rendered_context_tokens: int,
      safety_fraction: float = 0.70,
  ) -> CapacityAdmission
  validate_token_accounting(accounting: TokenAccounting) -> None
  calculate_tpot_s(turn_latency_s: float, ttft_s: float,
                   output_tokens: int) -> float
  stable_cell_id(...) -> str
  ```

  The module also exposes a `capacity` CLI subcommand. It accepts repeated
  `--service-log NAME=PATH`, repeated `--required-service NAME`, active
  conversations, actual maximum rendered context tokens, safety fraction, and
  `--output`; it writes the exact `CapacityAdmission.to_dict()` JSON atomically
  and exits 0 only for `admitted`.

  `CapacityAdmission.to_dict()` emits reported per-service capacities, their
  minimum usable capacity, required live tokens, budget tokens, fraction 0.70,
  decision (`admitted` or `admission-limited`), and a human-readable reason.

- [ ] **Step 1: Write failing parser, admission, accounting, and TPOT tests**

  Cover comma-formatted startup lines such as
  `GPU KV cache size: 145,632 tokens`, multiple log files, missing lines,
  duplicate consistent lines, and conflicting lines. Assert capacity uses the
  minimum required-service value. Test exact boundary admission, one-token-over
  rejection, zero/negative arguments, and the exclusive invariant:

  ```text
  local_reused_tokens + remote_loaded_tokens + recomputed_tokens = prompt_tokens
  ```

  Test TPOT with 1 and multiple output tokens using
  `(turn_latency - TTFT) / max(output_tokens - 1, 1)`.

- [ ] **Step 2: Verify the tests fail**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_multiturn_long_context_common.py -q
  ```

  Expected: module-not-found failure.

- [ ] **Step 3: Implement immutable dataclasses and fail-closed parsing**

  Use `@dataclass(frozen=True)` for `TokenAccounting` and
  `CapacityAdmission`. Match only the stable vLLM line:

  ```python
  KV_CAPACITY_RE = re.compile(
      r"GPU KV cache size:\s*([0-9][0-9,]*)\s+tokens"
  )
  ```

  Missing, non-positive, or conflicting capacities raise `ValueError`; no default
  capacity is permitted.

- [ ] **Step 4: Implement integer-safe admission and metric helpers**

  Compute `required_live_tokens` by multiplication and `budget_tokens` with
  `math.floor(min_capacity * Decimal("0.70"))`. Reject absent required services.
  Make accounting validation report all four counts in its exception.

- [ ] **Step 5: Run the common-module tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_multiturn_long_context_common.py -q
  ```

  Expected: all pass.

- [ ] **Step 6: Commit the common contract**

  ```bash
  git add benchmarks/multi_turn/long_context_common.py \
    tests/benchmarks/test_multiturn_long_context_common.py
  git commit --no-verify -m "Add long-context capacity admission" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 3: Make the PD Multi-turn Proxy Explicit and Observable

**Files:**
- Modify: `examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py`
- Modify: `docs/features/nixl_connector_usage.md`
- Create: `tests/pap/test_disagg_proxy_multiturn.py`

**Interfaces:**
- Consumes: upstream `usage.prompt_tokens_details.{cached_tokens,
  local_cached_tokens,external_cached_tokens}` and `kv_transfer_params`.
- Produces: CLI `--reuse-mode {oneway,bidirectional}`, `/stats`, and response
  headers:

  ```text
  X-VLLM-Prefill-Prompt-Tokens
  X-VLLM-Prefill-Local-Cached-Tokens
  X-VLLM-Prefill-External-Cached-Tokens
  X-VLLM-Prefill-Computed-Tokens
  X-VLLM-PD-Reuse-Mode
  X-VLLM-D2P-Transfer-Selected
  ```

- [ ] **Step 1: Write failing helper and route tests**

  Test that `oneway` never consumes or stores decoder entries and always reports
  zero external cached tokens. Test that `bidirectional` consumes a valid entry,
  attaches `do_remote_decode=True`, stores the new D entry, and reports the actual
  external count from the P response. Test expired entries, missing
  `conversation_id`, malformed usage, and both streaming/non-streaming response
  headers. Verify non-streaming responses preserve `prompt_token_ids`, output
  `token_ids`, usage, model, finish reason, and KV lifecycle data rather than
  rebuilding a lossy response.

  Add a retry-safety test: a cached D handle is consumed only after P succeeds, so
  a failed P request does not destroy the only reusable handle. Add a fail-closed
  test for the existing stale-payload hazard: if P returns no new
  `kv_transfer_params`, the old D→P params must never be forwarded to D.

- [ ] **Step 2: Verify the tests fail**

  ```bash
  .venv/bin/python -m pytest tests/pap/test_disagg_proxy_multiturn.py -q
  ```

  Expected: failures for the missing reuse mode and accounting adapter.

- [ ] **Step 3: Replace global implicit bidirectionality with explicit state**

  Parse `--reuse-mode`, place it and the conversation cache on `app.state`, and
  gate both cache lookup and decoder-entry storage on `bidirectional`. Keep the
  same P→D flow in both modes. Do not silently infer mode from whether an entry is
  present. Retain `bidirectional` as the compatibility default, but make the
  benchmark runner pass the mode explicitly and require `conversation_id`.
  Replace pop-on-lookup with peek/consume-after-success semantics.

- [ ] **Step 4: Preserve upstream response semantics**

  Factor prefill accounting into a pure validator that requires
  `prompt = local + external + computed`. Attach its generic headers to both
  `StreamingResponse` and `JSONResponse`. For non-streaming clients, forward the
  complete decoded OpenAI response fields captured from D, including token IDs and
  usage. For any timing not directly observed, write `null` plus a reason to
  `/stats`; never emit a fabricated zero.

- [ ] **Step 5: Add bounded counters and privacy-safe identifiers**

  `/stats` reports request counts, cache hit/miss/expired counts, D→P selected
  counts/tokens, P→D offered counts/tokens, failures, and reuse mode. Log only a
  SHA-256 prefix of `conversation_id`, not the raw value. Treat a proxy cache hit
  only as an offered handle; actual D→P load is the P engine's external-token
  field or metrics delta.

- [ ] **Step 6: Run proxy and payload tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/pap/test_disagg_proxy_multiturn.py \
    tests/pap/test_pd_payloads.py -q
  ```

  Expected: all pass.

- [ ] **Step 7: Document both proxy modes**

  Update the NIXL usage guide with explicit producer/consumer configs and proxy
  commands for one-way and bidirectional reuse. Document threshold 64 as the formal
  default and threshold 0 as a diagnostic-only path check.

- [ ] **Step 8: Commit the shared PD proxy**

  ```bash
  git add examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py \
    docs/features/nixl_connector_usage.md tests/pap/test_disagg_proxy_multiturn.py
  git commit --no-verify -m "Add explicit PD multi-turn reuse modes" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 4: Build the Deterministic Multi-turn Client

**Files:**
- Create: `benchmarks/multi_turn/benchmark_multiturn_long_context.py`
- Create: `tests/benchmarks/test_benchmark_multiturn_long_context.py`
- Modify: `examples/pap/multi_pap_proxy_server.py`
- Modify: `tests/pap/test_multi_pap_proxy_server.py`

**Interfaces:**
- Consumes: the generic headers from Task 3; PAP supplies the same headers from
  `multi_pap_proxy_server.py`.
- Produces: `conversation_metrics.jsonl`, `request_accounting.jsonl`,
  `correctness_audit.env`, and a deterministic `input_manifest.json` containing
  per-round prompt-token digests and exact token counts.
- CLI supports `--api exact|chat`, `--mode audit|performance`, `--base-url`,
  `--model`, `--corpus-path`, `--cell-id`, `--result-dir`, `--base-prompt-tokens`,
  `--first-output-tokens`, `--append-tokens`, `--output-tokens`, `--rounds`,
  `--active-conversations`, `--num-conversations`, `--seed`, and
  `--enable-thinking`. `--prepare-only` performs all local rendering and writes
  `input_manifest.json` without opening a server connection; `--cache-mode
  warm|cold|paired` makes the required cache state explicit.

- [ ] **Step 1: Write failing deterministic-workload tests**

  Use a temporary corpus with many numbered lines. Assert exact prompts equal the
  requested token count, differ across conversations, and remain byte-for-byte
  identical for the same seed/cell across lane names. Assert Chat rendering lands
  within one 16-token block of the target, uses `enable_thinking=True`, and carries
  the full assistant history into the next round. Verify the materialized reuse
  boundary excludes the last sampled token and aligns to block size 16.

- [ ] **Step 2: Write failing stream/metric tests**

  Feed synthetic SSE where one chunk contains multiple token IDs. Assert output
  token count comes from token IDs rather than chunk count, TTFT is the first
  non-empty token event, TPOT uses the approved formula, and ITL becomes
  `{value: null, reason: "token timestamps unavailable"}` when token-level timing
  cannot be proven. Assert malformed/missing accounting headers invalidate the
  request.

- [ ] **Step 3: Verify the tests fail**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_benchmark_multiturn_long_context.py -q
  ```

  Expected: module-not-found failure.

- [ ] **Step 4: Implement deterministic corpus windows**

  Load `benchmarks/sonnet.txt` by default, rotate lines using a hash of `(cell_id,
  conversation_index, seed)`, and add a unique numbered separator between cycles.
  Tokenize with `AutoTokenizer.from_pretrained(model, local_files_only=True,
  trust_remote_code=False)`. Exact mode sends token IDs directly. Chat mode uses a
  bounded search over document text until the rendered prompt is within 16 tokens
  of target; oversize prompts are rejected rather than silently truncated.

- [ ] **Step 5: Implement audit and closed-loop performance modes**

  Audit mode uses non-streaming `return_token_ids=True` and records complete token
  arrays for cross-lane comparison. Performance mode uses streaming and a semaphore
  of `active_conversations`; each conversation sends its next turn only after the
  prior turn finishes. Use stable `conversation_id` and `cache_salt` within a
  conversation and new namespaces per cell/repetition.

  In `--prepare-only`, render every planned conversation/round and store the actual
  maximum of prompt length plus that round's output reservation as
  `max_rendered_context_tokens_per_conversation`. The runner must use this measured
  value for admission; it may not substitute the nominal 4K/16K/32K base bucket.
  `paired` mode emits isolated warm and cold namespaces with identical token IDs.

- [ ] **Step 6: Adapt PAP to the generic accounting headers**

  Keep all existing `X-PAP-*` headers for compatibility. Add generic headers where
  PAP local cached tokens equal the existing cached count, external cached tokens
  are zero, and computed tokens retain the existing value. Validate the exclusive
  sum before returning.

- [ ] **Step 7: Run client and PAP proxy tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_benchmark_multiturn_long_context.py \
    tests/pap/test_multi_pap_proxy_server.py \
    tests/pap/test_pap_multiturn_prefix_cache.py \
    tests/pap/test_pap_multiturn_chat_prefix_cache.py -q
  ```

  Expected: all pass.

- [ ] **Step 8: Commit the common client**

  ```bash
  git add benchmarks/multi_turn/benchmark_multiturn_long_context.py \
    tests/benchmarks/test_benchmark_multiturn_long_context.py \
    examples/pap/multi_pap_proxy_server.py \
    tests/pap/test_multi_pap_proxy_server.py
  git commit --no-verify -m "Add deterministic multi-turn benchmark client" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 5: Add the Frozen Manifest and Memory-safe Lane Runners

**Files:**
- Create: `benchmarks/multi_turn/manifests/pap_pd_multiturn_long_context.json`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/long_context_runner_common.sh`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_long_context.sh`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_long_context.sh`
- Create: `.claude/skills/vllm-pap-benchmark/scripts/run_multiturn_long_context_cell.sh`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pd_same_workload.sh`
- Modify: `tests/pap/test_pap_launch_files.py`

**Interfaces:**
- Consumes: Task 2 capacity/admission CLI helpers and Task 4 benchmark client.
- Produces every artifact listed by the design, including
  `capacity_admission.json`, before any benchmark request is sent.

- [ ] **Step 1: Write failing static runner tests**

  Assert the dedicated wrappers contain 40960/4096 and context-specific sequence
  limits, never inherit old QPS/concurrency defaults, enable prefix caching and
  prompt details, set local proxy bypass, and use `.venv/bin/python`. Assert all
  service launches use `setsid`, store PID/PGID, and cleanup only negative recorded
  PGIDs. Assert `pkill`, bare `python`, `kv_both`, workload fallback, and capacity
  fallback strings are absent. Assert PD configs use `kv_producer`/`kv_consumer`
  and switch bidirectionality explicitly.

- [ ] **Step 2: Verify static tests fail**

  ```bash
  .venv/bin/python -m pytest \
    tests/pap/test_pap_launch_files.py -q
  ```

  Expected: failures for missing runners and contracts.

- [ ] **Step 3: Encode the approved matrix without expansion surprises**

  The manifest contains Gate 0; Matrix 1 values 4K/16K/32K × 128/512/2048;
  Matrix 2 values 16K/32K × R4/R8 × C1/C2; Matrix 3 mandatory 4K C1/C4/C8 and
  16K C4; conditional 32K C4; and the 16K/R4/C4 tail. Include repetitions,
  warm/cold requirements, conditional flags, and per-context scheduler profile.
  A schema check must expand to exactly 6 Gate cells, 63 mandatory architecture
  cells, and 3 conditional capacity cells.

- [ ] **Step 4: Implement fail-closed lifecycle helpers**

  `long_context_runner_common.sh` records commit, tracked status, effective config,
  ports, topology, and GPU processes. It invokes
  `long_context_common.py capacity` after health checks, parses each KV-owning
  service log, writes `capacity_admission.json` atomically, and exits before the
  client unless decision is `admitted`. Before admission it invokes the client with
  `--prepare-only` and reads the actual maximum rendered sequence reservation from
  `input_manifest.json`. PAP requires capacity from the PA/Prefill owner; PD
  requires both Prefill and Decode and uses their minimum. It scans logs for
  `CUDA out of memory`,
  `EngineDeadError`, and traceback; an OOM marks `run_status=invalid-oom`, kills the
  recorded process groups, and returns nonzero.

  Start a scoped resource sampler after health checks. It writes timestamped
  per-selected-GPU memory/utilization rows and process CPU/RSS rows to
  `resource_samples.csv`; sampler PID/PGID is recorded and cleaned with the run.
  Failure to sample is recorded as `null + reason` and may not be represented as
  zero utilization.

- [ ] **Step 5: Harden the existing PAP lifecycle foundation**

  Add a `PGIDS` array to `run_pap_same_pd_workload.sh`, launch Attention, Prefill,
  Projection, and proxy through `setsid`, and terminate only recorded groups. Add a
  `multiturn_long_context` client mode, but leave all legacy defaults and modes
  unchanged. The dedicated wrapper overrides every scale value; it never relies on
  the old `MAX_MODEL_LEN=512`, `MAX_NUM_SEQS=64`, or QPS defaults.

- [ ] **Step 6: Add PD producer/consumer long-context mode**

  Preserve the existing short benchmark behavior. For the dedicated wrapper,
  launch P with `kv_role=kv_producer`, D with `kv_role=kv_consumer`, and the shared
  proxy with the selected reuse mode. `PD-oneway` explicitly sets
  `bidirectional_kv_xfer=false`; `PD-bidir` sets true plus
  `decoder_kv_blocks_ttl=480` and `kv_recompute_threshold=64` on both services.
  Both set `kv_load_failure_policy=fail`, enable prefix caching, and enable
  detailed prompt-token usage. Snapshot `/metrics` immediately before and after
  each subrun; store raw snapshots plus deltas for
  `prompt_tokens_by_source`, local/external prefix-cache counters, and NIXL
  bytes/time. Merge the run-level cache deltas with the client's immutable
  `request_accounting.jsonl` into `cache_accounting.json`; never overwrite the
  client evidence. Store NIXL-specific deltas in `transfer_summary.json`.

- [ ] **Step 7: Add single-cell dispatch and progressive-order guards**

  The dispatcher accepts one cell ID and one lane. It checks that 16K/C1 has a
  passed 4K/C1 admission artifact and 32K/C1 has a passed 16K/C1 artifact. Higher
  concurrency likewise requires the lower point. It refuses 32K/C4 unless all
  three lanes have passed a restarted 32K/C1 preflight with `max_num_seqs=4`.
  There is no default `--all` behavior.

- [ ] **Step 8: Run shell syntax and static tests**

  ```bash
  bash -n .claude/skills/vllm-pap-benchmark/scripts/long_context_runner_common.sh
  bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_long_context.sh
  bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_long_context.sh
  bash -n .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_long_context_cell.sh
  .venv/bin/python -m pytest tests/pap/test_pap_launch_files.py -q
  ```

  Expected: syntax checks and tests pass.

- [ ] **Step 9: Commit the frozen runners**

  ```bash
  git add benchmarks/multi_turn/manifests/pap_pd_multiturn_long_context.json \
    .claude/skills/vllm-pap-benchmark/scripts/long_context_runner_common.sh \
    .claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_long_context.sh \
    .claude/skills/vllm-pap-benchmark/scripts/run_pd_multiturn_long_context.sh \
    .claude/skills/vllm-pap-benchmark/scripts/run_multiturn_long_context_cell.sh \
    .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh \
    .claude/skills/vllm-pap-benchmark/scripts/run_pd_same_workload.sh \
    tests/pap/test_pap_launch_files.py
  git commit --no-verify -m "Add OOM-safe multi-turn lane runners" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 6: Validate and Compare Three-lane Results

**Files:**
- Create: `benchmarks/multi_turn/summarize_multiturn_long_context.py`
- Create: `tests/benchmarks/test_summarize_multiturn_long_context.py`

**Interfaces:**
- Consumes: three lane run roots for a comparison group.
- Produces: `round_summary.json`, `comparison.json`, and Markdown tables with
  pairwise PAP/PD ratios, `best_PD`, invalid reasons, and noise/stability labels.

- [ ] **Step 1: Write failing aggregation tests**

  Build synthetic three-lane fixtures. Assert Round 1 and Round 2+ are separate;
  latency chooses `min(PD-oneway, PD-bidir)`, throughput chooses `max`; stable
  improvement requires three same-direction repetitions and median difference at
  least 5%; smaller differences are `parity/noise-band`. Assert any output digest,
  input digest, accounting, capacity, lifecycle, or OOM failure invalidates the
  entire comparison group.

- [ ] **Step 2: Verify the tests fail**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_summarize_multiturn_long_context.py -q
  ```

  Expected: module-not-found failure.

- [ ] **Step 3: Implement strict loading and comparison**

  Validate required artifacts before calculating metrics. Group by logical cell,
  lane, repetition, and round class. Use actual output-token counts. Emit null with
  a reason for unavailable ITL/server timing fields. Never include diagnostic or
  invalid runs in best-of or median calculations.

- [ ] **Step 4: Run aggregation tests**

  ```bash
  .venv/bin/python -m pytest \
    tests/benchmarks/test_summarize_multiturn_long_context.py -q
  ```

  Expected: all pass.

- [ ] **Step 5: Commit the aggregator**

  ```bash
  git add benchmarks/multi_turn/summarize_multiturn_long_context.py \
    tests/benchmarks/test_summarize_multiturn_long_context.py
  git commit --no-verify -m "Add strict multi-turn result comparison" \
    -m "Assisted-by: OpenAI Codex"
  ```

### Task 7: Freeze Code and Run the Safe Experimental Gates

**Files:**
- Create after evidence exists:
  `docs/experiments/pap-pd-multiturn-long-context-gate0-20260711.md`
- Modify after evidence exists: `benchmarks/pap/experiments/HISTORY.md`
- Raw output only:
  `/home/fei/research/PD/test/baseline/multiturn_pd_pap/results/runs/`

**Interfaces:**
- Consumes: committed code from Tasks 1-6 and one explicitly selected idle GPU
  pair at a time.
- Produces: accepted or rejected capacity preflights and Gate 0 evidence. It does
  not start the formal matrix automatically.

- [ ] **Step 1: Run the complete CPU/static verification suite**

  ```bash
  .venv/bin/python -m pytest \
    tests/pap/test_prompt_token_accounting.py \
    tests/benchmarks/test_multiturn_long_context_common.py \
    tests/benchmarks/test_benchmark_multiturn_long_context.py \
    tests/benchmarks/test_summarize_multiturn_long_context.py \
    tests/pap/test_disagg_proxy_multiturn.py \
    tests/pap/test_pap_launch_files.py \
    tests/pap/test_multi_pap_proxy_server.py \
    tests/pap/test_pd_payloads.py \
    tests/pap/test_pap_multiturn_prefix_cache.py \
    tests/pap/test_pap_multiturn_chat_prefix_cache.py -q
  ```

  Expected: all pass. Also run all four `bash -n` commands from Task 5.

- [ ] **Step 2: Confirm the exact tested commit and resource ownership**

  Record `git rev-parse HEAD`, `git status --porcelain --untracked-files=no`,
  `nvidia-smi` compute processes, GPU memory, `nvidia-smi topo -m`, and all chosen
  ports. Refuse to run if tracked files are dirty, a chosen GPU is occupied, or a
  port has an unexpected owner. Do not use GPU 0.

- [ ] **Step 3: Start each lane serially and perform capacity-only preflight**

  Start one lane on the same idle topology class, wait for health, parse capacity,
  and write admission decisions for C1 at 4K, 16K, and 32K without sending the
  long requests yet. Stop and preserve evidence if any capacity is missing or
  32K exceeds the 70% budget.

- [ ] **Step 4: Run Gate 0 serially**

  For each of the three lanes, run exact 128/48 and Chat 128/48 two-turn audits.
  Compare token arrays and accounting across lanes. Require PAP decode-derived
  local hits and PD-oneway external=0. Run PD-bidir first with the formal threshold
  64. If its reusable D-only suffix is below 64 and the engine records a threshold
  recompute decision, run one explicitly labeled diagnostic with threshold 0 to
  prove D→P; the diagnostic does not replace the formal-threshold result. Matrix
  1 keeps threshold 64, and actual external tokens must be positive whenever its
  eligible D-only suffix is at least 64. Any unexplained zero or lifecycle failure
  stops the experiment before long requests.

- [ ] **Step 5: Run progressive C1 memory probes**

  For each lane, restart from a clean process group and run one exact two-turn
  request at 4K, then 16K, then 32K. Before every request, re-evaluate admission;
  after it, capture peak GPU memory, KV occupancy, logs, accounting, output token
  audit, and lifecycle state. Do not start the next context if peak live-token use
  reaches 70%, free memory falls below the recorded workspace reserve, or any
  invalid marker appears.

- [ ] **Step 6: Write and commit the Gate report**

  The report maps each lane/cell to its absolute raw run root, tested commit,
  actual GPUs, parsed KV capacity, budget, required tokens, peak memory, result,
  and invalid reason. State clearly whether 32K is runtime-validated and whether
  the formal matrix is authorized. Update the experiment-history index.

  ```bash
  git add docs/experiments/pap-pd-multiturn-long-context-gate0-20260711.md \
    benchmarks/pap/experiments/HISTORY.md
  git commit --no-verify -m "Record PAP PD multi-turn long-context gates" \
    -m "Assisted-by: OpenAI Codex"
  ```

## Execution Boundary

This plan deliberately stops after code freeze, capacity preflight, Gate 0, and
progressive C1 probes. The 189 formal repetitions begin only if those gates pass.
The next execution plan will allocate GPU pairs, run the solo-vs-parallel
interference Gate, and schedule Matrix 1→2→3 in admission order; it will not modify
the benchmark implementation used by this plan.
