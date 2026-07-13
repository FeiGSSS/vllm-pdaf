# PAP Async Decode-Token D2H Implementation Plan

Status update (2026-07-14): implementation and C2/C4 validation completed.
The checklist below is retained as the original development plan; measured
results and the later default-enable decision are recorded in
`docs/design/pap-async-decode-token-d2h-results-20260713.md` and
`docs/design/pap-async-static-mps-c4-ab-results-20260714.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the synchronous Projection `input_ids` D2H from every PAP
decode forward, preserve decode-KV commit and multi-turn cache semantics, and
measure the net C4 70:30 TPOT change with a strict same-code A/B.

**Architecture:** `AsyncOutput` already copies sampled tokens to CPU on its
dedicated output-copy stream. When `PAP_ASYNC_DECODE_TOKEN=1` (the default), a
non-blocking callback enqueues those CPU tokens to an Attention-side HTTP inbox
instead of reading the next forward's GPU `input_ids`. Attention joins
token-ready and KV-ready records by `(session_request_id, new_seq_len)` and only
then enqueues the existing reliable `DecodeCommitClient`; session release waits
for KV-ready joins and acknowledged commits, while an unmatched final sampled
token is deliberately discarded because it has no KV cache.

**Tech Stack:** Python 3.12, PyTorch CUDA streams/events, FastAPI/Pydantic,
`httpx`, existing PAP local-fast transport and decode-commit protocol, pytest,
and the fixed five-round C2/C4 benchmark scripts.

## Global Constraints

- Work only on `feature/pap`; preserve all unrelated untracked experiment data.
- Use `.venv/bin/python` or `uv` for Python commands; never use system Python or
  bare pip.
- Do not run pre-commit, per the user's explicit instruction.
- Keep PAP MPS at 70:30 and use the fixed Qwen3-8B C4 workload; do not scan MPS.
- `PAP_ASYNC_DECODE_TOKEN=0` must retain the existing synchronous descriptor
  path so OFF/ON runs are a same-commit A/B.
- Queue overflow, token mismatch, missing token for KV-ready state, commit
  failure, and non-empty session drain must fail closed or emit an audit-fatal
  log.

---

## File Structure

- Create `vllm/pap/deferred_decode_token.py`: thread-safe token/KV rendezvous,
  duplicate validation, flush, cleanup, and counters.
- Create `vllm/pap/decode_token_client.py`: reliable background Projection to
  Attention token delivery and environment switch.
- Modify `vllm/v1/worker/gpu/async_utils.py`: optional post-copy callback after
  sampled-token trimming.
- Modify `vllm/v1/worker/gpu/model_runner.py`: remove the default synchronous
  D2H, capture per-request next sequence metadata, and enqueue copied tokens.
- Modify `examples/pap/pap_attention_executor.py`: token HTTP endpoint,
  rendezvous integration, lifecycle flush, and statistics.
- Modify `examples/pap/launch_pap_nixl.sh`: default/export the feature and client
  reliability settings.
- Modify `.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh`:
  record the effective A/B setting and keep strict audits.
- Create `tests/pap/test_async_decode_token.py`: unit and integration-contract
  tests for all new behavior.
- Modify `tests/pap/test_pap_attention_executor.py`: compute-path join and
  release/drain tests.
- Modify `docs/design/pap-experiment-history-index.md` and create a result note
  after the measured A/B.

---

### Task 1: Token/KV rendezvous state machine

**Files:**
- Create: `vllm/pap/deferred_decode_token.py`
- Create: `tests/pap/test_async_decode_token.py`

**Interfaces:**
- Produces: `DeferredDecodeCommit`, `DeferredDecodeTokenCommitter.record_token`,
  `record_kv_ready`, `flush_request`, `forget_request`, and `stats`.
- The dispatch callback receives one immutable commit only after both sides of
  the same `(request_id, new_seq_len)` are present.

- [ ] Write failing tests proving token-first and KV-first arrival both dispatch
  exactly once, a same-value retry is idempotent, a token mismatch raises, flush
  waits for a missing token, and forget drops token-only final-token state.
- [ ] Run
  `.venv/bin/python -m pytest tests/pap/test_async_decode_token.py -v` and verify
  collection fails because `vllm.pap.deferred_decode_token` does not exist.
- [ ] Implement the minimal condition-variable state machine. Dispatch outside
  the state lock, but count in-progress dispatch so `flush_request()` cannot
  return before the existing commit client has accepted the item.
- [ ] Re-run the test file and require all Task 1 tests to pass.

### Task 2: Reliable asynchronous token delivery

**Files:**
- Create: `vllm/pap/decode_token_client.py`
- Modify: `tests/pap/test_async_decode_token.py`

**Interfaces:**
- Produces: `async_decode_token_enabled()` and `DecodeTokenClient.publish`,
  `flush_request`, `forget_request`, and `shutdown`.
- Sends JSON `{request_id, new_seq_len, token_id}` to
  `/v1/pap/attention/decode-token` on the per-request Attention endpoint.

- [ ] Add a failing test with a blocked fake HTTP post proving `publish()`
  returns before delivery, `flush_request()` waits for acknowledgment, retries
  preserve the payload, and queue-full behavior raises instead of dropping a
  token.
- [ ] Run the new test selection and verify it fails for missing client symbols.
- [ ] Implement one daemon FIFO worker, bounded pending counters per request,
  retry/backoff, 2xx acknowledgment validation, and fail-closed flush state.
- [ ] Run the client tests and the existing decode-commit client tests.

### Task 3: Reuse `AsyncOutput` and remove Projection's default barrier

**Files:**
- Modify: `vllm/v1/worker/gpu/async_utils.py`
- Modify: `vllm/v1/worker/gpu/model_runner.py`
- Modify: `tests/pap/test_async_decode_token.py`
- Modify: `tests/pap/test_pap_contract.py`

**Interfaces:**
- `AsyncOutput(..., output_ready_callback=None)` invokes the callback once with
  the fully trimmed `ModelRunnerOutput`, after its existing copy event completes.
- The model runner callback maps each sampled token to the captured Prefill
  session handle, Attention HTTP endpoint, and `current_seq_len + 1`.

- [ ] Add a failing CPU-only `AsyncOutput.__new__` test proving the callback sees
  trimmed tokens and runs once after the fake copy event synchronizes.
- [ ] Add a failing model-runner contract test proving ON returns empty
  `pap_input_token_ids`, OFF retains the descriptor D2H, and live PAP sampled
  tokens enqueue one sideband item with the correct next sequence length.
- [ ] Implement the callback and lazy client creation. Capture request mappings
  before request-state cleanup; skip only synthetic warmup rows that have no PAP
  route, and reject multi-token speculative rows because PAP currently supports
  one decode token per request.
- [ ] Ensure the existing deferred trace records
  `token_boundary_input_ids_d2h_wall_ms` only in the OFF path.
- [ ] Run the focused tests and `py_compile` for both modified modules.

### Task 4: Attention rendezvous, API, and lifecycle correctness

**Files:**
- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `tests/pap/test_pap_attention_executor.py`
- Modify: `tests/pap/test_async_decode_token.py`

**Interfaces:**
- `POST /v1/pap/attention/decode-token` records a validated token and returns its
  join status.
- Unified KV append records KV-ready when descriptor token rows are empty; the
  OFF path with descriptor tokens retains existing direct commit behavior.

- [ ] Add failing tests for token-before-KV, KV-before-token, duplicate HTTP
  retry, mismatch rejection, direct descriptor fallback, final token cleanup,
  and release timeout when KV exists without a token.
- [ ] Implement registry-owned rendezvous state, endpoint validation, compute
  integration, stats exposure, and release/replacement ordering:
  rendezvous flush, existing commit-client flush, lease release, then forget.
- [ ] Run Attention, decode-commit, data-plane, routing, and contract test files.

### Task 5: Runtime defaults and strict A/B observability

**Files:**
- Modify: `examples/pap/launch_pap_nixl.sh`
- Modify:
  `.claude/skills/vllm-pap-benchmark/scripts/run_pap_multiturn_load.sh`
- Modify: benchmark validator files selected by the existing runner.

**Interfaces:**
- Default: `PAP_ASYNC_DECODE_TOKEN=1`.
- A/B override: `PAP_ASYNC_DECODE_TOKEN=0`.
- Attention stats after drain must report zero pending KV-ready items and zero
  mismatch/failure counters; token-only drops are expected to equal completed
  request sessions.

- [ ] Add/extend shell contract tests before changing launch scripts.
- [ ] Export and record the switch plus queue/retry/flush parameters.
- [ ] Extend strict audit to reject pending KV joins, mismatch, dispatch failure,
  or token-delivery failure; keep final-token-only drops informational.
- [ ] Run shell syntax checks, benchmark validator unit tests, and focused pytest.

### Task 6: C2 correctness canary and C4 70:30 performance A/B

**Files:**
- Create: `docs/design/pap-async-decode-token-d2h-results-20260713.md`
- Modify: `docs/design/pap-experiment-history-index.md`

**Interfaces:**
- OFF and ON use the same commit, workload, GPUs 1/2, model, exact-token
  continuations, QPS 2, C4, 16K first turn, four 120-token append turns, and
  256 output tokens per turn.

- [ ] Check branch, tracked status, PAP/vLLM processes, GPU occupancy, and proxy
  environment; do not kill unrelated jobs.
- [ ] Run ON C2 quick first and require 20/20 completion, exact digest/cache
  gates, zero fatal logs, zero pending KV joins, and zero active sessions.
- [ ] Run one OFF C4 and one ON C4 in alternating order with trace disabled.
- [ ] If the observed effect is within run noise, run two additional alternating
  OFF/ON pairs and report the across-pair median.
- [ ] Compare R1 and R2-R5 TTFT/TPOT, throughput, concurrency, correctness, token
  join counters, and session drain. Attribute only the strict OFF/ON delta to
  removing the synchronous D2H path.
- [ ] Record all absolute run directories, effective configs, audits, raw
  results, and the accepted/rejected conclusion in the result note and history
  index.
