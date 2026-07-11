# PAP Chat Multi-Turn Prefix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict Qwen3 Chat Completions two-turn audit that proves PAP
reuses at least one complete decode-derived KV block under the real chat
template.

**Architecture:** Keep the proven exact-token audit unchanged. Add a focused
Chat Completions client that sends the first assistant response back as normal
conversation history, computes the committed-token LCP from vLLM's returned
prompt/output token IDs, and compares warm and cache-salt-isolated cold
requests. Add a third benchmark client mode so startup, routing, correctness,
and session-drain audits stay identical to the exact-token experiment.

**Tech Stack:** Python 3.12 through `.venv/bin/python`, `httpx`, a local
Transformers tokenizer, vLLM OpenAI Chat Completions, Bash, and pytest.

## Global Constraints

- Use only `/data/ssd1/llm-models/Qwen3-8B`; never access Hugging Face.
- Run 1PA1P on GPU1/2 with Prefill/Attention MPS fixed at 70/30; do not scan MPS.
- Do not add Proxy affinity, persistent sessions, or cross-PA KV migration.
- Accept that the final sampled token and partial cache block may miss.
- Do not write message text or raw token IDs to result files; use counts and
  SHA-256 digests.
- Use Qwen3's thinking template. The non-thinking template inserts an empty
  reasoning scaffold into the generation prompt but omits it when the
  assistant response is rendered as history, so it cannot preserve the decode
  token prefix across turns.
- Never use system `python3`, bare `pip`, pre-commit, or commit hooks.

---

### Task 1: Strict Chat Completions audit client

**Files:**
- Create: `examples/pap/pap_multiturn_chat_prefix_cache.py`
- Create: `tests/pap/test_pap_multiturn_chat_prefix_cache.py`
- Reuse: `examples/pap/pap_multiturn_prefix_cache.py`

**Interfaces:**
- Consumes: `expected_prefix_hit_tokens(..., block_size: int) -> int`.
- Produces: `build_second_turn_messages(...) -> list[dict[str, str]]`,
  `chat_prefix_metrics(...) -> dict[str, int]`, and a CLI that writes
  `multiturn_chat_prefix_cache.json`.

- [x] **Step 1: Write the failing pure-function tests**

```python
def test_build_second_turn_messages_preserves_assistant_text():
    first = [{"role": "user", "content": "first"}]
    result = build_second_turn_messages(first, "answer", "follow-up")
    assert result == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]
    assert first == [{"role": "user", "content": "first"}]


def test_chat_prefix_metrics_use_true_retokenized_lcp():
    metrics = chat_prefix_metrics(
        list(range(32)),
        list(range(100, 133)),
        [*range(32), *range(100, 127), 999],
        block_size=16,
    )
    assert metrics == {
        "committed_lcp_tokens": 59,
        "expected_prefix_hit_tokens": 48,
        "decode_derived_hit_tokens": 16,
    }
```

- [x] **Step 2: Run the tests and verify red**

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_multiturn_chat_prefix_cache.py -q
```

Expected: collection fails because the Chat audit module does not exist.

- [x] **Step 3: Implement the pure functions**

```python
def build_second_turn_messages(first_messages, assistant_content,
                               second_user_content):
    return [
        *({"role": str(item["role"]), "content": str(item["content"])}
          for item in first_messages),
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": second_user_content},
    ]


def chat_prefix_metrics(first_prompt_token_ids, first_output_token_ids,
                        second_prompt_token_ids, *, block_size):
    committed = [*first_prompt_token_ids, *first_output_token_ids[:-1]]
    lcp = next(
        (i for i, pair in enumerate(zip(committed, second_prompt_token_ids))
         if pair[0] != pair[1]),
        min(len(committed), len(second_prompt_token_ids)),
    )
    expected = expected_prefix_hit_tokens(
        first_prompt_token_ids, first_output_token_ids,
        second_prompt_token_ids, block_size=block_size)
    prompt_boundary = len(first_prompt_token_ids) // block_size * block_size
    return {
        "committed_lcp_tokens": lcp,
        "expected_prefix_hit_tokens": expected,
        "decode_derived_hit_tokens": max(0, expected - prompt_boundary),
    }
```

- [x] **Step 4: Implement deterministic local prompt sizing and HTTP parsing**

Load the tokenizer with `local_files_only=True` and
`trust_remote_code=False`. Repeat the first user text until
`apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`
reaches `--min-first-prompt-tokens`. Send each request with this body:

```python
{
    "model": model,
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": 0,
    "seed": 0,
    "ignore_eos": True,
    "stream": False,
    "return_token_ids": True,
    "cache_salt": cache_salt,
    "chat_template_kwargs": {"enable_thinking": True},
}
```

Parse `prompt_token_ids` from the response root; parse `token_ids`,
`message.content`, and `finish_reason` from the single choice; parse Prefill
prompt/cached/computed counts from the PAP headers. Reject missing IDs, empty
assistant content, non-`length` finishes, and unexpected generated-token counts.

- [x] **Step 5: Implement strict warm/cold checks and safe JSON**

Use the same unique salt for turn 1 and warm turn 2, and a different salt for
cold turn 2. Require identical warm/cold prompt and output IDs, exact
LCP-derived warm hit, cold hit zero, and at least one decode-derived block.
Write only timings, counts, finish reasons, booleans, and token digests.

- [x] **Step 6: Verify the client**

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_multiturn_chat_prefix_cache.py \
  tests/pap/test_pap_multiturn_prefix_cache.py -q
.venv/bin/python -m py_compile \
  examples/pap/pap_multiturn_chat_prefix_cache.py
```

Expected: all tests pass and `py_compile` exits zero.

### Task 2: Benchmark-runner integration

**Files:**
- Modify: `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`
- Modify: `tests/pap/test_pap_launch_files.py`

**Interfaces:**
- Consumes: the Chat audit CLI from Task 1.
- Produces: `PAP_BENCH_CLIENT_MODE=multiturn_chat_prefix_cache` and
  `${RUN_ROOT}/multiturn_chat_prefix_cache.json`.

- [x] **Step 1: Extend the static runner test and verify red**

```python
assert "canonical | multiturn_prefix_cache | multiturn_chat_prefix_cache" in text
assert "pap_multiturn_chat_prefix_cache.py" in text
assert "multiturn_chat_prefix_cache.json" in text
```

Run the focused launch-file test. Expected: failure before runner changes.

- [x] **Step 2: Add the third mode without changing canonical defaults**

Treat both multi-turn modes as three-request 1PA1P audits with prompt-token
details and 64 decode-capacity tokens. Dispatch Chat mode with:

```bash
timeout "${BENCH_TIMEOUT}" "${PYTHON_BIN}" \
  examples/pap/pap_multiturn_chat_prefix_cache.py \
  --base-url "http://127.0.0.1:${PAP_PROXY_PORT}" \
  --model "${MODEL_PATH}" \
  --result-path "${RUN_ROOT}/multiturn_chat_prefix_cache.json" \
  --min-first-prompt-tokens "${INPUT_LEN}" \
  --first-output-tokens "${PAP_MULTITURN_FIRST_OUTPUT_TOKENS}" \
  --second-output-tokens "${OUTPUT_LEN}" \
  --block-size "${PAP_MULTITURN_BLOCK_SIZE}" \
  --min-decode-hit-blocks "${PAP_MULTITURN_MIN_DECODE_HIT_BLOCKS}"
```

- [x] **Step 3: Run focused and full PAP checks**

```bash
bash -n .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
.venv/bin/python -m pytest tests/pap -q
git diff --check
```

Expected: syntax and diff checks exit zero; PAP tests pass with existing skips.

### Task 3: Clean experiment and evidence

**Files:**
- Modify: `docs/design/pap-xpayp-multiturn-kv-affinity-20260710.md`
- Generate outside git:
  `/home/fei/research/PD/test/baseline/pap/results/runs/<RUN_ID>/`

- [x] **Step 1: Commit implementation with hooks disabled**

Stage only the two client/test files and two runner/test files, then use
`git commit --no-verify -m "Add PAP chat multi-turn cache audit"`.

- [x] **Step 2: Run a clean 1PA1P Chat audit**

Use client mode `multiturn_chat_prefix_cache`, GPU1/2, MPS 70/30, unified-KV
decode capacity 64, clean tracked-worktree enforcement, strict correctness,
and `VLLM_USE_FLASHINFER_SAMPLER=0`.

Expected: status passed; actual hit equals the true-LCP expected hit;
decode-derived hit is at least 16; cold hit is zero; warm/cold outputs match;
strict correctness has zero matches; all routes use `pa0:p0`; active sessions
are zero; service-log error scan is empty.

- [x] **Step 3: Record and commit the result summary**

Record commit/run directory, prompt/output/LCP/hit counts, warm/cold Prefill
times, correctness, routing, and drain status. Do not copy content or raw IDs.

- [ ] **Step 4: Run final verification**

```bash
.venv/bin/python -m pytest tests/pap -q
.venv/bin/python -m pytest tests/v1/core/test_prefix_caching.py -q
git diff --check
git status --porcelain --untracked-files=no
```

Expected: both suites pass; checks exit zero; tracked worktree is clean after
the documentation commit.
