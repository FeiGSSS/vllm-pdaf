# PAP Phase 1 Central Dispatcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace each PA peer's receive-compute-send thread with per-peer
ingress receivers and one PA-wide FIFO compute dispatcher, while preserving
exactly one Attention compute call per ingress batch.

**Architecture:** Each peer receiver owns only transport receive and transfers
the received message to a `PAPAttentionWorkItem`. A standalone
`PAPAttentionDispatcher` owns a FIFO queue and exactly one compute thread. The
dispatcher calls the existing Attention compute path once, sends the result to
the originating transport, and releases the input message exactly once. Each
receiver then wakes from a completion event before reading its next item, so a
Python doorbell busy-spin cannot starve the dispatcher. A receiver-stream CUDA
event transfers readiness to the dispatcher stream without CPU synchronization.
A `legacy`/`central_fifo` flag keeps same-code A/B available; Phase 1 performs
no same-layer combine and no deliberate coalescing wait.

**Tech Stack:** Python 3.12, PyTorch, FastAPI, vLLM PAP local-fast/NIXL mailbox,
pytest, Bash benchmark runner.

## Global Constraints

- Work on `feature/pap`; preserve unrelated untracked files and user changes.
- Run Python only through `.venv/bin/python` or `uv`; never system `python3`.
- Use the local `/data/ssd1/llm-models/Qwen3-8B`; do not access Hugging Face.
- Do not run pre-commit. Use focused pytest, shell syntax checks, and GPU smoke.
- Keep Python lines at or below 88 characters and use existing vLLM style.
- Keep `legacy` behavior available and make `central_fifo` opt-in in Phase 1.
- Do not add a coalescing window, same-layer combine, cross-layer scheduling,
  route-row reordering, or MPS parameter scan in this phase.
- Preserve strict correctness audit, decode-commit/lease-release checks, and
  session drain.

---

### Task 1: Standalone FIFO Work Item and Dispatcher

**Files:**

- Create: `vllm/pap/attention_scheduler.py`
- Create: `tests/pap/test_pap_attention_scheduler.py`

**Interfaces:**

- Produces `PAPAttentionWorkItem`, which carries `descriptor`, `qkv_batch`,
  `transport`, `peer_id`, `arrival_ns`, optional `input_message`, trace context,
  and an idempotent `release_input()` method.
- Produces `PAPAttentionDispatcher(handler, max_queue_size=0)`, with `start()`,
  `enqueue(item)`, `dispatch_next(timeout=None)`, `stop(drain, timeout)`, and
  `stats()`.
- The dispatcher owns an item only after `enqueue()` succeeds and guarantees
  one input release after handler success or failure.

- [x] **Step 1: Write the failing FIFO and ownership tests**

```python
def test_dispatcher_preserves_fifo_and_releases_each_input_once():
    handled = []
    released = []
    dispatcher = PAPAttentionDispatcher(
        handler=lambda item: handled.append(item.peer_id)
    )
    dispatcher.enqueue(make_item("p0", released))
    dispatcher.enqueue(make_item("p1", released))

    assert dispatcher.dispatch_next(timeout=0.1)
    assert dispatcher.dispatch_next(timeout=0.1)
    assert handled == ["p0", "p1"]
    assert released == ["p0", "p1"]


def test_dispatcher_releases_input_and_records_fatal_handler_error():
    item = make_item("p0", [])
    dispatcher = PAPAttentionDispatcher(
        handler=lambda _item: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    dispatcher.enqueue(item)
    with pytest.raises(RuntimeError, match="boom"):
        dispatcher.dispatch_next(timeout=0.1)

    assert item.input_released
    assert dispatcher.stats()["dispatch_failures"] == 1
```

- [x] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_scheduler.py -q
```

Expected: collection fails because `vllm.pap.attention_scheduler` does not
exist.

- [x] **Step 3: Implement the minimal FIFO dispatcher**

Implement a bounded/unbounded `queue.Queue`, one optional daemon worker, a
sentinel for shutdown, lock-protected counters, and input release in a
`finally` block. `dispatch_next()` is the single implementation used by both
unit tests and the worker thread. Do not inspect layer compatibility in Phase
1.

- [x] **Step 4: Verify GREEN and lifecycle behavior**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_scheduler.py -q
```

Expected: FIFO, release-on-success, release-on-error, worker start/stop, and
stats tests all pass.

---

### Task 2: Split Peer Ingress from Compute/Send

**Files:**

- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `tests/pap/test_pap_attention_executor.py`

**Interfaces:**

- Consumes `PAPAttentionWorkItem` and `PAPAttentionDispatcher` from Task 1.
- Produces `run_offload_exec_mailbox_receiver_loop(...)`, which receives,
  records a CUDA ready event, validates, records ingress, enqueues without
  running Attention, then waits for item completion before reading again.
- Produces `_execute_offload_exec_work_item(...)`, which records one compute,
  invokes `compute_offload_exec_batch_output()` once, sends one output batch to
  the source transport, and leaves input release to the dispatcher.

- [x] **Step 1: Write failing two-peer central-dispatch tests**

Use two fake transports with one descriptor each. Assert both receiver loops
can enqueue before the dispatcher runs, then assert FIFO dispatch performs two
compute calls, sends to the two original transports, and releases both input
messages once. Add a test that the receiver never invokes compute itself.

- [x] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  -k 'central_dispatch or mailbox_receiver' -q
```

Expected: tests fail because the receiver and work-item executor do not exist.

- [x] **Step 3: Extract common receive/compute trace state**

Move the current mailbox-loop receive result into a work item trace dictionary
containing receive start/done timestamps, receive breakdown, and arrival time.
Move current compute/send/logging into `_execute_offload_exec_work_item()`.
Keep the legacy loop calling the same helper so trace fields remain comparable.
Append `queue_wait_ms`, `peer`, and `arrival_ns` after existing trace fields so
the trace parser contract is unchanged.

- [x] **Step 4: Implement receiver-only loop and central callback**

The receiver loop must:

1. call `_recv_next_qkv_batch_message_or_tensor()`;
2. validate `qkv_batch.shape[0] == descriptor.item_count`;
3. record `record_offload_exec_peer_batch()`;
4. create and enqueue one work item;
5. wait on the item's CPU completion event after successful enqueue;
6. release the input itself only if enqueue fails.

The dispatcher callback must call
`record_offload_exec_compute(..., source_batches=1)` before exactly one existing
compute call and send the result to `item.transport`.

- [x] **Step 5: Verify GREEN and legacy compatibility**

Run:

```bash
.venv/bin/python -m pytest tests/pap/test_pap_attention_executor.py -q
```

Expected: all executor tests pass, including existing legacy trace and message
lifetime tests.

---

### Task 3: FastAPI Lifecycle, Feature Flag, and Stats

**Files:**

- Modify: `examples/pap/pap_attention_executor.py`
- Modify: `tests/pap/test_pap_attention_executor.py`
- Modify:
  `.claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh`
- Modify: `tests/pap/test_pap_launch_files.py`

**Interfaces:**

- Consumes the receiver and dispatcher from Tasks 1–2.
- Produces environment contract
  `PAP_ATTENTION_DISPATCH_MODE=legacy|central_fifo`, defaulting to `legacy`.
- Extends `/v1/pap/attention/stats` with dispatcher mode, queue depth,
  enqueued/dispatched items, maximum queue depth, queue-wait sum/max, and
  dispatch failures.

- [x] **Step 1: Write failing mode/lifecycle tests**

Test that two mailbox binds in `central_fifo` mode start two receiver threads
but one shared dispatcher. Test that `legacy` starts the existing full mailbox
loop. Test invalid mode fails closed before accepting a peer.

- [x] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  -k 'dispatch_mode or central_dispatcher' -q
```

Expected: central-mode lifecycle assertions fail.

- [x] **Step 3: Wire one lazy dispatcher per PA process**

Store the dispatcher on `app.state`. On first central-mode bind, construct and
start it under `offload_exec_lock`. Start one receiver thread per peer. Add a
shutdown handler that stops the dispatcher with drain enabled. Preserve all
legacy app-state fields and response payloads.

- [x] **Step 4: Propagate and record the runner flag**

Add a default `PAP_ATTENTION_DISPATCH_MODE=legacy`, export it to each Attention
process, and write it to `effective_config.env` and `run_metadata.json`.
Continue capturing `/v1/pap/attention/stats` after session drain.

- [x] **Step 5: Verify GREEN and shell syntax**

Run:

```bash
.venv/bin/python -m pytest \
  tests/pap/test_pap_attention_executor.py \
  tests/pap/test_pap_launch_files.py -q
bash -n \
  .claude/skills/vllm-pap-benchmark/scripts/run_pap_same_pd_workload.sh
```

Expected: all tests pass and Bash syntax check returns zero.

---

### Task 4: Phase 1 Correctness and Performance Checkpoint

**Files:**

- Results only under:
  `test/baseline/pap/results/runs/20260711_phase1_*`

**Interfaces:**

- Consumes `PAP_ATTENTION_DISPATCH_MODE` and the existing self-contained PAP
  benchmark runner.
- Produces strict legacy/central artifacts with the same local model,
  topology, transport, request workload, MPS 70/30 setting, and GPU placement.

- [x] **Step 1: Run focused verification without pre-commit**

Run all Task 1–3 tests, the proxy/Qwen routing tests, `git diff --check`, and
shell syntax checks. Require zero failures.

- [x] **Step 2: Run 2PA2P full-crossbar central correctness smoke**

Use GPUs 1–4, local-fast, fixed MPS 70/30, 8 requests, input 128, output 8,
QPS 2, and `crossbar_round_robin`. Require 8/0 completion, all four pair counts
equal 2, correctness audit passed, and session drain 0.

- [x] **Step 3: Check the Phase 1 invariant**

For each PA, require:

```text
offload_exec_peer_batches == offload_exec_compute_calls
offload_exec_source_batches_per_compute_sum == offload_exec_compute_calls
offload_exec_max_source_batches_per_compute == 1
dispatcher_enqueued == dispatcher_dispatched
dispatcher_failures == 0
```

This proves Phase 1 changed ownership but did not accidentally combine work.

- [x] **Step 4: Run same-code 1PA1P legacy/central QPS 4 A/B**

Run three no-trace repetitions per mode using the canonical i128/o32/prefix50,
128 requests, QPS 4 workload. Compare median TPOT across repetitions. Require
central correctness and no more than 3% median TPOT regression before enabling
central mode for Phase 2 work.

Completed on commit `d654f6011`: legacy three-run median TPOT was
`28.138 ms`; `central_fifo` was `28.514 ms` (`+1.34%`). Median TTFT changed
from `169.799 ms` to `170.455 ms` (`+0.39%`). All six runs completed `128/0`
and passed correctness, routing, and session-drain audits.

- [x] **Step 5: Checkpoint commit**

Stage only Phase 1 source, tests, plan/design documentation, and benchmark
script changes. Do not stage result directories or unrelated untracked files.
Commit with an AI-assistance trailer and exact test commands/results in the
handoff notes.
