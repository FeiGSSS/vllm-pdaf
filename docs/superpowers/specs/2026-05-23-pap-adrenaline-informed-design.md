# PAP Adrenaline-Informed Design

Date: 2026-05-23

## Purpose

This document narrows the PAP implementation plan after re-reading the
Adrenaline reference implementation. Its job is to prevent later implementation
from drifting back into a normal P/D split or a debug remote-attention RPC path.

PAP has three logical roles and two physical GPU roles:

```text
P+A GPU role:
  Prefill process: normal prompt forward, owns prompt compute
  Attention executor: parameter-free, stateful decode attention, owns decode KV

Projection GPU role:
  Projection process: model weights, QKV/O projection, MLP/MoE, logits/sampling
```

The non-negotiable performance invariant is:

```text
Projection does not read historical KV or run local decode attention.
Attention does not load model weights.
```

The current HTTP/base64 `true_split` path is a debug scaffold. It is useful only
to validate request ids, shape contracts, and fail-closed behavior. It is not the
performance design.

## What Adrenaline Contributes

Adrenaline should be read as an attention-executor reference, not as a final PAP
architecture.

Borrow these mechanisms:

- `AttentionRunner` replaces the full model with an attention-only module.
- `OffloadAttn` receives QKV, splits Q/K/V, runs only attention, and returns O.
- The attention executor owns a paged KV cache and scheduler-visible block
  state.
- Prompt KV is selected from Prefill's paged cache by slot mapping and written
  into Attention's paged cache with `reshape_and_cache_flash`.
- Decode K/V append also uses `reshape_and_cache_flash`, followed by paged
  FlashInfer/vLLM attention using block tables and seq-len metadata.
- The proxy tracks KV block pressure and uses measured attention bandwidth to
  make placement decisions.
- Prefill and Attention are co-located under MPS with different active-thread
  percentages.

Do not borrow these parts as final PAP design:

- Request-level offload semantics. PAP splits every decode layer into
  Projection plus Attention, not a subset of requests.
- A full Decode instance that still owns normal KV and local attention.
- HTTP stream markers such as `[Add Offload]` in the hot path.
- Barrier-heavy scatter/gather synchronization for every token/layer.
- Model-by-model patches as the long-term split point.

## Target Runtime Shape

The proxy is a control plane only:

```text
Client -> PAP proxy
  proxy selects sticky P+A group and Projection policy
  proxy sends prompt to Prefill
  proxy starts or resumes an Attention session
  proxy starts the decode token loop through the PAP coordinator
```

The per-token/layer tensor path is below the proxy:

```text
Projection worker:
  hidden state -> QKV projection
  send current-token Q/K/V and batch descriptor

Attention executor:
  append current K/V into local paged KV
  run paged decode attention over prompt + prior decode KV
  return attention output O

Projection worker:
  O projection -> MLP/MoE -> logits/sampling
```

Historical KV remains on the P+A GPU. Only current-token Q/K/V and O cross the
Projection/Attention boundary.

## Attention Instance Design

The Attention role should be implemented as an internal executor of a P+A group,
not as an OpenAI-compatible vLLM server.

Responsibilities:

- Allocate and own an Attention KV arena.
- Maintain session metadata: request id, conversation id, seq len, block table,
  slot mapping, imported prompt length, generated decode length, and owner P+A
  group.
- Import prompt KV from the co-located Prefill process.
- Append one decode K/V per scheduled token and layer.
- Compute decode attention output for each layer.
- Expose cheap introspection: active sessions, allocated blocks, imported prompt
  tokens, decode tokens, failed imports, and per-step timing.
- Free or roll back KV on request finish, abort, preemption, or speculative
  rejection.

Suggested executor API:

```text
create_session(session_id, request_id, conversation_id, max_seq_len)
allocate_blocks(session_id, block_ids, seq_len)
import_prefill_kv(session_id, layer_name, key, value, slot_mapping, seq_len)
append_and_compute(session_id, layer_name, q, k, v, slot, seq_len, descriptor) -> o
rollback(session_id, target_seq_len)
free_session(session_id)
```

The first performance implementation can keep this API behind a Python class,
but the tensors must not move through JSON/base64.

### Scheduler Visibility

Attention must be scheduler-visible. A stateless function
`attention(q, k, v, historical_kv_blob)` is the wrong abstraction.

Adrenaline's decisive pattern is that Decode broadcasts scheduled offload
request ids, and the Attention scheduler replays block allocation and append for
those same request ids. PAP should adapt this to vLLM V1:

- P+A scheduler owns the authoritative Attention block table.
- Scheduler output carries Attention session ids and block/slot descriptors.
- Projection receives descriptors, but does not allocate persistent KV.
- Attention validates that each append matches the scheduler descriptor before
  mutating KV.

This is what makes multi-turn reuse practical: turn 2 can resume the same
Attention session and append new prompt/decode tokens without transferring turn
1 decode KV back to Prefill.

### Prompt KV Import

Prompt KV import is a lifecycle step, not an optional optimization.

Required behavior:

```text
Prefill completes prompt forward
Prefill extracts prompt K/V from its paged KV cache using slot_mapping
Prefill sends prompt K/V to the co-located Attention executor through same-GPU IPC
Attention writes prompt K/V into its own paged KV cache with reshape_and_cache_flash
Attention marks prefix_len imported for the session
Projection decode may start only after this mark exists
```

The current fail-closed behavior is correct: if `prefix_len > 0` and Attention
has not imported prompt KV, decode must fail rather than silently attending over
only decode KV.

### Decode KV Append

For every decode layer and scheduled token:

```text
Projection computes q, k, v
Attention receives q, k, v plus slot/seq descriptor
Attention writes k/v into its paged KV cache with reshape_and_cache_flash
Attention runs paged decode attention over the session block table
Attention returns o
Projection continues with o_proj and the rest of the layer
```

This mirrors Adrenaline's `forward_with_blk` semantic shape. The implementation
should reuse vLLM/FlashInfer paged metadata rather than building a separate flat
KV history in Python.

## Projection Instance Design

Projection is a parameterized decode compute worker. It owns model weights and
stateless per-token compute, but it does not own historical KV.

Responsibilities:

- Batch decode requests from many P+A groups.
- Run QKV projection for each layer.
- Send only current-token Q/K/V to the correct Attention executor.
- Receive attention output O and run O projection.
- Run MLP/MoE, residual/norm, logits, and sampling.
- Report queue depth, merged batch size, GEMM/MoE time, tensor-transfer time,
  and sampled tokens back to the coordinator.

In the first Qwen3-only implementation, a layer-level hook in `qwen3.py` is
acceptable. The hook must remain isolated behind a PAP adapter so the next model
can move the split toward an attention backend or runner boundary.

Performance mode guardrails:

- decode-only batch
- one scheduled token per request
- valid Attention session for every request
- valid slot/seq descriptor for every layer
- no call to local decode attention in Projection
- no HTTP/base64 tensor transport

## vLLM V1 Integration Points

Use existing V1 surfaces before inventing new ones.

Scheduler side:

- `vllm/v1/core/sched/scheduler.py` already creates `SchedulerOutput` and knows
  scheduled request ids, new token counts, and KV blocks.
- `vllm/v1/core/sched/output.py` now preserves `NewRequestData.kv_transfer_params`.
- PAP should add scheduler metadata for Attention session id, P+A group id,
  Projection target, block ids, slot ids, and seq len.

Worker side:

- `vllm/v1/worker/gpu/model_runner.py` already builds request ids, positions,
  scheduled-token counts, and forward-context `additional_kwargs`.
- PAP should pass batch descriptors through this path until a cleaner backend
  adapter owns them.
- `Qwen3Attention` can remain the temporary split point only for the prototype.

Connector side:

- `KVConnectorBase_V1` already defines useful lifecycle hooks:
  `register_kv_caches`, `start_load_kv`, `wait_for_layer_load`, `save_kv_layer`,
  and `wait_for_save`.
- NIXL's scheduler/worker split shows how scheduler metadata becomes worker
  transfer metadata.
- PAP should create a connector/adapter for P+A local Attention import and
  Projection/Attention QKV/O transport, instead of putting tensor movement in
  the HTTP proxy.

## Data Plane Choices

The data plane can evolve in stages, but benchmark labels must stay honest.

Debug stage:

- HTTP JSON/base64 tensors
- CPU tensor KV history
- correctness and fail-closed tests only
- no performance claims

First performance stage:

- same-GPU IPC/shared buffer for Prefill -> Attention prompt KV
- NCCL point-to-point or CUDA IPC ring buffers for Projection <-> Attention
  QKV/O
- Attention paged KV arena on GPU
- Qwen3-only split hook

Target stage:

- pre-registered GPU buffers
- CUDA event synchronization
- bounded queues per Projection/Attention pair
- no per-token proxy calls
- backend/runner-level split rather than model-file split

## Proxy View

From the proxy's perspective, the system should look like logical PAP even when
Prefill and Attention share a physical GPU:

```text
P+A group 0: prefill endpoint, attention control endpoint, local tensor channel
P+A group 1: prefill endpoint, attention control endpoint, local tensor channel
...
Projection pool: projection workers with queue-depth and throughput telemetry
```

Proxy responsibilities:

- admission and request id creation
- conversation stickiness to a P+A group
- placement based on active Attention KV blocks and Projection queue depth
- prompt prefill dispatch
- Attention session creation and readiness checks
- result streaming to client
- cleanup on finish/abort

Proxy non-responsibilities:

- per-layer attention calls
- QKV/O tensor movement
- historical KV transport
- token-level synchronization in the hot path

## MPS Launch Model

A P+A physical GPU runs two logical executors under MPS:

```text
CUDA_VISIBLE_DEVICES=<pa_gpu> CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70 Prefill
CUDA_VISIBLE_DEVICES=<pa_gpu> CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30 Attention
```

The exact percentages should be tunable. The first benchmark matrix should vary
at least `80/20`, `70/30`, and `60/40`, because Attention pressure grows with
active KV blocks while Prefill pressure grows with prompt throughput.

Projection runs on separate GPUs and should maximize GEMM/MoE batching:

```text
CUDA_VISIBLE_DEVICES=<projection_gpu> Projection worker
```

MPS controls SM scheduling, not memory bandwidth. The P+A admission controller
must still cap active sessions/blocks to avoid Prefill and Attention bandwidth
interference.

## Multi-Turn Semantics

Conversation stickiness belongs to the P+A group. The same conversation id must
resume the same Attention session unless the session was explicitly evicted.

Turn 1:

```text
Prefill imports prompt KV into Attention
Attention appends generated decode KV token by token
session seq_len = prompt + generated
```

Turn 2:

```text
Proxy routes to same P+A group
New user tokens are appended through Prefill/Attention import path
Turn-1 decode KV remains in Attention; it is not copied back to Prefill
Projection computes only current-token parameterized decode work
```

If session eviction is required, the system should fail back to normal P/D or
re-prefill explicitly. Silent loss of decode KV is invalid.

## Bench Readiness Gate

Do not compare PAP against 6P2D as a performance result until all gates pass:

- prompt KV import uses same-GPU IPC/shared buffers, not HTTP/base64
- Attention owns GPU paged KV and logs active block count
- Projection logs prove local decode attention is skipped
- QKV/O transport avoids HTTP/base64
- single-turn greedy output matches normal vLLM
- multi-token decode matches normal vLLM within tolerance
- two-turn conversation reuses Attention-owned decode KV
- abort/finish frees Attention sessions and blocks
- benchmark output labels include P+A MPS split and Projection queue metrics

Until these pass, results should be named `debug_remote_attention`, not PAP
performance.

## Implementation Order

1. Keep debug mode fail-closed.
   - `true_split` must reject missing prompt KV for nonzero prefix length.
   - Debug runs must be excluded from performance result directories.

2. Build Attention executor skeleton.
   - Separate MPS process on the P+A GPU.
   - Real session/block table state.
   - GPU paged KV arena allocation.
   - Introspection for sessions, blocks, and import state.

3. Implement prompt KV import.
   - Extract Prefill K/V by slot mapping.
   - Transfer to Attention via same-GPU IPC/shared buffer.
   - Write into Attention paged KV with `reshape_and_cache_flash`.

4. Implement Projection/Attention tensor channel.
   - Replace HTTP QKV/O with NCCL P2P or CUDA IPC ring buffers.
   - Keep Qwen3-only layer split initially.
   - Enforce no local attention in Projection.

5. Add scheduler-visible Attention descriptors.
   - P+A scheduler allocates Attention blocks.
   - Projection receives descriptors.
   - Attention validates descriptors before appending K/V.

6. Add load-aware routing and benchmark harness.
   - Extend Adrenaline-style block/time estimator to PAP.
   - Route by Attention block pressure and Projection queue depth.
   - Run 6PA2P versus 6P2D only after readiness gates pass.

## Near-Term Code Landing Points

Current prototype files:

- `examples/pap/pap_attention_executor.py`: keep as debug control-plane and
  introspection scaffold; replace CPU KV history with paged GPU KV in the
  performance executor.
- `examples/pap/pap_proxy_server.py` and `multi_pap_proxy_server.py`: keep
  request routing and prefix-length registration; remove tensor hot-path calls
  from performance mode.
- `vllm/pap/shadow_attention.py`: keep debug HTTP helpers; introduce a separate
  tensor-channel adapter for performance mode.
- `vllm/model_executor/models/qwen3.py`: keep temporary Qwen3 split hook behind
  strict guards.
- `vllm/v1/worker/gpu/model_runner.py`: continue passing PAP descriptors through
  forward context until a backend/runner adapter replaces the model hook.
- `vllm/distributed/kv_transfer/kv_connector/v1/`: add or adapt a PAP connector
  for scheduler/worker metadata and local prompt-KV import.

## Main Risks

- Scheduler ownership: if Projection allocates or mutates persistent KV, PAP
  collapses back into decode.
- Tensor channel latency: QKV/O transfer must be lower than the saved local
  attention cost after batching.
- MPS interference: Prefill and Attention share memory bandwidth even with SM
  partitioning.
- Model-specific hooks: Qwen3-only is acceptable for first results, but the
  split boundary must not require rewriting every model.
- Multi-turn correctness: generated decode KV lives only in Attention, so
  eviction and resume semantics must be explicit.
