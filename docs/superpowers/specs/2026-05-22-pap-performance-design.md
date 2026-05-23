# PAP Performance Design

Date: 2026-05-22

## Goal

PAP stands for Prefill-Attention-Projection. The performance version is not an
optimization of the current remote-attention prototype. It changes the execution
contract:

- Prefill owns prompt processing and creates the initial KV state.
- Attention owns decode-time KV state and paged attention compute.
- Projection owns model parameters and parameterized decode compute: QKV
  projection, output projection, MLP/MoE, norm, logits, and sampling.

The key invariant is:

```text
Projection must not run local decode attention.
Attention must not load model weights.
```

The current prototype violates the first invariant intentionally for validation:
`Qwen3Attention.forward()` runs local `self.attn(q, k, v)` first and then
replaces the output with remote attention. That proves the dataflow, but it
cannot represent PAP's performance potential.

## Refined Direction After Re-reading Adrenaline

The main refinement from Adrenaline is that PAP should not treat Attention as a
normal externally scheduled service. The Attention role should be an internal
executor of the P+A group, with scheduler-visible KV blocks and a GPU tensor data
plane. The proxy may expose three logical roles, but it must not sit in the
per-token or per-layer path.

The resulting PAP boundary is:

```text
OpenAI proxy:
  request admission, conversation stickiness, P+A group selection

P+A group:
  Prefill worker, Attention internal executor, KV/session owner,
  decode-step coordinator

Projection pool:
  parameterized decode compute, batching across P+A groups, no persistent KV
```

This is different from the current debug implementation. A debug Attention
executor that receives historical KV over HTTP is useful for correctness tests,
but the performance design must send only current Q/K/V and must keep historical
KV resident in the Attention executor.

## Adrenaline Findings

Adrenaline is useful as a reference, but it should not be copied directly.

Useful mechanisms:

- It creates an attention-only runner. `OffloadAttn` contains an attention
  module and no full model weights.
- The decode model computes QKV, scatters QKV to the attention executor, and
  gathers attention output back.
- It keeps vLLM/FlashInfer paged KV metadata instead of inventing a separate KV
  layout.
- The attention backend writes K/V with `reshape_and_cache_flash` and then runs
  FlashInfer paged decode.
- It separates data, scheduling, finish, and synchronization communicators.
- The proxy tracks request sequence length, block consumption, and offload
  placement, then uses a simple load estimator based on KV blocks and measured
  attention bandwidth.
- It runs prefill and attention on the same physical GPU under MPS, with
  different active-thread percentages.

Important limits:

- It is partial attention offload, not PAP. Decode remains a full vLLM model
  instance and still owns normal projection/MLP/logits work.
- Offload is request-level and decided at the proxy. PAP needs decode-step and
  resource-level coordination inside the P+A group and Projection pool.
- Its scatter/gather plus barrier pattern is acceptable for a research
  prototype, but too synchronous for a low-latency PAP data plane.
- It wraps Llama model code directly. PAP should make the split at backend or
  runner boundaries so it does not become a model-by-model patch stack.

Conclusion: borrow Adrenaline's attention-only runner, paged KV metadata,
MPS/IPC discipline, and load-aware block accounting. Do not borrow its
request-level offload semantics as the final architecture.

### Control Plane Versus Tensor Data Plane

Adrenaline has a useful split that PAP should make explicit:

- The HTTP proxy is a request-level control plane. It assigns request ids,
  forwards prefill/decode requests, and tracks storage/load state.
- Tensor movement does not go through the proxy. Decode-to-Attention QKV and
  Attention-to-Decode O use distributed process groups, not HTTP payloads.
- Prefill-to-Attention KV transfer is same-device IPC through a transfer buffer.
- Attention has scheduler-visible state, because it receives the scheduled
  request ids and independently maintains block-table/KV state.

For PAP this becomes a hard rule:

```text
The proxy may route sessions, but the P+A coordinator owns the token loop.
The P+A coordinator and Projection runtime own tensor scheduling.
```

If a design requires the proxy to call Attention for each layer or token, it is
not the performance design.

### Code-Level Findings From Adrenaline

The important Adrenaline files are concentrated in its own `adrenaline/`
package plus a small set of vLLM patches:

- `adrenaline/entrypoints/adrenaline_proxy_server.py`
- `adrenaline/proxy/request_dispatcher.py`
- `adrenaline/proxy/storage_manager.py`
- `adrenaline/proxy/load_estimator.py`
- `adrenaline/model_runner/model_runner.py`
- `adrenaline/model_runner/attn_runner.py`
- `adrenaline/model_loader/models/offload_attn.py`
- `adrenaline/attention/backends/flashinfer.py`
- `vllm/core/scheduler.py`
- `vllm/distributed/parallel_state.py`
- `vllm/distributed/kv_transfer/agent/offload_exec_agent.py`
- `vllm/distributed/kv_transfer/agent/kv_transfer_in_device.py`

The architecture is effectively:

```text
client
  -> Adrenaline proxy
  -> Prefill vLLM producer
  -> Decode vLLM consumer
  -> Attention offload executor
```

The proxy assigns one `request_id`, chooses whether the request should offload
decode attention, runs Prefill with `max_tokens=1`, then forwards the full
decode request to the Decode instance. When offload is enabled, it first
broadcasts the request to Attention instances and waits for the special
`[Add Offload]` stream marker before starting Decode. This is a control-plane
handshake, not a tensor data plane.

The proxy keeps a `StorageManager` that tracks:

- total KV blocks on Decode and Attention locations
- used blocks and max-used blocks per location
- request-to-location mapping
- current and maximum block count per request

`LoadEstimator` estimates attention time as:

```text
num_active_blocks * block_size * token_size / measured_attention_bandwidth
```

and estimates total decode time as:

```text
projection_gemm_time(batch) + attention_time + measured_delta(batch)
```

This is directly useful for PAP, but the unit of placement changes. Adrenaline
chooses local decode attention versus offloaded attention. PAP chooses a P+A
group and a Projection worker, then must continuously balance Attention KV
pressure against Projection queueing.

The real attention split happens below the proxy:

- `AttentionRunner.load_model()` forces the Adrenaline FlashInfer backend and
  constructs `OffloadAttn` instead of a full model.
- `OffloadAttn` creates only an attention module and exposes
  `forward_with_offload(kv_cache, attn_metadata)`.
- Decode computes QKV inside the normal model. For offloaded requests,
  `get_offload_group().scatter_tensor_to_offload(qkv)` sends QKV to the
  Attention executor.
- Attention splits QKV, calls `forward_with_blk`, writes K/V into its paged KV
  cache with `reshape_and_cache_flash`, runs FlashInfer paged decode attention,
  and returns O with `gather_tensor_to_decode`.

The scheduler is also patched. Decode broadcasts scheduled offload request ids
to Attention via `Offload_attn_exec_agent.boardcast_sche_out`. The Attention
scheduler allocates or appends its own blocks for the same request ids, but only
for requests whose `prefill_rank` maps to that Attention instance. This is the
key design lesson: Attention must have scheduler-visible block-table state. It
cannot be a stateless service that receives arbitrary tensors.

Prefill-to-Attention KV movement uses `KV_transfer_agent_in_same_device`.
Prefill inserts K/V and hidden states into a same-device transfer buffer.
Attention receives them asynchronously, then writes received K/V into its own
paged cache with `reshape_and_cache_flash`. The code assumes prefill and
attention are paired by rank and co-located, which matches PAP's P+A group.

MPS launch scripts reinforce the intended physical placement:

- Prefill and Attention both use `CUDA_VISIBLE_DEVICES=0`.
- Prefill uses a larger MPS percentage, e.g. 70%.
- Attention uses a smaller MPS percentage, e.g. 30%.
- Decode runs on a separate GPU.

For PAP this becomes:

```text
P+A GPU:
  Prefill process, large SM budget
  Attention executor, smaller SM budget, owns decode KV

Projection GPU:
  model weights and parameterized decode compute
```

The launch scripts also show a practical asymmetry that PAP should preserve:
Prefill gets most SMs during prompt ingestion, while Attention gets a smaller
but persistent MPS share for decode KV attention. This does not isolate memory
bandwidth, so the P+A scheduler still needs admission control based on active KV
blocks and observed attention time.

### Implementation Decisions After Code Review

The Adrenaline code suggests four concrete PAP decisions.

First, the Attention role should be built as an executor/runner, not as an
OpenAI-compatible model server. Adrenaline's `AttentionRunner` swaps the loaded
model for `OffloadAttn`, and `OffloadAttn` constructs only an attention module
from `VllmConfig`. PAP should do the same in V1: create a
`PAPAttentionExecutor` that initializes the attention backend, KV cache arena,
FlashInfer/vLLM metadata state, and lifecycle methods, but never calls the
normal model loader for weights. The executor API should be:

```text
create_session(session_id, request_id, max_seq_len, block_table)
import_prefill_kv(session_id, layer_kv_handles, slot_mapping, seq_len)
append_and_compute(batch[layer_id, session_id, q, k, v, slot, seq_len]) -> o
free_session(session_id)
rollback(session_id, num_tokens)
```

Second, PAP should mirror schedule decisions into Attention, not send arbitrary
attention calls from Projection. Adrenaline patches the Decode scheduler to
broadcast scheduled offload request ids, and the Attention scheduler replays
allocation/append for those ids. PAP should adopt the pattern with different
roles:

```text
P+A scheduler decides the decode batch and KV slots
Projection scheduler merges parameterized work
Attention executor receives the same batch descriptor and appends K/V into
its own paged KV according to scheduler-owned slots
```

The critical contract is that Attention's block table is scheduler-visible.
Attention cannot be a stateless function of `(q, k, v, historical_kv_blob)`.

Third, prompt KV import must be a first-class lifecycle step. Adrenaline's
same-device transfer agent exports an IPC buffer, copies prompt K/V selected by
`slot_mapping`, then the Attention side writes those tensors into its paged KV
with `reshape_and_cache_flash`. PAP should borrow this exact semantic shape.
Without this step, `true_split` can append decode K/V but cannot correctly
attend over the prompt or over prior conversation turns.

Fourth, Adrenaline's NCCL/Gloo scatter/gather and global barriers are a useful
correctness reference but not the final PAP hot path. The first PAP performance
version may use a simple collective channel to avoid HTTP serialization, but
the target path should be event-driven GPU buffers:

```text
Projection writes Q/K/V into a pre-registered buffer
Projection records a CUDA event
Attention waits, appends K/V, computes O, records completion
Projection waits and continues O projection/MLP
```

Barriers should only appear at startup, graph capture, and error recovery.

### What PAP Should Borrow

PAP should borrow these mechanisms:

- Attention-only executor construction. The Attention executor should load no
  model weights and should expose a narrow append-and-compute API.
- Scheduler-visible Attention KV state. The P+A scheduler must allocate,
  append, free, and account Attention KV blocks explicitly.
- vLLM/FlashInfer paged KV metadata. Reuse block tables, slot mappings,
  paged-kv indices, indptr, and last-page lengths instead of inventing a new
  layout.
- `reshape_and_cache_flash` style K/V append. Decode K/V should be appended to
  Attention-owned paged KV before paged decode attention runs.
- Separate control and data synchronization. Adrenaline uses separate channels
  for schedule, data, finish, and barriers. PAP should keep that separation,
  but replace global barriers on the hot path with CUDA events and bounded
  queues.
- Proxy-side capacity awareness. PAP proxy/coordinator should track KV block
  pressure and Projection queue pressure rather than using pure round-robin.
- MPS co-location discipline. Prefill and Attention should start as separate
  MPS clients on the same GPU, with explicit SM percentages.

The highest-value borrowing is the `OffloadAttn` idea, not the exact class.
`OffloadAttn` proves that an attention-only vLLM runner can be built without
loading model parameters. PAP should build an equivalent V1-side internal
executor that owns:

- one paged KV arena per active model/parallel shard
- per-request session metadata
- per-layer block tables
- FlashInfer/vLLM attention metadata builders
- an append-and-compute entry point for decode tokens

The executor should not know about QKV projection, O projection, MLP/MoE, logits,
or sampling.

### First Performance Milestone

The current `true_split` prototype removes local decode attention from the
Projection path and sends only current-token Q/K/V. That is necessary but not
yet sufficient for a performance claim. The first PAP performance milestone
should require all of the following:

- Attention imports prompt KV from Prefill and uses it in the first decode
  token.
- Attention appends decode K/V into an Attention-owned paged KV cache on every
  token.
- Projection does not own or read historical KV for decode attention.
- Q/K/V and O move through a tensor channel, not HTTP JSON/base64.
- The P+A scheduler can report active sessions, allocated KV blocks, and decode
  sequence lengths.
- A single-turn greedy correctness test matches normal vLLM within numerical
  tolerance.
- A two-turn sticky-session test reuses turn-1 generated KV in turn 2 without
  transferring decode KV back to Prefill.

Until these are true, benchmark results should be labeled as prototype or
debug-mode PAP, not PAP performance mode.

### What PAP Should Not Copy

PAP should not copy these Adrenaline choices as final design:

- Request-level offload semantics. PAP is not deciding whether to offload
  attention; PAP always separates decode attention from Projection in
  performance mode.
- Full Decode vLLM as the Projection instance. In PAP, Projection should not
  own persistent KV and must not run local decode attention.
- Per-request stream marker handshakes. They are fine for a prototype but too
  coarse for per-token, per-layer PAP execution.
- Global barrier-heavy execution. Barriers are acceptable for startup, graph
  capture, and failure recovery, but the per-layer path needs event-based
  producer/consumer synchronization.
- Model-by-model invasive wrappers. Adrenaline wraps Llama-oriented pieces.
  PAP may begin Qwen3-only, but the boundary should move toward runner/backend
  abstractions.

PAP should also avoid copying Adrenaline's "full Decode worker plus offloaded
attention" interpretation. In PAP, Projection is not a normal Decode instance.
It is closer to a parameter server for decode-layer compute: it loads weights,
runs the parameterized operations, and calls the P+A group's Attention executor
for attention output.

### Current vLLM V1 Integration Points

The current vLLM V1 codebase gives PAP three useful insertion points:

1. Request control metadata
   - `vllm/v1/request.py` stores `kv_transfer_params`.
   - `vllm/v1/core/sched/output.py` can preserve those params in
     `NewRequestData`.
   - This is enough for proxy-selected P+A group, Attention endpoint, session
     id, and Projection policy hints.

2. Worker forward metadata
   - `vllm/v1/worker/gpu/model_runner.py` builds `InputBatch`, block tables,
     slot mappings, and `forward_context`.
   - `forward_context.additional_kwargs` can carry PAP mode and request ids.
   - `forward_context.slot_mapping` and per-layer `attn_metadata` are the
     natural bridge from scheduler state into attention-layer code.

3. Attention backend boundary
   - Model code currently calls `self.attn(q, k, v)`.
   - A true performance split must intercept before that local call.
   - The durable implementation should move from Qwen3 model hooks toward an
     attention backend or runner-level PAP adapter so other models are not
     patched layer by layer.

The current PAP prototype already uses the first two points, but only for a
debug path. In `debug_remote_attention`, Projection still computes local
attention, gathers local historical KV, and sends a serialized attention
request to the Attention executor. That path is useful for correctness
instrumentation only. It is not a performance architecture.

## Recommended Architecture

```text
OpenAI client
    |
    v
PAP proxy
    |
    v
P+A group 0..N
  - Prefill vLLM worker
  - Attention executor on the same GPU
  - KV/session/block owner
  - Decode coordinator
    |
    | PAP tensor data plane
    v
Projection pool 0..M
  - model weights
  - QKV/O/MLP/MoE/logits/sampling
  - no persistent KV
```

### Optimized Logical Roles

PAP should expose three logical roles but only two physical GPU roles:

```text
Physical P+A GPU:
  Prefill role
    full prompt forward, prompt KV creation
  Attention role
    decode KV owner, paged attention, no model weights

Physical Projection GPU:
  Projection role
    decode QKV/O/MLP/MoE/logits/sampling, no persistent KV
```

The Attention role should be launched as an internal executor of the P+A group.
It may be a separate MPS process in the first performance implementation, but it
should be supervised by the P+A group rather than treated as an independent
OpenAI endpoint. This preserves independent SM budgeting while keeping KV
ownership local to P+A.

### Optimized Hot Path

The per-token hot path should be:

```text
P+A coordinator:
  pick decode batch
  send compact batch descriptor to Projection

Projection:
  for each layer:
    produce q,k,v for the current token batch
    write q,k,v to a GPU-direct send buffer
    signal Attention

Attention:
  append k,v to Attention-owned paged KV
  run paged decode attention over historical KV
  write o to the return buffer
  signal Projection

Projection:
  run o_proj, residual, MLP/MoE
  sample next token
  return token ids and minimal sampling side effects
```

No historical KV moves from Projection to Attention. No q/k/v/o tensors are
serialized through HTTP or JSON. Projection can batch work from multiple P+A
groups, but every layer call must preserve the request-to-Attention mapping.

### Borrowing Map From Adrenaline

| Adrenaline mechanism | PAP use |
| --- | --- |
| `AttentionRunner` + `OffloadAttn` | Build an Attention internal executor with no model weights. |
| `AdrenalineFlashInferMetadata` | Reuse paged-KV metadata concepts: slot mapping, block tables, paged indices, indptr, last-page lengths. |
| `reshape_and_cache_flash` before paged decode | Append current decode K/V into Attention-owned KV before computing O. |
| `KV_transfer_agent_in_same_device` | Guide Prefill-to-Attention KV import using same-GPU IPC/shared buffers. |
| `Offload_attn_exec_agent` scatter/gather | Use as a correctness reference for QKV/O tensor collectives, but replace barrier-heavy hot path with event-driven queues. |
| `StorageManager` + `LoadEstimator` | Track Attention KV pressure and estimate attention time from active blocks and measured bandwidth. |
| MPS launch scripts | Start Prefill and Attention as separate MPS clients with asymmetric SM percentages. |

### Design Guardrails

Future implementation should satisfy these guardrails before running PAP/PD
performance comparisons:

- `pap_mode=true_split` must not call local decode attention inside Projection.
- Attention must maintain decode KV across tokens and across turns of a sticky
  conversation.
- Projection must send only current-token Q/K/V to Attention.
- The proxy must not perform per-layer or per-token tensor calls.
- Benchmark scripts must reject the debug HTTP/base64 attention mode when the
  run is labeled as PAP performance mode.
- P+A logs must report Attention KV blocks, active sessions, and MPS settings.
- Projection logs must report queue depth and merged decode batch sizes.

The proxy only handles ingress, session stickiness, and group selection. It
should not orchestrate per-layer compute. The per-token layer loop must stay
inside the PAP runtime because each decode token has one dependency chain per
layer.

The P+A group owns the request state:

- request id and conversation id
- generated token history
- sampling state or sampling metadata
- KV block table
- prefix-cache ownership
- Attention executor endpoint and buffer handles
- Projection placement for the current decode step

The Projection worker owns the parameterized layer loop for a decode step. It
calls Attention for each layer through a GPU-direct tensor channel and returns
sampled token ids, not full logits, in the normal path.

## Decode Data Flow

For one decode step over a batch of requests:

```text
P+A coordinator
  -> Projection.decode_step(batch descriptors, attention handles)

Projection:
  h = embedding(input_token) or previous pipeline input
  for layer in layers:
      q, k, v = qkv_proj + q/k norm + rope(h)
      send q,k,v descriptor to Attention
      wait/read attention output o
      h = o_proj + residual + mlp/moe
  logits = lm_head(norm(h))
  token = sample(logits, sampling metadata)
  -> P+A coordinator: token ids and compact sampling side effects

Attention:
  for each layer request:
      append k/v to paged KV cache using slot_mapping
      compute paged decode attention from q and historical KV
      write o to Projection output buffer
```

For multi-turn conversations, session stickiness is mandatory in the first
performance version: the same conversation must return to the same P+A group.
That avoids transferring decode KV back into prefill on the second turn.

## Component Contracts

### PAP Proxy

Responsibilities:

- assign a new conversation to a P+A group
- keep conversation stickiness
- forward OpenAI requests to the selected P+A coordinator
- optionally choose a Projection pool policy for the group

Non-responsibilities:

- no per-layer calls
- no tensor transport
- no KV block management

### P+A Coordinator

Responsibilities:

- own the vLLM scheduler-facing request state
- allocate and free KV blocks for Attention
- import prefill KV into Attention after prompt processing
- maintain request-to-block-table mapping
- build decode batches
- send decode-step work to Projection
- apply backpressure when Attention KV capacity or Projection queues are full

The coordinator should be implemented near the KV connector/model-runner
boundary, not as an OpenAI HTTP proxy feature.

### Attention Executor

Responsibilities:

- run on the same physical GPU as Prefill
- own or import the paged KV cache
- expose `append_and_compute(layer, q, k, v, metadata) -> o`
- support `create_session`, `import_prefill_kv`, `free_session`, and
  `rollback_tokens`
- use vLLM attention metadata semantics: block table, slot mapping,
  paged-kv indptr/indices/last-page-len

Recommended implementation:

- first version: separate process under MPS, launched as an internal executor
  by the P+A group supervisor
- control plane: local IPC/gRPC/HTTP only for handles and lifecycle
- data plane: CUDA IPC for same-GPU buffers; NIXL/UCX/GPUDirect for remote
  Projection traffic
- attention kernel: reuse vLLM V1 attention backend where possible; otherwise
  port the Adrenaline FlashInfer metadata pattern to V1

### Projection Worker

Responsibilities:

- load model weights
- run parameterized decode compute
- call Attention for decode attention
- batch work from multiple P+A groups when shapes and layer ids match
- run logits and sampling in the normal path

Non-responsibilities:

- no persistent KV cache
- no prefix cache ownership
- no request admission based on KV capacity

For the first implementation, Qwen3-only is acceptable. The model-specific code
must be isolated behind a `PAPProjectionRunner` so the next model does not
require invasive edits throughout vLLM.

## Transport Design

The current prototype uses HTTP JSON/base64 and gathers full historical KV from
Projection. The performance version must remove both.

Data plane:

- preallocate GPU ring buffers for QKV and attention output
- exchange buffer descriptors and CUDA event handles on the control plane
- send only QKV and attention output per layer, not historical KV
- keep KV resident in Attention
- support batched descriptors: `(session_id, layer_id, qkv_offset, out_offset,
  slot_mapping, block_table, seq_len)`

Same-GPU Prefill to Attention:

- prefer shared KV arena or CUDA IPC handles
- Prefill writes prompt KV once
- Attention imports block handles and becomes the decode owner

Projection to Attention:

- if same node with peer access: CUDA IPC or NCCL point-to-point
- if different node: NIXL/UCX/GPUDirect RDMA
- avoid per-layer Python serialization

Synchronization:

- use CUDA events for buffer readiness and completion
- avoid global barriers on the hot path
- reserve barriers for startup, graph capture, and failure recovery

## Scheduler Design

PAP needs two schedulers:

1. P+A scheduler
   - admits requests based on KV capacity and SLO
   - owns continuous batching and block allocation
   - sends decode steps to Projection

2. Projection scheduler
   - groups requests from many P+A groups into projection microbatches
   - preserves layer dependency order
   - applies queueing limits so one P+A group cannot monopolize the worker

Initial policy:

- conversation-sticky P+A group
- least-queue Projection selection per decode step
- greedy or simple sampling first
- no session migration

Later policy:

- load-aware Projection selection using measured GEMM/MoE time
- load-aware P+A admission using Attention KV block pressure and bandwidth
- optional migration only when the cost of KV movement is explicitly modeled

## MPS And Resource Partitioning

Prefill and Attention should run on the same physical GPU but with unequal
resource budgets.

Initial practical launch:

- one MPS daemon per P+A GPU
- Prefill process: larger SM percentage
- Attention process: smaller SM percentage
- short MPS pipe directory paths to avoid Unix socket length issues
- each client process sees the MPS GPU as local ordinal 0

Better version:

- enable per-context device multiprocessor partitioning
- create CUDA contexts with explicit SM affinity
- keep separate CUDA streams for KV import, attention compute, and control

MPS does not provide strict HBM bandwidth isolation. Admission control and
queueing are still needed to prevent Prefill from starving Attention or the
reverse.

## Requirements

### R1: No Local Decode Attention In Projection

WHEN PAP performance mode is enabled
THE SYSTEM SHALL bypass local vLLM decode attention in Projection.

WHEN a Projection layer computes QKV
THE SYSTEM SHALL obtain the attention output from the Attention executor before
running O projection.

### R2: Attention Owns Decode KV

WHEN prefill finishes for a request
THE SYSTEM SHALL make the prompt KV visible to the co-located Attention executor.

WHEN decoding appends a token
THE SYSTEM SHALL write the new K/V into the Attention-owned paged KV cache.

### R3: Projection Is Stateless With Respect To KV

WHEN a decode step finishes
THE SYSTEM SHALL allow Projection to drop all per-token hidden intermediates for
that request.

WHEN a request is resumed for the next token
THE SYSTEM SHALL reconstruct the required compute from input token, positions,
sampling metadata, and Attention session handles, not from Projection KV state.

### R4: Session Stickiness

WHEN a conversation has existing KV state
THE SYSTEM SHALL route the next turn to the same P+A group unless explicit KV
migration is implemented.

### R5: GPU-Direct Data Path

WHEN PAP performance mode transfers QKV or attention output
THE SYSTEM SHALL use a tensor data path based on CUDA IPC, NCCL, UCX, NIXL, or
equivalent GPU-direct transport.

WHEN the hot path executes a layer
THE SYSTEM SHALL NOT serialize tensors through JSON/base64.

## Implementation Phases

### Phase 0: Freeze The Prototype Boundary

- Keep the current prototype as `pap_remote_attention_debug`.
- Mark it as correctness/debug mode in config and docs.
- Add checks that performance benchmarks do not accidentally use the debug path.

### Phase 1: True-Split Correctness, Qwen3 Only

- Add a `PAPProjectionRunner` for Qwen3 decode.
- Remove local `self.attn(q, k, v)` from the PAP performance path.
- Add an Attention executor that receives QKV and returns attention output.
- Let Attention own decode KV.
- Support greedy decode first.
- Verify token-by-token equality or bounded numerical drift against baseline.

Transport can be simple in this phase, but it must not transfer historical KV
from Projection to Attention.

### Phase 2: Same-GPU KV Import

- Co-locate Prefill and Attention under MPS.
- Share or import prefill KV via CUDA IPC/shared arena.
- Implement request lifecycle: create, import prefill KV, append decode KV,
  free.
- Add multi-turn tests where the second turn reuses the first turn's decode KV.

### Phase 3: GPU-Direct Projection-Attention Data Plane

- Replace Python/HTTP tensor movement with ring buffers and GPU events.
- Add buffer descriptor protocol.
- Add backpressure and timeout handling.
- Measure per-layer QKV/O transfer cost.

### Phase 4: Projection Pool Batching

- Add Projection scheduler queue.
- Merge decode steps from multiple P+A groups.
- Batch QKV/O/MLP/MoE operations.
- Keep Attention calls grouped by target Attention executor and layer.

### Phase 5: Broader vLLM Integration

- Move model-specific hooks toward an attention backend or runner-level
  abstraction.
- Add non-greedy sampling support.
- Add TP/PP compatibility checks.
- Evaluate MoE models separately because routing and expert parallelism change
  the Projection workload.

## Main Difficulties

- vLLM assumes scheduler, model runner, KV cache, and attention backend live in
  one worker. PAP breaks that assumption.
- Every decode token has a strict per-layer dependency chain, so latency is
  dominated by repeated Projection-Attention round trips unless transport is
  GPU-direct and overlapped.
- KV block ownership must stay consistent across prefill, decode, prefix cache,
  cancellation, preemption, and multi-turn reuse.
- Sampling is awkward: returning logits to P+A is too expensive, but sampling
  inside Projection means Projection needs enough sampling state.
- CUDA graph capture is harder because the graph crosses logical executors.
- MPS can partition SMs, but not cleanly partition memory bandwidth.
- Failure cleanup is nontrivial: leaked KV blocks or stale IPC handles will
  corrupt later sessions.

## Testing Strategy

Correctness:

- one-token decode equivalence against normal vLLM
- multi-token greedy equivalence
- multi-turn session reuse
- cancellation frees Attention KV blocks
- Projection restart fails requests cleanly without leaking Attention sessions

Microbenchmarks:

- QKV/O transfer latency per layer
- Attention append-and-compute throughput by batch and sequence length
- Projection GEMM/MoE throughput by merged batch size
- Prefill/Attention MPS interference under different SM percentages

End-to-end:

- compare 6PA2P PAP performance mode against 6P2D PD baseline
- use both long-prompt and decode-bound workloads
- report TTFT, TPOT, ITL, request throughput, token throughput, queueing time,
  Attention KV usage, and Projection queue depth

Adrenaline's public scripts mostly use single-request-style benchmark samples
from sonnet/random/dataset serving. Even when ShareGPT-style data is prepared,
the serving benchmark issues independent OpenAI requests rather than preserving
a live multi-turn session with reused decode KV. PAP should therefore include an
explicit multi-turn benchmark in addition to the existing baseline matrix. The
minimum useful multi-turn test is: turn 1 generates N tokens, turn 2 appends a
new user prompt to the same conversation id, and the P+A group reuses the turn-1
decode KV without moving it back to Prefill.

## Open Decisions

- Whether Phase 1 Attention executor should be a separate MPS process or an
  in-process internal executor. Separate process matches the target resource
  model; in-process is easier for correctness.
- Whether Projection or P+A owns sampling for non-greedy decoding.
- Whether the first GPU-direct data path should use NIXL, NCCL point-to-point,
  or a minimal CUDA IPC extension.
- How much of Adrenaline's FlashInfer backend should be ported to vLLM V1
  versus adapting the current vLLM V1 attention backend.

## 2026-05-23 Adrenaline Re-read Addendum

After re-reading the reference code, the most important Adrenaline lesson is
that Attention is not an OpenAI-serving peer. It is a scheduler-mirrored
executor with its own paged KV cache. The proxy decides placement and starts a
handshake, but the tensor path is below the proxy.

The strongest code-level patterns to borrow are:

- `adrenaline/model_runner/attn_runner.py`: `AttentionRunner.load_model()`
  forces the Adrenaline attention backend and constructs `OffloadAttn` instead
  of loading a full model. PAP should build the V1 equivalent: an internal
  `PAPAttentionExecutor` that initializes attention backends, metadata
  builders, and KV cache arenas, but never loads QKV/O/MLP/logits weights.
- `adrenaline/model_loader/models/offload_attn.py`: `OffloadAttn` receives QKV
  through a distributed group, splits Q/K/V, calls an attention-only module, and
  gathers O back. PAP should borrow the role boundary, not the exact
  scatter/gather API.
- `adrenaline/attention/backends/flashinfer.py`: the real stateful append is
  `reshape_and_cache_flash(key, value, kv_cache, slot_mapping, ...)` followed by
  paged FlashInfer decode using `paged_kv_indices`, `paged_kv_indptr`, and
  `paged_kv_last_page_len`. PAP must preserve this semantic shape. A CPU tensor
  list of historical KV is only a debug substitute.
- `vllm/distributed/kv_transfer/agent/kv_transfer_in_device.py`: Prefill
  selects prompt KV from its paged cache by slot mapping, puts it in an IPC
  buffer, and Attention writes it into its own paged cache with
  `reshape_and_cache_flash`. This is the right prompt-KV import model for PAP's
  co-located P+A GPU.
- `vllm/core/scheduler.py`: Adrenaline broadcasts scheduled offload request ids
  from Decode to Attention, and the Attention scheduler replays block
  allocation/append decisions for only the requests mapped to its prefill rank.
  PAP should implement the V1 version of this as scheduler-visible Attention
  block state, not as arbitrary RPC calls from Projection.
- `adrenaline/proxy/storage_manager.py` and `load_estimator.py`: storage and
  time estimates are based on active KV blocks, block size, token KV size, and
  measured attention bandwidth. PAP should extend this to two queues:
  Attention KV pressure on each P+A group and Projection queue/GEMM pressure on
  each Projection worker.

The reference also clarifies what not to copy:

- Do not copy `[Add Offload]` stream markers or request-level HTTP handshakes
  into the hot path.
- Do not copy the old scheduler patches directly into V1. The current fork
  already has cleaner boundaries in `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/core/sched/output.py`, `vllm/v1/worker/gpu/model_runner.py`, and
  `KVConnectorBase_V1`.
- Do not keep Projection as a full Decode instance with local historical KV.
  That is Adrenaline's architecture, not PAP.
- Do not rely on global barriers except for startup, graph capture, and error
  recovery. PAP's per-layer path needs bounded GPU buffers and event
  synchronization.

### Optimized V1 Integration Design

The implementation should use three V1 surfaces:

1. Scheduler output and block ownership
   - `SchedulerOutput.num_scheduled_tokens` tells whether a batch is decode-only
     and whether every request has exactly one scheduled token.
   - `NewRequestData.kv_transfer_params` carries proxy-selected P+A group,
     Attention session id, Attention handle, and Projection policy hints.
   - The P+A scheduler owns the authoritative Attention block table. Projection
     receives descriptors; it does not allocate persistent decode KV.

2. Worker model runner and forward context
   - `GPUModelRunner.prepare_inputs()` already builds request ids, positions,
     sequence lengths, block tables, and slot mappings.
   - `set_forward_context(..., additional_kwargs=...)` is the current bridge for
     `pap_mode`, request ids, scheduled-token counts, and Attention handles.
   - Performance mode must be enabled only for decode-only batches where
     `max_query_len == 1` and every scheduled request is mapped to a valid
     Attention session.

3. Attention/connector boundary
   - The short-term Qwen3 hook can call a `PAPAttentionBackend`-style adapter,
     but the durable split should move out of model files.
   - A new PAP V1 connector should own lifecycle metadata:
     `create_session`, `import_prefill_kv`, `append_decode_kv`,
     `free_session`, `rollback`.
   - Worker-side connector methods already have useful lifecycle hooks:
     `register_kv_caches`, `start_load_kv`, `wait_for_layer_load`,
     `save_kv_layer`, and `wait_for_save`. PAP should reuse these concepts
     instead of inventing an unrelated KV-transfer API.

The concrete target is:

```text
P+A scheduler:
  allocate Attention KV blocks
  build Attention batch descriptors
  choose Projection worker per decode step

Prefill worker:
  run normal prompt forward
  export prompt KV to co-located Attention executor

Attention executor:
  import prompt KV into its paged KV cache
  append decode K/V with vLLM slot mapping
  compute paged decode attention and return O

Projection worker:
  run QKV projection and all parameterized decode compute
  send only current-token Q/K/V
  receive O
  never read historical KV in performance mode
```

### Fail-Closed Rules

These rules prevent debug PAP from being mistaken for performance PAP:

- `pap_mode=true_split` must fail if Prefill reports a nonzero prefix length
  but Attention has not imported prompt KV for that session.
- `pap_mode=true_split` must fail during benchmark launch if the tensor data
  plane is HTTP JSON/base64.
- Projection logs must prove that no local decode attention path was executed.
- Attention logs must report prompt-KV import, decode KV append, active session
  count, and allocated block count.
- Benchmark scripts must label HTTP/base64 runs as `debug_remote_attention`;
  they must not write those results under a PAP performance directory.
- A two-turn test is mandatory before any PAP-vs-PD claim: turn 2 must reuse
  turn-1 decode KV in the same P+A group without copying decode KV back to
  Prefill.

### Revised Implementation Order

The safest implementation order is now:

1. Freeze debug mode and add fail-closed guards.
   - Keep the current HTTP executor only as `debug_remote_attention`.
   - Register real `prefix_len` from Prefill metadata.
   - Make true-split decode reject missing prompt KV instead of silently
     attending over decode-only KV.

2. Build a scheduler-visible Attention executor skeleton.
   - Use a separate MPS process on the P+A GPU.
   - Allocate a real paged KV arena.
   - Mirror request/session/block metadata from the P+A scheduler.
   - Add introspection endpoints or local IPC for active sessions and blocks.

3. Implement same-GPU prompt KV import.
   - Start from Adrenaline's slot-mapping extraction and
     `reshape_and_cache_flash` writeback semantics.
   - Prefer a same-device IPC/shared-buffer path over HTTP tensor payloads.
   - Add single-turn correctness before enabling multi-token decode.

4. Replace QKV/O HTTP with a tensor channel.
   - First acceptable performance path: NCCL point-to-point or CUDA IPC ring
     buffers on one node.
   - Target path: pre-registered GPU buffers plus CUDA events.
   - No historical KV is transferred in this phase.

5. Move the Qwen3 hook toward a backend/runner adapter.
   - The Qwen3-only path is acceptable for the first paper-quality prototype.
   - The adapter must be isolated enough that the next model does not require
     duplicating the full layer implementation.

6. Add Projection batching and load-aware routing.
   - Route by both Attention block pressure and Projection queue depth.
   - Record merged Projection batch sizes, QKV/O transfer cost, and Attention
     append-and-compute time.

### Benchmark Readiness Gate

PAP is ready for the 6PA2P versus 6P2D comparison only when all of these pass:

- single-token greedy output matches normal vLLM for a non-empty prompt
- multi-token greedy output matches normal vLLM within numerical tolerance
- two-turn sticky conversation reuses Attention-owned decode KV
- no Projection-local historical KV reads are executed in performance mode
- data plane avoids HTTP JSON/base64 for QKV/O
- P+A logs expose active KV blocks and sessions
- Projection logs expose queue depth and merged decode batch size

Until then, PAP benchmark runs should be described as connectivity or debug
measurements, not performance results.

## Recommended Next Step

Start with Phase 1, Qwen3-only, separate config flag:

```text
pap_mode = debug_remote_attention | true_split
```

The acceptance gate for `true_split` is simple: Projection must not call local
decode attention, Attention must own decode KV, and greedy outputs must match
baseline for single-turn and multi-turn prompts.
