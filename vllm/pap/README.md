# PAP source layout

PAP separates Prefill, Attention and Projection. A PA combines a vLLM Prefill
worker with a separate Attention service on the same GPU. Projection uses vLLM
for the remaining Decode model computation and scheduling.

## Ownership

| Module | Responsibility |
| --- | --- |
| `gateway/` | HTTP requests, Dynamo PA selection, Prefill-to-Decode orchestration, cancellation and request-wide cleanup |
| `integration/` | Adapt PAP request and KV state to vLLM EngineCore, scheduler, worker and model runner |
| `model/` | Model-facing Prefill KV publication and Projection whole-step CUDA Graph execution |
| `attention/` | Attention service execution, whole-step CUDA Graphs and backend selection |
| `kv/` | KV mapping, readiness, paged metadata, Decode updates, token commits and block leases |
| `transport/` | NVSHMEM Attention–Projection communication and tracing |
| `protocol/` | Shared descriptors, service request models and KV control-message codecs |

These are responsibility boundaries, not a strict dependency hierarchy.
Import KV objects from their owning modules: `kv/__init__.py` deliberately
does not initialize the Attention registry when a Prefill worker imports leases.

## Main paths

- Request: `gateway/app.py` → `gateway/request_pipeline.py` → Prefill →
  request-specific KV readiness → Projection → completion/cancellation cleanup.
- Prefill KV: `kv_connector.py` → `model/prefill.py` → `kv/handoff.py`
  → Attention's session registry. Prefill publishes only allocator-owned prompt
  blocks. CUDA IPC maps the storage;
  HTTP/TCP carries control information, not the full KV tensors.
- Decode: `integration/runner.py` → `model/step_graph.py` →
  `model/projection.py` → NVSHMEM → `attention/step_graph.py`
  → the selected Attention backend → NVSHMEM return.
  This is the logical execution path; Attention step preparation precedes QKV.
- KV lifecycle: Projection publishes accepted tokens through
  `kv/decode_token_client.py`; Attention joins token and KV readiness in
  `kv/decode_commit.py`, then uses `kv/control_client.py` to submit commits or
  release leases to Prefill. `kv/lease.py` protects blocks still in use.

Decode KV capacity uses per-request low-watermark prefetch. Prefill readiness
starts the first asynchronous allocation, and subsequent steps start another
256-token allocation when writable headroom drops below 256 tokens. Each request
has at most one allocation in flight, normally keeping future headroom between
256 and 512 tokens. Attention waits only if Decode reaches the current writable
end before that request completes. Background threads perform only the HTTP
round trip; authoritative block IDs are installed at an Attention preflight
boundary, where all layer views are updated atomically and step/PAT metadata is
invalidated before the new blocks are used.

The owning Prefill EngineCore serves allocations through
`/v1/pap/prefill/decode-allocate`; the vLLM KV manager remains the only allocator.
Allocation changes capacity, not the committed-token watermark.
`PAP_UNIFIED_KV_DECODE_CAPACITY_TOKENS` is the fallback per-request Decode limit
when the client supplies no output limit; it is not eagerly allocated capacity.

Chunked-Prefill preemption uses generation fencing. The scheduler requests an
Attention revocation and waits for acknowledgement before it mutates the running
queue or returns blocks to vLLM. Revocation is accepted only before Decode claims
the layout. Resumed Prefill publishes the next generation, and late manifests
from older generations are ignored. Decode-owned layouts are never reclaimed
through this Prefill-preemption path.

The allocator grows by 256 tokens per request. Near a request's declared output
limit, the final growth may be smaller. If no evictable or free block can satisfy
a request before Decode reaches its writable end, the step is rejected before
QKV is accepted; active Decode KV is not silently preempted or aliased. A future
admission/backpressure policy can improve availability under complete
non-evictable KV saturation without weakening this ownership rule.

`gateway/lifecycle.py` remains separate: it owns the entire distributed
request, including engine cancellation and router reservations, not only KV.

PAP selects a PA with Dynamo on every conversation turn; it has no other
routing policy. The native Dynamo selector owns the KV event index. Each PA has
exactly one Attention endpoint. PAP registers both the live vLLM KV-event endpoint and its replay
endpoint, so Dynamo can recover a missing ZMQ batch instead of retaining an
incomplete prefix index.

Same-host Dynamo routing can copy a conversation prefix when a later turn moves
to another PA. Set `PAP_ENABLE_PA_KV_MIGRATION=1`. The Gateway tracks the
previous PA, asks it to pin the longest matching block-hash prefix, and passes
its process-lifetime CUDA IPC catalog to the selected target. Export requests
use an allocator-synchronized snapshot side channel in the EngineCore process;
they do not wait behind model steps in the EngineCore utility queue. The target vLLM
allocator creates its own blocks, the Prefill connector copies only blocks not
already cached locally, and source pins are released by tracked asynchronous
Gateway tasks. New turns fail closed after a release failure. Cross-host PA KV
migration is not implemented by this CUDA IPC path.
`gateway/tokenizer.py` renders prompt tokens and block hashes for routing and KV
migration, without a second Python KV-event subscriber.
`gateway/topology.py` maps the chosen PA to its fixed Projection owner.

Within vLLM integration, `engine.py` owns control validation and application;
`scheduler.py` also owns accepted-token publication, and `projection.py` owns
Decode route grouping. `model/projection.py` includes the generic Attention
execution binding; endpoint installation lives in `endpoint_plugin.py`.

## Upstream vLLM touch points

PAP state machines stay in `vllm/pap/`. Upstream files only expose
registration points or narrow delegation:

| File | Hook |
| --- | --- |
| `entrypoints/openai/{chat_completion,completion}/serving.py` | PAP tokenized-input preparation when model hooks are enabled |
| `model_executor/layers/attention/{attention,execution}.py` | Attention execution factory/override registration |
| `forward_context.py` | Additional forward-context keys |
| `v1/core/sched/scheduler.py` | `PAPSchedulerAdapter`, Decode lookahead and token notifications |
| `v1/core/sched/output.py`, `v1/outputs.py` | Per-request `kv_transfer_params` and step `extension_data` carriers |
| `v1/core/kv_cache_manager.py` | `allocate_local_slots=False` for compute-only Projection requests |
| `v1/core/block_pool.py` | Allocator-synchronized cached-prefix pinning for KV snapshot export |
| `v1/worker/gpu/model_runner.py` | `PAPModelRunnerAdapter` / `PAPKVCacheAdapter` |

## Attention backends

- `attention/backend.py`: plan/selector contracts, PAT-or-Triton selection and
  the unified `run_pap_decode_attention` entry point.
- `attention/pat_backend.py`: PAT-specific plans, fixed-address metadata,
  incremental updates and backend execution.
- `attention/triton_backend.py`: Triton paged-decode launch settings,
  workspace/metadata caches and execution.

Selection reuses a valid previous decision/plan, otherwise selects PAT for
physical KV reuse when available, or uses Triton. Graph entries retain the plan
state they capture.

## Entry points and shared helpers

- `plugin.py` and `endpoint_plugin.py`: vLLM model/control integration.
- `service.py`: standalone Attention service.
- `prefill_control_router.py`: ordered Prefill controls and Projection abort
  acknowledgement.
- `config.py`: runtime configuration and model-hook activation.
- `transport/binding.py`: transport construction, Projection cache and peer
  binding; `transport/nvshmem/` owns the implementation and native CUDA bridge.
- `deferred_cuda_trace.py`, `runtime_cuda_context_audit.py` and the NVSHMEM
  step recorder: optional diagnostics.

The default `local` NVSHMEM mode uses same-host P2P and a UID file. The
`ibgda` mode uses TCP UID bootstrap and symmetric source staging for device-side
puts across hosts. Both modes execute the mandatory whole-step CUDA Graph
protocol; the runtime does not fall back between them.

`attention/compute.py` only prepares step state for the Graph executor. Graph
execution uses the full row-stable `graph_slot_tensor`, with `-1` for rows that
must not append KV.

Static dead-code reports are review candidates, not deletion instructions.
HTTP decorators, enum/serialized fields, native-resource lifetime references
and test cleanup helpers may have no ordinary Python call site.

Tests live in [tests/pap](../../tests/pap/README.md); workloads and experiments
live in [benchmarks/pap](../../benchmarks/pap/README.md). A source-layout change
does not establish an end-to-end performance result or require replaying every
dataset. Run the relevant existing tests; obtain approval before a new workload
run.
