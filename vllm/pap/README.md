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
| `transport/` | Same-host NVSHMEM Attention–Projection communication and tracing |
| `protocol/` | Shared descriptors, service request models and KV control-message codecs |

These are responsibility boundaries, not a strict dependency hierarchy.
Import KV objects from their owning modules: `kv/__init__.py` deliberately
does not initialize the Attention registry when a Prefill worker imports leases.

## Main paths

- Request: `gateway/app.py` → `gateway/request_pipeline.py` → Prefill →
  request-specific KV readiness → Projection → completion/cancellation cleanup.
- Prefill KV: `kv_connector.py` → `model/prefill.py` → `kv/handoff.py`
  → Attention's session registry. CUDA IPC maps the storage;
  HTTP/TCP carries control information, not the full KV tensors.
- Decode: `integration/runner.py` → `model/step_graph.py` →
  `model/projection.py` → NVSHMEM → `attention/step_graph.py`
  → the selected Attention backend → NVSHMEM return.
  This is the logical execution path; Attention step preparation precedes QKV.
- KV lifecycle: Projection publishes accepted tokens through
  `kv/decode_token_client.py`; Attention joins token and KV readiness in
  `kv/decode_commit.py`, then uses `kv/control_client.py` to submit commits or
  release leases to Prefill. `kv/lease.py` protects blocks still in use.

`gateway/lifecycle.py` remains separate: it owns the entire distributed
request, including engine cancellation and router reservations, not only KV.

PAP supports only Dynamo PA selection, on every conversation turn. Retired
round-robin and conversation-affinity policies fail validation; there is no
fallback selector. The native Dynamo selector owns the KV event index.
`gateway/tokenizer.py` renders prompt tokens and hashes for routing and local
load accounting, without a second Python KV-event subscriber.
`gateway/topology.py` maps the chosen PA to its fixed Projection owner.

Within vLLM integration, `engine.py` owns control validation and application;
`scheduler.py` also owns accepted-token publication, and `projection.py` owns
Decode route grouping. `model/projection.py` includes the generic Attention
execution binding. The unused `integration/api.py` and `worker.py` adapters
were removed; endpoint installation remains in `endpoint_plugin.py`.

## Attention backends

- `attention/backend.py`: plan/selector contracts, PAT-or-Triton selection and
  the unified `run_pap_decode_attention` entry point.
- `attention/pat_backend.py`: PAT-specific plans, fixed-address metadata,
  incremental updates and backend execution.
- `attention/triton_backend.py`: Triton paged-decode launch settings,
  workspace/metadata caches and execution.

Backend selection remains unchanged: reuse a valid previous decision/plan,
otherwise select PAT for physical KV reuse when available, or use Triton.
Graph entries retain the plan state they capture. This layout change does not
change kernel math, Graph ownership, wire formats or asynchronous delivery.

## Entry points and shared helpers

- `plugin.py` and `endpoint_plugin.py`: vLLM model/control integration.
- `service.py`: standalone Attention service.
- `prefill_control_router.py`: ordered Prefill controls and Projection abort
  acknowledgement.
- `config.py`: runtime configuration and model-hook activation.
- `transport/binding.py`: transport construction, Projection cache and peer
  binding; `transport/nvshmem/` owns the implementation and native CUDA bridge.
- `deferred_cuda_trace.py` and `runtime_cuda_context_audit.py`: optional
  diagnostics. The unconnected prefix-cache audit and old eager-Attention
  locality logger were removed; active cached-token reporting remains.

`attention/compute.py` now only prepares step state for the Graph executor.
The unused per-batch Attention execution path, adapter logging wrappers and
redundant transport views are gone. `PAP_OFFLOAD_EXEC_TRACE`,
`PAP_OFFLOAD_EXEC_TRACE_LAYER`, `PAP_PREFIX_CACHE_AUDIT`,
`PAP_KV_LOCALITY_PROFILE` and `PAP_KV_LOCALITY_PROFILE_MIN_BATCH` are rejected
as retired settings. Use the active NVSHMEM step recorder and deferred tracing
for new diagnostics; they do not claim to reproduce the retired formats.

Recursive cleanup also removed the old non-Graph KV append/slot-cache path,
its unused compact slot and active-index tensors, and unconnected fan-in event
collection. Graph execution still uses the full row-stable `graph_slot_tensor`
with `-1` for rows that must not append KV. Current slot-topology counters,
KV leases, readiness checks and the NVSHMEM PA/kernel/Projection recorder remain.
Legacy always-zero append counters are no longer advertised as live metrics;
the deferred trace's empty `fanins` output key is retained for report readers.

Static dead-code reports are review candidates, not deletion instructions.
HTTP decorators, enum/serialized fields, native-resource lifetime references
and test cleanup helpers may have no ordinary Python call site. In particular,
the resident dispatch stream is retained to own the native dispatcher's stream.

There are no forwarding modules at the removed `lifecycle/`, `topology/`,
`model/hooks.py`, `transport/factory.py` or `transport/projection.py` paths.
Use the current modules directly. Historical experiment snapshots retain the
source paths of their recorded revisions.

Tests live in [tests/pap](../../tests/pap/README.md); workloads and experiments
live in [benchmarks/pap](../../benchmarks/pap/README.md). A source-layout change
does not establish an end-to-end performance result or require replaying every
dataset. Run the relevant existing tests; obtain approval before a new workload
run.
