---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260724-SINGLE-PROJECTION-BATCH
  - PAP-20260722-AIPERF-PROJECTION-AUTO
  - PAP-20260722-AIPERF-PA090-EAGER
  - PAP-20260722-AIPERF-CONVERGENCE
  - PAP-20260721-AIPERF-PIECEWISE-CUDAGRAPH
  - PAP-20260721-AIPERF-AUDITED-CAPACITY
  - PAP-20260716-4GPU-CONV-AFFINITY
  - PAP-20260710-ARBITRARY-XY
  - PAP-20260711-ATTENTION-COMBINE
  - PAP-20260714-SEAL-HANDOFF-KV
last_validated_commit: cb6fe35009905c32b52a8ad10b7e00d778c03679
---

# PAP architecture

PAP separates prompt processing, attention state, and decode projection while
keeping one ownership rule: Prefill owns the physical paged KV cache for the
whole request.

## Roles and data flow

1. The Gateway selects a Prefill–Attention group and a Projection endpoint.
2. Prefill processes prompt chunks and owns prompt plus decode KV blocks.
3. Prefill publishes a static KV catalog and one generation-bound request
   manifest to its colocated Attention service.
4. Projection runs the KV-unaware decode model path. For each step it sends
   Q/K/V to Attention and receives the attention output.
5. Attention opens the Prefill-owned KV through CUDA IPC or NIXL, appends the
   current decode K/V, runs vLLM's Triton paged-decode kernel, and returns the
   output.
6. Sampled tokens, decode commits, ACKs, leases, and session drain close the
   request without transferring KV ownership to Projection.

Projection never owns prompt history. Attention is an execution service over
Prefill-owned state, not a second durable KV owner.

## Topology and execution

`PAPTopology` accepts any positive `<x>pa<y>p` shape. One PA group contains a
Prefill role and its Attention service; Projection count is independent.
The normal round-robin policy balances requests. The
`conversation_affinity` policy instead assigns each new conversation to the
next PA and reuses that PA for every later turn; Projection selection remains
independent.

- The Gateway admits each PA to one Projection source for a complete request
  wave. Requests from that source may batch; another source takes ownership
  only after the active wave drains.
- PAP never splits one vLLM scheduler batch into independently inflight
  microbatches. vLLM async scheduling remains enabled: it may prepare the next
  scheduler step while the current full model forward runs, but
  `UniProcExecutor` does not submit that next batch into the model until the
  current 36-layer forward returns.
- A global batch may contain requests routed to several PA groups. Projection
  computes QKV once for the whole batch, splits it only for same-step fan-out,
  lets the independent Attention services run concurrently, gathers every
  shard, and then continues the layer. These route groups are not microbatches
  and cannot interleave layer execution with another Projection batch.
- Different PA groups remain independent, so x:y routing does not serialize
  unrelated Attention services.
- Conversation-affinity state lives in the Gateway and contains only the
  conversation-to-PA owner plus token-free counters. KV locality and physical
  ownership remain inside the selected PA.
- Attention retains topology-derived combine/scatter mechanics, but the current
  Gateway path does not use opportunistic cross-source per-layer cohorts.

| Capability | Implementation | Milestone validation |
| --- | --- | --- |
| 3PA1P, same host | `local_fast` + CUDA IPC | Current AIPerf matrix |
| Other xPAyP, same host | Gateway wave admission over `local_fast` | Controlled smoke |
| xPAyP, cross host | NIXL mailbox/backend | Preserved, contract only |
| TP or other models | Existing integration boundary | Outside the current matrix |

## Execution modes

Eager execution is the default AIPerf mode. The
optional `piecewise` mode reuses vLLM's compile and token-count CUDA Graph
dispatch without attempting to capture PAP host side effects.

- Prefill and Projection receive explicit process-static Graph roles. Model
  code never selects a role from per-request Python metadata during capture or
  replay.
- Projection remote Attention and Prefill KV publication are registered as
  graph-unsafe opaque operations. They divide graph-safe QKV, normalization,
  output projection, residual, and MLP regions without replaying network or
  lifecycle operations.
- Synthetic capture forwards produce shape-correct Projection outputs but do
  not open sessions or send transport messages.
- Prefill captures scheduled-token sizes `1,2,4,8,16,32,64,128`; Projection
  and PD Decode capture `1,2,4,8,12,16,20,24,28,32`. Other shapes fall back to
  normal execution and are not rejected.

Full-model CUDA Graph is unsupported because OFFLOAD_EXEC, remote Attention,
KV publication, and request-generation state are dynamic host-controlled
operations. The accepted boundary is implemented in `model/cudagraph.py` and
connected to vLLM piecewise splitting operations; it is not a separate graph
executor.

## Module ownership

- `config.py`: typed topology, placement, transport, MPS, lifecycle, and
  feature configuration; retired selectors fail closed.
- `gateway/`: the OpenAI-compatible request boundary, role clients, payload
  construction, topology selection, Projection wave admission, and
  Prefill–Attention–Projection request orchestration. `1PA1P` is the `x=1,
  y=1` form of the same gateway path.
- `integration/`: the only PAP-to-vLLM glue boundary. `scheduler.py`,
  `engine.py`, `worker.py`, `kv_cache.py`, and `api.py` translate PAP state for
  their matching vLLM owners; `runner.py` owns Projection request state,
  forward-context construction, peer activity, and asynchronous sampled-token
  delivery. `settings.py` parses shared process settings once per owner.
- `model/`: typed forward-batch access, Projection Attention execution,
  Prefill sealed-KV publication, piecewise CUDA Graph boundaries, and the
  Projection checkpoint-weight memory planner used by launchers.
- `protocol/`: wire models, descriptors, sealed KV codec, and transport
  contracts.
- `topology/`: route groups and Projection peer membership.
- `lifecycle/`: asynchronous sampled-token delivery and joining, decode commits,
  ACKs, and lease release/ownership.
- `kv/`: `registry.py` owns session/catalog/lifecycle state;
  `decode_state.py` owns unified-KV slot planning, append, and readiness waits;
  the package also contains sealed handoff, metadata, models, IPC opening, and
  optional observability.
- `attention/`: `runtime.py` owns the service-facing runtime;
  `execution.py` owns direct/combine mailbox execution; `peers.py` owns
  Projection membership, transports, and receiver threads; `dispatcher.py`
  and `compute.py` own queueing and Attention compute; `kernels.py` owns the
  selected decode kernel and its step-level workspace.
- `transport/`: `factory.py` is the composition boundary. `local/` owns the
  same-host CUDA IPC endpoint, stream-ordered protocol, and send/receive hot
  path. `nixl/` owns the cross-host mailbox message, endpoint, and OFFLOAD_EXEC
  adapter. The shared ownership-bearing interface lives in `protocol/`.
- `service.py`: thin Attention HTTP/TCP composition and process entry point.

Legacy top-level compatibility façades have been removed; the retirement record
is [compatibility.md](compatibility.md). Runtime code, launchers, and tests now
import their owning modules directly. The vLLM scheduler, engine, worker, API
server, KV manager, and both model runners call surface-specific `integration/`
adapters instead of parsing PAP metadata, environment settings, or lease state
themselves. KV block allocation and decode-commit mutation remain in the vLLM
KV manager because that owner controls its internal block structures. Qwen3's
thin tensor interception points delegate PAP execution to `model/`; none of
these entry points define alternate runtime algorithms.

## Import ownership rule

New top-level forwarding modules are not allowed. Launchers use
`python -m vllm.pap.service` and `python -m vllm.pap.gateway.app`; code and
tests import `attention/`, `gateway/`, `kv/`, `lifecycle/`, `protocol/`,
`topology/`, `transport/`, and surface-specific `integration/` owners directly.
Historical experimental branches are reproduced from their Git commits and raw
artifacts, not from selectable code paths in the current runtime.

The Attention kernel selection is one main path, not a runtime experiment switch.
`PAPAttentionStepContext` prepares block metadata and a fixed split-4 Triton
workspace once per decode step, then reuses them across all model layers. The
runtime does not retain the former per-layer FA2 fallback. Historical
`paged_flash_*` trace field names remain stable so old and new runs can be
compared without rewriting experiment data.

Offline trace analysis and remote-Attention reports live under
`benchmarks/pap/tooling/` and are invoked through `tools/pap_*`; the runtime
package does not import benchmark tooling. The obsolete
`decode_commit_router.py` was removed because the current Prefill control path
is exclusively owned by `prefill_control_router.py`.
