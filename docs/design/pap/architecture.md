---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260715-ARCHITECTURE-MILESTONE
  - PAP-20260715-RUNTIME-BOUNDARY-E2E
  - PAP-20260715-MODEL-ADAPTER-E2E
  - PAP-20260715-INTEGRATION-E2E
  - PAP-20260715-P17-POST-REFACTOR
  - PAP-20260710-ARBITRARY-XY
  - PAP-20260711-ATTENTION-COMBINE
  - PAP-20260714-SEAL-HANDOFF-KV
last_validated_commit: 9fb642937d27f8871ce653216f8b70d64176679a
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
   current decode K/V, runs paged FlashAttention, and returns the output.
6. Sampled tokens, decode commits, ACKs, leases, and session drain close the
   request without transferring KV ownership to Projection.

Projection never owns prompt history. Attention is an execution service over
Prefill-owned state, not a second durable KV owner.

## Topology and execution

`PAPTopology` accepts any positive `<x>pa<y>p` shape. One PA group contains a
Prefill role and its Attention service; Projection count is independent.
Request routing selects a stable `(PA, Projection)` pair for a turn.

- One Projection source uses direct Attention execution.
- Multiple active Projection sources use topology-derived combine/scatter.
- Active membership is request-cohort state; removed global dispatch and
  adaptive-coalescing selectors are not supported runtime modes.

| Capability | Implementation | Milestone validation |
| --- | --- | --- |
| 1PA1P, same host | `local_fast` + CUDA IPC | P17 release gate |
| xPAyP, same host | direct/combine over `local_fast` | Preserved, contract only |
| xPAyP, cross host | NIXL mailbox/backend | Preserved, contract only |
| TP or other models | Existing integration boundary | Outside P17 gate |

## Module ownership

- `config.py`: typed topology, placement, transport, MPS, lifecycle, and
  feature configuration; retired selectors fail closed.
- `gateway/`: the OpenAI-compatible request boundary, role clients, payload
  construction, topology selection, and Prefill–Attention–Projection request
  orchestration. `1PA1P` is the `x=1, y=1` form of the same gateway path.
- `integration/`: the only PAP-to-vLLM glue boundary. `scheduler.py`,
  `engine.py`, `worker.py`, `kv_cache.py`, and `api.py` translate PAP state for
  their matching vLLM owners; `runner.py` owns Projection request state,
  forward-context construction, peer activity, and asynchronous sampled-token
  delivery. `settings.py` parses shared process settings once per owner.
- `model/`: typed forward-batch access, Projection Attention execution, and
  Prefill sealed-KV publication used by model implementations.
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
  and `compute.py` own queueing and Attention compute.
- `transport/`: backend-neutral contract plus NIXL; `local_fast.py` owns peer
  binding and lifecycle, while `local_fast_io.py` owns wire encoding and the
  send/receive hot path.
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

Offline trace analysis and remote-Attention reports live under
`benchmarks/pap/tooling/` and are invoked through `tools/pap_*`; the runtime
package does not import benchmark tooling. The obsolete
`decode_commit_router.py` was removed because the current Prefill control path
is exclusively owned by `prefill_control_router.py`.
