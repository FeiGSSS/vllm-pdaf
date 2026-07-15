---
pap_doc_schema: 1
status: current
canonical: null
superseded_by: null
related_experiments:
  - PAP-20260715-RUNTIME-BOUNDARY-E2E
  - PAP-20260715-MODEL-ADAPTER-E2E
  - PAP-20260715-INTEGRATION-E2E
  - PAP-20260715-P17-POST-REFACTOR
  - PAP-20260710-ARBITRARY-XY
  - PAP-20260711-ATTENTION-COMBINE
  - PAP-20260714-SEAL-HANDOFF-KV
last_validated_commit: 7f732f5bea71d2bd4698f7bd04c1415cc77115cc
---

# PAP architecture

PAP separates prompt processing, attention state, and decode projection while
keeping one ownership rule: Prefill owns the physical paged KV cache for the
whole request.

## Roles and data flow

1. The proxy selects a Prefill–Attention group and a Projection endpoint.
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
- `integration/`: typed request metadata plus one model-runner adapter owning
  Projection request state, forward-context construction, peer activity, and
  the asynchronous sampled-token bridge shared with vLLM.
- `model/`: typed forward-batch access, Projection Attention execution, and
  Prefill sealed-KV publication used by model implementations.
- `protocol/`: wire models, descriptors, sealed KV codec, and transport
  contracts.
- `topology/`: route groups and Projection peer membership.
- `lifecycle/`: asynchronous sampled tokens, decode commits, ACKs, and leases.
- `kv/`: sealed handoff, Prefill-owned registry state, paged metadata, data
  models, CUDA IPC opening, and optional KV observability.
- `attention/`: the service-facing runtime façade plus direct/combine dispatch
  and compute runtime.
- `transport/`: backend-neutral contract plus `local_fast` and NIXL backends.
- `service.py`: thin Attention HTTP/TCP and transport composition.

`attention_executor.py`, `remote_attention.py`, `shadow_attention.py`, and old
client modules may remain compatibility façades. Both vLLM model runners call
the same `integration/` owner; V1 fails closed for PAP while V2 provides the
asynchronous sampled-token callback. Qwen3 delegates its PAP model path to
`model/`; none of these entry points define alternate runtime algorithms.

## Compatibility rule

Stable launch/import entry points may forward to their new owners. Private
module layout can continue to change. Historical experimental branches are
reproduced from their Git commits and raw artifacts, not from selectable code
paths in the current runtime.
