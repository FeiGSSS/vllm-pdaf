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
  - PAP-20260713-ASYNC-DECODE-TOKEN-D2H
  - PAP-20260714-REGISTRY-LOCK-SAFE-ASYNC
  - PAP-20260714-SEAL-HANDOFF-KV
last_validated_commit: 7f732f5bea71d2bd4698f7bd04c1415cc77115cc
---

# PAP runtime

The current runtime has one accepted correctness path. Path-selecting variables
are parsed or rejected at service/launcher composition roots through
`PAPRuntimeConfig`; core components cannot use them to revive experimental
algorithms. Backend sizing, endpoint, timeout, and observability knobs still
have some compatibility reads inside their owner modules and remain explicit
cleanup debt rather than alternate runtime paths.

## KV publication and readiness

The sealed handoff has two levels:

- A static catalog describes each stable Prefill KV backing tensor and its
  CUDA IPC/NIXL metadata. Attention opens and reuses it.
- A request manifest binds a request and session generation to its catalog,
  block layout, prefix length, and GPU ready event.

Attention may activate a request only after the complete manifest is ready for
the expected generation. Stale generations and incomplete prefixes fail
closed. Registry locks protect validation and snapshots; CUDA handle opening,
copies, and kernel work stay outside the global control lock.

The unified paged state remains Prefill-owned. Input-driven cases such as a
ragged or partial batch may use a conservative correctness fallback, but no
environment variable can select the retired per-layer descriptor or
non-unified ownership paths.

## Decode-step lifecycle

For every active step:

1. Projection builds the topology-derived route plan and sends Q/K/V.
2. Attention waits for matching KV readiness, appends decode K/V, and executes
   direct or combine/scatter paged attention.
3. Attention publishes the result to Projection.
4. Sampled-token delivery is always asynchronous. Token and KV completion join
   by request/session generation before the step commits.
5. Decode commit and ACK advance the Prefill scheduler/cache transaction.
6. Lease release happens after the committed tail is safe to release.
7. Shutdown/drain requires no pending token, commit, lease, or Attention
   session state.

Session release retains a bounded tombstone for known generation-bound handles.
This makes a queued late sampled-token notification idempotent after DELETE,
while a request ID that was never registered still fails closed with 404.

Retries, bounded queues, timeouts, and failure propagation remain operational
controls. Synchronous sampled-token D2H, synchronous Prefill KV import,
diagnostic barriers, manual dispatch modes, dynamic MPS, and handoff mode
selectors have been removed or explicitly rejected as retired flags.

## Transport boundary

The execution transport is built through `transport/factory.py`:

- `local_fast` uses same-host CUDA IPC endpoints and a stream-ordered slot/
  doorbell protocol.
- NIXL uses the backend-neutral mailbox actor/message layer plus NIXL endpoint
  metadata, notification, and transfer progress.

Protocol and lifecycle callers depend on the transport interface rather than
backend-specific endpoint types. Cross-host NIXL is preserved but is not an E2E
gate in this milestone.

## Observability

Correctness logs, routing audit, decode-token join, commit/lease accounting,
MPS visibility, session drain, and optional deferred CUDA traces are evidence;
they must not change normal scheduling semantics. Trace-on timings are
diagnostic and cannot be promoted to normal performance results.

## Known refactor boundary

The major responsibilities now have explicit packages, and Qwen3 delegates PAP
forward-batch, Projection Attention, and Prefill KV publication to `model/`.
One `integration/runner.py` owner now supplies both model runners, and
`attention/controller.py` shields the service from registry/dispatcher
internals. KV data models, IPC opening, and observability are separate from the
registry. The remaining large state is the performance-sensitive unified-KV
registry and the local-fast/NIXL backend internals. Further splits should be
mechanical and owner-driven; they must not reintroduce removed runtime
selectors or broaden the P17 gate.
