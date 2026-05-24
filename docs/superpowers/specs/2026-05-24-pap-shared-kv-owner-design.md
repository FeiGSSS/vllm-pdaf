# PAP Shared KV Owner Design

## Purpose

The next PAP phase removes the remaining KV duplication between Prefill and
Attention on a PA node.

The current implementation already keeps Projection KV/data-plane stateless, but
Prefill and Attention still do not share one KV store. Prefill gathers prompt KV
from its vLLM paged cache and exports CUDA IPC descriptors; Attention opens the
descriptors and copies the data into its own registry storage. Decode K/V
arriving from Projection is also appended to Attention-local buffers. That means
the PA node can hold redundant KV copies, and Prefill cannot see decode KV
written during a previous turn.

The target is a single PA-owned KV pool:

- Prefill owns the real vLLM paged KV blocks.
- Attention reads and writes those blocks through local IPC.
- Decode K/V produced from Projection is appended into Prefill-owned blocks, not
  into an Attention-private KV copy.
- A later request in the same session can reuse first-turn prompt plus decode KV
  without copying it back from an external cache.

## Core Decision

Use one vLLM-facing connector, `PAPSharedKVConnector`, with internal backends
instead of multiple independent vLLM KV connectors.

```text
vLLM scheduler / worker hooks
        |
PAPSharedKVConnector
        |
PAKVOwner
        |
        +-- LocalResidentBackend
        |   - Same PA node / same GPU path.
        |   - Attaches existing Prefill-owned blocks.
        |   - Exposes CUDA IPC descriptors to Attention.
        |   - Does not copy KV.
        |
        +-- RemoteMigrationBackend
            - Cross-PA path.
            - Moves KV from one PA owner to another only when routing requires it.
            - Can use NIXL, RDMA, NCCL/P2P, LMCache/FlexKV-style transports.
```

To vLLM, this remains one connector. Internally, the connector asks `PAKVOwner`
where the session KV lives and chooses a backend:

- `LOCAL_RESIDENT`: attach and refcount existing Prefill-owned blocks.
- `REMOTE`: initial implementation reroutes or misses; later implementation
  migrates into local Prefill-owned blocks.
- `MISS`: normal Prefill allocation and compute.

## Why Not LMCache Semantics

LMCache is the right inspiration for vLLM integration, but not the exact data
model we want.

The default KVConnector/LMCache pattern is:

```text
external cache hit
  -> vLLM allocates local paged KV blocks
  -> connector loads/copies KV into those blocks
  -> attention computes from local blocks
```

That is correct for external storage, but it recreates the redundancy we are
trying to remove on a colocated PA node.

PAP shared KV should use resident attach semantics:

```text
session hit in Prefill-owned resident KV
  -> connector pins/refcounts existing blocks
  -> request block table attaches those blocks
  -> Attention reads/writes the same blocks through IPC
```

The connector should not allocate a second KV pool and copy resident KV into it
for the local path.

## Components

### PAKVOwner

`PAKVOwner` is the source of truth for session KV ownership on a PA node.

Responsibilities:

- Map `session_id` to resident block ids, sequence length, layer readiness, and
  placement.
- Own leases/refcounts that prevent vLLM from freeing session blocks while
  Attention or a later turn can reuse them.
- Export per-layer or cross-layer KV block descriptors for local IPC.
- Allocate or reserve decode slots in Prefill-owned blocks.
- Track which decode slots have been materialized by Attention.
- Release session blocks when the session expires or is explicitly closed.

`PAKVOwner` should not perform remote transport itself. It returns placement and
block metadata; transport backends perform data movement when needed.

### PAPSharedKVConnector

`PAPSharedKVConnector` adapts vLLM scheduler and worker hooks to `PAKVOwner`.

Scheduler-side responsibilities:

- Identify requests that carry PAP session metadata.
- Ask `PAKVOwner` whether a prefix is locally resident, remote, or missing.
- For local resident hits, return matched token counts without requesting
  load-copy semantics.
- Attach existing resident blocks to the request's logical block table.
- Keep blocks pinned after request finish when the session should remain alive.

Worker-side responsibilities:

- Register Prefill vLLM KV cache tensors with `PAKVOwner`.
- Publish CUDA IPC descriptors for resident block ranges.
- Coordinate readiness metadata with Attention.
- Avoid `start_load_kv()` copy behavior for `LOCAL_RESIDENT` requests.

The connector can still use `KVConnectorBase_V1` as the vLLM integration point,
but its local path must be an attach path, not a load path.

### LocalResidentBackend

This backend handles colocated Prefill and Attention.

Data-plane semantics:

- Prefill-owned paged KV tensors are the only real KV storage.
- Attention opens CUDA IPC handles for those tensors.
- Attention computes over Prefill-owned KV blocks directly.
- Attention writes decode K/V into Prefill-owned slots.
- No prompt KV or decode KV copy is kept in Attention-local registry storage.

Synchronization:

- Prefill writes must be visible before Attention reads prompt KV.
- Attention writes must be visible before any later Attention read or Prefill
  session reuse.
- The implementation should use CUDA events or stream-ordered waits where
  possible, and avoid global device synchronization on the hot path.

### RemoteMigrationBackend

This backend handles cross-PA placement.

If session KV lives on PA-A and a later turn is scheduled to PA-B, the system has
three choices:

- Route back to PA-A.
- Recompute on PA-B.
- Migrate KV from PA-A to PA-B.

The first implementation should prefer route-back or recompute. Migration can be
added later behind the same backend interface.

Remote migration is copy/move semantics, not zero-copy shared ownership:

```text
PA-A resident blocks -> transport -> PA-B allocated blocks
```

After migration, PA-B becomes the local resident owner or a managed replica.

## Transport Interface

Transport protocol details should live below `PAPSharedKVConnector`.

```python
class PAPKVTransport:
    def export_blocks(self, session_id, block_refs) -> TransportDescriptor: ...
    def import_blocks(self, descriptor, dst_blocks) -> TransferHandle: ...
    def wait(self, handle) -> None: ...
    def supports_zero_copy(self) -> bool: ...
    def locality(self) -> Locality: ...
```

Expected implementations:

- `CudaIpcTransport`: same GPU, cross process. Supports local zero-copy
  read/write and is the first target.
- `NcclTransport`: useful for same-node cross-GPU migration/copy, not ideal for
  long-lived random access shared ownership.
- `NixlTransport`: useful for cross-node migration over NIXL/RDMA-style paths.
- `RdmaTransport`: direct RDMA/UCX/Mooncake-style migration backend.
- `LmcacheTransport`: optional fallback for external store semantics; should
  not be used for the local resident no-copy path.

## Data Flows

### First Turn on a PA Node

1. Proxy sends a PAP request to a PA group.
2. Prefill vLLM computes prompt KV into its normal paged KV blocks.
3. `PAPSharedKVConnector` registers the session with `PAKVOwner`.
4. `PAKVOwner` pins the resident blocks and exports local descriptors for
   Attention.
5. Projection receives only PAP metadata and remains KV/data-plane stateless.
6. Attention reads prompt KV directly from Prefill-owned blocks.

### Decode Step

1. Projection sends one token's Q/K/V to Attention.
2. Attention requests or receives the target decode slot from `PAKVOwner`.
3. Attention writes K/V into the Prefill-owned block slot through CUDA IPC.
4. Attention computes attention over prompt plus decode blocks.
5. `PAKVOwner` records the token as materialized for the session.

### Later Turn on Same PA

1. Scheduler receives a request with the same PAP session id.
2. `PAPSharedKVConnector` asks `PAKVOwner` for resident prefix coverage.
3. For a local hit, the connector attaches existing blocks and pins them.
4. No external KV copy is performed.
5. Prefill computes only the new suffix that is not already resident.

### Later Turn on Different PA

1. Scheduler discovers the session is remote.
2. Initial behavior: route back to the owner PA if capacity allows, otherwise
   recompute.
3. Future behavior: use `RemoteMigrationBackend` to copy KV to the new PA and
   then attach it as local resident KV.

## Scheduler and Block Lifecycle

The most important invariant is that vLLM must not free resident session blocks
while `PAKVOwner` owns a lease.

Required scheduler/block-manager changes:

- A way to attach existing block ids to a request without allocating duplicate
  blocks.
- A way for request finish to transfer block lifetime to `PAKVOwner` instead of
  immediately freeing blocks.
- Session-level release to return leased blocks to vLLM.
- Validation that attached blocks match token prefix, block size, dtype, layout,
  and model identity.

This is closer to session-aware prefix-cache block reuse than to ordinary
external KV load.

## Performance Considerations

Attention TPOT is sensitive to this path, but CUDA IPC itself is not the main
risk. CUDA IPC maps the same GPU memory into the Attention process; it is not a
PCIe copy.

Performance risks:

- Opening IPC handles repeatedly on the hot path.
- Python per-layer/per-token descriptor work.
- Building contiguous temporary KV segments before attention.
- Using global CUDA synchronization instead of stream/event ordering.
- Fragmented block access patterns that current attention kernels cannot consume
  efficiently.

The implementation should cache opened handles, use block-table-style metadata,
and keep the attention kernel path as close as possible to normal paged
attention.

## Phased Implementation

### Phase 1: Local Descriptor and Ownership Skeleton

- Add `PAKVOwner` session metadata with no data movement.
- Register Prefill KV cache tensors and block ids.
- Export stable CUDA IPC descriptors for Prefill-owned KV tensors.
- Keep existing copy path as fallback.

### Phase 2: Zero-Copy Prefill KV Read

- Replace Attention `_prefill_kv` copies with views into Prefill-owned KV blocks.
- Compute attention over resident Prefill KV without copying to registry storage.
- Verify 1PA1P output and trace counts.

### Phase 3: Decode KV Write-Back

- Allocate/reserve decode slots in Prefill-owned blocks.
- Let Attention write Projection-provided K/V into those slots.
- Remove Attention-local `_decode_kv` as the source of truth.
- Verify that first-turn prompt plus decode KV is visible in `PAKVOwner`.

### Phase 4: Same-PA Multi-Turn Reuse

- Add session id metadata through proxy and scheduler.
- Attach resident session blocks for later turns.
- Compute only new suffix tokens.
- Verify the second turn reuses first-turn prompt plus decode KV without copy.

### Phase 5: Remote Placement Policy

- Add remote placement status to `PAKVOwner`.
- Start with route-back or recompute.
- Add `RemoteMigrationBackend` only after the local resident path is stable.

## Testing

Unit tests:

- `PAKVOwner` lease/refcount behavior.
- Resident block attach and release.
- Decode slot reservation and materialization.
- CUDA IPC descriptor serialization without tensor payloads.
- Connector lookup returning local, remote, and miss states.

Contract tests:

- Local resident hits do not call a load-copy path.
- Attention does not keep `_prefill_kv` / `_decode_kv` as the source of truth in
  shared mode.
- Projection payload remains metadata-only and KV/data-plane stateless.
- Non-PAP vLLM and ordinary KVConnector paths remain unchanged.

E2E tests:

- 1PA1P first turn with zero-copy Prefill KV read.
- 1PA1P decode K/V write-back into Prefill-owned blocks.
- Same-PA second turn reuses first-turn prompt plus decode KV.
- X:Y routing stays correct when route-back policy is enabled.

## Open Boundaries

- The first implementation should not attempt cross-PA migration.
- The first implementation should not attempt to support all transport
  protocols.
- The first implementation may require session stickiness to the PA node that
  owns the resident KV.
- Remote migration should be designed after local no-copy correctness is proven.
