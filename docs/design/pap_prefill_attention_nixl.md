# PAP Data Plane Contract

This note records the intended PAP performance contract, the current prototype
gap, and the Adrenaline patterns we should reuse.

## Performance Contract

PAP has three logical roles and two physical GPU roles:

- Prefill: normal vLLM prefill. It computes prompt hidden states and prompt KV.
- Attention: decode attention executor colocated with Prefill. It owns prompt
  and decode KV state, has no model weights, and computes attention from Q/K/V.
- Projection: decode projection executor. It owns model weights for QKV/O
  projection, MLP/MoE, logits, and sampling, but it should not own prompt KV.

The performance data plane is:

1. Prefill to Attention: same-GPU CUDA IPC, shared KV cache handles, or an
   equivalent local OFFLOAD_KV path. This transfer must stay on the Prefill /
   Attention GPU and should avoid CPU serialization.
2. Projection to Attention: NCCL/P2P/NVLink OFFLOAD_EXEC path for per-step Q/K/V
   and attention output O.
3. Prefill to Projection: no prompt KV transfer in performance PAP. Projection
   receives only control metadata and the decode task routed by the proxy.

The control plane is separate:

- Proxy to Prefill, Attention, and Projection may use HTTP or ZMQ.
- Control messages carry request ids, session handles, sequence lengths, routing
  decisions, and readiness signals.
- Control messages must not carry prompt KV or per-step Q/K/V/O tensors in the
  performance path.

Projection should not receive Prefill KV. The first decode step is driven by the
proxy scheduling Projection for the last prefill token, while Attention already
has the prompt prefix KV colocated with Prefill.

## Current Prototype Gap

The current runnable prototype is not the performance path:

- Prefill to Attention imports prompt KV through
  `/v1/pap/attention/import-prefill-kv` using serialized tensor payloads after
  local prefill attention has populated Prefill's KV cache.
- Projection to Attention decode attention can use the PAP TCP binary path for
  current-token Q/K/V and attention output.
- Projection intentionally skips receiving prefill KV in
  `true_split_performance`.

Therefore current benchmarks should be labeled as
`PAP-prototype-http-tcp-data-plane`. They validate control flow and functional
splitting, but they cannot validate final PAP performance because tensor data is
still crossing CPU HTTP/TCP paths.

Projection-to-Attention data plane must not be TCP/HTTP in performance mode.
Prefill-to-Attention data plane must not be HTTP tensor serialization in
performance mode.

## Adrenaline Patterns to Reuse

Adrenaline has the right separation of communication concerns:

- `OFFLOAD_KV`: Prefill sends prompt KV to the Attention/offload role.
- `OFFLOAD_EXEC`: Decode/Projection exchanges Q/K/V, attention output, and other
  execution tensors with the Attention/offload role.
- Proxy/control APIs remain separate from these tensor paths.

For PAP, the direct mapping is:

- `OFFLOAD_KV` becomes Prefill-to-colocated-Attention KV installation. On the
  same GPU this should prefer CUDA IPC/shared cache ownership over networked
  copy.
- `OFFLOAD_EXEC` becomes Projection-to-Attention per-step Q/K/V/O exchange over
  NCCL/P2P/NVLink.
- The Attention role should be an internal executor with KV ownership, not an
  HTTP tensor store.

## Minimal Implementation Route

The next useful code milestone is an internal Attention executor with a real
GPU data plane:

1. Refactor the Attention process into an internal executor that can allocate or
   reference GPU KV cache blocks.
2. Add a Prefill-to-Attention KV install API that passes CUDA IPC/shared-cache
   handles or `OFFLOAD_KV` metadata over the control plane, not tensor bytes.
3. Replace Projection-to-Attention TCP with an `OFFLOAD_EXEC` communicator using
   NCCL/P2P/NVLink.
4. Keep HTTP/ZMQ only for register, route, ready, abort, and completion control.
5. Add a hard performance-mode guard that rejects HTTP/TCP tensor transports
   when `PAP_MODE=true_split_performance`.

The next benchmark worth comparing to PD is the first version where Attention
receives Prefill KV through CUDA IPC/shared cache ownership and exchanges
Projection Q/K/V/O through NCCL/P2P/NVLink.

## Concrete vLLM/Adrenaline Implementation Hooks

Adrenaline's implementation gives us three concrete hooks:

1. Group initialization in `parallel_state.py`:
   - `_OFFLOAD_KV` builds one Prefill-Attention group per colocated pair.
   - `_OFFLOAD_EXEC` builds one Decode/Projection plus all Attention ranks group
     per tensor-parallel rank.
2. Projection-side model boundary:
   - after `qkv_proj` and rotary/qk norm, split Q/K/V by target Attention role;
   - call `get_offload_group().scatter_tensor_to_offload(...)`;
   - after Attention computes O, call
     `get_offload_group().gather_tensor_to_decode(...)`;
   - then run `o_proj`, MLP/MoE, logits, and sampling locally on Projection.
3. Attention-side executor:
   - load an `OffloadAttn`-style no-weight model containing only attention
     module metadata;
   - receive Q/K/V from `OFFLOAD_EXEC`;
   - run attention against local KV cache;
   - gather O back to Projection.

The PAP implementation should not literally preserve Adrenaline's single
Decode-instance assumption. It should generalize group construction to many
Projection nodes:

- each Projection rank joins one or more `OFFLOAD_EXEC` groups;
- each Attention rank joins the groups for the Projection nodes it serves;
- Proxy routing chooses the Projection group for a request and sends only the
  group/session id over the control plane.

For Prefill-to-Attention KV, the preferred design is stricter than ordinary
PD NIXL:

- if Prefill and Attention are MPS processes on the same physical GPU, export
  KV-cache CUDA IPC handles or share a common KV allocation owner;
- the control-plane message carries request id, block table, seq len, and handle
  metadata;
- Attention registers the handle and records the request-to-block mapping;
- no Projection process participates in this KV installation.

This gives the intended PAP split:

- Prefill: writes prompt KV once.
- Attention: keeps prompt/decode KV and runs attention kernels.
- Projection: streams Q/K/V and consumes O over NVLink, while amortizing model
  weights across many PA nodes.
