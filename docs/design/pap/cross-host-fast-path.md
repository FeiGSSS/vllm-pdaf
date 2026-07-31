---
pap_doc_schema: 1
status: proposal
canonical: null
superseded_by: null
related_experiments: []
last_validated_commit: null
---

# Cross-host step-planned transport proposal

This document records the planned cross-host extension of PAP's same-host
`local_fast` data path. It is a TODO and design constraint, not a statement of
current performance or validation.

## Motivation and current boundary

PAP replaces a bulk Prefill-to-Decode KV handoff with QKV fan-out and
Attention-output fan-in at every model layer. The same-host path keeps this
critical path practical by preparing peer pointers, contiguous row ranges,
byte counts, base generations, and per-layer signal batches once per Decode
step. Each layer still has a data dependency and therefore performs one
lightweight batched launch, but it does not rebuild metadata or submit each
peer independently.

Cross-host NIXL remains contract-covered but performance-unverified. The
existing mailbox path includes synchronization, message publication, and
completion polling intended for general transfers; it must not be treated as
the eventual layer-wise fast path without measurement and redesign.

## Required execution contract

A future `PAPStepTransferPlan` should preserve these properties across hosts:

1. Register fixed GPU QKV and output buffers and exchange remote metadata once
   during process or peer initialization.
2. At Decode-step preparation, freeze the active PA set, contiguous row
   ranges, local and remote offsets, byte counts, and base generations for all
   layers.
3. Reuse one buffer per direction in strict layer order. Generations, rather
   than multiple ring slots, distinguish successive layers.
4. For each layer, expose one Projection-side batched fan-out submission, not
   one Python/RPC submission per PA.
5. Let each PA write its Attention result directly into its assigned
   Projection output range. Fan-in completion is an active-peer bitmap or
   equivalent joined signal, not a sequential receive loop.
6. Keep model-thread execution asynchronous. A native progress thread may
   advance network completions, but it must not rebuild request metadata,
   synchronize the model CUDA stream, or insert Python polling into the layer
   path.
7. Fail closed unless GPU-direct RDMA is verified. TCP emulation, host staging,
   or an unknown transport must not silently produce performance evidence.

The plan can precompute all metadata, but it cannot submit all 36 layer
transfers at step start: layer \(L+1\)'s QKV depends on layer \(L\)'s returned
Attention output. The intended boundary is one planning operation per step and
one lightweight asynchronous data submission per layer.

## Backend decision

The first implementation candidate is a NIXL/UCX fast path. NIXL already
provides registered GPU-memory descriptions, cached remote metadata, backend
selection, and asynchronous transfer handles. The new path should cache
descriptors and transfer handles at step preparation and provide a native
multi-peer submission wrapper instead of reusing the mailbox hot path.

NCCL is the matched-shape alternative. Grouped point-to-point operations can
express scatter and gather, and NCCL 2.29 or newer also exposes registered
one-sided `PutSignal`/`WaitSignal` operations. It requires a focused check for
GPU-SM interference, fixed-rank matching and ordering constraints, installed
version support, and dynamic active-peer behavior before it can become the
runtime backend.

Raw `ibverbs`/DEVX is a last resort. It is justified only if a controlled
microbenchmark shows that NIXL/UCX or NCCL itself contributes the dominant
latency and cannot expose the required batching or asynchronous progress.
Writing verbs directly would also make PAP responsible for QP/CQ progress,
memory-key lifetime, GPU/NIC topology, remote visibility, congestion,
reconnection, and multi-rail behavior.

Primary interface references:

- [NIXL design and metadata caching](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)
- [NCCL point-to-point and one-sided communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)
- [NCCL CUDA Graph constraints](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)

## Development and validation order

1. Build a two-host, matched-shape microbenchmark for the actual Qwen3-8B
   QKV/output sizes. Compare NIXL/UCX and NCCL for one peer and concurrent
   multi-PA fan-out/fan-in.
2. Record p50/p95/p99 layer round-trip latency, native/Python submission time,
   achieved network bandwidth, GPU-SM interference under PAP's MPS split, and
   verified GPU-direct transport selection. Bandwidth alone is insufficient
   because PAP pays a sequential dependency at every layer.
3. Implement the NIXL/UCX `PAPStepTransferPlan` only after the microbenchmark
   identifies a viable path. Keep the current cross-host backend as
   `preserved-unverified` until correctness and end-to-end tests pass.
4. Run single-PA correctness, multi-PA concurrent fan-out/fan-in, generation
   wrap/reuse, timeout/error injection, and the canonical AIPerf end-to-end
   comparison.

The success question is not whether a 400-Gbit/s link reaches a high bulk
number. It is whether GPU-direct communication plus software progress keeps
the per-layer critical-path penalty small and stable enough across all model
layers and active PA counts.
