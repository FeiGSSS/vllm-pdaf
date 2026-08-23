# PAP vLLM 0.26 forward port

## Objective

Port the complete supported PAP runtime from `a1d8ec918` onto `v0.26.0`
without copying the old vLLM hot-path forks. The candidate must preserve PAP's
current correctness and lifecycle semantics and must be performance
non-inferior on the frozen S128/C32 workload.

The supported runtime is Qwen3-8B FP16 TP1, same-host xPA1P with one
Projection, Prefill-owned KV shared through CUDA IPC, Attention--Projection
communication through the minimal NVSHMEM whole-step Graph, static
conversation affinity, and no PA migration. Removed multi-Projection,
`local_fast`, NIXL mailbox, non-Graph NVSHMEM, and migration paths are not part
of this port.

## Integration principles

1. `vllm/pap/` owns PAP algorithms, request state, lifecycle, transport, and
   Graph control.
2. Upstream vLLM files expose generic registration or narrow delegation
   points; they do not own PAP state machines.
3. Qwen3 and Qwen2 remain upstream. Normalized/rotated Q/K/V is intercepted at
   the generic opaque Attention operation. A later optional packed-QKV hook may
   optimize Qwen3 without becoming the correctness path.
4. Prefill KV publication uses the v0.26 KV-connector layer callback rather
   than model-specific post-Attention code.
5. HTTP control routes use endpoint plugins. Engine control uses the generic
   utility channel and an isolated PAP lifecycle adapter.
6. With PAP disabled, vLLM behavior and performance remain unchanged.

The first generic extension is an Attention execution-factory registry. PAP's
general plugin binds a Projection executor to each Attention instance during
model construction; no Qwen3 source change is required.

## Validation gates

- **L0 structure:** review every non-`vllm/pap` hunk. Only generic hooks or
  narrow delegates are allowed.
- **L1 CPU contracts:** supported config, protocol, topology, gateway,
  lifecycle, transport, and model-extension behavior pass without hidden CUDA
  skips.
- **L2 vLLM seams:** scheduler/KV/async/OpenAI/compile focused upstream tests
  pass with PAP disabled and PAP metadata reaches the active V2 runner intact.
- **L3 GPU components:** paged-Attention numerical parity, CUDA-IPC readiness,
  NVSHMEM ABI/capture/replay/shutdown, and live MPS partition audits pass.
- **L4 topology smoke:** 1PA1P, 2PA1P, then 7PA1P pass routing, token, lease,
  Graph, drain, and cleanup audits.
- **L5 full function:** frozen S128/C32 completes 455/455 requests with the
  exact dataset/token totals, lifecycle audits, and a deterministic token
  parity subset.
- **L6 non-inferiority:** paired isolated old/new repetitions use fresh
  processes and worktree-local NVSHMEM bridges. Candidate medians must satisfy
  ITL <= 60.68054 ms, throughput >= 2.288499 requests/s, and TTFT <= 4705.389
  ms. Paired bounds additionally require ITL <= 1.02x, throughput >= 0.98x,
  and TTFT <= 1.05x the old runtime.

One run is not sufficient for L6. Start with three balanced pairs, extend to
five when no clear result exists, and to ten if the confidence interval still
crosses a non-inferiority boundary.

## Review gates

- R1: model/compute/transport/lifecycle interface design.
- R2: isolated PAP module review.
- R3: independent review of every upstream hunk and the PAP-disabled path.
- R4: CUDA Graph, NVSHMEM concurrency, and resource-lifecycle review.
- R5: end-to-end evidence and paired performance-statistics review.

Capture without replay, configured NIXL without a live data-path audit,
455 requests with length-only correctness, or stale tests that assert removed
architectures do not satisfy these gates.
