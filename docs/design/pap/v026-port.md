# PAP vLLM 0.26 forward port

## Objective

Port the complete supported PAP runtime from `a1d8ec918` onto `v0.26.0`
without copying the old vLLM hot-path forks. The candidate must preserve PAP's
current correctness and lifecycle semantics and must be performance
non-inferior on the frozen S128/C32 workload.

The source milestone reports package version 0.23.1 and was developed through
the project's 0.24-era experiments. It is not a v0.26 runtime with a different
launcher.

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
7. Prefill workers explicitly disable v0.26 asynchronous scheduling. PAP uses
   Prefill completion as a control-plane handoff, and the asynchronous output
   pipeline delayed that handoff and increased closed-loop Prefill queueing.

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

## Runtime lifecycle boundary

The same-host NVSHMEM world and its symmetric allocations are process-lifetime
resources. A normal launcher shutdown first drains requests, wakes any blocked
Attention Graph receiver through its device abort signal, joins the receiver,
and then exits every PE process. An abort that wakes a pending Graph replay
also suppresses that step's KV commit. The runtime intentionally does not call a
collective `nvshmem_free`/`nvshmem_finalize` after a peer has crashed, because
collective teardown cannot be made fail-safe when that PE is gone.

The device abort guarantees that a Graph still waiting for QKV can be woken and
will not start a later output put. It cannot recall a remote put that had
already passed its abort check before shutdown began.

Attention health is fail-closed after binding: a stopped or dead Graph receiver
changes `/health` from `ok` to `error`.

## Current validation evidence

- Qwen3 GQA paged decode matches a PyTorch reference for 32 Q heads, 8 KV
  heads, head dimension 128, mixed sequence lengths, and non-contiguous block
  tables.
- A fixed greedy subset matched native vLLM output; one additional FP16 prompt
  diverged after 27 matching tokens, so cross-kernel bitwise parity is not a
  supported contract.
- 1PA1P and 7PA1P completed live CUDA-IPC/NVSHMEM Graph, routing, drain, and
  correctness audits. The frozen S128/C32 input completed 455/455 requests in
  every accepted repetition.
- PAP-disabled scheduler/KV tests and a native v0.26 Qwen3 service run passed.
