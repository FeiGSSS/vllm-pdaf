# PAP NIXL/NVLink Optimization Idea Book

## Goal

Reduce PAP TPOT on the NIXL mailbox path without optimizing the local IPC
transport. Each idea must be independently testable. If an idea does not show a
measurable win on the same warmed workload, revert the behavior change and keep
only the experiment note if it prevents repeated work.

Reference run:

- `test/baseline/pap/results/runs/20260706_nixl_root_trace_v3.md`
- Qwen3-8B, `1pa1p`, random `i128/o32`, request rate 16,
  max concurrency 64, 16 warmups
- Median TPOT: `308.52 ms`
- Per-layer median remote attention: `3.70 ms`
- Attention-side median `recv_qkv_ms`: `1.60 ms`
- Attention-side median `metadata_build_ms`: `1.84 ms`

Important caveat: `recv_qkv_ms` starts after the attention worker sends the
previous layer output and waits for the next layer QKV. That wait can include
projection-side work between layers, not only NIXL/NVLink data movement.

## Ranking Method

Ideas are sorted by the combined score:

1. Expected TPOT impact.
2. Implementation size and rollback safety.
3. Probability that the change affects the current NIXL/NVLink hot path.
4. Measurement clarity.

## Ranked Ideas

| Rank | Idea | Area | Why first or later | Expected effect | Validation |
|---:|---|---|---|---|---|
| 1 | Reuse paged FlashAttention metadata across layers for the same decode step | Metadata | Largest measured attention-side operation and low protocol risk | Remove most repeated `metadata_build_ms` work when request order, block IDs, and seq lens are unchanged across layers | Unit test cache hit behavior; compare `metadata_build_ms`, `remote_total_ms`, TPOT |
| 2 | Split `recv_qkv_ms` into idle wait, notification decode, payload materialize, and transfer/sync | Measurement for communication | Needed before treating the full 1.60 ms as NVLink cost | No direct speedup, but prevents optimizing projection idle time as communication | Trace summary shows the subcomponents and their medians |
| 3 | NIXL mailbox low-latency defaults A/B: telemetry off, inline poll, slot count, direct output | Communication | Existing knobs, easy rollback, affects NIXL only | Reduce notification/poll/ACK fixed cost if those dominate | Same warmed run with one knob at a time; keep only winning default |
| 4 | Persistent NIXL compact notification for OFFLOAD_EXEC batches | Communication | Removes repeated nested metadata encoding/decoding, but likely smaller than wait/metadata costs | Lower notification decode and Python allocation cost | Add binary/compact payload tests; compare mailbox wait/read traces |
| 5 | Step-level remote-attention batch plan ID | Communication/control | Bigger protocol change, but directly attacks per-layer descriptor traffic | Per-layer message carries layer ID plus plan ID instead of request arrays | Correctness tests for plan lifecycle; compare notify payload size and recv trace |
| 6 | Projection writes K/V directly into final unified KV slots and sends only Q | Communication/data movement | Highest data-path upside, but hardest consistency surface | Shrinks QKV payload and removes attention-side append copy | Lease correctness tests; A/B Q-only path against full QKV |
| 7 | CUDA/Triton metadata fill kernel | Metadata | Useful only after metadata reuse; otherwise it optimizes work that should not repeat | Faster cache misses and block-boundary updates | Microbenchmark metadata miss path; compare miss-only traces |
| 8 | Cross-layer pipelining of projection and attention | Scheduling/protocol | Potentially high upside but large model execution change | Overlap projection post-attention work with attention receive/compute | Requires correctness and scheduler ordering tests plus full benchmark |

## Implementation Policy

- Do not modify `PAP_OFFLOAD_EXEC_TRANSPORT=local_fast` behavior.
- Keep every optimization behind normal correctness tests.
- For performance experiments, use the same model, workload, warmup, request
  rate, concurrency, and trace settings as the reference run.
- A change is worth committing only if it has either:
  - a warmed end-to-end win, or
  - a clearly measured operation-level win on the targeted overhead and no
    regression in end-to-end correctness.
- If a result is neutral or worse, revert the code change and record the
  rejected idea in this document or a run note.

## Current Next Step

Implement Rank 1 first. It is not a transport change, but it removes the largest
confirmed per-layer attention-side overhead while preserving the NIXL mailbox
path. Rank 2 then separates true NIXL/NVLink transfer cost from projection idle
wait before deeper communication changes.
