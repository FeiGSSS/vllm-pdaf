# PAP Remote-Attention Optimization Design

## Goal

Use fine-grained PAP timing profiles to identify the current Attention↔Projection bottleneck, quantify the gap against a simple per-layer lower bound, and drive small A/B optimizations that reduce PAP TPOT toward the PD baseline.

The first target is Qwen3-8B `1PA1P` vs `1P1D` on the warmed `i128/o32/q16/c64/w32` workload. The current reference is PD median TPOT `24.9 ms` and PAP median TPOT `294.8 ms` from `benchmarks/pap/experiments/legacy/reports/pap-pd-comparison-methodology-20260701.md`.

## Current Evidence

Existing code already includes batched remote Attention, mailbox slot protocol, direct-slot QKV send, push-write handle caching, prefetch hooks, and paged FlashAttention hooks. The next step is not to repeat those features, but to verify which fast paths actually run and where measured time still exceeds the theoretical lower bound.

A rough per-layer lower bound is:

```text
T_lb_layer = bytes(QKV + attention_output) / P2P_bandwidth + attention_compute
```

For Qwen3-8B, bf16, batch 64, and about 21 GB/s P2P bandwidth, the lower bound is roughly `0.18 ms/layer`, or about `6.6 ms/token` over 36 layers. Existing micro traces are around `1.05 ms/layer`, while the warmed E2E PAP result is about `8.2 ms/layer` effective. This gap points to Projection recv/wait, queueing, scheduler effects, and state-maintenance overhead rather than raw wire time or Attention kernel time alone.

## Diagnostic Loop

Every experiment must produce a self-contained run directory with:

- benchmark JSON;
- service logs;
- trace summary;
- effective environment and command;
- current git commit and short diff name;
- one summary row comparing theoretical lower bound, micro trace, and E2E TPOT.

The core trace fields are:

- Projection: `send_ms`, `recv_ms`, `remote_total_ms`, `calls`, batch key, resume/recv timestamps;
- Attention: `recv_qkv_ms`, `compute_ms`, `send_output_ms`, `total_ms`, `calls`, `paged_flash_ms`, `fallback_ms`;
- mailbox: queue/publish/ACK/read/transfer/slot wait timings by message kind;
- scheduler evidence: benchmark concurrency, Projection `Running`, and actual call distribution.

## Implementation Scope

Phase 1 adds analysis and experiment plumbing only:

1. Extend or add a summary tool that reads benchmark JSON plus service logs and emits one markdown/CSV row per run.
2. Add a lower-bound calculator for PAP remote Attention using model dimensions, batch size, dtype bytes, P2P bandwidth, and measured Attention compute.
3. Make the summary tool report whether important fast paths appear active from logs: paged FlashAttention, fallback SDPA, prefetch, direct output, batch calls, and mailbox wait/read/ACK timing.

Phase 2 runs low-risk A/B experiments using existing flags:

1. baseline current default;
2. `PAP_OFFLOAD_EXEC_USE_PAGED_FLASH_ATTN=1` and native paged append if not already active;
3. `PAP_ATTENTION_MAILBOX_PREFETCH=1`;
4. direct mailbox output if safe;
5. slot count / async send / piggyback variants only when Phase 1 says wait/ACK is dominant.

Phase 3 considers larger code changes only if Phase 2 proves the remaining gap is still the serialized P/A layer boundary. Candidate larger changes include cross-layer pipelining, a more persistent ring protocol, or Projection resume scheduling changes.

## Validation

A change is successful only if it has both correctness and performance evidence:

- focused unit tests for any modified parser/tool/code path;
- one warmed `1PA1P` Qwen3-8B run with the full diagnostic artifact set;
- comparison against the same-workload PD row;
- explanation of whether the win came from lower per-layer remote time, better batch density, lower queueing, or lower memory pressure.

Unverified code changes must be marked as implemented but not validated. No result should be called better unless benchmark config, warmup, concurrency, and trace evidence match the comparison baseline.

## Git and Experiment Management

Use small commits or clearly named working-tree checkpoints per experiment family. Do not mix unrelated protocol changes with diagnostic tooling. Record run directories in the relevant design or experiment note after each benchmark so the next session can reproduce the comparison.
