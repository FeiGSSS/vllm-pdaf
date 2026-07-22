# Canonical AIPerf cutover regression

## Scope

This run verifies the first clean commit after retiring the P17 runner and the
project-owned multi-turn performance client.

- vLLM/PAP commit: `894b81ae9238373ac0950fc7932bed7bfb3dd74c`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0, commit `854ff91a4a221f899b806e7660a89b41b80d5689`
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Topology: PAP 3PA1P, eager mode, static 72/20-SM PA partition
- Load: 32 conversations, ten turns, 320 requests, concurrency 12
- Lengths: randomized 8K initial input, roughly 512 appended input tokens,
  and 16-64 output tokens
- Timing: conversation concurrency with think/tool delays
- Dataset seed: 42
- Dataset SHA-256:
  `718d3c309929f9c1d9b55a894f2c5e92059aafb977816e25ef345dffeeeed114`

## Validity

All 320 requests completed. Output-length validation, conversation-affine
routing across three PA nodes, asynchronous decode-token joining, correctness
log audit, all three static-MPS audits, and zero-session drain passed. The
launcher exited with status 0.

## Result

| TTFT p95 | ITL p95 | Request throughput | Strict goodput | Standard goodput | Relaxed goodput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,080.84 ms | 39.92 ms | 2.593 req/s | 2.496 req/s | 2.593 req/s | 2.593 req/s |

The strict good fraction was 96.25%; standard and relaxed good fractions were
100%. The point therefore passed all three SLO tiers.

This is a one-repetition runtime regression, not a new PAP-versus-PD capacity
claim. It establishes that the canonical AIPerf path works on the cutover
commit; milestone performance claims still require the documented matched
matrix and three repetitions.

The complete local evidence bundle is under
[`20260722_894b81ae9_aiperf_pap_c12_regression`](runs/20260722_894b81ae9_aiperf_pap_c12_regression/raw/capacity_results.md).
