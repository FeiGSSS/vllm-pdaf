# AIPerf-only PAP convergence regression

## Scope

This run validates the committed cleanup that removed active P17 coupling,
made the PAP and PD runners AIPerf-only, and relocated benchmark diagnostics.

- vLLM/PAP commit: `aafcfb1ea1800b4dc0bcd1ea8299d9984e9624aa`
- Tracked worktree at launch: clean
- AIPerf: 0.11.0
- Model and hardware: Qwen3-8B FP16 on four NVIDIA L20 GPUs
- Topology: PAP 3PA1P, eager mode, static 72/20-SM PA partition
- Load: 32 conversations, ten turns, 320 requests, concurrency 12
- Lengths: randomized 8K initial input, roughly 512 appended input tokens,
  and 16-64 output tokens
- Timing: conversation concurrency with think/tool delays
- Dataset seed: 42
- Dataset SHA-256:
  `7340d27ab601b4a77f14487b384d5ada5a61821dac6d890ab98d5bdfa8026318`

## Validity

All 320 requests completed. Output/correctness validation, conversation-affine
routing across three PA nodes, asynchronous decode-token joining, all three
static-MPS audits, and zero-session drain passed. The launcher exited with
status 0 and released all four GPUs.

## Result

| TTFT p95 | ITL p95 | Request throughput | Strict goodput | Standard goodput | Relaxed goodput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,904.08 ms | 40.25 ms | 2.563 req/s | 2.443 req/s | 2.539 req/s | 2.539 req/s |

The strict good fraction was 95.31%; standard and relaxed were both 99.06%.
The point passed all three SLO tiers.

Relative to the preceding single-point cutover run, ITL changed by +0.83%,
request throughput by -1.14%, and strict goodput by -2.11%. TTFT p95 was
39.56% higher, but remained below the 5-second strict SLO and below the
3,552.38-ms audited eager C12 result. These one-repetition tails are retained
as observations, not promoted to a precise performance-regression claim.

The complete local evidence bundle is colocated under
[`raw/`](runs/20260722_aafcfb1ea_aiperf_pap_c12_convergence/raw/).
