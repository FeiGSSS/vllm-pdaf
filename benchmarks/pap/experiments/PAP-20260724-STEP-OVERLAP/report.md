# PAP step-scoped control overlap

## Scope

This targeted AIPerf C12 run validates commit
`31e0b3882b5fadf63d2d68b52855dfc7307c11fd` against the accepted scheduler
overlap baseline `29cc69029554ec6ff3b0a438dab8a03cee6b2815`.

The treatment preserves one Projection scheduler batch and the existing
36-layer compute order while moving independent work off the critical path:

- the Projection runner prepares session-aware route metadata before model
  forward;
- Projection publishes one payload-free step descriptor before layer-0 QKV;
- Attention prepares the step context, append slots, paged metadata, and
  workspace on a separate CUDA stream;
- same-layer output waits and disjoint scatters use independent peer-local
  streams when a batch is routed to multiple PAs.

No Attention computation, Projection computation, layer, or scheduler batch is
interleaved by this change.

## Workload and validity

- Hardware/model: four NVIDIA L20 GPUs, Qwen3-8B FP16.
- Topology: 3PA1P, static 72/20-SM Prefill/Attention split.
- Client: AIPerf 0.11.0.
- Load: 32 conversations, ten turns, 320 requests, C12.
- Input: randomized 8,192-token initial and 512-token appended means.
- Output: randomized 32-token mean, 16-64-token bounds.
- Projection memory: automatic `model_weights_x1.20`, resolving to 0.4070.
- PA Prefill memory: 0.90.

The clean run completed 320/320 requests and 32/32 conversations with zero
errors or cancellations. Strict correctness, conversation-affinity routing,
decode-token join, session drain, and static-MPS audits passed.

Across the three Attention instances, the runtime recorded 6,120 step-context
misses and exactly 220,320 hits (`6,120 * 36`). This confirms that each context
was created by step preparation and then reused by all 36 layers. Metadata was
built once per step.

## Results

| Configuration | TTFT avg ms | TTFT p90 ms | TTFT p95 ms | ITL avg ms | ITL p95 ms | Req/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scheduler-overlap baseline | 794.08 | 1,745.76 | 3,121.90 | 35.20 | 41.89 | 2.529 |
| Step/control overlap | 790.54 | 1,505.33 | 3,954.70 | 35.07 | 39.72 | 2.542 |
| Relative change | -0.45% | -13.77% | +26.68% | -0.37% | -5.19% | +0.50% |

The decode-focused metrics show no regression: mean and p95 ITL improve, and
request throughput increases slightly. Mean and p90 TTFT also improve. The
single-run TTFT p95 worsens while p99 changes only +4.32%; the affected records
are concentrated in the initial concurrent Prefill burst, outside the
step-overlap decode path. Treat that isolated tail movement as inconclusive,
not as a TTFT improvement claim.

## Decision

Accept the control-overlap implementation as the new development baseline. It
preserves correctness and improves the primary decode-tail metric without
changing the one-batch execution invariant.

This remains controlled, single-repetition evidence. A release-level
performance claim still requires three interleaved repetitions, and the
multi-PA stream benefit should be isolated with a transport trace before
attributing a specific fraction of the E2E change to it.
