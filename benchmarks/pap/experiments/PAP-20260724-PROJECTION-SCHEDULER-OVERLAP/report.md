# PAP Projection scheduler-overlap regression

## Scope

This targeted AIPerf C12 run validates the corrected PAP batch boundary at
runtime commit `29cc69029554ec6ff3b0a438dab8a03cee6b2815`:

- PAP does not split a vLLM scheduler batch into independently inflight
  microbatches;
- same-step requests may still fan out as route shards to independent PAs;
- vLLM async scheduling remains enabled for next-step scheduler, metadata,
  output, and sampling overlap;
- `UniProcExecutor` submits the complete layer sequence in stream order, so
  adjacent PAP batches cannot interleave their GPU layer execution.

The run also includes a conservative CPU-path cleanup: request and token
metadata is normalized once per model forward and reused by all 36 layers,
and each layer reuses the forward context already validated by
`should_execute()`.

## Workload and validity

- Hardware/model: four NVIDIA L20 GPUs, Qwen3-8B FP16.
- Topology: 3PA1P, static 72/20-SM Prefill/Attention split.
- Client: AIPerf 0.11.0.
- Load: 32 conversations, ten turns, 320 requests, C12.
- Input: randomized 8,192-token initial and 512-token appended means.
- Output: randomized 32-token mean, 16-64-token bounds.
- Delays: `0,3,3,1,3,3,1,3,3,1` seconds per conversation.
- Projection memory: automatic `model_weights_x1.20`, resolving to 0.4070.
- PA Prefill memory: 0.90.

The clean run completed 320/320 requests and 32/32 conversations with zero
errors or cancellations. Strict output, conversation-affinity,
decode-token-join, session-drain, and static-MPS audits passed. The scheduling
audit recorded:

```text
ASYNC_SCHEDULING=1
SCHEDULER_QUEUE_DEPTH=2
PAP_RUNNER_MICROBATCH_PIPELINE=0
```

## Results

| Configuration | TTFT avg ms | TTFT p95 ms | ITL avg ms | ITL p95 ms | Req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier async C12 baseline | 809.91 | 3,205.91 | 34.76 | 40.72 | 2.534 |
| Rejected no-async mean | 766.75 | 2,811.63 | 38.24 | 44.85 | 2.484 |
| Corrected async C12 | 794.08 | 3,121.90 | 35.20 | 41.89 | 2.529 |

Relative to the rejected no-async mean, restoring scheduler overlap improves
mean ITL by 7.96%, ITL p95 by 6.60%, and request throughput by 1.80%.
Relative to the earlier async baseline, mean ITL is 1.25% higher, ITL p95 is
2.89% higher, and throughput is 0.22% lower. This single repetition therefore
shows no material E2E regression from the metadata cleanup.

## Decision

Accept the corrected boundary and keep vLLM async scheduling enabled. Preserve
only the PAP-specific invariant: one scheduler batch is not split into
layer-interleaved microbatches.

Safe follow-up optimizations should overlap or reuse CPU/control work without
changing GPU batch order. Existing examples include step-scoped route plans,
Attention metadata/workspace reuse, stream-ordered local transport, async
sampled-token delivery, async Prefill KV import, and the forward-context cache
validated here.
