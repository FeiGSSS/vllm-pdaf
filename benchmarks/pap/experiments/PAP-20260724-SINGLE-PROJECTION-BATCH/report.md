# PAP Projection no-async diagnostic

## Scope

This experiment measured an overly strict interpretation of the PAP
single-batch requirement:

- runtime commit: `cb6fe35009905c32b52a8ad10b7e00d778c03679`;
- Projection runs with vLLM asynchronous scheduling disabled;
- one Projection step may fan QKV shards out to several PA peers;
- PAP-specific microbatching remained removed.

It is retained as a negative A/B diagnostic, not as a current runtime
milestone or a new PAP-versus-PD capacity scan. The latest full capacity
comparison remains
`PAP-20260722-AIPERF-PROJECTION-AUTO`.

## Interpretation correction

In this vLLM tree, asynchronous scheduling changes
`VllmConfig.max_concurrent_batches` from one to two. `EngineCore` then creates
a two-entry scheduler queue. However, PAP uses `UniProcExecutor`, whose
`execute_model` call runs the complete model forward before returning its
asynchronous output handle. GPU work from adjacent steps remains stream
ordered. The queue can overlap CPU scheduler, metadata, output, and sampling
work; it cannot place two PAP microbatches at different model layers.

Disabling vLLM asynchronous scheduling was therefore not required by the
single-batch constraint.

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

After removing run-specific `session_id` and `cache_salt` fields, the previous
baseline and both new AIPerf input files have the identical SHA-256
`9bc50f0fb27d90da0f4826a4f5fb339a7416e618a7fbab9fd1c22571fe58dfbc`.

Both treatment repetitions:

- completed 320/320 requests and 32/32 conversations;
- reported zero request errors or cancellations;
- passed strict output, conversation-affinity, session-drain, and static-MPS
  audits;
- confirmed `ASYNC_SCHEDULING=0`;
- used a clean tracked worktree at runtime commit `cb6fe3500`.

## Results

The baseline is the eager 3PA1P C12 point at `4aaf9dd77`, where vLLM
asynchronous scheduling was enabled by default.

| Run | TTFT avg ms | TTFT p95 ms | ITL avg ms | ITL p95 ms | Req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous two-batch baseline | 809.91 | 3,205.91 | 34.76 | 40.72 | 2.534 |
| Single batch R1 | 773.75 | 2,873.44 | 38.88 | 45.65 | 2.467 |
| Single batch R2 | 759.75 | 2,749.82 | 37.59 | 44.05 | 2.502 |
| Single batch mean | 766.75 | 2,811.63 | 38.24 | 44.85 | 2.484 |

Mean treatment change relative to the previous baseline:

| Metric | Change |
| --- | ---: |
| TTFT average | -5.33% |
| TTFT p50 | -8.21% |
| TTFT p95 | -12.30% |
| ITL average | +10.00% |
| ITL p50 | +10.77% |
| ITL p95 | +10.16% |
| Request latency average | +3.70% |
| Request latency p95 | -7.62% |
| Request throughput | -1.98% |
| Output-token throughput | -1.98% |

Both treatment runs remain within the Strict thresholds of TTFT p95 <= 5 s
and ITL p95 <= 50 ms. R1 had an isolated ITL tail
(`p99=82.27 ms`); R2 did not reproduce it (`p99=50.24 ms`).

The observed E2E change is consistent with removing safe CPU/GPU scheduling
overlap: steady decode latency increases by about 3.5 ms while total
throughput remains within 2%. This result does not justify disabling async
scheduling.

## Decision

Reject the no-async treatment. Keep vLLM asynchronous scheduling enabled and
retain this record only as a negative control.

The actual invariant is narrower: PAP must not split one vLLM scheduler batch
into independently inflight microbatches or interleave their layer execution.
Same-step PA route shards and vLLM's CPU-side next-step preparation remain
allowed.
