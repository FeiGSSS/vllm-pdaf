# PAP remote-Attention trace archive

Date: 2026-07-02

Status: historical diagnostic. The measured per-layer mailbox runtime has
since been replaced by the current step/batch path; these numbers explain the
optimization decision and are not a current performance baseline.

## Workload

- Qwen3-8B FP16, 1PA1P, TP1;
- input 128, output 16, request rate 16;
- 64 measured requests and 8 warmups;
- 64/64 requests completed in each retained run.

Raw summaries and filtered trace JSON are retained under
`benchmarks/pap/experiments/legacy/runs/20260702_profile_output/`.

## Accepted diagnostic

The final valid cross-node run `20260702_133208` measured median TPOT
`171.43 ms`, or `4.762 ms/layer`. Its Projection receive span was
`2.274 ms/layer`, decomposed over 2,376 matched batches as follows:

| Phase | Median ms/layer |
| --- | ---: |
| QKV mailbox delivery | 0.405 |
| Attention receive, KV append, pack, and pre-compute | 1.102 |
| Attention output publish | 0.019 |
| Output mailbox delivery | 0.632 |
| Untraced/overlapped portion | 0.116 |

Attention-side detail attributed about `0.367 ms` to KV append, `0.491 ms`
to packing, and `0.141 ms` to SDPA. The measured mailbox round trip was about
`1.037 ms/layer`; raw byte movement alone did not explain the end-to-end gap.
Another `2.038 ms/layer` sat outside the traced layer path, primarily in the
worker/model-runner and engine loop.

An earlier run (`20260702_085453`) reached the same qualitative conclusion:
wait and scheduling gaps dominated raw mailbox materialization, while paged
FlashAttention was inactive. The `20260702_124226` run is retained only as
failure evidence because its modified Attention trace emitted no batch rows;
its cross-node correlation must not be used.

## Decision

The result motivated eliminating per-layer control-plane scheduling and
building the current step/batch descriptorless path. It remains useful as
root-cause evidence, but comparisons against current PAP must use normalized
P17 or AIPerf experiment bundles.
