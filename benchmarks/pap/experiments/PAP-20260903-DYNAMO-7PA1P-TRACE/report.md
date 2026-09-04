# 7PA1P 2K Dynamo bottleneck trace

## Configuration

- Qwen3-8B, one L20 per process, 7 Attention/Prefill workers and one
  Projection worker;
- Dynamo PA routing with `prefill_load_scale=2.0`;
- Prefill `max_num_batched_tokens=2048`;
- 60 conversations, three turns each, 180 requests, Poisson 0.9 requests/s,
  concurrency 60, no warmup;
- Agentic Coding half-length dataset, SHA256
  `258b72c85772c9d372f1b63ee0bf6d710f27cb00234027e2c750c82a5fa9563c`.

Attempt 003 is the valid GPU trace. All 180 requests completed without errors,
all correctness/whole-step Graph/routing/drain audits passed, and 512 contiguous
steps (global step 3256 through 3767) were aligned by each PA's NVSHMEM epoch.
Attempts 001 and 002 were used to correct trace retention and alignment. Their
invalid raw artifacts were removed during the final experiment cleanup.

## End-to-end result

| Metric | Mean | P50 | P90 | P99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 1623.72 ms | 1106.58 ms | 4003.72 ms | 6248.18 ms |
| Per-request mean TBT | 49.25 ms | 49.40 ms | 57.91 ms | 67.59 ms |
| End-to-end latency | 24.37 s | 19.06 s | 41.09 s | 90.29 s |

The GPU marker trace adds two Attention markers per layer and Projection-side
markers. Its mean TBT is 10--12% above the existing untraced 2K Dynamo result
of 43.88 ms. Use the trace for attribution and the untraced run for production
absolute performance.

## TTFT

All 180 requests were joined between the Gateway log and AIPerf records.

| Phase | Mean | P99 | Share of mean client TTFT |
| --- | ---: | ---: | ---: |
| Prefill HTTP request | 1389.28 ms | 6033.82 ms | 85.6% |
| Projection admission to first streamed token | 143.43 ms | 228.68 ms | 8.8% |
| Gateway tokenization | 38.09 ms | 66.12 ms | 2.3% |
| Client/Gateway transport and parsing delta | 42.01 ms | 91.52 ms | 2.6% |
| Attention registration | 1.81 ms | 4.66 ms | 0.1% |
| Dynamo routing | 1.95 ms | 5.27 ms | 0.1% |
| Attention KV readiness | 1.88 ms | 6.79 ms | 0.1% |
| Projection admission wait | 0.016 ms | 0.022 ms | negligible |

TTFT and Prefill latency have correlation 0.9997. Prefill latency and the
actual cache-miss prompt token count have correlation 0.9468. The requests
average 29,092 prompt tokens, 25,275 cache-read tokens, and 3,817 Prefill
tokens. A simple linear fit gives about 321 ms fixed cost plus 0.280 ms per
Prefill token.

The seven Prefill logs contain 17--23 ten-second scheduler samples each. None
reports a waiting request. Their maximum reported KV-cache occupancy is
24.7--34.4%. This sampling can miss a short queue, but it rules out persistent
Prefill backlog and KV capacity pressure. The TTFT bottleneck is Prefill
service time for cache-miss tokens while Prefill shares each PA GPU with
Decode Attention, not Dynamo routing, readiness, admission, or KV exhaustion.

## TBT

The aligned GPU tensors have shapes:

- PA fan-out-to-return latency: `[512, 36, 7]`;
- PA-local Attention kernel latency: `[512, 36, 7]`;
- Projection return-to-next-dispatch latency: `[512, 36]`.

For each step, the traced cycle is
`sum_layer(max_PA(PA latency)) + sum_layer(Projection latency)`.

| Additive mean component | Time per step | Share |
| --- | ---: | ---: |
| Projection path | 28.07 ms | 46.6% |
| Mean PA Attention kernel work | 15.36 ms | 25.5% |
| Attention-kernel imbalance at the barrier | 7.11 ms | 11.8% |
| Mean PA non-Attention path | 3.51 ms | 5.8% |
| Additional non-Attention PA skew | 6.14 ms | 10.2% |
| Total traced cycle | 60.19 ms | 100% |

Therefore, perfect Attention load balance alone has an upper bound of about
7.11 ms, or 11.8%, in this heavy window. Perfectly eliminating all seven-PA
fan-out imbalance has an upper bound of 13.25 ms, or 22.0%, and would still
leave about 46.94 ms in this instrumented window. The global barrier is
material, but it is not the sole explanation for PAP TBT.

### Step boundaries

Layers 1--35 have about 0.484 ms mean PA fan-out-to-return latency. Layer 0's
slowest PA takes 7.27 ms on average. Replacing it with an ordinary-layer value
would remove 6.56 ms per step. The layer-0 PA barrier alone contributes 5.36
ms; only 0.20 ms of that is Attention-kernel imbalance.

Ordinary Projection layer transitions take about 0.694 ms. The final layer to
the next step's first dispatch takes 3.78 ms, adding 3.09 ms. Together, the
Attention-side step launch and Projection-side next-step preparation add about
9.64 ms relative to ordinary layer transitions.

This matches the implementation: 36 layer exchanges are GPU-driven inside
each whole-step CUDA Graph, but every decode step still sends CPU control
metadata, prepares a context, and launches a Graph independently in the seven
Attention processes and the Projection process. Schedule changes make the
problem worse: the slowest layer-0 PA averages 13.54 ms on changed-batch steps
versus 6.66 ms on stable steps. The stable-step cost shows that rebuild/capture
is not the only source; host wakeup, metadata publication, stream readiness,
and independent multi-process launches remain in the boundary.

The final Projection boundary also causes the long tail. All 512 steps have
their largest Projection interval at layer 35. Its P50/P99/max are
2.71/30.60/146.65 ms. Across the full AIPerf output, individual positive ITLs
have P99 119.35 ms, P99.9 248.64 ms, and max 1546.98 ms.

### Attention placement and PAT

| PA rank | Requests | Logical context | Unique physical context | PAT steps | Attention kernel |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.71 | 79,940 | 41,299 | 99.0% | 361.7 us |
| 1 | 4.51 | 125,932 | 45,072 | 100% | 449.1 us |
| 2 | 7.16 | 187,441 | 38,041 | 100% | 570.6 us |
| 3 | 2.43 | 81,197 | 47,673 | 100% | 413.9 us |
| 4 | 2.69 | 79,935 | 41,032 | 100% | 350.1 us |
| 5 | 2.05 | 79,463 | 48,060 | 83.8% | 399.0 us |
| 6 | 1.00 | 50,008 | 49,616 | 0% | 442.0 us |

PAT is selected for 83.3% of PA-step cells. The mean unique-context effective
bandwidth is 446 GB/s, while logical context is amplified 2.26x by sharing.
Attention latency correlates strongly with logical context (0.816) and request
count (0.779), but weakly with unique context alone (0.135). PA 2 is the
slowest in 315 of 512 steps even though it has the lowest unique physical KV:
it has the most requests and by far the largest logical context.

The current Dynamo placement is effective at physical KV/prefix reuse, but
physical unique blocks are not a sufficient predictor for PAT execution time.
A TBT-oriented placement score must also include logical context, request
count, and measured PAT latency. This is the concrete source of the remaining
Attention barrier imbalance.

## Bottleneck conclusion

1. TTFT is Prefill compute/service time for cache-miss tokens. Routing,
   readiness, admission, and KV capacity are not bottlenecks in this run.
2. TBT's largest additive component is the Projection path (28.07 ms), but the
   largest avoidable architectural cost is the per-step host/Graph boundary
   across Projection and seven Attention processes (about 9.64 ms in the
   traced heavy window, with a large tail).
3. The global PA barrier adds 13.25 ms in total. About 7.11 ms comes from real
   Attention-kernel imbalance; Dynamo's physical-KV-aware placement overloads
   PAT with too many logical requests on PA 2. The remaining skew is dominated
   by layer-0 step-launch coordination.
4. The next optimization priorities are a GPU-resident step dispatcher (or a
   graph-to-graph handoff that removes independent host launches), then a
   PAT-aware latency placement score. Optimizing only KV bandwidth or only
   balancing unique KV tokens cannot remove the measured bottleneck.

Raw artifacts are under `results/qps_0p9_2k/attempt_003/`, notably
`detailed_trace.pt`, `detailed_trace.json`, `gateway_phase_trace.json`, and the
AIPerf and service logs.

## Step-boundary successor optimization

A follow-up same-PA trace split the slowest layer-0 path further:

| Slowest-PA phase | Mean |
| --- | ---: |
| Control wait and D2H | 0.778 ms |
| Control decode | 0.026 ms |
| Context and PAT prepare | 0.801 ms |
| Graph lookup/capture amortization | 0.252 ms |
| Graph replay API | 0.111 ms |
| GPU replay marker to first Graph node | 0.018 ms |

The GPU submission queue is not the bottleneck. CPU control and rebuilding the
36-layer context structure explain nearly all of the graph-external layer-0
delay.

The implemented synchronous successor fast path reuses the preceding step's
session entries, 36-layer state tuples, and topology only when the request
order, session epochs, topology, shape, scale, and committed sequence state
still match. Slot tensors, paged-attention metadata, workspaces, and PAT plans
are still prepared for the real new step. Request membership changes,
re-registration, or topology changes use the complete original path.

Matched no-trace A/B results on the same 180-request workload:

| Metric | Original | Successor | Change |
| --- | ---: | ---: | ---: |
| Mean TBT | 44.719 ms | 42.886 ms | -1.833 ms (-4.10%) |
| Mean TTFT | 1602.44 ms | 1604.31 ms | +0.12% |
| Mean end-to-end latency | 22.364 s | 21.530 s | -3.73% |
| Output throughput | 284.85 token/s | 288.83 token/s | +1.40% |

Both runs completed 180/180 requests with no errors and passed correctness,
Graph, routing, decode-token, and drain audits. Successor hits cover
98.7--99.5% of PA decode steps. A follow-up short trace reduced slowest-PA
context-prepare mean from 0.801 ms to 0.259 ms and its P50 from 0.719 ms to
0.170 ms.

The remaining stable boundary is dominated by roughly 0.6--0.8 ms of control
wait/D2H. Removing it requires a GPU-resident dispatcher or an equivalent
persistent device-side control path rather than another metadata-cache tweak.
