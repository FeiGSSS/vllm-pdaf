# GPU-resident 7PA1P bottleneck trace

## Configuration

- Qwen3-8B on eight L20 GPUs, 7 Attention/Prefill workers and one
  Projection worker;
- GPU-resident Attention dispatcher enabled with a one-step hardware-wait
  window;
- Dynamo routing, Prefill `max_num_batched_tokens=2048`;
- 60 conversations, three turns, 180 requests, Poisson 0.9 requests/s,
  concurrency 60;
- Agentic Coding half-length dataset used by the preceding matched trace.

Attempt 002 is the valid run. All 180 requests completed without errors and
all correctness, Graph, routing, decode-token, and drain audits passed. The
merged tensor contains 512 aligned global Projection steps, 3399 through 3910:

- PA fan-out-to-return latency: `[512, 36, 7]`;
- PA-local Attention kernel latency: `[512, 36, 7]`;
- Projection return-to-next-dispatch latency: `[512, 36]`.

Attempt 001 found and corrected a resident-dispatch trace-marker omission. Its
invalid raw artifacts were removed during the final experiment cleanup.

## End-to-end result

| Metric | Mean | P50 | P90 | P99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 1602.20 ms | 1109.99 ms | 3826.31 ms | 6253.77 ms |
| Per-request mean TBT | 45.05 ms | 44.26 ms | 49.98 ms | 57.29 ms |
| End-to-end latency | 22.45 s | 17.35 s | 38.69 s | 80.90 s |

Output throughput is 279.05 token/s. GPU marker instrumentation makes the
absolute TBT higher than a no-trace run, so use this run for attribution.

Across all 83,730 positive token intervals, TBT has P50/P90/P99/P99.9/max of
43.03/50.37/97.11/150.87/571.91 ms. There are 731 intervals above 100 ms and
four above 250 ms.

## TTFT

| Phase | Mean | Share of client TTFT |
| --- | ---: | ---: |
| Prefill HTTP service | 1382.05 ms | 86.3% |
| Projection admission to first token | 134.56 ms | 8.4% |
| Client/Gateway delta | 38.35 ms | 2.4% |
| Gateway tokenization | 36.09 ms | 2.3% |
| Attention registration | 1.89 ms | 0.1% |
| Dynamo routing | 1.95 ms | 0.1% |
| Attention KV readiness | 1.95 ms | 0.1% |
| Projection admission wait | 0.017 ms | negligible |

TTFT and Prefill latency have correlation 0.9997. Each request averages
29,092 prompt tokens, 25,299 cache-read tokens, and 3,793 actually computed
Prefill tokens. Prefill latency and effective Prefill tokens have correlation
0.9467.

The seven Prefill workers provide 136 ten-second scheduler samples. Maximum
observed state is one running request, zero waiting requests, and 35% KV-cache
occupancy. This rules out persistent Prefill queueing and KV exhaustion. TTFT
is dominated by cache-miss Prefill compute while Prefill shares each GPU with
Decode Attention under MPS.

## TBT additive breakdown

The 512-step traced cycle is
`sum_layer(max_PA(PA latency)) + sum_layer(Projection latency)`.

| Component | Mean per step | Share |
| --- | ---: | ---: |
| Projection path | 27.37 ms | 51.5% |
| Mean PA Attention kernel work | 13.47 ms | 25.3% |
| Attention-kernel barrier imbalance | 5.72 ms | 10.8% |
| Mean PA non-Attention path | 2.73 ms | 5.1% |
| Additional non-Attention PA skew | 3.90 ms | 7.3% |
| Total traced cycle | 53.19 ms | 100% |

Perfectly balancing only Attention kernels has an upper bound of 5.72 ms, or
10.8%. Perfectly eliminating all PA fan-out imbalance has an upper bound of
9.62 ms, or 18.1%. These are upper bounds rather than achievable gains.

### Step boundary

The slowest PA takes 4.29 ms on layer 0 versus 0.615 ms on layers 1--35. The
Attention-side step boundary therefore adds 3.68 ms on average. Its P50/P90/
P99/max are 0.74/2.64/77.72/95.57 ms, so a few large stalls dominate the mean.

The final Projection transition takes 3.06 ms versus 0.695 ms for ordinary
transitions, adding another 2.37 ms. Its P50 and P99 extras are tightly grouped
at 2.01 and 2.08 ms, but one 108.62 ms outlier remains.

Together these two boundary views expose about 6.05 ms per traced step. This
overlaps the PA-skew decomposition above and must not be added to it.

The GPU replay marker to the first child-Graph node is only 2.90 us. On the
slowest PA, Graph start to kernel start has a 4.35 us P50. The common GPU
device-launch path is therefore not the remaining bottleneck. Rare scheduling
stalls raise the latter phase's mean to 0.37 ms and maximum to 106.48 ms.

The Attention host phases before launch average:

| Host phase | Mean across PA steps |
| --- | ---: |
| Control wait and D2H | 0.623 ms |
| Control decode | 0.021 ms |
| Context/PAT preparation | 0.188 ms |
| Graph lookup/capture amortization | 0.045 ms |
| Total | 0.876 ms |

The maximum pre-launch host total across seven PAs averages 1.80 ms per step,
with P50/P90/P99 of 1.39/1.65/10.84 ms. The `graph_replay_submit` trace field
cannot be interpreted as submission overhead in resident mode because the
current one-step dispatcher call waits for the whole child Graph to finish.

### Attention load and PAT

PAT is selected for 82.2% of aligned PA-step cells. The mean PA-step has 2.93
requests, 87,051 logical context tokens, 40,417 unique physical tokens, and
46,634 shared tokens. Mean Attention latency is 0.374 ms per layer and mean
effective unique-KV bandwidth is 455.5 GB/s.

Physical KV balance is not compute balance. PA4 averages 135,795 logical
tokens over 5.32 requests but only 36,680 unique tokens; it is the slowest
average Attention worker at 0.444 ms per layer. PA5 averages 117,060 logical
tokens and 0.426 ms. A routing score based only on unique physical KV misses
PAT's request-count and logical-context costs.

Across the full run, 37,679 of 38,013 PA steps use the successor context path
(99.1%). Only 334 steps recheck prefix topology, producing 264 PAT rebuilds
and 70 Triton selections. There are no pending-KV records or dispatch failures.

## Comparison with the preceding trace

The preceding trace predates both the successor-context fast path and the
resident dispatcher, so this is directional rather than a clean one-variable
A/B.

| Metric | Previous trace | Current trace | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 1623.72 ms | 1602.20 ms | -1.3% |
| Mean TBT | 49.25 ms | 45.05 ms | -8.5% |
| Traced GPU cycle | 60.19 ms | 53.19 ms | -11.6% |
| Slowest-PA layer-0 latency | 7.27 ms | 4.29 ms | -41.0% |
| Estimated two-sided step boundary | 9.64 ms | 6.05 ms | -37.2% |
| Individual-token TBT P99 | 119.35 ms | 97.11 ms | -18.6% |
| Individual-token TBT maximum | 1546.98 ms | 571.91 ms | -63.0% |

## Conclusion

1. TTFT is still a Prefill-compute problem, not routing, readiness, admission,
   queueing, or KV capacity.
2. Projection is the largest required TBT component at 27.37 ms per step.
   A dispatcher cannot remove this computation.
3. The largest measured avoidable PA cost is the 9.62 ms barrier imbalance.
   PAT-aware routing must consider logical context and request count as well as
   physical KV.
4. The common GPU dispatcher launch costs only a few microseconds. The current
   one-step implementation still has 0.88 ms average host preparation per PA
   and a 1.80 ms mean seven-PA maximum, plus rare step-boundary stalls.
5. The next dispatcher optimization must batch-arm stable schedules and use an
   independent completion mailbox. Merely replacing `graph.replay()` with a
   device launch cannot remove the remaining CPU control/preparation path.

Primary artifacts are in `results/qps_0p9_2k/attempt_002/`: `detailed_trace.pt`,
`detailed_trace.json`, `gateway_phase_trace.json`, raw PA traces, AIPerf output,
and service logs.
