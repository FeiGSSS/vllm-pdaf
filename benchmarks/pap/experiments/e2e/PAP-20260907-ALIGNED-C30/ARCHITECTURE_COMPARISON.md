# DP, PD, and PAP architecture comparison

## Scope and metric definitions

The primary comparison uses concurrency 30 because all six architectures have
two independent service starts under that load. Each run contains 60 sessions,
three turns per session, and 180 requests. The primary table therefore covers
360 requests per architecture and 2,160 requests in total.

The model is Qwen3-8B FP16 with TP1, native 32K context, no YaRN, and eight
NVIDIA L20 GPUs. Input bytes, conversation/turn identities, and output lengths
match across architectures. Every included run passed effective-configuration,
live process topology, GPU mapping, correctness, and cleanup audits.

- TTFT is client request start to first output token.
- ITL is AIPerf `inter_token_latency`, the mean generation interval for each
  request. ITL percentiles are percentiles across request-level means.
- P95 and P99 below use all 360 request records and AIPerf's linear percentile
  method.
- GPU utilization is the mean of physical-GPU telemetry samples inside the
  client request windows.

DP and PD use isolated Dynamo 1.4.1 with vLLM 0.26.0. PAP uses the development
implementation recorded by the experiment. Adding 7P1D changed only the shared
runner's supported topology list; it did not change the DP/PD model code.

## Concurrency-30 results

| Architecture | TTFT mean | ITL mean | Request throughput | GPU utilization |
| --- | ---: | ---: | ---: | ---: |
| DP8 | **2.208 s** | 77.383 ms | **2.948 req/s** | 89.5% |
| PD 7P1D | 8.888 s | 66.967 ms | 1.824 req/s | 54.0% |
| PD 6P2D | 5.698 s | 48.991 ms | 2.680 req/s | 90.1% |
| PD 4P4D | 10.980 s | 28.375 ms | 2.034 req/s | 89.5% |
| PD 2P6D | 34.734 s | **25.114 ms** | 0.699 req/s | 43.3% |
| PAP 7PA1P | **5.364 s** | **44.850 ms** | **2.892 req/s** | **90.1%** |

PAP is the best measured balance of TTFT, ITL, and throughput. Relative to PD
6P2D, PAP has 5.86% lower mean TTFT, 8.45% lower mean ITL, and 7.94% higher
request throughput. Relative to DP8, PAP has 42.04% lower mean ITL and only
1.88% lower throughput, but its mean TTFT is 3.157 seconds higher.

PD 7P1D is dominated by both PAP and PD 6P2D: either alternative has lower
TTFT, lower ITL, and higher throughput. PD 2P6D obtains the lowest ITL at the
cost of very high TTFT and low throughput. PD 4P4D is the practical option
when an ITL target near 30 ms is more important than TTFT.

The measured TTFT/ITL Pareto frontier is DP8, PAP 7PA1P, PD 4P4D, and PD 2P6D.
Which point is usable depends on the service SLO rather than a single latency
average.

## Tail latency

| Architecture | TTFT P95 | TTFT P99 | ITL P95 | ITL P99 |
| --- | ---: | ---: | ---: | ---: |
| DP8 | 5.661 s | 7.909 s | 132.822 ms | 162.697 ms |
| PD 7P1D | 13.654 s | 14.603 s | 74.498 ms | 79.063 ms |
| PD 6P2D | 9.822 s | **10.703 s** | 66.042 ms | 69.246 ms |
| PD 4P4D | 14.446 s | 15.930 s | 33.553 ms | 35.623 ms |
| PD 2P6D | 60.652 s | 65.560 s | **27.223 ms** | **27.488 ms** |
| PAP 7PA1P | **9.557 s** | 12.245 s | **53.583 ms** | **57.026 ms** |

PAP improves mean, P95 TTFT, and ITL tails over 6P2D. Its TTFT P99 is 14.41%
higher than 6P2D, so a small set of PAP requests still has an unexplained tail.
The current aggregate data localizes this as a tail problem but does not prove
its mechanism.

## Multi-turn behavior

| Architecture | Turn 1 TTFT | Turn 2 TTFT | Turn 3 TTFT |
| --- | ---: | ---: | ---: |
| DP8 | 3.159 s | 1.736 s | 1.727 s |
| PD 7P1D | 7.905 s | 7.978 s | 10.781 s |
| PD 6P2D | 4.663 s | 6.159 s | 6.273 s |
| PD 4P4D | 8.878 s | 11.719 s | 12.343 s |
| PD 2P6D | 27.903 s | 25.272 s | 51.028 s |
| PAP 7PA1P | 5.951 s | 5.180 s | 4.962 s |

PAP and DP8 improve after the first turn. All measured PD splits except the
second turn of 2P6D become slower in later turns. This is a directly measured
trend; assigning it to queueing, cache reuse, or transfer behavior requires a
request-ID timeline across those boundaries.

## Resource-balance evidence

Per-GPU utilization identifies both extreme PD bottlenecks:

- In 7P1D, the sole Decode GPU averages 96.5% utilization while the seven
  Prefill GPUs average between 43.7% and 55.0%. Decode is saturated while
  Prefill capacity is idle.
- In 2P6D, the two Prefill GPUs average 97.3% and 96.0%; the six Decode GPUs
  average between 21.0% and 30.9%. Prefill is saturated while Decode capacity
  is idle.
- 6P2D and 4P4D keep all physical GPUs relatively busy but choose different
  TTFT/ITL tradeoffs.
- PAP's eight physical GPUs average between 87.4% and 93.0%. The telemetry is
  physical-GPU level and cannot attribute utilization separately to PAP
  Prefill and Attention processes sharing a card.

DP8 has the best TTFT and throughput but the worst ITL and ITL tail. Mixed
Prefill/Decode execution is consistent with this result, but the current GPU
telemetry alone does not establish interference as the cause.

## Concurrency sensitivity

The earlier concurrency-60 comparison contains the same five original
architectures but no 7P1D run:

| Architecture | C60 TTFT | C60 ITL | C30 TTFT change | C30 ITL change |
| --- | ---: | ---: | ---: | ---: |
| DP8 | 3.822 s | 144.372 ms | -42.24% | -46.40% |
| PD 6P2D | 15.084 s | 45.904 ms | -62.22% | +6.73% |
| PD 4P4D | 30.134 s | 26.368 ms | -63.56% | +7.61% |
| PD 2P6D | 65.443 s | 25.113 ms | -46.92% | +0.01% |
| PAP 7PA1P | 14.537 s | 43.524 ms | -63.10% | +3.05% |

Reducing concurrency mainly improves TTFT for disaggregated architectures;
their mean ITL changes by 0% to 8%. DP8 improves both TTFT and ITL sharply,
showing much stronger load sensitivity in token cadence.

The C60 and C30 PAP measurements straddle the recorded replacement of
`torch.cuda.synchronize()` with the required
`torch.accelerator.synchronize()` API at the Prefill retirement boundary. The
CUDA behavior was validated, but the cross-concurrency PAP comparison is not a
strict one-variable causality experiment.

## SLO interpretation

The architecture ranking under simultaneous TTFT, ITL, and E2E constraints is
reported in [GOODPUT.md](GOODPUT.md). The bundled AIPerf example SLO is not
suitable for this native-32K input distribution. Diagnostic long-context SLO
points show that PAP wins when ITL up to 75 ms is acceptable, whereas 4P4D is
the only architecture meeting 95% attainment at the tested 50 ms ITL point.

Only concurrency 30 and 60 have been measured. These fixed-load results do not
establish maximum sustainable goodput; that requires a load sweep around one
product-defined SLO knee.
