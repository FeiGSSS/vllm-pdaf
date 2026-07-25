# PAP versus PD versus fused DP: eight-GPU capacity scan

> Controlled development evidence. Every point is one clean repetition at
> commit `fb3a622b9`; final release claims still require three repetitions of
> the selected boundary points.

Date: 2026-07-25

## Result

Under this long-input, short-output multi-turn workload, PAP has the highest
SLO-compliant request goodput under Strict and Standard. Under Relaxed, fused
DP has the highest goodput, while PAP remains substantially ahead of PD.

| SLO | Best PAP | Best PD | Fused DP | PAP vs PD | PAP vs DP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict: TTFT 5 s, ITL 50 ms | **3.347 rps**, 7PA1P C16 | 2.442 rps, 6P2D C12 | 2.294 rps, C8 | **+37.0%** | **+45.9%** |
| Standard: TTFT 10 s, ITL 75 ms | **4.339 rps**, 6PA2P C32 | 3.596 rps, 6P2D C24 | 4.006 rps, C16 | **+20.7%** | **+8.3%** |
| Relaxed: TTFT 20 s, ITL 100 ms | 4.894 rps, 7PA1P C32 | 3.785 rps, 6P2D C24 | **5.181 rps**, C24 | **+29.3%** | -5.5% |

Goodput is eligible only when the point is complete and correct and at least
95% of all 640 requests meet both the TTFT and ITL limits. The table never
substitutes a high numerical goodput from an ineligible point.

The observed PAP Relaxed winner, 7PA1P C32, is a boundary result rather than a
stable default. Four C32 observations have relaxed-good fractions of 94.22%,
96.25%, 93.59%, and 97.81%. The repeat-stable PAP candidate is 6PA2P C32:
4.443 relaxed-good requests/s, 17.4% above PD and 14.2% below fused DP.

## Workload and validity

- Qwen3-8B FP16, TP1, eight NVIDIA L20 GPUs.
- 128 conversations, five turns, 640 requests at every point.
- Same AIPerf dataset SHA-256 in both scan phases:
  `4196b1f1b20afe38849c4c31926975dd14aa5b547241b72e146ac0c3f31ac028`.
- Mean initial user text: 8,020.7 tokens.
- Mean later-turn user text: 1,408.2 tokens.
- Randomized output: mean 32.4, median 31, range 16-64 tokens.
- Think/tool schedule: 0/3/3/1/3 seconds.
- Eager execution, `max_model_len=32768`, `max_num_seqs=256`.
- PA, PD, and fused replicas use `gpu_memory_utilization=0.90`.
  Projection uses automatic checkpoint-weight sizing.
- Conversation ownership is sticky for PAP, PD, and the fused replica pool.

Every included point completed 640/640 requests and passed correctness and
routing validation. PAP 7PA1P C48 completed the request records but contained
one `ClientPayloadError`; it is reported only as an overload boundary and is
ineligible for performance selection.

## Common-concurrency scaling

The following table shows all topologies at the common C16/C24/C32 points.
“ITL” is AIPerf's request-level mean inter-token latency, the TPOT-equivalent
decode metric used by this testbed.

| Architecture | Topology | C | Req/s | Output tok/s | TTFT avg / p95 | ITL avg / p95 | Strict | Standard | Relaxed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| PAP | 7PA1P | 16 | 3.416 | 110.70 | 1.09 / 2.90 s | 33.61 / 40.18 ms | pass | pass | pass |
| PAP | 7PA1P | 24 | 4.311 | 139.67 | 1.41 / 3.93 s | 42.89 / 76.07 ms | fail | fail | pass |
| PAP | 7PA1P | 32 | 5.003 | 162.10 | 1.88 / 5.98 s | 47.84 / 82.59 ms | fail | fail | pass* |
| PAP | 6PA2P | 16 | 3.268 | 105.88 | 1.46 / 4.05 s | 30.17 / 35.25 ms | pass | pass | pass |
| PAP | 6PA2P | 24 | 4.048 | 131.16 | 2.08 / 6.36 s | 32.12 / 38.35 ms | fail | pass | pass |
| PAP | 6PA2P | 32 | 4.443 | 143.96 | 3.14 / 8.56 s | 32.81 / 39.64 ms | fail | pass | pass |
| PD | 4P4D | 16 | 2.878 | 93.24 | 2.44 / 6.96 s | 25.75 / 29.91 ms | fail | pass | pass |
| PD | 4P4D | 24 | 3.430 | 111.15 | 3.61 / 10.29 s | 27.05 / 31.29 ms | fail | fail | pass |
| PD | 4P4D | 32 | 3.755 | 121.67 | 4.98 / 21.82 s | 27.73 / 32.14 ms | fail | fail | fail |
| PD | 6P2D | 16 | 3.059 | 99.10 | 2.02 / 5.50 s | 28.76 / 34.43 ms | fail | pass | pass |
| PD | 6P2D | 24 | 3.785 | 122.65 | 2.91 / 9.76 s | 30.99 / 38.67 ms | fail | pass | pass |
| PD | 6P2D | 32 | 2.580 | 83.60 | 7.95 / 21.54 s | 41.07 / 106.72 ms | fail | fail | fail |
| DP | replicas ×8 | 16 | 4.156 | 134.65 | 0.64 / 1.80 s | 32.12 / 67.90 ms | fail | pass | pass |
| DP | replicas ×8 | 24 | 5.383 | 174.41 | 0.80 / 2.21 s | 39.95 / 93.58 ms | fail | fail | pass |
| DP | replicas ×8 | 32 | 6.567 | 212.79 | 0.98 / 2.87 s | 47.86 / 112.09 ms | fail | fail | fail |

`pass*` marks the repeat-unstable 7PA1P C32 Relaxed point.

From C16 to C32, raw request throughput changes by:

| Topology | Raw throughput change |
| --- | ---: |
| Fused DP ×8 | +58.0% |
| PAP 7PA1P | +46.4% |
| PAP 6PA2P | +36.0% |
| PD 4P4D | +30.5% |
| PD 6P2D | -15.6% |

The raw-throughput winner is fused DP at every common concurrency. Its TTFT is
also the lowest. Its SLO capacity is limited by decode latency: ITL p95 grows
from 67.9 ms at C16 to 112.1 ms at C32.

PAP 7PA1P gives the best PAP raw throughput and TTFT. Its single
seven-PA fan-in domain creates a substantially larger join tail, so ITL grows
more quickly and C32 is not repeat-stable. PAP 6PA2P trades 11.2% C32 raw
throughput for a much lower ITL p95, 39.64 versus 82.59 ms, and is the only
topology that passes Standard at C32.

PD 4P4D keeps ITL nearly flat as concurrency grows, but TTFT p95 rises from
6.96 to 21.82 seconds; Prefill queueing defines its capacity. PD 6P2D is
better at C16/C24 for this long-input workload, but two Decode nodes saturate
at C32: throughput falls 31.8% from C24, while ITL p95 rises to 106.72 ms.

## Strict-boundary refinement

The main C16/C24/C32/C48 scan left PD and DP without a Strict passing point.
The same dataset and commit were therefore used for a small C8/C12/C20
refinement:

| Architecture | Topology | C | Req/s | TTFT avg / p95 | ITL avg / p95 | Strict good fraction | Strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PAP | 7PA1P | 20 | 3.991 | 1.25 / 3.56 s | 37.60 / 50.58 ms | 93.28% | fail |
| PAP | 6PA2P | 20 | 3.665 | 1.84 / 4.67 s | 30.84 / 37.25 ms | 93.91% | fail |
| PD | 4P4D | 8 | 1.771 | 1.65 / 4.81 s | 24.53 / 27.08 ms | 95.94% | pass |
| PD | 4P4D | 12 | 2.406 | 1.98 / 5.74 s | 25.08 / 28.19 ms | 91.72% | fail |
| PD | 6P2D | 8 | 1.785 | 1.57 / 4.56 s | 25.83 / 29.37 ms | 96.72% | pass |
| PD | 6P2D | 12 | 2.533 | 1.69 / 4.76 s | 27.12 / 30.89 ms | 96.41% | pass |
| DP | replicas ×8 | 8 | 2.353 | 0.53 / 1.53 s | 25.29 / 26.23 ms | 97.50% | pass |
| DP | replicas ×8 | 12 | 3.275 | 0.63 / 1.66 s | 29.33 / 56.18 ms | 93.44% | fail |

PAP's Strict advantage comes from admitting more simultaneous conversations
while keeping the joint TTFT/ITL pass fraction above 95%. DP keeps excellent
TTFT, but its C12 ITL tail crosses the 50-ms Strict limit. PD 6P2D reaches C12
with lower ITL but less aggregate throughput.

## Overload boundary

- PAP 6PA2P C48 remains correct but drops to 4.047 requests/s. Its ITL p95
  stays at 42.95 ms; TTFT p95 reaches 20.73 seconds, so Relaxed misses by five
  requests.
- PAP 7PA1P C48 reaches 5.118 raw requests/s, but ITL p95 reaches 175.40 ms
  and one payload error makes the point ineligible.
- PD 4P4D and 6P2D both fail Relaxed at C32, so C48 is skipped.
- Fused DP fails Relaxed at C32, so C48 is skipped.

## Decisions

1. Use 6PA2P C32 as the latency-stable PAP development baseline.
2. Retain 7PA1P as a throughput-oriented topology; do not use its C32 Relaxed
   result for a release claim until the fan-in tail is improved or three
   repetitions pass.
3. Use 6P2D as the best PD topology at C12/C24 for this workload. Its C32
   Decode collapse makes 4P4D the safer high-concurrency PD shape, but neither
   passes Relaxed there.
4. Report fused DP as the Relaxed winner and raw-throughput winner. PAP's
   demonstrated advantage is Strict/Standard goodput and capacity, not a
   universal win over fused deployment.
5. If promoting these results, repeat only the selected points three times:
   PAP 7PA1P C16, PAP 6PA2P C32, PD 6P2D C12/C24, and DP C8/C16/C24.

Raw artifacts remain machine-local under:

- `benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_fullscan_c16_48`
- `benchmarks/pap/experiments/_staging/capacity/20260725_8gpu_capacity_strict_refinement`

