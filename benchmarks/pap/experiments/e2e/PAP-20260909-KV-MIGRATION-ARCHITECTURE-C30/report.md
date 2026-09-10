# Single-host architecture comparison after PAP KV migration

## Scope and comparability

This comparison replaces the earlier PAP point with two new
PAP 7PA1P runs that enable same-host PA KV migration. The other points are the
immutable DP and PD archives from the aligned concurrency-30 experiments.

Every architecture has two independent service starts and 360 requests. All
runs use Qwen3-8B FP16, eight NVIDIA L20 GPUs, native 32K context, the same
60-session three-turn input, closed-loop concurrency 30, block size 16,
Prefill budget 2048, Decode/Projection budget 256, and no Hugging Face model
override. The AIPerf `inputs.json` SHA256 is
`ece4ff1e413301990a84a1a4f43b7a65ae45b150c023a2f43e725c99ef7032c5`
for all twelve runs. Conversation/turn identities and output lengths match.

This audit found that experiment 3's original r5/r6 performance runs inherited
the PAP runner's YaRN factor-4 default, while the DP/PD controls explicitly
disabled model overrides. Those runs remain valid for their one-variable
KV-migration A/B, but they are excluded here. The replacement PAP r7/r8 runs
set `PAP_HF_OVERRIDES=''` and passed service, topology, routing, migration,
correctness, output, and cleanup audits.

## Mean performance

| Architecture | TTFT | ITL | E2E | Request throughput |
| --- | ---: | ---: | ---: | ---: |
| DP8 | **2.208 s** | 77.383 ms | 8.381 s | 2.947 req/s |
| PD 7P1D | 8.888 s | 66.967 ms | 14.280 s | 1.824 req/s |
| PD 6P2D | 5.698 s | 48.991 ms | 9.681 s | 2.680 req/s |
| PD 4P4D | 10.980 s | 28.375 ms | 13.275 s | 2.034 req/s |
| PD 2P6D | 34.734 s | **25.114 ms** | 36.759 s | 0.699 req/s |
| PAP 7PA1P + KV migration | 4.504 s | 45.227 ms | **8.164 s** | **3.194 req/s** |

The new PAP result has the highest request and output throughput. Relative to
DP8, PAP has 8.37% higher request throughput, 41.55% lower ITL, and 2.60% lower
E2E latency, while TTFT is 104.02% higher. DP8 remains the low-TTFT point.

PAP now dominates both Decode-heavy 7P1D and the balanced 6P2D split on all
four mean metrics:

- Versus PD 6P2D: TTFT -20.96%, ITL -7.68%, E2E -15.67%, throughput +19.19%.
- Versus PD 7P1D: TTFT -49.32%, ITL -32.46%, E2E -42.83%, throughput +75.11%.

PD 4P4D and 2P6D retain lower ITL because they dedicate more GPUs to Decode,
but both pay much higher TTFT, E2E latency, and lower throughput. The measured
TTFT/ITL/throughput Pareto frontier is DP8, PAP, PD 4P4D, and PD 2P6D.

For context, replacing the older aligned PAP point with this result changes
mean TTFT from 5.364 to 4.504 seconds and throughput from 2.892 to 3.194 req/s.
That historical difference is not a one-variable A/B; experiment 3's paired
control/migration runs remain the causal evidence for the KV-copy benefit.

## Tail latency

| Architecture | TTFT P95 | TTFT P99 | ITL P95 | ITL P99 | E2E P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DP8 | **5.661 s** | **7.909 s** | 132.822 ms | 162.697 ms | 21.393 s |
| PD 7P1D | 13.654 s | 14.603 s | 74.498 ms | 79.063 ms | 25.876 s |
| PD 6P2D | 9.822 s | 10.703 s | 66.042 ms | 69.246 ms | 18.754 s |
| PD 4P4D | 14.446 s | 15.930 s | 33.553 ms | 35.623 ms | 19.666 s |
| PD 2P6D | 60.652 s | 65.560 s | **27.223 ms** | **27.488 ms** | 67.581 s |
| PAP 7PA1P + KV migration | 8.613 s | 11.497 s | 51.774 ms | 52.667 ms | **16.274 s** |

PAP improves P95 TTFT by 12.31% and ITL P99 by 23.94% over PD 6P2D. Its TTFT
P99 is still 7.42% higher than 6P2D, so the last percentile remains a PAP
optimization target. PAP has the lowest E2E P99 of all six architectures.

## Multi-turn behavior

| Architecture | Turn 0 TTFT | Turn 1 TTFT | Turn 2 TTFT |
| --- | ---: | ---: | ---: |
| DP8 | 3.159 s | 1.736 s | **1.727 s** |
| PD 7P1D | 7.905 s | 7.978 s | 10.781 s |
| PD 6P2D | 4.663 s | 6.159 s | 6.273 s |
| PD 4P4D | 8.878 s | 11.719 s | 12.343 s |
| PD 2P6D | 27.903 s | 25.272 s | 51.028 s |
| PAP 7PA1P + KV migration | 5.688 s | 4.344 s | 3.481 s |

PAP is the only disaggregated architecture whose TTFT improves monotonically
across turns. Turn-2 TTFT is 38.81% below turn 0. Across the two PAP runs, 73 PA
changes triggered 73 successful exports with no miss or failure. Estimated
cache read was 64.90%, leaving 2,366,960 computed Prefill tokens over 360
requests.

## Diagnostic goodput

The SLO points are the same diagnostics used in the earlier aligned report:

- D1: TTFT <= 5 s, ITL <= 50 ms, E2E <= 10 s.
- D2: TTFT <= 10 s, ITL <= 75 ms, E2E <= 20 s.
- D3: TTFT <= 15 s, ITL <= 50 ms, E2E <= 20 s.
- D4: TTFT <= 15 s, ITL <= 75 ms, E2E <= 25 s.

Each entry is `goodput req/s (attainment)`. Bold entries meet the 95% default
feasibility threshold.

| Architecture | D1 | D2 | D3 | D4 |
| --- | ---: | ---: | ---: | ---: |
| DP8 | 0.516 (17.5%) | 1.465 (49.7%) | 0.598 (20.3%) | 1.465 (49.7%) |
| PD 7P1D | 0.106 (5.8%) | 1.008 (55.3%) | 0.117 (6.4%) | 1.697 (93.1%) |
| PD 6P2D | 0.454 (16.9%) | **2.553 (95.3%)** | 1.280 (47.8%) | **2.680 (100%)** |
| PD 4P4D | 0.136 (6.7%) | 0.559 (27.5%) | **1.955 (96.1%)** | **1.955 (96.1%)** |
| PD 2P6D | 0.025 (3.6%) | 0.052 (7.5%) | 0.070 (10.0%) | 0.070 (10.0%) |
| PAP + KV migration | 1.446 (45.3%) | **3.079 (96.4%)** | 2.475 (77.5%) | **3.194 (100%)** |

No architecture is feasible at D1. At D2, PAP and PD 6P2D are feasible, and
PAP goodput is 20.58% higher. At D3, PAP has the highest raw goodput but misses
the 95% attainment requirement; PD 4P4D is the only feasible point because of
its lower ITL. At D4, PAP has the highest feasible goodput.

## Repetition stability and evidence

The two aligned PAP runs produced mean TTFT of 4.585 and 4.423 seconds,
mean ITL of 44.780 and 45.673 ms, and throughput of 3.184 and 3.204 req/s.
The result is therefore not driven by one service start.

`results/archives/` contains the two new PAP runs. The DP/PD raw bytes remain
in `PAP-20260907-ALIGNED-C30` and `PAP-20260907-ALIGNED-C30-7P1D`; the verifier
checks those immutable manifests, extracts all twelve runs, confirms the common
configuration and output identities, and reproduces `summary.json` and
`requests.csv`.
